import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from safari.models.sequence.h3 import H3

from safari.models.sequence.hyena_components import HyenaOperator

try:
    from safari.ops.fftconv import fftconv_func

    HAS_FFTCONV = True
except ImportError:
    print(
        "Warning: fftconv CUDA extension not available, using reference implementation"
    )
    HAS_FFTCONV = False
    fftconv_func = None

from s4dd.module_library.sequence_model import SequenceModel

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba3 import Mamba3Config, Mamba3LMHeadModel


class Mamba3Model(nn.Module):
    """CLM wrapper around the Mamba-3 SSM architecture.

    Mamba-3 improves over Mamba-2 with:
      - Trapezoidal discretization (second-order accurate state update)
      - Complex-valued SSM via data-dependent RoPE (enables state-tracking)
      - MIMO formulation (better hardware utilisation during decode)
      - QK-Normalisation on B, C projections
      - Learnable BC bias (head-specific, channel-wise, init to ones)
      - No short convolution (trapezoidal + bias removes the need for conv1d)

    The backbone is imported from the mamba3-minimal package:
      https://github.com/GuptaVishu2002/mamba3-minimal/tree/fix-packaging

    Architecture follows Llama design:
      Embedding → N × [RMSNorm → Mamba3 → RMSNorm → SwiGLU] → RMSNorm → LM Head

    Interface mirrors MambaModel / Mamba2Model so it is a drop-in replacement
    inside train_models_RNN.py.
    """

    def __init__(
        self,
        vocabulary,
        n_layers: int = 4,
        model_dim: int = 256,
        d_state: int = 128,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        dropout: float = 0.1,
        max_len: int = 250,
        use_mimo: bool = False,
        mimo_rank: int = 4,
        **kwargs,
    ):
        super(Mamba3Model, self).__init__()

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # Hyperparameters (stored for repr / checkpointing)
        self.model_dim = model_dim
        self.d_state = d_state
        self.headdim = headdim
        self.expand = expand
        self.chunk_size = chunk_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.max_len = max_len
        self.use_mimo = use_mimo
        self.mimo_rank = mimo_rank

        # ── Validate headdim ──────────────────────────────────────────────────
        d_inner = expand * model_dim
        assert (
            d_inner % headdim == 0
        ), f"d_inner (expand*model_dim = {d_inner}) must be divisible by headdim ({headdim})"
        assert (
            d_state % 2 == 0
        ), f"d_state ({d_state}) must be even for complex SSM / RoPE pairing"

        # ── Build Mamba-3 backbone ────────────────────────────────────────────
        # Mamba3LMHeadModel has its own embedding + LM head, but we replace the
        # embedding with one that uses our vocabulary's padding index, and we
        # repurpose the LM head as our output projection.
        cfg = Mamba3Config(
            d_model=model_dim,
            n_layer=n_layers,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            chunk_size=chunk_size,
            vocab_size=self.vocabulary_size,
            pad_vocab_size_multiple=1,  # exact size; no padding of vocab
            use_mimo=use_mimo,
            mimo_rank=mimo_rank,
        )
        self.backbone = Mamba3LMHeadModel(cfg, device=str(self.device))

        # Replace the embedding so we can honour padding_idx.
        # (Mamba3LMHeadModel does not expose padding_idx in its Embedding.)
        self.backbone.backbone.embedding = nn.Embedding(
            self.vocabulary_size,
            model_dim,
            padding_idx=self.padding_idx,
        ).to(self.device)

        # Re-tie lm_head weights to the new embedding.
        self.backbone.lm_head.weight = self.backbone.backbone.embedding.weight

        # ── Dropout (applied after each residual block output) ───────────────
        self.dropout_layer = nn.Dropout(dropout)

        # ── Loss function ─────────────────────────────────────────────────────
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # Move everything to the target device
        if torch.cuda.is_available():
            self.cuda()

    # ──────────────────────────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch_size, seq_len) — token indices

        Returns
        -------
        logits : (batch_size, seq_len, vocab_size)
        """
        # Mamba3LMHeadModel.forward returns (logits, inference_caches).
        # We only need logits during training / teacher-forced evaluation.
        logits, _ = self.backbone(x, h=None)
        return logits

    # ──────────────────────────────────────────────────────────────────────────
    # loss
    # ──────────────────────────────────────────────────────────────────────────

    def loss(self, batch) -> torch.Tensor:
        """Compute mean cross-entropy loss over non-padding positions.

        The collate function returns tensors in (seq_len, batch_size) layout;
        we transpose to (batch_size, seq_len) and call .contiguous() to avoid
        stride errors inside the Mamba-3 SSD kernels.
        """
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate returns (seq_len, batch_size) → (batch_size, seq_len).
        # .contiguous() is required: transpose() only swaps strides without
        # copying memory, which can cause stride-alignment errors in SSD.
        padded = padded.transpose(0, 1).contiguous()

        # Mamba-3's chunked SSD requires seqlen to be a multiple of chunk_size.
        # Pad the sequence dimension if necessary.
        seq_len = padded.shape[1]
        remainder = seq_len % self.chunk_size
        if remainder != 0:
            pad_len = self.chunk_size - remainder
            padded = F.pad(padded, (0, pad_len), value=self.padding_idx)

        logits = self(padded)  # (batch_size, padded_seq_len, vocab_size)

        # Teacher-forced targets: shift right by one position.
        # Trim both to the original (unpadded) seq_len - 1 so padding tokens
        # never contribute to the loss (CrossEntropyLoss ignores padding_idx).
        targets = padded[:, 1:seq_len]  # (batch_size, seq_len - 1)
        logits = logits[
            :, : seq_len - 1, :
        ]  # (batch_size, seq_len - 1, vocab_size)

        loss = 0.0
        for char_idx in range(targets.shape[1]):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    # ──────────────────────────────────────────────────────────────────────────
    # sample
    # ──────────────────────────────────────────────────────────────────────────

    def sample(
        self,
        *,
        n_sequences: int,
        max_len: int = None,
        return_smiles: bool = True,
        return_losses: bool = False,
        descriptors=None,
    ):
        """Auto-regressively sample sequences from the model.

        Uses the Mamba-3 inference cache (constant-time per step) so generation
        is efficient: the full prefix is processed in one chunked forward pass,
        then each new token is decoded in O(1) via the recurrent step path.

        Parameters
        ----------
        n_sequences : int
            Number of sequences to generate in parallel.
        max_len : int, optional
            Maximum generation length (defaults to self.max_len).
        return_smiles : bool
            Decode token sequences to SMILES strings if True.
        return_losses : bool
            Also return per-sequence NLL losses if True.
        descriptors : ignored
            Accepted for API compatibility with ConditionalRNN; has no effect.
        """
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        self.eval()

        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)
        finished = torch.zeros(
            n_sequences, dtype=torch.uint8, device=self.device
        )
        log_probs = torch.zeros(n_sequences, device=self.device)
        sequences: list[torch.Tensor] = []

        with torch.no_grad():
            # ── Initialise with SOS token; build inference cache ──────────────
            # Shape: (n_sequences, 1)
            current_ids = torch.full(
                (n_sequences, 1),
                start_token,
                dtype=torch.long,
                device=self.device,
            )

            # Process the SOS token through the chunked (non-inference) path to
            # initialise the per-layer caches h.  chunk_size=1 is allowed because
            # seqlen=1 is divisible by 1; alternatively we pad to chunk_size.
            # We pad to chunk_size and then use only the last logit.
            pad_len = self.chunk_size - 1
            if pad_len > 0:
                padded_ids = F.pad(current_ids, (pad_len, 0), value=pad_token)
            else:
                padded_ids = current_ids

            logits_init, h = self.backbone(padded_ids, h=None)
            # h is now a list of InferenceCache, one per layer.

            # Get the logit for the position corresponding to SOS (last position).
            logits_step = logits_init[:, -1, :]  # (n_sequences, vocab_size)
            logits_step = torch.clamp(logits_step, min=-1e4, max=1e4)
            prob = F.softmax(logits_step, dim=-1)

            if not (torch.isnan(prob).any() or torch.isinf(prob).any()):
                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                log_prob = F.log_softmax(logits_step, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses

                finished = torch.ge(finished + (outputs == stop_token), 1)
            else:
                # Fallback: emit SOS again (will be cleaned up by vocabulary.decode)
                outputs = current_ids.squeeze(1)

            # ── Auto-regressive generation using the recurrent step ───────────
            for _ in range(max_len - 1):
                if torch.prod(finished) == 1:
                    break

                # Shape: (n_sequences, 1) — the previously sampled token
                next_ids = outputs.unsqueeze(1)

                # Constant-time step via inference cache
                logits_step, h = self.backbone(next_ids, h=h)
                logits_step = logits_step[:, -1, :]  # (n_sequences, vocab_size)
                logits_step = torch.clamp(logits_step, min=-1e4, max=1e4)
                prob = F.softmax(logits_step, dim=-1)

                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                log_prob = F.log_softmax(logits_step, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses

                finished = torch.ge(finished + (outputs == stop_token), 1)

        seqs = (
            torch.cat(sequences, dim=1)
            if sequences
            else torch.full(
                (n_sequences, 1),
                start_token,
                dtype=torch.long,
                device=self.device,
            )
        )

        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            smiles = sequences

        if was_training:
            self.train()

        if return_losses:
            return smiles, log_probs.detach().cpu().numpy()
        return smiles


class MambaModel(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_layers=4,
        model_dim=256,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        max_len=250,
        **kwargs,
    ):
        super(MambaModel, self).__init__()

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # Hyperparameters
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layers = n_layers
        self.expand = expand
        self.dropout = dropout
        self.max_len = max_len

        # Model components
        padding_t = torch.tensor(self.padding_idx).to(self.device)
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.model_dim, padding_idx=padding_t
        )

        # Stack of Mamba layers
        # This module uses roughly 3 * expand * d_model^2 parameters
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=self.model_dim,  # Model dimension d_model
                    d_state=self.d_state,  # SSM state expansion factor
                    d_conv=self.d_conv,  # Local convolution width
                    expand=self.expand,  # Block expansion factor
                )
                for _ in range(n_layers)
            ]
        )

        # Layer norm for each layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.model_dim) for _ in range(n_layers)]
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Output projection
        self.output_embedding = nn.Linear(self.model_dim, self.vocabulary_size)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # Final layer norm applied after all Mamba layers, before output projection
        self.final_norm = nn.LayerNorm(self.model_dim)

        # Move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Embed
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)

        # Apply Mamba layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x)  # Mamba expects (B, L, D)
            x = self.dropout_layer(x)
            x = x + residual

        x = self.final_norm(x)
        # Project to vocabulary
        logits = self.output_embedding(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def loss(self, batch):
        """Compute loss for a batch."""
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate always returns (seq_len, batch_size); transpose to (batch_size, seq_len)
        padded = padded.transpose(0, 1)

        logits = self(padded)

        targets = padded[:, 1:]
        logits = logits[:, :-1, :]

        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])
        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        self.eval()

        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        inputs = (
            torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        )
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []

        with torch.no_grad():
            for step in range(max_len):
                if step == 0:
                    current_seq = inputs.unsqueeze(1)
                else:
                    seq_list = [inputs.unsqueeze(1)] + sequences
                    current_seq = torch.cat(seq_list, dim=1)

                logits = self(current_seq)[:, -1, :]
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)

                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses

                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # ← added empty-sequence fallback, matches H3/Hyena/S4
        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )

        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            smiles = sequences

        if was_training:
            self.train()

        if return_losses:
            return smiles, log_probs.detach().cpu().numpy()
        else:
            return smiles


class Mamba2Model(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_layers=4,
        model_dim=256,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1,
        max_len=250,
        **kwargs,
    ):
        super(Mamba2Model, self).__init__()

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # Hyperparameters
        self.model_dim = model_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_layers = n_layers
        self.expand = expand
        self.dropout = dropout
        self.max_len = max_len

        # Model components
        padding_t = torch.tensor(self.padding_idx).to(self.device)
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.model_dim, padding_idx=padding_t
        )

        # Stack of Mamba2 layers
        # causal_conv1d_cuda requires d_in_proj % 8 == 0, where:
        #   d_in_proj = 2*expand*d_model + 2*d_state + nheads
        #   nheads    = (expand*d_model) // headdim
        # Find the largest headdim (power-of-2) that satisfies this.
        d_inner = int(self.expand * self.model_dim)
        _headdim = None
        for hd in [64, 32, 16, 8]:
            if d_inner % hd != 0:
                continue
            nheads_candidate = d_inner // hd
            if (2 * d_inner + 2 * self.d_state + nheads_candidate) % 8 == 0:
                _headdim = hd
                break
        if _headdim is None:
            raise ValueError(
                f"No valid headdim found for model_dim={self.model_dim}, "
                f"expand={self.expand}, d_state={self.d_state}. "
                f"Ensure (2*expand*d_model + 2*d_state + nheads) is divisible by 8."
            )
        self.mamba2_layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=self.model_dim,  # Model dimension d_model
                    d_state=self.d_state,  # SSM state expansion factor, typically 64 or 128
                    d_conv=self.d_conv,  # Local convolution width
                    expand=self.expand,  # Block expansion factor
                    headdim=_headdim,
                )
                for _ in range(n_layers)
            ]
        )

        # Layer norm for each layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.model_dim) for _ in range(n_layers)]
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Output projection
        self.output_embedding = nn.Linear(self.model_dim, self.vocabulary_size)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # Final layer norm applied after all Mamba2 layers, before output projection
        self.final_norm = nn.LayerNorm(self.model_dim)

        # Move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Embed
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)

        # Apply Mamba2 layers with residual connections
        for mamba2_layer, layer_norm in zip(
            self.mamba2_layers, self.layer_norms
        ):
            residual = x
            x = layer_norm(x)
            x = mamba2_layer(x)  # Mamba2 expects (B, L, D)
            x = self.dropout_layer(x)
            x = x + residual

        x = self.final_norm(x)
        # Project to vocabulary
        logits = self.output_embedding(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def loss(self, batch):
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate returns (seq_len, batch_size); transpose to (batch_size, seq_len).
        # .contiguous() is critical: transpose() only swaps strides without copying
        # memory. Non-standard strides propagate through nn.Embedding into
        # causal_conv1d, causing: RuntimeError: strides must be multiples of 8
        padded = padded.transpose(0, 1).contiguous()

        logits = self(padded)

        targets = padded[:, 1:]
        logits = logits[:, :-1, :]

        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])
        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        self.eval()

        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        inputs = (
            torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        )
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []

        with torch.no_grad():
            for step in range(max_len):
                if step == 0:
                    current_seq = inputs.unsqueeze(1)
                else:
                    seq_list = [inputs.unsqueeze(1)] + sequences
                    current_seq = torch.cat(seq_list, dim=1)

                logits = self(current_seq)[:, -1, :]
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)

                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses

                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )

        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            smiles = sequences

        if was_training:
            self.train()

        if return_losses:
            return smiles, log_probs.detach().cpu().numpy()
        else:
            return smiles


class H3Model(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_layers=4,
        model_dim=256,
        state_dim=64,
        head_dim=1,
        dropout=0.1,
        max_len=250,
        use_fast_fftconv=False,
        # SSM kernel parameters
        measure="diag-lin",
        mode="diag",
        lr=None,
        **kernel_args,
    ):
        super(H3Model, self).__init__()

        # Device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # Hyperparameters
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.head_dim = head_dim
        self.dropout = dropout
        self.max_len = max_len
        self.use_fast_fftconv = use_fast_fftconv and HAS_FFTCONV
        self.measure = measure
        self.mode = mode

        # Model components
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.model_dim, padding_idx=self.padding_idx
        )

        # Stack of H3 layers using actual Safari implementation
        self.h3_layers = nn.ModuleList(
            [
                H3(
                    d_model=self.model_dim,
                    d_state=self.state_dim,
                    l_max=max_len,
                    head_dim=self.head_dim,
                    use_fast_fftconv=self.use_fast_fftconv,
                    dropout=self.dropout,
                    layer_idx=i,
                    mode=self.mode,  # Use S4D variant
                    measure=self.measure,
                    lr=None if lr == 0.0 else lr,
                    **kernel_args,
                )
                for i in range(n_layers)
            ]
        )

        # Layer norm for each layer
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(self.model_dim) for _ in range(n_layers)]
        )

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Final layer norm applied after all H3 layers, before output projection.
        self.final_norm = nn.LayerNorm(self.model_dim)

        # Output projection
        self.output_embedding = nn.Linear(self.model_dim, self.vocabulary_size)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # Move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Embed
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)

        # Apply H3 layers with residual connections
        for h3_layer, layer_norm in zip(self.h3_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = h3_layer(x)  # H3 expects (B, L, H)
            x = self.dropout_layer(x)
            x = x + residual

        # Final layer norm before output projection
        x = self.final_norm(x)

        # Project to vocabulary
        logits = self.output_embedding(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def loss(self, batch):
        """Compute loss for a batch."""
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate always returns (seq_len, batch_size); transpose to (batch_size, seq_len)
        padded = padded.transpose(0, 1)

        # Forward pass
        logits = self(padded)  # (batch_size, seq_len, vocab_size)

        # Calculate loss (predict next token)
        targets = padded[:, 1:]  # (batch_size, seq_len-1)
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)

        # Compute loss
        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])

        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        """Sample sequences from the model."""
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        self.eval()

        # Get tokens
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        # Initialize
        inputs = (
            torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        )

        # Loss function
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        # Sampling loop
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []

        with torch.no_grad():
            for step in range(max_len):
                # Get logits for all sequences so far
                if step == 0:
                    current_seq = inputs.unsqueeze(1)  # (n_sequences, 1)
                else:
                    # Build full sequence so far
                    seq_list = [inputs.unsqueeze(1)] + sequences
                    current_seq = torch.cat(
                        seq_list, dim=1
                    )  # (n_sequences, step+1)

                # Forward pass
                logits = self(current_seq)  # (n_sequences, step+1, vocab_size)
                logits = logits[
                    :, -1, :
                ]  # Get last position (n_sequences, vocab_size)

                # Clamp and sample
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)

                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                # Calculate NLL
                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)

                # Zero losses if finished
                losses[finished.bool()] = 0
                log_probs += losses

                # Check if finished
                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # Concatenate sequences and decode
        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )
        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            smiles = sequences

        if was_training:
            self.train()

        if return_losses:
            return smiles, log_probs.detach().cpu().numpy()
        else:
            return smiles


class HyenaModel(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_layers=4,
        d_model=256,
        order=2,
        filter_order=64,
        n_order_heads=1,
        dropout=0.25,
        max_len=250,
        **hyena_args,
    ):
        super(HyenaModel, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vocabulary = vocabulary
        self.vocabulary_size = len(vocabulary)
        self.padding_idx = vocabulary.dictionary["<PAD>"]
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.max_len = max_len

        self.embedding = nn.Embedding(
            self.vocabulary_size, d_model, padding_idx=self.padding_idx
        )

        self.hyena_layers = nn.ModuleList(
            [
                HyenaOperator(
                    d_model=d_model,
                    l_max=max_len,
                    order=order,
                    filter_order=filter_order,
                    num_heads=n_order_heads,
                    dropout=dropout,
                    **hyena_args,
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(n_layers)]
        )

        self.dropout_layer = nn.Dropout(dropout)
        # Final layer norm applied after all Hyena layers, before output projection.
        self.final_norm = nn.LayerNorm(d_model)
        self.output_embedding = nn.Linear(d_model, self.vocabulary_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        x = self.embedding(x)

        for hyena_layer, layer_norm in zip(self.hyena_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = hyena_layer(x)
            x = self.dropout_layer(x)
            x = x + residual

        # Final layer norm before output projection
        x = self.final_norm(x)
        return self.output_embedding(x)

    def loss(self, batch):
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)
        # Collate always returns (seq_len, batch_size); transpose to (batch_size, seq_len)
        padded = padded.transpose(0, 1)

        logits = self(padded)
        targets = padded[:, 1:]
        logits = logits[:, :-1, :]

        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])
        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        self.eval()

        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        inputs = (
            torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        )
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []

        with torch.no_grad():
            for step in range(max_len):
                if step == 0:
                    current_seq = inputs.unsqueeze(1)
                else:
                    seq_list = [inputs.unsqueeze(1)] + sequences
                    current_seq = torch.cat(seq_list, dim=1)

                logits = self(current_seq)
                logits = logits[:, -1, :]

                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)

                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)
                sequences.append(outputs.view(-1, 1))

                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)
                losses[finished.bool()] = 0
                log_probs += losses

                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # Concatenate sequences and decode
        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )
        if return_smiles:
            outputs = [
                self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs
            ]
        else:
            outputs = sequences

        if was_training:
            self.train()

        if return_losses:
            return outputs, log_probs.detach().cpu().numpy()
        else:
            return outputs


class StructuredStateSpaceSequenceModel(nn.Module):
    def __init__(
        self,
        vocabulary,
        model_dim=256,
        state_dim=64,
        n_blocks=2,
        n_ssm=1,
        dropout=0.25,
        max_len=250,
    ):
        super(StructuredStateSpaceSequenceModel, self).__init__()

        # detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # hyperparams
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.n_blocks = n_blocks
        self.n_ssm = n_ssm
        self.dropout = dropout
        self.max_len = max_len

        # S4 layer configuration
        self.layer_config = [
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {
                "_name_": "s4",
                "d_state": self.state_dim,
                "n_ssm": self.n_ssm,
            },
            {"_name_": "ff"},
        ]
        self.pool_config = {"_name_": "pool", "stride": 1, "expand": None}

        # model components
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.model_dim, padding_idx=self.padding_idx
        )

        self.model = SequenceModel(
            d_model=self.model_dim,
            n_layers=self.n_blocks,
            transposed=False,  # Changed to False - expect (batch, length, dim)
            dropout=self.dropout,
            layer=self.layer_config,
            pool=self.pool_config,
        )

        self.output_embedding = nn.Linear(self.model_dim, self.vocabulary_size)
        self.recurrent_state = None

        # loss function (ignoring padding)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        batch_size, seq_len = x.size()

        # Embed the input
        x = self.embedding(x)  # (batch_size, seq_len, model_dim)

        # Pass through S4 model (without state in training mode)
        x, _ = self.model(x, state=None)

        # Project to vocabulary
        logits = self.output_embedding(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def reset_state(self, batch_size, device=None):
        if device is None:
            device = self.device
        self.recurrent_state = self.model.default_state(
            batch_size, device=device
        )

    def recurrent_step(self, x_t):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)

        x_t = self.embedding(x_t).squeeze(1)  # (batch_size, model_dim)
        x_t, state = self.model.step(x_t, state=self.recurrent_state)
        self.recurrent_state = state
        x_t = self.output_embedding(x_t)  # (batch_size, vocab_size)

        return x_t

    def loss(self, batch):
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate always returns (seq_len, batch_size); transpose to (batch_size, seq_len)
        padded = padded.transpose(0, 1)

        # batch_size = padded.shape[0]
        # seq_len = padded.shape[1]

        # Don't use recurrent state during training - use full convolution mode
        self.recurrent_state = None

        # Forward pass
        logits = self(padded)  # (batch_size, seq_len, vocab_size)

        # Calculate loss
        # Shift targets: predict next token
        targets = padded[:, 1:]  # (batch_size, seq_len-1)
        logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)

        # Reshape for loss calculation
        loss = 0.0
        actual_len = min(logits.shape[1], targets.shape[1])

        for char_idx in range(actual_len):
            loss += self.loss_fn(logits[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=None,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        if max_len is None:
            max_len = self.max_len

        was_training = self.training
        # IMPORTANT: Set model to eval mode before sampling
        self.eval()

        # Setup for recurrent mode
        for module in self.model.modules():
            if hasattr(module, "setup_step"):
                module.setup_step()

        # Reset state
        self.reset_state(n_sequences, device=self.device)

        # Get start/stop tokens
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        # Create start token tensor
        inputs = (
            torch.empty(n_sequences).fill_(start_token).long().to(self.device)
        )

        # Setup loss function
        loss_fn = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        # Sample sequences
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []

        with torch.no_grad():  # Also add no_grad for efficiency
            for step in range(max_len):
                # Get logits for current input
                logits = self.recurrent_step(inputs)

                # Clamp logits to prevent inf/nan
                logits = torch.clamp(logits, min=-1e4, max=1e4)

                # Sample from distribution
                prob = F.softmax(logits, dim=-1)

                # Check for invalid values
                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1).squeeze(1)

                sequences.append(outputs.view(-1, 1))

                # Calculate NLL
                log_prob = F.log_softmax(logits, dim=-1)
                losses = loss_fn(log_prob, outputs)

                # Zero losses if we are finished sampling
                losses[finished.bool()] = 0
                log_probs += losses

                # Update inputs for next step
                inputs = outputs

                # Track whether sampling is done for all molecules
                finished = torch.ge(finished + (outputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # Concatenate sequences and decode
        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )

        if return_smiles:
            outputs = [
                self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs
            ]
        else:
            outputs = sequences

        if was_training:
            self.train()

        # Optionally return losses
        if return_losses:
            return outputs, log_probs.detach().cpu().numpy()
        else:
            return outputs


class RNN(nn.Module):
    def __init__(
        self,
        vocabulary,
        rnn_type="LSTM",
        n_layers=3,
        embedding_size=1024,
        hidden_size=512,
        dropout=0,
    ):
        super(RNN, self).__init__()

        # detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)

        # embedding layer
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            self.vocabulary_size,
            self.embedding_size,
            padding_idx=self.padding_idx,
        )

        # RNN architecture
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
        )

        # dropout
        self.dropout = nn.Dropout(dropout)
        # decoder
        self.decoder = nn.Linear(self.hidden_size, self.vocabulary_size)

        # loss function (ignoring padding)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # initialize weights
        # self.init_weights()

        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def loss(self, batch):
        # extract the elements of a single minibatch
        padded, lengths, _ = batch  # ignore descriptors, if any
        # move to the gpu
        padded = padded.to(self.device)

        # embed the padded sequence
        embedded = self.embedding(padded)
        # -> embedded: max_len x batch_size x emb_size
        if self.dropout.p > 0:
            embedded = self.dropout(embedded)

        # now pack the embedded sequences
        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed)
        # unpack the output
        padded_output, output_lens = pad_packed_sequence(packed_output)
        # -> packed_output: max_len x batch_size x hidden_size

        # run LSTM output through decoder
        if self.dropout.p > 0:
            padded_output = self.dropout(padded_output)
        decoded = self.decoder(padded_output)
        # -> decoded: max_len x batch_size x vocab_len

        # finally, calculate loss
        loss = 0.0
        max_len = max(lengths)
        targets = padded[1:, :]
        for char_idx in range(max_len - 1):
            loss += self.loss_fn(decoded[char_idx], targets[char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        max_len=250,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        was_training = self.training
        self.eval()

        # get start/stop tokens
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        # create start token tensor
        inputs = (
            torch.empty(n_sequences)
            .fill_(start_token)
            .long()
            .view(1, n_sequences)
            .to(self.device)
        )
        # initialize hidden state
        if self.rnn_type == "LSTM":
            hidden = torch.zeros(
                self.n_layers, n_sequences, self.hidden_size
            ).to(self.device), torch.zeros(
                self.n_layers, n_sequences, self.hidden_size
            ).to(
                self.device
            )
        else:
            hidden = torch.zeros(
                self.n_layers, n_sequences, self.hidden_size
            ).to(self.device)

        # setup loss function
        loss = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        # sample sequences
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []
        with torch.no_grad():
            for step in range(max_len):
                embedded = self.embedding(inputs)
                output, hidden = self.rnn(embedded, hidden)
                logits = self.decoder(output)
                prob = F.softmax(logits, dim=2)
                inputs = torch.multinomial(prob.squeeze(0), num_samples=1).view(
                    1, -1
                )
                sequences.append(inputs.view(-1, 1))
                # calculate NLL too
                log_prob = F.log_softmax(logits.squeeze(0), dim=1)
                losses = loss(log_prob, inputs.squeeze(0))
                # zero losses if we are finished sampling
                losses[finished.squeeze(0).bool()] = 0
                log_probs += losses
                # track whether sampling is done for all molecules
                finished = torch.ge(finished + (inputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # concatenate sequences and decode
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            outputs = [
                self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs
            ]
        else:
            outputs = sequences

        if was_training:
            self.train()

        # optionally return losses
        if return_losses:
            return outputs, log_probs.detach().cpu().numpy()
        else:
            return outputs


class CausalSelfAttention(nn.Module):
    """adapted from nanoGPT, minGPT, jmtomczak"""

    def __init__(self, embedding_size=256, max_len=250, n_heads=8, dropout=0.1):
        super().__init__()

        self.embedding_size = embedding_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.dropout = dropout
        assert self.embedding_size % self.n_heads == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        # output projection
        self.c_proj = nn.Linear(self.embedding_size, self.embedding_size)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # from nanoGPT:
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )
        if not self.flash:
            print("WARNING: using slow attention")
            # causal mask to ensure that attention is only applied to the
            # left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(self.max_len, self.max_len)).view(
                    1, 1, self.max_len, self.max_len
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch_size, seq_len, emb_size

        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_size, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # -> (B, nh, T, hs)

        # causal self-attention; Self-attend:
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        )


class MLP(nn.Module):
    def __init__(
        self, embedding_size=256, exp_factor=4, dropout=0.1, bias=True
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.bias = bias
        self.c_fc = nn.Linear(
            self.embedding_size,
            self.exp_factor * self.embedding_size,
            bias=self.bias,
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            self.exp_factor * self.embedding_size,
            self.embedding_size,
            bias=self.bias,
        )
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        embedding_size=256,
        max_len=250,
        n_heads=8,
        exp_factor=4,
        dropout=0.1,
        bias=True,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.max_len = max_len
        self.n_heads = n_heads
        self.exp_factor = exp_factor
        self.dropout = dropout
        self.bias = bias

        self.ln_1 = LayerNorm(self.embedding_size, bias=self.bias)
        self.attn = CausalSelfAttention(
            embedding_size=self.embedding_size,
            max_len=self.max_len,
            n_heads=self.n_heads,
            dropout=self.dropout,
        )
        self.ln_2 = LayerNorm(self.embedding_size, bias=self.bias)
        self.mlp = MLP(
            embedding_size=self.embedding_size,
            exp_factor=self.exp_factor,
            dropout=self.dropout,
            bias=self.bias,
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocabulary,
        n_blocks=8,
        n_heads=8,
        embedding_size=256,
        max_len=250,
        dropout=0.1,
        exp_factor=4,
        bias=True,
    ):
        super(Transformer, self).__init__()

        # detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # vocabulary
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.padding_idx = self.vocabulary.dictionary["<PAD>"]

        # hyperparams
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.dropout = dropout
        self.exp_factor = exp_factor
        self.bias = bias
        # model itself
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    self.vocabulary_size,
                    self.embedding_size,
                    padding_idx=self.padding_idx,
                ),
                wpe=nn.Embedding(self.max_len, self.embedding_size),
                drop=nn.Dropout(self.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            embedding_size=self.embedding_size,
                            max_len=self.max_len,
                            n_heads=self.n_heads,
                            exp_factor=self.exp_factor,
                            dropout=self.dropout,
                            bias=self.bias,
                        )
                        for _ in range(self.n_blocks)
                    ]
                ),
                ln_f=LayerNorm(self.embedding_size, bias=self.bias),
            )
        )
        self.lm_head = nn.Linear(
            self.embedding_size, self.vocabulary_size, bias=False
        )
        # skip weight tying per MolGPT

        # loss function (ignoring padding)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )

        # initialize weights
        self.apply(self.init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_blocks)
                )

        # move to GPU
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        batch_size, seq_len = x.size()
        assert seq_len <= self.max_len

        # embeddings
        tok_emb = self.transformer.wte(x)
        # -> batch_size * seq_len * emb_size
        # position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos_emb = self.transformer.wpe(pos)
        # -> 1 * seq_len * emb_size

        # combine embeddings with dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # now forward embeddings through the transformer itself
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def loss(self, batch):
        # Collate always returns (padded, lengths, descriptors); descriptor ignored here
        padded, lengths, _ = batch

        padded = padded.to(self.device)

        # Collate always returns (seq_len, batch_size); transpose to (batch_size, seq_len)
        padded = padded.transpose(0, 1)

        # Get actual sequence length from batch
        actual_seq_len = padded.shape[1]

        decoded = self(padded)  # batch_size x seq_len x vocab_size

        loss = 0.0
        targets = padded[:, 1:]  # batch_size x (seq_len-1)

        # Loop only up to actual decoded sequence length minus 1
        for char_idx in range(
            min(actual_seq_len - 1, decoded.shape[1], targets.shape[1])
        ):
            loss += self.loss_fn(decoded[:, char_idx, :], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
        # Reset recurrent state before sampling
        # self.reset_state(n_sequences, device=self.device)

        was_training = self.training
        self.eval()
        torch.cuda.empty_cache()

        # get start/stop tokens
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        # create start token tensor
        inputs = (
            torch.empty(n_sequences)
            .fill_(start_token)
            .long()
            .view(n_sequences, 1)
            .to(self.device)
        )

        # setup loss function
        loss = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        # sample sequences
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []
        with torch.no_grad():
            for step in range(self.max_len):
                logits = self(inputs)[:, -1, :]
                # Clamp logits to prevent inf/nan
                logits = torch.clamp(logits, min=-1e4, max=1e4)
                prob = F.softmax(logits, dim=-1)

                # Check for invalid values and skip if found
                if torch.isnan(prob).any() or torch.isinf(prob).any():
                    break

                outputs = torch.multinomial(prob, num_samples=1)
                # append to growing sequence
                inputs = torch.cat((inputs, outputs), dim=1)
                sequences.append(outputs)
                # calculate NLL too
                log_prob = F.log_softmax(logits, dim=1)
                losses = loss(log_prob, outputs.squeeze(1))
                # zero losses if we are finished sampling
                losses[finished.bool()] = 0
                log_probs += losses
                # track whether sampling is done for all molecules
                finished = torch.ge(
                    finished + (outputs.squeeze(1) == stop_token), 1
                )
                if torch.prod(finished) == 1:
                    break

        # Concatenate sequences and decode
        seqs = (
            torch.cat(sequences, 1)
            if sequences
            else torch.empty(n_sequences, 1, dtype=torch.long)
            .fill_(start_token)
            .to(self.device)
        )
        if return_smiles:
            outputs = [
                self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs
            ]
        else:
            outputs = sequences

        torch.cuda.empty_cache()

        if was_training:
            self.train()

        # optionally return losses
        if return_losses:
            return outputs, log_probs.detach().cpu().numpy()
        else:
            return outputs

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ConditionalRNN(nn.Module):
    def __init__(
        self,
        vocabulary,
        rnn_type="LSTM",
        n_layers=3,
        embedding_size=128,
        hidden_size=512,
        dropout=0,
        num_descriptors=2,
        conditional_emb=False,
        conditional_emb_l=True,
        conditional_dec=False,
        conditional_dec_l=True,
        conditional_h=False,
    ):
        super(ConditionalRNN, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.conditional_emb = conditional_emb
        self.conditional_emb_l = conditional_emb_l
        self.conditional_dec = conditional_dec
        self.conditional_dec_l = conditional_dec_l
        self.conditional_h = conditional_h
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.num_descriptors = num_descriptors

        # Assert that conditional_emb_l and conditional_emb cannot both be true at the same time
        assert not (
            self.conditional_emb_l and self.conditional_emb
        ), "Both conditional_emb_l and conditional_emb cannot be true at the same time."

        # Assert that conditional_dec_l and conditional_dec cannot both be true at the same time
        assert not (
            self.conditional_dec_l and self.conditional_dec
        ), "Both conditional_dec_l and conditional_dec cannot be true at the same time."

        # Assert that at least one of the conditional* flags is set
        assert (
            self.conditional_emb_l
            or self.conditional_emb
            or self.conditional_dec_l
            or self.conditional_dec
            or self.conditional_h
        ), "At least one conditional parameter must be set for the conditional model"

        # set up input/output sizes for RNN
        rnn_input_size = self.embedding_size  # Default: embedding size
        rnn_output_size = self.hidden_size  # Default: hidden size

        # Determine rnn_input_size based on the conditions for conditional_emb_l and conditional_emb
        if self.conditional_emb_l and not self.conditional_emb:
            rnn_input_size = (
                self.embedding_size + self.embedding_size
            )  # Only add self.embedding_size
        elif not self.conditional_emb_l and self.conditional_emb:
            rnn_input_size = (
                self.embedding_size + self.num_descriptors
            )  # Add num_descriptors to self.embedding_size

        # Determine rnn_output_size based on the conditions for conditional_dec_l and conditional_dec
        if self.conditional_dec_l and not self.conditional_dec:
            rnn_output_size = (
                self.hidden_size + self.embedding_size
            )  # Only add self.embedding_size
        elif not self.conditional_dec_l and self.conditional_dec:
            rnn_output_size = (
                self.hidden_size + self.num_descriptors
            )  # Add num_descriptors to self.hidden_size

        self.padding_idx = self.vocabulary.dictionary["<PAD>"]
        self.embedding = nn.Embedding(
            self.vocabulary_size,
            self.embedding_size,
            padding_idx=self.padding_idx,
        )
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=rnn_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(rnn_output_size, self.vocabulary_size)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.padding_idx, reduction="none"
        )
        # instantiate hidden states
        if self.conditional_h:
            self.descriptor_to_hs = []
            self.descriptor_to_cs = []
            for layer in range(self.n_layers):
                self.descriptor_to_hs.append(
                    nn.Linear(self.num_descriptors, self.hidden_size)
                )
                self.descriptor_to_cs.append(
                    nn.Linear(self.num_descriptors, self.hidden_size)
                )
            # compatibility with cuda
            self.descriptor_to_hs = nn.ModuleList(self.descriptor_to_hs)
            self.descriptor_to_cs = nn.ModuleList(self.descriptor_to_cs)
        # full descriptor embedding instead of 1d scalar
        if self.conditional_emb_l:
            self.conditional_to_emb = nn.Linear(
                self.num_descriptors, self.embedding_size
            )
        # same for decoder
        if self.conditional_dec_l:
            self.conditional_to_dec = nn.Linear(
                self.num_descriptors, self.embedding_size
            )
        # self.init_weights()
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, inputs, hidden):
        return False

    def loss(self, batch):
        # extract the elements of a single minibatch
        padded, lengths, descriptors = batch
        # move to the gpu
        padded, descriptors = padded.to(self.device), descriptors.to(
            self.device
        )

        # embed the padded sequence, along with the descriptors
        embedded = self.embedding(padded)
        # -> embedded: max_len x batch_size x emb_size
        if self.dropout.p > 0:
            embedded = self.dropout(embedded)
        # cat descriptors along dimension of emb_size
        # descriptors_repeating: max_len x batch_size x 1
        descriptors_repeating = descriptors.unsqueeze(0).repeat(
            max(lengths), 1, 1
        )

        if self.conditional_emb_l:
            combined_embedding_features = self.conditional_to_emb(
                descriptors_repeating.float()
            )
            # combined_embedding_features: max_len x batch_size x emb_size
            embedded = torch.cat(
                [embedded, combined_embedding_features], axis=2
            ).float()
            # embedded: max_len x batch_size x (2x emb_size)
        elif self.conditional_emb:
            embedded = torch.cat(
                [embedded, descriptors_repeating], axis=2
            ).float()
        # -> embedded: max_len x batch_size x (emb_size + number of features)

        # now pack the embedded sequences [packed : sum of len x embedding_size]
        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        # optionally, instantiate h_0/c_0 from descriptor
        if self.conditional_h:
            h_0, c_0 = self.init_hidden(descriptors)
            # states: num_layers x batch_size x hidden_size
            packed_output, hidden = self.rnn(packed, (h_0, c_0))
            # sum(len)x hidden_size
        else:
            packed_output, hidden = self.rnn(packed)
        # unpack the output

        padded_output, output_lens = pad_packed_sequence(packed_output)
        # -> packed_output: max_len x batch_size x hidden_size
        # run LSTM output through decoder
        if self.dropout.p > 0:
            padded_output = self.dropout(padded_output)
        # cat descriptors along dimension of emb_size
        if self.conditional_dec_l:
            combined_embedding_features = self.conditional_to_dec(
                descriptors_repeating.float()
            )
            padded_output = torch.cat(
                [padded_output, combined_embedding_features], axis=2
            ).float()
        elif self.conditional_dec:
            padded_output = torch.cat(
                [padded_output, descriptors_repeating], axis=2
            ).float()
        # -> padded_output: max_len x batch_size x (hidden_size + number of features)
        decoded = self.decoder(padded_output)
        # -> decoded: max_len x batch_size x vocab_len

        # finally, calculate loss
        loss = 0.0
        max_len = max(lengths)
        targets = padded[1:, :]
        for char_idx in range(max_len - 1):
            loss += self.loss_fn(decoded[char_idx], targets[char_idx])

        loss = loss.mean()
        return loss

    def sample(
        self,
        *,
        descriptors=None,
        n_sequences=None,
        max_len=250,
        return_smiles=True,
        return_losses=False,
    ):
        assert (
            descriptors is not None
        ), "descriptors must be provided for sampling from a Conditional RNN model"
        assert (
            n_sequences is None or len(descriptors) == n_sequences
        ), "When providing descriptor values, either omit n_sequences or make them conform to the number of descriptors"

        was_training = self.training
        self.eval()

        # get start/stop tokens
        start_token = self.vocabulary.dictionary["SOS"]
        stop_token = self.vocabulary.dictionary["EOS"]
        pad_token = self.vocabulary.dictionary["<PAD>"]

        # create start token tensor
        n_sequences = len(descriptors)
        inputs = (
            torch.empty(n_sequences)
            .fill_(start_token)
            .long()
            .view(1, n_sequences)
            .to(self.device)
        )
        # initialize hidden state
        if self.conditional_h:
            hidden = self.init_hidden(
                descriptors
            )  # Initializing hidden state based on number of layers
        else:
            hidden = torch.zeros(
                self.n_layers, n_sequences, self.hidden_size
            ).to(self.device), torch.zeros(
                self.n_layers, n_sequences, self.hidden_size
            ).to(
                self.device
            )

        # repeat descriptors
        descriptors = descriptors.view(1, n_sequences, descriptors.shape[1])

        loss = nn.NLLLoss(reduction="none", ignore_index=pad_token)

        # sample sequences
        finished = torch.zeros(n_sequences).byte().to(self.device)
        log_probs = torch.zeros(n_sequences).to(self.device)
        sequences = []
        with torch.no_grad():
            for step in range(max_len):
                embedded = self.embedding(inputs)
                if self.conditional_emb_l:
                    combined_embedding = self.conditional_to_emb(
                        descriptors.float()
                    )
                    embedded = torch.cat(
                        [embedded, combined_embedding], axis=2
                    ).float()
                elif self.conditional_emb:
                    embedded = torch.cat(
                        [embedded, descriptors], axis=2
                    ).float()

                output, hidden = self.rnn(embedded, hidden)
                if self.conditional_dec_l:
                    combined_embedding = self.conditional_to_dec(
                        descriptors.float()
                    )
                    output = torch.cat(
                        [output, combined_embedding], axis=2
                    ).float()
                elif self.conditional_dec:
                    output = torch.cat([output, descriptors], axis=2).float()

                logits = self.decoder(output)
                prob = F.softmax(logits, dim=2)
                inputs = torch.multinomial(prob.squeeze(0), num_samples=1).view(
                    1, -1
                )
                sequences.append(inputs.view(-1, 1))
                log_prob = F.log_softmax(logits.squeeze(0), dim=1)
                losses = loss(log_prob, inputs.squeeze(0))
                # zero losses if we are finished sampling
                losses[finished.squeeze(0).bool()] = 0
                log_probs += losses
                # track whether sampling is done for all molecules
                finished = torch.ge(finished + (inputs == stop_token), 1)
                if torch.prod(finished) == 1:
                    break

        # concatenate sequences and decode
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            smiles = [self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs]
        else:
            smiles = sequences

        if was_training:
            self.train()

        if return_losses:
            return smiles, log_probs.detach().cpu().numpy()
        else:
            return smiles

    def init_hidden(self, descriptors):
        h_0s = []
        c_0s = []
        for layer in range(self.n_layers):
            descriptor_to_h = self.descriptor_to_hs[layer]
            descriptor_to_c = self.descriptor_to_cs[layer]
            h_0 = descriptor_to_h(descriptors.float())
            c_0 = descriptor_to_c(descriptors.float())
            h_0s.append(h_0)
            c_0s.append(c_0)
        # stack into correct dimensionality
        h_0 = torch.stack(h_0s)
        c_0 = torch.stack(c_0s)
        return h_0, c_0
