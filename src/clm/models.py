import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        padding_t = torch.tensor(self.padding_idx).to(self.device)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, padding_idx=padding_t
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
        padding_t = torch.tensor(self.padding_idx).to(self.device)

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
                    padding_idx=padding_t,
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
        # extract the elements of a single minibatch
        padded, lengths = batch
        # tranpose padded to batch_size * seq_len
        padded = padded.transpose(0, 1)
        # move to the gpu
        padded = padded.to(self.device)

        # pass through the entire transformer model
        decoded = self(padded)
        # -> decoded: batch_size x max_len x vocab_size

        # finally, calculate loss
        loss = 0.0
        max_len = max(lengths)
        targets = padded[:, 1:]
        for char_idx in range(max_len - 1):
            loss += self.loss_fn(decoded[:, char_idx], targets[:, char_idx])

        return loss.mean()

    def sample(
        self,
        *,
        n_sequences,
        return_smiles=True,
        return_losses=False,
        descriptors=None,
    ):
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
        for step in range(self.max_len):
            logits = self(inputs)[:, -1, :]
            prob = F.softmax(logits, dim=-1)
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

        # concatenate sequences and decode
        seqs = torch.cat(sequences, 1)
        if return_smiles:
            outputs = [
                self.vocabulary.decode(seq.cpu().numpy()) for seq in seqs
            ]
        else:
            outputs = sequences

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
        padding_t = torch.tensor(self.padding_idx).to(self.device)
        self.embedding = nn.Embedding(
            self.vocabulary_size, self.embedding_size, padding_idx=padding_t
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
                embedded = torch.cat([embedded, descriptors], axis=2).float()

            output, hidden = self.rnn(embedded, hidden)
            if self.conditional_dec_l:
                combined_embedding = self.conditional_to_dec(
                    descriptors.float()
                )
                output = torch.cat([output, combined_embedding], axis=2).float()
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
