import os
import pandas as pd
import torch
from rdkit import Chem
from tqdm import tqdm


class EarlyStopping:
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset, and save the
    best model.

    Note that patience here is measured in steps, rather than in epochs,
    because the size of an epoch will not be consistent if the size of the
    dataset changes.

    adapted from:
    https://github.com/Bjarten/early-stopping-pytorch
    https://github.com/fastai/fastai/blob/master/courses/dl2/imdb_scripts/finetune_lm.py
    """

    def __init__(self, patience=100):
        """
        Args:
            model: the PyTorch model being trained
            output_file: (str) location to save the trained model
            patience: (int) if the validation loss fails to improve for this
              number of consecutive batches, training will be stopped
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.step_at_best = 0
        self.stop = False
        print("instantiated early stopping with patience=" + str(self.patience))

    def __call__(self, val_loss, model, output_file, step_idx):
        # do nothing if early stopping is disabled
        if self.patience > 0:
            if self.best_loss is None:
                self.best_loss = val_loss
                self.step_at_best = step_idx
                self.save_model(model, output_file)
            elif val_loss >= self.best_loss:
                # loss is not decreasing
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
                    print("stopping early with best loss " + str(self.best_loss))
            else:
                # loss is decreasing
                self.best_loss = val_loss
                self.step_at_best = step_idx
                # reset counter
                self.counter = 0
                self.save_model(model, output_file)

    def save_model(self, model, output_file):
        torch.save(model.state_dict(), output_file)


def track_loss(
    output_file, epoch, batch_no, value, outcome=("training loss", "validation loss")
):
    sched = pd.DataFrame(
        {
            "epoch": epoch,
            "minibatch": batch_no,
            "outcome": outcome,
            "value": value,
        }
    )

    # write training schedule (write header if file does not exist)
    if not os.path.isfile(output_file) or batch_no == 0:
        sched.to_csv(output_file, index=False)
    else:
        sched.to_csv(output_file, index=False, mode="a", header=False)


def print_update(
    model, epoch, batch_idx, training_loss, validation_loss, n_smiles=64, masses=None
):
    # print message
    tqdm.write("*" * 50)
    tqdm.write(
        "epoch {} -- step {} -- loss: {:5.2f} -- ".format(
            epoch, batch_idx, training_loss
        )
        + "validation loss: {:5.2f}".format(validation_loss)
    )

    # sample a batch of SMILES and print them
    if masses is not None:
        # masses = torch.tensor(masses) # Masses are already a tensor
        smiles = model.sample(masses, return_smiles=True)
    else:
        smiles = model.sample(n_smiles, return_smiles=True)

    valid = 0
    for idx, sm in enumerate(smiles):
        if idx < 5:
            tqdm.write(sm)
        if sm is not None and Chem.MolFromSmiles(sm):
            valid += 1

    pct_valid = 100 * valid / n_smiles
    tqdm.write("{:>4.1f}% valid SMILES".format(pct_valid) + "\n")
