import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names=['input'],     # the model's input names
              output_names=['output'],   # the model's output names
              dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                            'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients
        #  when backward() is called, which we do not want here.
        self._optim.zero_grad()
        # -propagate through the network
        pred = self._model(x)
        # -calculate the loss
        loss = self._crit(pred, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss.item()

    def val_test_step(self, x, y):
        # predict / propagate through the network and calculate the loss and predictions
        pred = self._model(x)
        loss = self._crit(pred, y)
        # return the loss and the predictions
        return loss.item(), pred

    def train_epoch(self):
        # set training mode
        self._model.train()
        total_loss = 0.0
        # iterate through the training set
        for x, y in tqdm(self._train_dl, desc='train', leave=False):
            # transfer the batch to the gpu if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            # perform a training step
            total_loss += self.train_step(x, y)
        # calculate the average loss for the epoch and return it
        return total_loss / len(self._train_dl)

    def val_test(self):
        # set eval mode (important for BatchNorm / Dropout)
        self._model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        # disable gradient computation (not needed for evaluation)
        with t.no_grad():
            # iterate through the validation set
            for x, y in tqdm(self._val_test_dl, desc='val', leave=False):
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                total_loss += loss
                # save the predictions and the labels for each batch
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())

        # average loss over all batches
        avg_loss = total_loss / len(self._val_test_dl)

        # stack all batches and threshold the sigmoid outputs at 0.5
        preds = t.cat(all_preds, dim=0).numpy()
        labels = t.cat(all_labels, dim=0).numpy()
        preds_binary = (preds >= 0.5).astype(np.int32)

        # mean (macro) F1 over the two labels -- the challenge metric
        f1 = f1_score(labels, preds_binary, average='macro', zero_division=0)
        print('Validation loss: {:.4f} | mean F1: {:.4f}'.format(avg_loss, f1))

        # return the loss
        return avg_loss

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0

        # lists for the train and validation losses, an epoch counter and
        # bookkeeping for early stopping
        train_losses = []
        val_losses = []
        epoch_counter = 0
        best_val_loss = np.inf
        epochs_without_improvement = 0

        while True:
            # stop by epoch number
            if epochs > 0 and epoch_counter >= epochs:
                break

            # train for an epoch and validate
            train_loss = self.train_epoch()
            val_loss = self.val_test()

            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # save the model on validation improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_checkpoint(epoch_counter)
            else:
                epochs_without_improvement += 1

            print('Epoch {}: train loss {:.4f} | val loss {:.4f}'.format(
                epoch_counter, train_loss, val_loss))

            # early stopping check
            if (self._early_stopping_patience > 0 and
                    epochs_without_improvement >= self._early_stopping_patience):
                print('Early stopping triggered after {} epochs.'.format(epoch_counter + 1))
                break

            epoch_counter += 1

        # return the losses for both training and validation
        return train_losses, val_losses
