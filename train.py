import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Hyperparameters (see the explanation in the accompanying message).
# ---------------------------------------------------------------------------
CSV_PATH = 'data.csv'
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 100
VAL_FRACTION = 0.2
USE_CUDA = t.cuda.is_available()


# load the data from the csv file and perform a train-test-split
data = pd.read_csv(CSV_PATH, sep=';')
train_df, val_df = train_test_split(
    data, test_size=VAL_FRACTION, random_state=42
)

# set up data loading for the training and validation set
train_dl = t.utils.data.DataLoader(
    ChallengeDataset(train_df, 'train'),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_dl = t.utils.data.DataLoader(
    ChallengeDataset(val_df, 'val'),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# create an instance of our ResNet model
net = model.ResNet()

# set up a suitable loss criterion. Our model already applies Sigmoid, so we
# use plain BCELoss (NOT BCEWithLogitsLoss, which would sigmoid a second time).
crit = t.nn.BCELoss()

# set up the optimizer
optim = t.optim.Adam(
    net.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(
    net,
    crit,
    optim=optim,
    train_dl=train_dl,
    val_test_dl=val_dl,
    cuda=USE_CUDA,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
)

# go, go, go... call fit on trainer
res = trainer.fit(epochs=MAX_EPOCHS)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
