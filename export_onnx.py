import torch as t
from trainer import Trainer
import sys
import torchvision as tv

epoch = int(sys.argv[1])
import model as model_module
model = model_module.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit, cuda=False)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
