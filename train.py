import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import OrderedDict
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    def prepare_data(self):
        datasets.FashionMNIST('F_MNIST_data', download=True, train=True)
        datasets.FashionMNIST('F_MNIST_data', download=True, train=False)

    def setup(self, stage: str):
        train = datasets.FashionMNIST('F_MNIST_data', download=True, train=True, transform=self.transform)
        test = datasets.FashionMNIST('F_MNIST_data', download=True, train=False, transform=self.transform)
        train_set_size = int(len(train) * 0.8)
        valid_set_size = len(train) - train_set_size
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(train, [train_set_size, valid_set_size])
        self.mnist_test = test
        #print(self.mnist_train)
        #self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        #self.val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)


    def train_dataloader(self):
        #print(self.train_sampler)
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size)

    #def predict_dataloader(self):
    #    return torch.utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size)
#data = MNISTDataModule()
#print(data)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mo = nn.Sequential(nn.Linear(784, 392),
                                        nn.ReLU(),
                                       nn.Dropout(0.25),
                                       nn.Linear(392, 196),
                                       nn.ReLU(),
                                       nn.Dropout(0.25),
                                       nn.Linear(196, 98),
                                        nn.ReLU(),
                                        nn.Dropout(0.25),
                                        nn.Linear(98, 49),
                                        nn.ReLU(),
                                        nn.Linear(49, 10),
                                        nn.LogSoftmax(dim=1))
    def forward(self,x):
        return self.mo(x)

class MNISTModel(pl.LightningModule):
    def __init__(self,model,loss,optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss = loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.shape[0], -1)
        #print("HJERE IS THE SHAPE of X", x.shape)
        x_hat = self.model(x)
        #print("HER IS THE SHAPE OF X HAT", x_hat.shape)
        loss = self.loss(x_hat, y)
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.shape[0], -1)
        #print("HJERE IS THE SHAPE of X", x.shape)
        x_hat = self.model(x)
        #print("HER IS THE SHAPE OF X HAT", x_hat.shape)
        loss = self.loss(x_hat, y)
        self.log("val_loss", loss)
        return loss
    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.shape[0], -1)
        #print("HJERE IS THE SHAPE of X", x.shape)
        x_hat = self.model(x)
        #print("HER IS THE SHAPE OF X HAT", x_hat.shape)
        loss = self.loss(x_hat, y)
        self.log("test_loss", loss)
        return loss
    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer

@hydra.main(config_name="config")
def run_train(cfg: DictConfig):
    data = MNISTDataModule(batch_size=cfg.batch_size)
    #print(cfg.batch_size)
    model = Model()
    #print("Learning rate",cfg.lr)
    optimizer = cfg.optimizer
    if optimizer =='sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    #print(optimizer)
    module = MNISTModel(model,nn.NLLLoss(),optimizer)

    trainer = pl.Trainer(max_epochs=cfg.epochs)
    #print("EPochs",cfg.epochs)
    trainer.fit(module,data)

if __name__ == '__main__':
    run_train()
## HERE WE CALL THE OBJECTS CREATED



#optimizer = optim.Adam(model.parameters(), lr=0.0007)
#print("hello")
