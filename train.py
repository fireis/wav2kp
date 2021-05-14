from os import name
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from data.base_data_module import BaseDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from pytorch_lightning.callbacks import LearningRateMonitor

def get_data_loaders(upsampled=True, pca=True, windowed=True):
    if upsampled:
        if pca:
            train_keypoints = torch.load("dataset/Train_keypoints_upsampled_pca10")
            val_keypoints = torch.load("dataset/Val_keypoints_upsampled_pca10")
            test_keypoints = torch.load("dataset/Test_keypoints_upsampled_pca10")
        
        elif not pca:
            train_keypoints = torch.load("dataset/Train_keypoints_upsampled")
            val_keypoints = torch.load("dataset/Val_keypoints_upsampled")
            test_keypoints = torch.load("dataset/Test_keypoints_upsampled")
    
    else:
        train_keypoints = np.load("dataset/Train_keypoints.npy",  allow_pickle=True)
        val_keypoints = np.load("dataset/Val_keypoints.npy",  allow_pickle=True)
        test_keypoints = np.load("dataset/Test_keypoints.npy",  allow_pickle=True)
    if windowed:
        train_mfccs = torch.load("dataset/Train_mfccs_windowed.pt" )
        val_mfccs = torch.load("dataset/Val_mfccs_windowed.pt" )
        test_mfccs = torch.load("dataset/Test_mfccs_windowed.pt" )    
    else:
        train_mfccs = torch.load("dataset/Train_mfccs.pt" )
        val_mfccs = torch.load("dataset/Val_mfccs.pt" )
        test_mfccs = torch.load("dataset/Test_mfccs.pt" )

    train_set  = BaseDataset(data=train_mfccs, targets=train_keypoints)
    train_loader = DataLoader(train_set)

    val_set  = BaseDataset(data=val_mfccs, targets=val_keypoints)
    val_loader = DataLoader(val_set)

    test_set  = BaseDataset(data=test_mfccs, targets=test_keypoints)
    test_loader = DataLoader(test_set)

    return (train_loader, val_loader, test_loader)


# Here we define our model as a class
class LSTM(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.25)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input)   
        lstm_out = self.dropout(lstm_out)  
        y_pred = self.linear(lstm_out)
        return y_pred


    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self.forward(X_batch.float())
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y_batch.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}   
    
    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self.forward(X_batch.float())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y_batch.float())

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

        return ({"optimizer": optimizer, "scheduler": scheduler, "monitor":"val_loss"})


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_data_loaders(upsampled=True, pca=False, windowed=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    model = LSTM(2000, 200, batch_size=10, output_dim=136, num_layers=2)
    model_name = 'i2000_o200_b10_ups_s2000'
    logger = pl.loggers.TensorBoardLogger("training/logs", name=model_name)
    trainer = pl.Trainer(logger=logger, weights_save_path="training/chkpt", gpus=1, max_epochs=16000, callbacks=[lr_monitor])#, overfit_batches=0.02)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)
    trainer.save_checkpoint(f"{model_name}_16000.ckpt")
