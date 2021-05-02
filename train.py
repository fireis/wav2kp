from os import name
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from data.base_data_module import BaseDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def get_data_loaders(upsampled=True):
    if upsampled:
        train_keypoints = torch.load("dataset/Train_keypoints_upsampled")
        val_keypoints = torch.load("dataset/Val_keypoints_upsampled")
        test_keypoints = torch.load("dataset/Test_keypoints_upsampled")
    else:
        train_keypoints = np.load("dataset/Train_keypoints.npy",  allow_pickle=True)
        val_keypoints = np.load("dataset/Val_keypoints.npy",  allow_pickle=True)
        test_keypoints = np.load("dataset/Test_keypoints.npy",  allow_pickle=True)
    
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# Here we define our model as a class
class LSTM(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2, dnn_shape=1000):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dnn_shape = dnn_shape

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        # Define the output layer
        self.linear = nn.Linear(self.dnn_shape, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input)
        
        lstm_out = self.dropout(lstm_out)
        
        y_pred = self.linear(lstm_out)
        #print(f"y_pred.shape in net: {y_pred.shape}")
        return y_pred


    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        # print(f" X_batch:{X_batch.shape}, y_batch: {y_batch.shape}")
        y_pred = self.forward(X_batch.float())
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y_batch.float())
        #print(f"y_pred.shape: {y_pred.shape}, y_batch.shape(): {y_batch.shape}")
        # loss_fn = torch.nn.CTCLoss()
        # loss = loss_fn(y_pred, y_batch.float())
        # criterion = nn.BCEWithLogitsLoss()
        # loss = criterion(y_pred, y_batch.unsqueeze(1))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
        


    
    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self.forward(X_batch.float())
        # print(f" X_batch:{X_batch.shape}, y_batch: {y_batch.shape} \n {y_pred.shape}")

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y_batch.float())
        # print(y_batch.squeeze().shape[0])
        # loss_fn = torch.nn.CTCLoss()
        # loss = loss_fn(y_pred, y_batch.float(), [X_batch.squeeze().shape[0]],[y_batch.squeeze().shape[0]] )
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

        return [optimizer], [scheduler]


if __name__ == '__main__':

    train_loader, val_loader, test_loader = get_data_loaders()

    model = LSTM(40, 20, batch_size=80, output_dim=136, num_layers=1, dnn_shape=20)
    logger = pl.loggers.TensorBoardLogger("training/logs", name="i40_o20_b80_ups_s300")
    trainer = pl.Trainer(logger=logger, weights_save_path="training/chkpt", gpus=1, max_epochs=10000)

    trainer.fit(model, train_loader, val_dataloaders=val_loader)

