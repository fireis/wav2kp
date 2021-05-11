import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from data.base_data_module import BaseDataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pandas as pd
from tqdm import tqdm

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
        #print(f" X_batch:{X_batch.shape}, y_batch: {y_batch.shape} \n {y_pred.shape}")

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(y_pred, y_batch.float())
        # print(y_batch.squeeze().shape[0])
        # loss_fn = torch.nn.CTCLoss()
        # loss = loss_fn(y_pred, y_batch.float(), [X_batch.squeeze().shape[0]],[y_batch.squeeze().shape[0]] )
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience = 50)

        return ({"optimizer": optimizer, "scheduler": scheduler, "monitor":"val_loss"})

def draw_keypoints(frames, file_name, max_size=(256, 256)):
    
    for idx, frame in zip(range(0, len(frames)), frames):
        # print(frame.shape)
        img_d = Image.new("RGB", (max_size[0], max_size[1]) )

        draw = ImageDraw.Draw(img_d)
        points = np.empty([68, 2], dtype=int)
        # frame = frame.detach().numpy()
        for p in range(0, 136, 2):
            current_coord = int(p/2)
            points[current_coord, 0] = frame[p]
            points[current_coord, 1] = frame[p+1]
            draw.ellipse((points[current_coord, 0], points[current_coord, 1], points[current_coord, 0] + 1, points[current_coord, 1] + 1), fill='white', outline='white')


        img_d.save(f"{file_name}/{idx:05}.png")

from utils.v2v_utils import *
def draw_face_edges_v2v(frames, file_name):
    
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]],  # face
                 [range(17, 22)],  # right eyebrow
                 [range(22, 27)],  # left eyebrow
                 [[28, 31], range(31, 36), [35, 28]],  # nose
                 [[36, 37, 38, 39], [39, 40, 41, 36]],  # right eye
                 [[42, 43, 44, 45], [45, 46, 47, 42]],  # left eye
                 [range(48, 55), [54, 55, 56, 57, 58, 59, 48]],  # mouth
                 [range(60, 65), [64, 65, 66, 67, 60]]  # tongue
                 ]

    w, h = 256, 256
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    # im_edges = np.zeros((h, w), np.uint8) # edge map for all edges


    color = 0
    color_edge = (255, 255, 255)

    for idx, frame in zip(range(0, len(frames)), frames):

        im_edges = np.full((h, w), color, np.uint8)
        keypoints = np.empty([68, 2], dtype=int)
        # frame = frame.detach().numpy()
        for p in range(0, 136, 2):
            current_coord = int(p/2)
            keypoints[current_coord, 0] = frame[p]
            keypoints[current_coord, 1] = frame[p+1]

        np.savetxt(f"{file_name}/{idx:05}.txt", keypoints, delimiter=",", fmt="%05d")
        # print(keypoints.shape)
        # print(frame[48])
    
        # from v2v
        pts = keypoints[:17, :].astype(np.int32)
        baseline_y = (pts[0, 1] + pts[-1, 1]) / 2
        upper_pts = pts[1:-1, :].copy()
        upper_pts[:, 1] = baseline_y + (baseline_y - upper_pts[:, 1]) * 2 // 3
        keypoints = np.vstack((keypoints, upper_pts[::-1, :]))

        for edge_list in part_list:
            for edge in edge_list:

                # im_edge = np.full((h, w), color, np.uint8)
                im_edge = np.zeros((h, w), np.uint8)  # edge map for the current edge
                for i in range(0, max(1, len(edge) - 1),
                            edge_len - 1):  # divide a long edge into multiple small edges when drawing
                    sub_edge = edge[i:i + edge_len]
                    x = keypoints[sub_edge, 0]
                    y = keypoints[sub_edge, 1]

                    curve_x, curve_y = interpPoints(x, y)  # interp keypoints to get the curve shape

                    drawEdge(im_edges, curve_x, curve_y, color=color_edge)

        # np.save(im_edges,A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
        im_edges = Image.fromarray(im_edges)
        # im_edges = resize_img(im_edges, keypoints)
        im_edges.save(f"{file_name}/{idx:05}.png")

        # Image.fromarray(crop(im_edges, keypoints)).save(A_path.replace("txt", "png").replace("keypoints", "gambi_image"))
    return

def rescale_keypoints(keypoints, max_size=(256, 256)):
    # multiply all x and y coordinates by the corresponding axis max
    # this is defined on the image processing stage and should be the
    # inverse of the scale_keypoints function
    for p in range(0, 136, 2):
        keypoints[:, :, p] = keypoints[:, :, p] * max_size[0]
        keypoints[:, :, p+1] = keypoints[:, :, p+1] * max_size[1]
    
    
    return keypoints

def upsample_keypoints(kps, scale=1/3, mode='linear'):
    t = torch.Tensor(kps)
    m = torch.nn.Upsample(scale_factor=scale, mode=mode)
    tt = torch.transpose(t, 0, 2)
    utt = m(tt)
    uttt = torch.transpose(utt, 0, 2)
    return uttt

def subsample(kps, scale=3):

	# Subsample the points
    kps = kps.squeeze()
    print(kps.shape)
    new_shape = int(kps.shape[0]/scale)
    
    new_y = np.zeros((new_shape,136))
    for idx in range(new_y.shape[0]):
        if not (idx*scale > kps.shape[0]-1):
            # Get into (x, y) format
            new_y[idx] = kps[int(idx*scale)]
        else:
            break
    # print('Subsampled y:', new_y.shape)
    new_y = [np.array(each) for each in new_y.tolist()]
    # print(len(new_y))
    return new_y

if __name__ == '__main__':
    
    model_name = 'i2000_o200_b10_ups_s2000'
    model = LSTM.load_from_checkpoint(f"{model_name}_16000.ckpt",  input_dim=2000, hidden_dim=200, batch_size=10, output_dim=136, num_layers=1)

    train_keypoints = torch.load("dataset/Train_keypoints_upsampled")
    train_mfccs = torch.load("dataset/Train_mfccs.pt" )

    train_mfccs = torch.load("dataset/Train_mfccs_windowed.pt" )
    val_mfccs = torch.load("dataset/Val_mfccs_windowed.pt" )
    test_mfccs = torch.load("dataset/Test_mfccs_windowed.pt" )   

    train_set  = BaseDataset(data=train_mfccs, targets=train_keypoints)
    train_loader = DataLoader(train_set)

    file_path = os.path.abspath("") + ""
    train_test_dist = pd.read_excel("docs/train_val_test_dist.xlsx")

    set_names = ["Train", "Test", "Val"]
    for set_name in set_names:
        file_names = sorted(train_test_dist[train_test_dist.set == set_name].video.unique())
        new_path = f"results/{set_name}"
        try:
            os.mkdir(new_path)
        except:
            pass
        for idx, name in tqdm(enumerate(file_names, 0)):
            # print(name, idx)

            infered = model(train_mfccs[idx].unsqueeze(0))
            infered = rescale_keypoints(infered.detach())
            infered = subsample(infered)

            new_path = f"results/{set_name}/{name}"
            try:
                os.mkdir(new_path)
            except:
                pass
            # print(len(infered), infered[0].shape)
            draw_face_edges_v2v(infered, new_path)
            # os.system(f"docker run -v C:/studies/wav2kp/results/{set_name}/{name}:/t/  jrottenberg/ffmpeg  -start_number 0 -i /t/%05d.png -c:v libx264 -vf \"fps=24.97,format=yuv420p\" /t/{name}.mp4 -y")