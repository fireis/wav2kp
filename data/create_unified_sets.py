import torchaudio
import pandas as pd
import glob
import numpy as np
from tqdm import tqdm
from torchaudio import transforms
import torch
from typing import Any, Callable, Dict, Sequence, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
import os
from sklearn.decomposition import PCA

SequenceOrTensor = Union[Sequence, torch.Tensor]

def scale_keypoints(keypoints, max_size=(256, 256)):
    # divide all x and y coordinates by the corresponding axis max
    # this is defined on the image processing stage

    keypoints[ :, 0] = keypoints[ :, 0]/max_size[0]
    keypoints[ :, 1] = keypoints[ :, 1]/max_size[1]
    
    return keypoints

def assemble_path(file_name):
    file_path = os.path.abspath("") +  "/raw_videos/" + file_name + "/"
    audio_file = file_path + "audio/" + file_name + ".wav"
    keypoints_folder = file_path + "keypoints_reduced/" 

    return audio_file, keypoints_folder

def upsample_keypoints(kps, scale=4, mode='linear'):
    t = torch.Tensor(kps)
    m = torch.nn.Upsample(scale_factor=scale, mode=mode)
    tt = torch.transpose(t, 0, 2)
    utt = m(tt)
    uttt = torch.transpose(utt, 0, 2)
    return uttt

def find_keypoints(keypoints_folder):
    keypoints = []
    # open the files while sorting as the order matters
    frames_list = sorted(glob.glob(keypoints_folder + "*.txt"))
    # read each txt with the samples
    for frame in frames_list:
        kps = read_keypoint_txt(frame)
        kps = scale_keypoints(kps)
        # make x and y a single sequence to make predictions easier
        # kps = np.reshape(kps, (1, 136)).squeeze()
        # print(f"find_keypoints, after reading and making xy continuous: {kps.shape}")
        # kps = torch.Tensor(kps)
        keypoints.append(kps)
        # print(f"find_keypoints, after reading keypointss: {kps.shape}")
    print(f"pre upsampling {len(keypoints)} {keypoints[0].shape}")
    keypoints = upsample_keypoints(keypoints)
    print(f"post upsampling {keypoints.shape}")
    keypoints = torch.reshape(keypoints, (keypoints.shape[0], 136))
    print(f"post reshaping {keypoints.shape}")
    # kps = np.reshape(kps, (1, 136)).squeeze()
        
    print(f"find_keypoints, before padding: {len(keypoints)} , {keypoints[0].shape}")
    # keypoints = pad_sequence(keypoints, batch_first=True)
    #print(f"find_keypoints, after padding: {keypoints.shape}")
    # transform to numpy array to make processing easier
    # keypoints = np.array(keypoints) 
    
    return keypoints

def read_keypoint_txt(keypoints_folder):
    keypoints = np.loadtxt(keypoints_folder, delimiter=",")
    return keypoints

def read_audio_file(path=""):
    """
    Reads the specifies file in path and returns it in Tensor format
    """
    # fix to work on windows
    path = path.replace("\\", "\\")
    return torchaudio.load(path)

def extract_mfcc(audio_file_path, resample=True):
    audio, sr =  read_audio_file(audio_file_path)
    if resample:
        resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
        sr = 16000
        audio = resampler(audio)
    mfcc = transforms.MFCC(sample_rate=sr, melkwargs={"n_mels": 40})
    coefs = mfcc(audio)
    #print(f"1mfcc coefs initial shape: {coefs.shape}")
    coefs = torch.transpose(coefs, 1, 2)
    coefs = torch.squeeze(coefs)
    #print(f"1mfcc coefs after transposing and squeezing shape: {coefs.shape}")
    return coefs

def extract_mfcc_masking(audio_file_path, resample=True, set_name="Train"):
    audio, sr =  read_audio_file(audio_file_path)
    if resample:
        resampler = transforms.Resample(orig_freq=sr, new_freq=16000)
        sr = 16000
        audio = resampler(audio)
    if set_name == "Train":
        aud_transf = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
            torchaudio.transforms.TimeMasking(time_mask_param=35)
            )
    else:
        aud_transf =  torchaudio.transforms.MelSpectrogram()

    coefs = aud_transf(audio)
    c2 = aud_transf(audio)
    # print(f"mfcc coefs initial shape: {coefs.shape}")
    coefs = torch.transpose(coefs, 1, 2)
    coefs = torch.squeeze(coefs)

    return coefs


def apply_window(mfccs, kps):
    past_window = 40
    time_delay = 20
    X=[]
    y=[]
    kp2aud = mfccs.shape[0] / kps.shape[0]
    for i in range(0, kps.shape[0] - 1):
        idx = int(round(i * kp2aud, 0))

        # handle last frames. I dont like this, but its good enough for now
        if idx + past_window >= mfccs.shape[0] - 1:
            idx = mfccs.shape[0] - past_window - 1

        # print(idx, idx + past_window)
        aud = mfccs[idx : idx + past_window].reshape((-1))
        kp = kps[i] #.reshape((1, -1))
        
        X.append(aud)
        y.append(kp)
    X = pad_sequence(X)
    X = torch.transpose(X, 0, 1)
    y = pad_sequence(y)
    y = torch.transpose(y, 0, 1)
    return X, y

def assemble_set(train_test_dist, set_name="Train"):
    file_names = sorted(train_test_dist[train_test_dist.set == set_name].video.unique())
    mfccs_list = []
    keypoints_list = []
    aud_len = []
    kp_len = []
    ds = []
    file_names = tqdm(file_names)
    i = 0
    for file_name in file_names:
        i+=1
        # get  the files to process
        audio_file_path, keypoints_folder = assemble_path(file_name)
        # extract keypoints
        keypoints = find_keypoints(keypoints_folder)
        # extract mfccs
        mfccs = extract_mfcc(audio_file_path)
        # mfccs = extract_mfcc_masking(audio_file_path, set_name)
        aud_len.append(mfccs.shape[0]//2)
        kp_len.append(len(keypoints))
        # print(mfccs.shape)
        # print(keypoints.shape)
        #mfccs, keypoints = apply_window(mfccs=mfccs, kps=keypoints)
        # print(mfccs.shape)
        # print(keypoints.shape)
        # pca = PCA(n_components=200)
        # mfccs = torch.from_numpy(pca.fit_transform(mfccs))

        # kp_pca = PCA(n_components=30)
        # keypoints = torch.from_numpy(kp_pca.fit_transform(keypoints))
        

        # append to the dataset list
        mfccs_list.append(mfccs)
        keypoints_list.append(keypoints)

    return mfccs_list, keypoints_list, aud_len, kp_len


    
if __name__ == '__main__':

    # TODO: MAKE THIS READ THE PATH FROM ARGPARSE
    
    file_path = os.path.abspath("") + "/docs/train_val_test_dist.xlsx"
    train_test_dist = pd.read_excel(file_path)
    train_files = train_test_dist[train_test_dist.set == "Train"].video.unique()
    test_files = train_test_dist[train_test_dist.set == "Test"].video.unique()
    val_files = train_test_dist[train_test_dist.set == "Val"].video.unique()

    for set_dist in ["Train", "Val", "Test"]:
        dataset_folder = os.path.abspath("") +  "/dataset/"
        print(dataset_folder)
        mfccs, keypoints, aud_len, kp_len = assemble_set(train_test_dist, set_name=set_dist)
        
        print(f"set creation, before mfcc padding: {len(mfccs)}, {mfccs[0].shape}")
        #mfccs = pad_sequence(mfccs, batch_first=True)
        #print(f"set creation, after mfcc padding: {mfccs.shape}")

        print(f"set creation, before kp padding: {len(keypoints)} {keypoints[0].shape}")
        #keypoints = pad_sequence(keypoints, batch_first=True)
        #print(f"set creation, after kp padding: {keypoints.shape}")
        
        torch.save(mfccs, f"{dataset_folder}{set_dist}_mfccs.pt" )
        torch.save(f"{dataset_folder}{set_dist}_keypoints_not_padded", keypoints, allow_pickle=True)
        # break