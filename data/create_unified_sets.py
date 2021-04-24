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

def find_keypoints(keypoints_folder):
    keypoints = []
    # open the files while sorting as the order matters
    frames_list = sorted(glob.glob(keypoints_folder + "*.txt"))
    # read each txt with the samples
    for frame in frames_list:
        kps = read_keypoint_txt(frame)
        kps = scale_keypoints(kps)
        #print(f"find_keypoints, after reading keypointss: {kps.shape}")
        # make x and y a single sequence to make predictions easier
        kps = np.reshape(kps, (1, 136)).squeeze()
        #print(f"find_keypoints, after reading and making xy continuous: {kps.shape}")
        kps = torch.Tensor(kps)
        keypoints.append(kps)
        
    print(f"find_keypoints, before padding: {len(keypoints)} , {keypoints[0].shape}")
    keypoints = pad_sequence(keypoints, batch_first=True)
    print(f"find_keypoints, after padding: {keypoints.shape}")
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
    print(f"mfcc coefs initial shape: {coefs.shape}")
    coefs = torch.transpose(coefs, 1, 2)
    coefs = torch.squeeze(coefs)
    print(f"mfcc coefs after transposing and squeezing shape: {coefs.shape}")
    return coefs

def assemble_set(train_test_dist, set_name="Train"):
    file_names = train_test_dist[train_test_dist.set == set_name].video.unique()
    mfccs_list = []
    keypoints_list = []
    ds = []
    file_names = tqdm(file_names)
    for file_name in file_names:
        # get  the files to process
        audio_file_path, keypoints_folder = assemble_path(file_name)
        # extract keypoints
        keypoints = find_keypoints(keypoints_folder)
        # extract mfccs
        mfccs = extract_mfcc(audio_file_path)
        # append to the dataset list
        mfccs_list.append(mfccs)
        keypoints_list.append(keypoints)
        
    return mfccs_list, keypoints_list


    
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
        mfccs, keypoints = assemble_set(train_test_dist, set_name=set_dist)
        
        print(f"set creation, before mfcc padding: {len(mfccs)}, {mfccs[0].shape}")
        mfccs = pad_sequence(mfccs, batch_first=True)
        print(f"set creation, after mfcc padding: {mfccs.shape}")

        print(f"set creation, before kp padding: {len(keypoints)} {keypoints[0].shape}")
        keypoints = pad_sequence(keypoints, batch_first=True)
        print(f"set creation, after kp padding: {keypoints.shape}")
        
        torch.save(mfccs, f"{dataset_folder}{set_dist}_mfccs.pt" )
        np.save(f"{dataset_folder}{set_dist}_keypoints", keypoints, allow_pickle=True)
        break