from PIL import Image, ImageDraw
import numpy as np
import glob
import face_alignment
# import dlib
from skimage import io
import os
import subprocess
from joblib import Parallel, delayed
from v2v_utils import (resize_orig_v2v, draw_face_edges_v2v)
from tqdm import tqdm 

# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "-i", "--input", type=str, required=True, help="path to input image folder"
#     )
#     parser.add_argument(
#         "-s",
#         "--size",
#         type=tuple,
#         required=False,
#         help="path to predictor .dat file",
#         default=(256, 256),
#     )
#     args = parser.parse_args()
#     return args

def find_kp(path, method="fa", save_img=True, resize=False, cropped=False):

    # print("Processing files at {}".format(path))
    fpath = path + "*.png"
    frames_path = glob.glob(fpath)
    if method == "dlib":
        save_path_txt = path + "dlib_keypoints/"
        save_path_img = path + "dlib_img/"

        if not os.path.exists(save_path_txt):
            os.makedirs(save_path_txt)
        if not os.path.exists(save_path_img):
            os.makedirs(save_path_img)

        predictor = dlib.shape_predictor(
            os.path.abspath('predictor/shape_predictor_68_face_landmarks.dat'))
        detector = dlib.get_frontal_face_detector()

    elif method == "fa":
        save_path_txt = path
        if cropped:
            save_path_txt = save_path_txt.replace("images_reduced", "keypoints_reduced")
        else:
            save_path_txt = save_path_txt.replace("images", "keypoints")
        
        detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                flip_input=False, device='cuda')

    for frames in frames_path:
        save_name_txt = os.path.join(save_path_txt, frames[-9:-4] + '.txt')

        img = io.imread(frames)

        points = np.empty([68, 2], dtype=int)
        if method == "dlib":
            dets = detector(img, 1)
            if len(dets) > 0:
                pass

        elif method == "fa":
            shape = detector.get_landmarks(img)
            # print(shape)
            if shape != None:
                if True:
                    for b in range(68):
                        points[b, 0] = shape[0][b][0]
                        points[b, 1] = shape[0][b][1]

        np.savetxt(save_name_txt, points, fmt='%05d', delimiter=',')


def generate_imgs(video):
    path_img = video.replace(".mp4", "_%05d.png")
    subprocess.call(["ffmpeg", "-i", video, path_img, "-hide_banner"])


def create_folders(folders):

    for folder in folders:
       
        reduced_img_path = folder.replace("images", "images_reduced")
        reduced_kp_path = folder.replace("images", "keypoints_reduced")
        if not os.path.exists(reduced_img_path):
            os.makedirs(reduced_img_path)
        if not os.path.exists(reduced_kp_path):
            os.makedirs(reduced_kp_path)

if __name__ == "__main__":
    # This code uses sections from base prep of vid2vid

    folders = glob.glob("C:/studies/wav2kp/raw_videos/*/images/")

    folders = tqdm(folders)
    
    # find keypoints on original images
    print("\nFinding keypoints on original images")
    Parallel(n_jobs=2)(delayed(find_kp)(folder) for folder in folders)

    # create the necessary folders if needed
    print('\nCreating folders')
    create_folders(folders)

    # crop images
    for folder in folders:
   
        images = glob.glob(folder+"*.png")
        images = tqdm(images)
        
        Parallel(n_jobs=12)(delayed(resize_orig_v2v)(image) for image in images)

    reduced_folders = glob.glob("C:/studies/wav2kp/raw_videos/*/images_reduced/")
    reduced_folders = tqdm(reduced_folders)
    print("\nFinding keypoints on reduced images")
    Parallel(n_jobs=2)(delayed(find_kp)(folder, save_img=True, method="fa", cropped=True) for folder in reduced_folders)

