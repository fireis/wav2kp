# Dataset Preprocessing Utils

Here we present some functions used to preprocess the CH-Unicamp dataset. These scripts were created to allow easy reproduction of the experiments and are not necessarily optimized and made to work on multiple computers, so you may need to change some things to reflect your reality.

## Available Scripts

[Create Folders](create_folders.bat): This script creates separate folders for each video within the CH-Unicamp Dataset. It assumes that you are running it from within a folder with every mp4 video available. It will create folders for audio, keypoints, video, and images for each dataset sample. It is a bat file meant to be used on Windows but may be easily adapted to run on Linux.

[Extract Audio](extract_audio.bat): This script will use a docker image of ffmpeg to extract the audio into a wav file for each CH-Unicamp sample.

[Extract Frames](extract_frames.bat): This script uses a docker image of ffmpeg to extract frames from each video within the CH-Unicamp dataset.

