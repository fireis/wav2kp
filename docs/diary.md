# WAV2KP

- This diary is meant to help me keep organized through this project. I plan to write my development here, separating by project day.

## Day 1 - 04/03/2021

- The first stage of this project will be to prepare our dataset. The CH-Unicamp is a dataset composed of mp4 files containing footage and the corresponding audio. For this project, we need to get a separate WAV file for each video and turn each video into a sequence of images to allow individual frame processing. For this purpose, we are going to use the FFMPEG library. As I may run this project wither on windows, mac, or Linux, I'm going to prepare the installation of FFMPEG using a docker image;

- I worked towards finding a good docker image of ffmpeg to start the extraction script;
- I extracted the audio from the videos of the CH-Unicamp dataset.

## Day 2 - 04/04/2021

- Install Face Alignement libs to extract the keypoints from the images;
- Convert the videos to individual png frames;
- Created a script to extract the facial keypoints and crop the images to the format used when training the original network.

## Day 3 - 04/10/2021

- My main objective today is to give wav2vec2 a fisrt try;
- When opening the wav files produced previously, I had to deal with format issues, as I had extracted the wav with mp3 compression, and scipy doesn't accept it. The solution was to try some conversion methods and update the audio extraction script after I found a working config;
