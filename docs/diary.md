# WAV2KP

- This diary is meant to help me keep organized through this project. I plan to write my development here, separating by project day.

## Day 1

- The first stage of this project will be to prepare our dataset. The CH-Unicamp is a dataset composed of mp4 files containing footage and the corresponding audio. For this project, we need to get a separate WAV file for each video and turn each video into a sequence of images to allow individual frame processing. For this purpose, we are going to use the FFMPEG library. As I may run this project wither on windows, mac, or Linux, I'm going to prepare the installation of FFMPEG using a docker image.

- I worked towards finding a good docker image of ffmpeg to start the extraction script.