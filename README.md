# WAV2KP

## Intro
This project aims to generate a sequence of facial keypoints representing the speech using the audio as input. 
We created this report to share the work done as a final project of the [Full Stack Deep Learning course](https://fullstackdeeplearning.com/spring2021/). There is still much to be done, but this is the point at which we are now. 

For the full description of my journey, and even learn with some mistakes, please check my [diary](https://github.com/fireis/wav2kp/blob/main/docs/diary.md)

## Data

### Dataset used in our tests
We use the CH-Unicamp Dataset, which contains videos of an actress speaking carefully designed sentences in Brazilian Portuguese. This dataset includes videos of the actress speaking the sentences performing each of the 22 emotion categories of the OCC emotion model.

## Preprocessing - Initial Transformation
As the data is in the video format, the first step to extract the frames in an image format and the audio as a single file. We have used the FFmpeg tool to perform this extraction, generating a sequence of images and audio files. 

### Preprocessing - Feature Extraction (Audio)
To use the audio information, we first extract the features using an MFCC feature extractor. As we were already using the Torch library, we have used the Torchaudio implementation of the MFCC coefficient extraction. The code for this process is available on the [extract_mfcc](https://github.com/fireis/wav2kp/blob/f1751eb31aa3ad4a5d391528df5ccc3a833d534c/data/create_unified_sets.py#L80) section of the create_unified_sets file. 

### Preprocessing - Feature Extraction (Facial Keypoints)
We need to search for the facial keypoints in each of the frames as the input to our system is the sequence of keypoints. To perform this extraction, we use the [extract_keypoints](https://github.com/fireis/wav2kp/blob/main/utils/extract_keypoints.py) script. 

## Preprocessing -  Windows

### Preprocessing - Audio Windows
During the development process, we have tried a different set of approaches to achieve good results. Our most recent approach followed some ideas presented in the (Obamanet project)[https://github.com/karanvivekbhargava/obamanet] and used a windowing strategy in the MFCC data. This strategy consists of creating a sliding window that considers a more extensive timeframe, resulting in a bigger input. For example, the first input contains the MFCCs extracted from the 1st until the 40th sample, while the second contains the data of the 2nd to the 41th sample. The exact code for this process is available in (this function.)[https://github.com/fireis/wav2kp/blob/f1751eb31aa3ad4a5d391528df5ccc3a833d534c/data/create_unified_sets.py#L118] 

### Preprocessing - Keypoints Upsampling
Our audio data is sampled at 16kHz and windowed, and our images are 29.97fps. To make the number of images closer to the number of MFCC windows, we upsample the keypoints, increasing the number of image samples three times. This upsampling help in the training process, as we are not using any CTC-like loss strategy and need to link a single input to a single output to calculate the loss. We have tried to use some loss similar to CTC to avoid this upsampling, but we did not succeed. 

## Network
As the audio data is highly temporal-dependant and repetitive, we have followed other approaches, such as [Obamanet](https://github.com/karanvivekbhargava/obamanet) and used an LSTM followed by a fully connected neural network layer to obtain our results. 

## Results
Our initial results show that we are on a good path, but the road is long. The main issue we are facing now is that the results are not as realistic as needed to achieve good results on the GAN synthesis. The current results are available in the section below as a sample. 

### Results - WAV2KP

https://user-images.githubusercontent.com/20783380/118335337-ccb73000-b4e5-11eb-9ca0-9b09179a5b64.mp4


### Results - WAV2KP with Vid2vid

https://user-images.githubusercontent.com/20783380/118335371-e0fb2d00-b4e5-11eb-9f57-5d702661725e.mp4


