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
- Concluded that I will probably need to downsample the audio to 16k, as the pretrained model used this rate. For future reference, I will use the post: https://huggingface.co/blog/fine-tune-wav2vec2-english.

## Day 4 - 04/11/2021

- My main objective today is to make a first feature extraction applying wav2vec2 to the CH-Unicamp dataset;
- Tested the facebook/wav2vec2-large-xlsr-53-portuguese model and found out that it performs well in speech recognition. My conclusion is that I can the actual sentences extracted from the audio by this model as input to my LSTM  in a semi-supervised manner. I also tested the facebook/wav2vec2-base-960h and realized that, although the words were incorrect, the phonemes I imagine when reading the generated text make sense when comparing to the actual sentence. It seems to be a good option for a multilingual project, but we might stick to Portuguese this time as the dataset is fully in pt-br.
- Now, I'm going to research a bit more on LSTMs with varying input and output sizes. This will help if using either MFCC extracted from the audio or the words from the wav2vec2 model.

## Day 5 - 04/12/2021

- Understand if we could use wav2vec2 to determine the timing for each word. This could be then used to crop the audio signals to be used as input to the LSTM.

## Days 6, 7 and 8 - 04/12/2021 - 04/15/2021

- Tried to find if wav2vec2 would be able to produce the time period for each word, but I concluded that this would be unfeasible due to its structure;
- Researched LSTM and audio processing option. Decided to try using MFCC with window of 33ms as the videos are 29.97 FPS, so we should have a window per frame. I believe this wouldn't be necessary, but at least it is a way to justify the window;

## Day 9 - 04/17/2021

- Process audio files, extracting MFCCs;
- Create the train, test and validation division of the dataset;
- Ideally, I would find a way to make my network input size variable. But, as this adds complexity and I want to have at least something done at the end of the development period, I'm deciding to work with fixed size for now and leave this as an improvement path for later;
- Creation of the torch lightning data model;
- Read a bit about [NeMo](https://github.com/NVIDIA/NeMo) and [asteroid](https://asteroid-team.github.io/), which could help with ready encoder modules. But I think I'll leave this for later as I want to have a bit more control on what I'm actually doing; 

## Day 10 - 04/18/2021

- Objective: finish the data loader structure and train a first network to validate the process;
- Realized that there was a bug in the keypoint extraction that made the reduced keypoints to be identical to the original ones. Then I fixed and ran the process again;
- I was trying to use the mfcc transformation on the dataset class, but I was getting to confuse and couldn't make progress. So I decided to preprocess the complete dataset and use pytorch only to load the files. This makes changes to the transform more challenging and maybe more difficult to implement an inference pipeline in the future, but I think that by the time I have something working it will be easier to make things the way I want, moving this part to the dataset class;
- Started exploring the network. Found that [pack padded sequence may solve my problem](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html#torch.nn.utils.rnn.pack_padded_sequence). I have to understand it a bit better, [this seems to be a nice source](https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch);
- Finished the day with issues on padding the sequence to make the size consistent;

## Day 11 - 04/19/2021

- Tried to fix the issues with the padding of the input vectors to start using a proper LSTM;
- Used padding to shape the input arrays to the LSTM network.

## Day 12 - 04/20/2021

- Written the fundamental parts for torch lightning;
- Experimented with LSTM, but still could not fix the configs to match our input.

## Day 13 - 04/21/2021

- Could get the input shapes for the LSTM right and after lots of debugging, realized that my issue now was with the fully connected layer;
- Got the network working, but I think that I need to change the output shape. Im confused on wheter the LSTM will ouput the time sequence as a single output (so each time step would be a piece of a single vector) or if there would be one output per time step

## Day 14 - 04/22/2021

- Studied LSTMs, losses and the output shape to asses what should I do to make my model work.

## Day 15 - 04/23/2021

- Concluded that I should use CTC loss and should not pad my Y vectors (keypoints);
- Reviewed the dimensions after each transform I am applying to the data.

## Day 16 - 04/24/2021

- Today I'll focus on rethinking my approach;
- Followed an approach similar to obamanet and standardized my X and Y vector sizes;
- Performed single training to check whether the structure is working.

## Day 17 - 04/25/2021

- Started by coding the drawing section of the code to be able to see the results from the latest training;
- Added validation and training steps to model code;
- Take model code to python file to improve trackability now that there is something that at leat seems to work;

## Day 18 - 04/26/2021

- Verified the results using 2 different models trained with the data that repeats audio segments for a single keypoint. The results are bad as there is no movement within the mouth region. Now I think I'll try to follow obamanet and repeat a single keypoint set for different audio periods;

## Day 19 - 04/27/2021

- Verified that adding PCA didn't help much to get diverse outputs. 
- Decided to focus on trying CTC or some variation of that.

## Day 20 - 05/01/2021

- Worked on the upsampling stage beacuse I'm certain that I have an issue with my data

## Day 21 - 05/02/2021

- Fixed some code messes;
- Review data processing;
- Tested upsacaled data and PCA after upscaling;

## Day 22 - 05/03/2021

- Got the models results and realised that we still can't get a mouth moving

## Day 23 - 05/04/2021

- Tested how the different lstm structures were performing when overfitting minibatches. Found that they can't improve much beyond the point in which I was getting to in the past;
- Decided to try a transformer instead of LSTM

## Day 24 - 05/05/2021

- Tested Transformer with no good results;
- Tested applying the time window similar to obamanet's;

## Day 25 - 05/06/2021

- Tested starting the training with way smaller learning rate;
- Got to 16k epochs with windowing and used the model to infer training samples and realized that we have at least somewhat usable results;

## Day 26 - 05/07/2021

- Worked on the inference script to assess the results;

## Day 27 - 05/10/2021

- Starting evaluating the results and planning the presentation



days 1 and 2 ~8h
days 3 and 4 ~6h
days 5 to 8 ~3h
day 9  5h
day 10  6h
day 11 1h30m
day 12 1h
day 13 3h
day 14 .5h
day 15 1h30m
day 16 3h
day 17  3h
day 18  2h
day 19 1h
day 20  14
day 21 3h
day 22 and 23 1h +
day 24 1h30
day 25 30m
day 26 30m
day 27 1h30m


 