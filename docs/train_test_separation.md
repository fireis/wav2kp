# Train Test Separation

The CH-Unicamp dataset contains videos of an actress speaking carefully designed sentences while expressing a given emotion. In this work we currently do not use the emotion information to transform the audio to keypoints, but this is an important task that we may pursue in the future. Hence, we perform our train/test separation trying to have videos of every emotion on both training and testing.

Our initial strategy is to have 1 video per emotion in the test set, which results roughly on a 70/30 split between train/test sets. As we have 1 video for a given text while asking the actress to express an emotion and being neutral, we follow the same directive for the neutral video distribution. 

Within the training set, we also separate a portion to validation. Using the same strategy, we consider 1 video per emotion in the validation set

Considering that this would result on a poorly distributed set, 