# Ear-Recognition-Challenge

UERC Ear recognition challenge with 1500 images of 150 subjects.(10 images per subject)

In this challenge we splitted 60-40 % from every subject and used for train and test. 
Among training images again splitted 80-20 % for training and validation. 
Image Augmentation like fliping,rotating,adding noise, etc where done using imgaug libray.
Used pretrained Xception architecture for imagenet and changed the output layer for 150 classes instead of 1000 in pretrineed network.
