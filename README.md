# Classifying Mushroom Images by Genus
## Problem Description & Goal
My goal is to classify mushroom images by genus because I'm trying to get into mushroom foraging and I thought it'd be fun! I used a [dataset from Kaggle](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) that has mushroom images from 9 different genera. 

Constraints: to make the problem more interesting, I limited myself to the existing data (no downloading other images, but augmentations are allowed). I also did not allow myself to finetune an existing model. I am also running the model locally on my laptop, so it has to be lightweight enough to train in a few hours to be practical to iterate on. 

Test statistic: % of images classified as the correct genus. 

## Inputs, Outputs, and Model Predictions
The original images were many different shapes and sizes. I resized all or them to 150x150 by downsizing them then center cropping. I originally considered picking the most common aspect ratio, but after examining the images squares captured the data fine and were simple to implement.

The model's input is a square image. Its output is a 1D tensor with the log odds of each of the 9 classes. To use the model, we can softmax those logits to convert them to probabilities if we want confidence levels in our predictions and take the most likely class as the model's prediction.

I split the input data into training, test, and validation data. I added a copy of the training data with 1-2 random augmentations on each image to help address the limited data size. 

## Original Architecture & Training Setup
Original setup:
- A CNN with 2 conv layers, 3 residual blocks (2 conv layers each), average pooling, and a fully connected layer 
- Cross entropy loss (suitable for multi-class categorization)
- Adam with a learning rate of .001 and a step learning rate scheduler

## Architecture and Training Improvements
My first challenge was that the model performance was plateauing early on the training data. I made the following changes:
- Switched to having dilation alternating with no dilation rather than consecutive layers with the same dilation to avoid gridding effects. (I didn't increase dilation progressively, as mushroom classification seems to require more fine-grained than holistic information)
- Increased the learning rate and reduced the speed of its decay because the loss was jumping up and down after quickly plateauing 
- Included both max pooling and average pooling before the fully connected layer as average pooling may have been losing too much fine feature data 

By this point, the model was learning the training dataset well but was now overfitting. To address that, I:
- Added new augmentations each epoch rather than reusing the same data
- Added dropout layers

That got the model to not overfit. To further improve performance, I:
- Switched my optimizer to SGD, which tends to have better performance than Adam on CNN tasks (whereas Adam tends to converge faster) - since the model was plateauing before the end of training, the slower convergance and better performance was a good tradeoff 
- Looked through the data manually. Upon examining a confusion matrix and looking through common confusions, one hypothesis is that the model is missing out on fine-grained data. (The main things it messed up up were mushrooms that differed primarily texturally but otherwise looked similar.) To address that, I increased the image size (originally made it small since I'm running things locally), reduced the stride in early layers, and pushed the dilation to later layers.

## Ideas to Further Improve Accuracy
First, I would examine the loss curves to determine the area of improvement: are we plateauing on training data, overfitting, converging slowly, etc.? 

Based on what I've seen so far, some ideas for improvement include:
- Not reusing the original training data after the 1st epoch (use new augmentations only for all successive epochs)—this would help with overfitting 
- Automated hyperparameter optimization for key architecture and training parameters—like I did in the [circle detection project](https://github.com/k-reddy/circle_detection)
- Weight the loss function based on classes - this would help if less common classes seem to be doing worse
- Adding extra augmented data of rare classes to equalize the number of samples for each class - this would address the same problem 

## Files & Running the Code
To run the code, you can use run_model.py. This file trains and saves the model. You will need to un-comment the lines that download the kaggle data for your first run and comment out the provided base_dir. 

If you want to run the model on the training data and examine the outputs, use the bottom of data_exploration.ipynb, or use the trainer.run_model() function, depending on what kind of outputs you need. 

The other files are:
- cnn.py: this specifies the architecture for the CNN (a MushroomClassifier object)
- trainer.py: defines a MushroomTrainer object that can be used to train and run the model 
- utils.py: has some data cleaning utils