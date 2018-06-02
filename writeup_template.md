# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/center_2018_05_09_23_14_22_317.jpg "Center lane driving"
[image2]: ./output_images/center_2018_05_09_23_14_22_317_flipped.jpg "Flipped"
[image3]: ./output_images/track1_reverse_center1.jpg "Recovery Image"
[image4]: ./output_images/track1_reverse_center2.jpg "Recovery Image"
[image5]: ./output_images/track1_reverse_center3.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./output_images/model.png "Keras Model"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Note: I have modified the Udacity provided drive.py. The image sent to predict steering angle is cropped and resized to adapt with the model input size.

```
# Before
steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
```
```
# After
cropped = image_array[60:-10,20:-20,:]
resized = cv2.resize(cropped,(140, 50), interpolation = cv2.INTER_AREA)        
steering_angle = float(model.predict(resized[None, :, :, :], batch_size=1))
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64 (model.py lines 96-114) 

The data is normalized in the model using a Keras lambda layer (code line 98). The model includes RELU layers to introduce nonlinearity (code line 99~102).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 103). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 35-41). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 112).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

Mainly 3 dataset is used:
- Udacity
- Track 1 - Normal
- Track 2 - Reverse


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a well known architecture.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because original LeNet architecture is very effective in image classification problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The mean squared error on the training set and validation set was not much different. However, the vehicle was falling off the track and later it was found that `cv2.imread()` reads the image in BGR color space, but the model was trained for RGB image.

After the color space mismatch was fixed, while the vehicle could drive better than before, still it was falling off. As no image augmentation and steering angle adjustment was not done, using track2 driving log didn't give better result as well.

Then I tried to include Udacity provided data. With these improved training data, finally model became good enough for the vehicle to drive autonomously around the track 1 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 198-110) is as follows:

![alt text][image8]

The model closely follows LeNet architecture with last conv layer removed (as the input image size is smaller than the original model).


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle on track one revese using center lane driving:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Note: Track two driving log was also recorded. However, the model didn't generalize well. So this dataset is not used for the training.

After the collection process, I had 12,184 number of data points.

To augment the data set, I also flipped images and angles to left/right avoid biases in driving log. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

Batch generator was used to increase training data with more samples (`model.py` line #53-#84).

I finally randomly shuffled the data set and put 20% of the data into a validation set.

* Total driving log records: 12187
* Training dataset records: 9749
* Validation dataset records: 2438

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 10 with batch size 64. I used an adam optimizer so that manually training the learning rate wasn't necessary.
