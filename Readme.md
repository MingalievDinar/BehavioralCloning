# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/output_5_1.png "Normal Image"
[image2]: ./Images/output_5_1_flip.png "Flipped Image"
[image3]: ./Images/output_15_0.png "MSE"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model architecture is similar to Nvidia with small corrections. My model consists of a convolution neural network with filter sizes between 3x3 and 5x5; depths between 8 and 20 in Convolution layers and between 1-100 in Dense layers (model.py lines 91-105) 

The model includes RELU layers to introduce nonlinearity.
Additionally the data is:
* Normalized in the model using a Keras lambda layer (code line 94)
* Cropped (code line 92)
* Decrease image depth to 1 instead of 3 (in line 93 with helping functions in lines 83-87).

#### 2. Attempts to reduce overfitting in the model

The model doesn't contain dropout layers in order to reduce overfitting. Instead of that I checked the model MSE on training and validation set. 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 116-122). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

![alt text][image3]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 105).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road with steering angles correction 0.3 (model.py line 27).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make it simple (number or layers and depth), be able to keep the car on the road, to have low validation loss

My first step was to use a convolution neural network model similar to the AlexNet. I thought this model might be appropriate because it is common practice to start with simple enough model which has already showed good results in many tasks. But later I realized that Nvidia model trains faster and takes less size. In addition, I decreased original convolution layers depth because input images are not three dimensions but only one. As the result, the model trains faster.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Usually I had less validation loss in the first one-two epochs, but of course later training loss became lower than validation loss. I kept training as well as validation and training losses were decreasing. It took me 5 iterations. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I made correction (model.py lines 43-46). 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 91-105) consisted of five convolution layers followed by three dense layers. The depth of layers:
* Convolutions: 8, 12, 16, 20 and 20
* Dense: 100, 50, 10


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle driving two laps in the opposite direction.

To augment the data sat: flipped images and angles thinking that this would help to have more data without actually collecting it. For example, here is an image that has then been flipped:

![alt text][image2]

After the collection process, I had 15738x2 number of data points (x2 - to count augmented images).

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 (in more epochs validation loss starts rising). I used an adam optimizer so that manually training the learning rate wasn't necessary.
