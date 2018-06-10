import csv
import cv2
import numpy as np
from numpy.random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths to folders with data
# I have images from different folders in order to have a way to easily replace one track by other one
file_name = 'driving_log.csv'
folders = ['1_track_1/', '1_track_2/', '1_track_11/', '1_track_21/']
img_= 'IMG/'

# Function to read all images directories 
def read_obs(file):
    lines = []
    with open(file) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)
    return lines
    
# Creating a sample with image location and angle of rotation. Adding correction to left/right images
sample = []
correction = 0.3
for folder in folders:
    img_folder = folder + img_
    file_csv = folder + file_name
    lines = read_obs(file_csv)
    for line in lines:
        measurement = float(line[3])
        for i in range(3):
            addr = img_folder + line[i].split('/')[-1]
            if i==1:
                measurement = measurement+correction
            elif i==2:
                measurement = measurement-correction
            sample.append([addr,measurement])

# To make additional correction because in several turns to left, the car turns to right
for i in range(2000, 2210):
    sample[i][1] -= 0.3
for i in range(1100, 1250):
    sample[i][1] -= 0.3
    
# Split sample to train and validation
from sklearn.model_selection import train_test_split
shuffle(sample)
train_samples, validation_samples = train_test_split(sample, test_size=0.3)

# Creating generator
# Including augmented images to generator
import sklearn
def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_samples = samples[offset:offset+int(batch_size/2)]

            images = []
            angles = []
            for batch_sample in batch_samples:
                image = cv2.imread(batch_sample[0])
                angle = batch_sample[1]
                images.append(image)
                angles.append(angle)
                # Augmented observations
                images.append(cv2.flip(image,1))
                angles.append(-1.0*angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=34)
validation_generator = generator(validation_samples, batch_size=34)

# to to make image gray (to have one dimension instead of three), because colors are not important
def output_of_lambda(input_shape):
    return (input_shape[0], input_shape[1], input_shape[2], 1)

def mean(x):
    return K.max(x, axis=3, keepdims=True)
    
# Build model. Because instead of having color image I have gray image I decided to decrease layers depths (in comparison with Nvidia model) . So, training time decreased proportionally 3 times as well as model size
from keras import backend as K
model3 = Sequential()
model3.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=(160,320,3)))
model3.add(Lambda(mean, output_shape=output_of_lambda))
model3.add(Lambda(lambda x: x/255.5 - 0.5))
model3.add(Convolution2D(8,   5, 5, subsample=(2,2), activation='relu'))
model3.add(Convolution2D(12,  5, 5, subsample=(2,2), activation='relu'))
model3.add(Convolution2D(16,  5, 5, subsample=(2,2), activation='relu'))
model3.add(Convolution2D(20,  3, 3, activation='relu'))
model3.add(Convolution2D(20,  3, 3, activation='relu'))
model3.add(Flatten())
model3.add(Dense(100))
model3.add(Dense(50))
model3.add(Dense(10))
model3.add(Dense(1))
model3.compile(loss='mse', optimizer='adam')

history3 = model3.fit_generator(train_generator,
                              samples_per_epoch=(2*len(train_samples)),
                             validation_data=validation_generator,
                              nb_val_samples=(2*len(validation_samples)),
                              nb_epoch=5)
model3.save('model.h5',overwrite=True)


### Plot the training and validation loss for each epoch
plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
