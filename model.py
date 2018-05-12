import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


class DrivingLogReader:
    def __init__(self, base_path, data_dirs):
        self.base_path = base_path
        self.data_dirs = data_dirs
        
        self.driving_log = self.read_all(self.base_path, self.data_dirs)
        self.record_count = len(self.driving_log)       
        

    def read_csv_data(self, path):
        lines = []

        with open(path) as csvfile:
            reader = csv.reader(csvfile)

            for line in reader:
                lines.append(line)
        return lines

    def read_all(self, base_path, data_dirs):
        tmp = []
        
        for track in data_dirs:
            path = '{}/{}/driving_log.csv'.format(base_path, track)
            
            print("Reading driving log from {}".format(path))      
            tmp.append(self.read_csv_data(path))

        driving_log = np.concatenate(tmp)
        return driving_log
    
    def train_valid_split(self, ratio=0.8):
        np.random.shuffle(self.driving_log)

        n_train = int(ratio * self.record_count)
        train, valid = self.driving_log[:n_train], self.driving_log[n_train:]
        
        return train, valid

def read_image_rgb(source_path):    
    image_bgr = cv2.imread(source_path)        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb

    
# Generators
import random

class Generator:
    def next_batch(self, driving_log, batch_size, shape=(140, 50)):    
        correction_factor = [0, 0.2, -0.2]

        # Create empty arrays to contain batch of features and labels#
        batch_images = np.zeros((batch_size, shape[1], shape[0], 3))
        batch_measurements = np.zeros((batch_size, 1))

        while True:
            i = 0
            while i < batch_size:
                # choose random index in features
                log = random.choice(driving_log)            
                img_index = random.choice(range(0, 3))

                source_path = log[img_index]            

                image_rgb = read_image_rgb(source_path)
                cropped = image_rgb[60:-10,20:-20]
                image = cv2.resize(cropped, shape, interpolation=cv2.INTER_AREA)

                measurement = float(log[3]) + correction_factor[img_index]

                batch_images[i] = image
                batch_measurements[i] = measurement

                batch_images[i+1] = np.fliplr(image)
                batch_measurements[i+1] = -measurement

                i += 2            

            yield batch_images, batch_measurements

from keras.models import Sequential
from keras.layers import Flatten, Dense,Lambda, Cropping2D, Dropout, Conv2D
from keras.layers.convolutional import Convolution2D    

class SteeringAnglePredictor:
    def __init__(self, input_shape=(50,140, 3)):
        self.input_shape = input_shape
        self.model = self.create_model(self.input_shape)
        self.generator = Generator()
    
    def create_model(self, input_shape):
        model = Sequential()
        model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
        model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(1164))
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

        model.compile(loss='mse', optimizer='adam')

        return model    
    
    def train(self, train_data, validation_data, batch_size=64, epochs=10):
        samples_per_epoch = (int(len(train_data)/batch_size + 1) * batch_size)
        print(samples_per_epoch)
        
        self.model.fit_generator(self.generator.next_batch(train_data, batch_size),
                        validation_data=self.generator.next_batch(validation_data, batch_size),
                        samples_per_epoch=samples_per_epoch,
                        nb_val_samples=len(validation_data), 
                        nb_epoch=epochs
                       )
        
    def save_model(self, file_name):
        self.model.save(file_name)


def train_model():
    # Load driving log
    driving_log_reader = DrivingLogReader(
        base_path = '/home/bibagimon/nanodegree/data',
        data_dirs = ['track1_normal', 'track1_reverse', 'udacity']
    )   
    print("Total driving log records: {}".format(driving_log_reader.record_count))
    
    # Split dataset to train & validation
    train, valid = driving_log_reader.train_valid_split()    
    
    print("Training dataset shape: {}".format(train.shape))
    print("Validation dataset shape: {}".format(valid.shape))

    # Define model (Nvidia based)
    predictor = SteeringAnglePredictor()
    
    # Train the model
    predictor.train(train, valid, batch_size=64, epochs=5)
    predictor.save_model('model.h5')
    
if __name__== "__main__":
    train_model()
