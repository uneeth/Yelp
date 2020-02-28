# -*- coding: utf-8 -*-

import io
import PIL
import h5py
import keras
import pandas as pd
import numpy as np
import simplejson as jsn
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from keras import optimizers
from keras import backend as k
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten


def load_json_data(data_dir):
    json_image_data = []
    json_location = data_dir + '/json/'
    with open(json_location + 'photos.json', 'r') as file:
        for line in file:
            json_image_data.append(jsn.loads(line))
    json_image_data = pd.DataFrame(json_image_data)
    json_image_data = json_image_data[['label', 'photo_id']]
    return json_image_data[0:50]

def create_h5py(data_dir):
    image_df = load_json_data(data_dir)
    h5file_name = data_dir + '/dataset/' + 'yelp_dataset.h5'
    with h5py.File(h5file_name, 'w') as h5file:
        for image in image_df['photo_id']:
            _file = open(data_dir + '/photos/' + image + '.jpg', 'rb')
            binary_data = _file.read()
            img = np.frombuffer(binary_data, dtype='uint8')
            h5file[image] = img


def load_h5py(data_dir, width, height):
    image_df = load_json_data(data_dir)
    image_array = np.zeros((len(image_df), height, width, 3), dtype=np.uint8)
    h5file_name = data_dir + '/dataset/' + 'yelp_dataset.h5'
    counter = 0
    with h5py.File(h5file_name, "r") as h5file:
        for image in image_df['photo_id']:
            raw = np.array(h5file[image])
            img = Image.open(io.BytesIO(np.asarray(raw)))
            img = img.resize((width, height), resample=PIL.Image.NEAREST)
            img = np.array(img).astype(np.uint8)
            image_array[counter] = img
            plt.imshow(img)
            plt.show()
            counter += 1
    label_actual = np.sort(np.unique(image_df['label']))
    label_encode = image_df['label'].replace(label_actual, range(len(label_actual)))
    label_array = keras.utils.to_categorical(label_encode, len(label_actual))
    return np.array(image_array), np.array(label_array)
#     return np.array(image_array), np.array(label_encode)

def build_model(input_shape, outputs, activation, learn_rate, loss, optimizer, drop_rate):
    # Build model
    yelp_model = Sequential()
    yelp_model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, data_format='channels_last'))
    yelp_model.add(Activation(activation))
    yelp_model.add(MaxPooling2D((2, 2), strides=1))

    yelp_model.add(Conv2D(64, (3, 3)))
    yelp_model.add(Activation(activation))
    yelp_model.add(Activation(activation))
    yelp_model.add(MaxPooling2D((2, 2), strides=1))

    yelp_model.add(Conv2D(128, (3, 3)))
    yelp_model.add(Activation(activation))
    yelp_model.add(MaxPooling2D((2, 2), strides=1))

    yelp_model.add(Dropout(drop_rate))

    yelp_model.add(Conv2D(256, (3, 3), padding='same'))
    yelp_model.add(Activation(activation))
    yelp_model.add(Conv2D(256, (3, 3)))
    yelp_model.add(Activation(activation))
    yelp_model.add(Conv2D(256, (3, 3)))
    yelp_model.add(Activation(activation))
    yelp_model.add(MaxPooling2D((2, 2), strides=1))

    yelp_model.add(Dropout(drop_rate))

    yelp_model.add(Conv2D(512, (3, 3), padding='same'))
    yelp_model.add(Activation(activation))
    yelp_model.add(Conv2D(512, (3, 3)))
    yelp_model.add(Activation(activation))
    yelp_model.add(Conv2D(512, (5, 5)))
    yelp_model.add(Activation(activation))
    yelp_model.add(MaxPooling2D((2, 2), strides=1))

    yelp_model.add(Dropout(drop_rate))

    yelp_model.add(Flatten())

    yelp_model.add(Dense(4096))
    yelp_model.add(Activation(activation))
    yelp_model.add(Dense(2048))
    yelp_model.add(Activation(activation))
    yelp_model.add(Dense(1024))
    yelp_model.add(Activation(activation))

    yelp_model.add(Dropout(drop_rate))

    yelp_model.add(Dense(outputs))
    yelp_model.add(Activation(k.softmax))

    # Optimize model
    if optimizer == 'SGD':
        opt = optimizers.SGD(lr=learn_rate)
    elif optimizer == 'RMSprop':
        opt = optimizers.RMSprop(lr=learn_rate)
    elif optimizer == 'Adagrad':
        opt = optimizers.Adagrad(lr=learn_rate)
    else:
        opt = optimizers.Adam(lr=learn_rate)

    # Compile model
    yelp_model.compile(loss=loss, optimizer=opt)

    return yelp_model

# dataset details
data_dir = '/Users/akula/Documents/ML/Fall 2019/CSE 574 - Machine Learning/Final Project/yelp_photos'
# image details
height = 32
width = 32

# create data
""" create only if h5py file is not present """
create_h5py(data_dir)

# load data
x, y = load_h5py(data_dir, width, height)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# Grid Search
model = KerasClassifier(build_fn=build_model, verbose=0)

# define the grid search parameters
batch_size = [32, 64, 128, 256]
epochs = [30, 50, 100]
loss = ['mean_squared_error', 'categorical_crossentropy']
optimizer = ['Adam', 'SGD', 'RMSprop', 'Adagrad']
learn_rate = [0.001, 0.01, 0.1]
drop_rate = [0.25, 0.5]
activation = ['relu', 'sigmoid', 'tanh']

# param_grid = dict(input_shape=[x_train.shape[1:]],
#                     outputs=[1], 
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    activation=activation,
#                    learn_rate=learn_rate,
#                    optimizer=optimizer,
#                    loss=loss,
#                    drop_rate=drop_rate
#                    )

# fold = StratifiedKFold(n_splits=5).split(x_train, y_train)

# grid = GridSearchCV(estimator=model,
#                     param_grid=param_grid,
#                     cv=fold)

# grid_result = grid.fit(x_train, y_train)

# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# create model
model = build_model(x_train.shape[1:],
                    y_train.shape[-1],
                    activation[0],
                    learn_rate[0],
                    loss[0],
                    optimizer[0],
                    drop_rate[0])

print('built model..')

# Data Augmentation
datagen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=0.0,
                            fill_mode='nearest',
                            horizontal_flip=True,
                            vertical_flip=True,
                            rescale=1. / 255,
                            preprocessing_function=None,
                            validation_split=0.25)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size[0]),
                   epochs=epochs[0])

# Save model with weights
model.save(data_dir + '/models/yelp_model.h5')

score = model.evaluate(x_test, y_test)
print('Model loss:', score)
#print('Model accuracy:', score)