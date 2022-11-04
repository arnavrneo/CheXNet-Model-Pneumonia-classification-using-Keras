!pip install kaggle
!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json  # download and move your kaggle.json to the root directory

!kaggle datasets download paultimothymooney/chest-xray-pneumonia

!unzip chest-xray-pneumonia

import re
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = [180, 180]
EPOCHS = 25

"""As validation data and training data are imbalanced, we will join them together and then create training and validation dataset out of them."""

filenames = tf.io.gfile.glob('chest_xray/train/*/*')
filenames.extend(tf.io.gfile.glob('chest_xray/val/*/*'))

train_filenames, val_filenames = train_test_split(filenames, test_size=0.2)

train_filenames[:5]

"""How many are there?"""

NORMAL = len([file for file in train_filenames if "NORMAL" in file])
PNEUMONIA = len([file for file in train_filenames if "PNEUMONIA" in file])

print("Normal images in training set: ", NORMAL)
print("Pneumonia images in training set: ", PNEUMONIA)

"""As classes are imbalanced, we will correct this later on.

`tf.data` is used for reading and transformations.
"""

train_list_ds = tf.data.Dataset.from_tensor_slices(train_filenames)
val_list_ds = tf.data.Dataset.from_tensor_slices(val_filenames)

for i in train_list_ds.take(5):
    print(i.numpy())

"""`tf.data.experimental.cardinality` returns the cardinality of the dataset, if known.

**Checking the no. of images in our training & validation sets.**
"""

TRAIN_COUNT = tf.data.experimental.cardinality(train_list_ds).numpy() # numpy otherwise it returns a tensor object.
VAL_COUNT = tf.data.experimental.cardinality(val_list_ds).numpy() 

print(f'Training images count: {TRAIN_COUNT}\nValidation images count: {VAL_COUNT}')

"""The two labels are as follows:

**1 indicates Pneumonia**

**0 indicates Normal**
"""

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep) # convert the path to a list of path components
    
    return parts[-2] == "PNEUMONIA" # The second to last is the class-directory

    """ I think whats happening above is this:
        The paths are in the form : 'chest_xray/train/PNEUMONIA/person30_bacteria_155.jpeg'
        So, I split them into individual words and then I try to return the second 
        word in backwards manner. So, path[-2] gives the class name 
    """

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3) # convert the compressed string to a 3d int tensor
    img = tf.image.convert_image_dtype(img, tf.float32) # converting to floats in range[0, 1]
    
    return tf.image.resize(img, IMAGE_SIZE) # resizing

def process_path(path):
    label = get_label(path)
    
    img = tf.io.read_file(path)
    img = decode_img(img)
    
    return img, label

train_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

val_ds = val_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

print(train_ds)

"""Checking the shape of (image, label) pair."""

for image, label in train_ds.take(1):
    print(f'image shape: {image.numpy().shape}')
    print(f'label: {label.numpy()}')

"""# Formatting test dataset"""

test_list_ds = tf.data.Dataset.list_files('chest_xray/test/*/*')
TEST_IMAGE_COUNT = tf.data.experimental.cardinality(test_list_ds).numpy()
test_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(32)

TEST_IMAGE_COUNT

"""# Viz the dataset

We can also cache the dataset to speed up processing.
"""

def pre(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
            
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    
    ds = ds.repeat() # repeat forever
    ds = ds.batch(32)
    
    
    ds = ds.prefetch(buffer_size=AUTOTUNE) # prefetch lets the dataset fetch batches in the
                                           # background while the model is training
    
    return ds

"""Calling the next batch iteration of training data"""

train_ds = pre(train_ds)
val_ds = pre(val_ds)

image_batch, label_batch = next(iter(train_ds))

"""Plotting the dataset"""

def show_img(image_batch, label_batch):
    plt.figure(figsize=[10, 10])
    for i in range(25):
        ax = plt.subplot(5, 5, i+1)
        plt.imshow(image_batch[i])
        
        if label_batch[i]:
            plt.title("PNEUMONIA")
        else:
            plt.title("NORMAL")
        plt.axis("off")
        
show_img(image_batch.numpy(), label_batch.numpy())

"""# Model building

The weights are of CheXnet
"""

from keras.models import Sequential, Model
from keras.applications import densenet
from keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D, AveragePooling2D

chexnet_weights = '/content/brucechou1983_CheXNet_Keras_0.3.0_weights.h5'

with tf.distribute.get_strategy().scope():
    
    base = densenet.DenseNet121(weights=None,
                               include_top=False,
                               input_shape=(180,180,3)
                               )
    
    pred = keras.layers.Dense(14, activation='sigmoid', name='pred')(base.output)
    
    base = keras.Model(inputs=base.input, outputs=pred)
    
    base.load_weights(chexnet_weights)
    print("CheXNet loaded")
    
    base.trainable = False
    new_model = tf.keras.layers.GlobalAveragePooling2D()(base.layers[-3].output) 

    output = tf.keras.layers.Dense(1, activation='sigmoid')(new_model) 

    model = Model(base.input, output)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC()])

base.layers[-5:-1]

history = model.fit(
    train_ds,
    steps_per_epoch= TRAIN_COUNT // 32,
    epochs=5,
    validation_data=val_ds,
    validation_steps=VAL_COUNT // 32,
)

model.metrics_names

fig, ax = plt.subplots(1, 2, figsize=[20, 3])
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss']):
  ax[i].plot(history.history[met])
  ax[i].plot(history.history['val_' + met])
  ax[i].set_title('Model {}'.format(met))
  ax[i].set_xlabel('epochs')
  ax[i].set_ylabel(met)
  ax[i].legend(['train', 'val'])

"""# Predicting the results"""

loss, auc = model.evaluate(test_ds)

"""# Saving the model"""

model.save('my_model.h5')

model.save_weights('my_weights.ckpt')
