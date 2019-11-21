import pandas as pd
import numpy as np
import os
import sys
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

# Dronet files import
import utils
from common_flags import FLAGS
def _main():
    print('inside')
    base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

    # Print mobilenet summary
    print(base_model.summary())

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dense(1024,activation='relu')(x) #dense layer 2
    x=Dense(512,activation='relu')(x) #dense layer 3
    preds=Dense(2,activation='softmax')(x) #final layer with softmax activation

    model=Model(inputs=base_model.input,outputs=preds)

    # Print mobilenet summary
    print(model.summary())
    for i,layer in enumerate(model.layers[:(len(model.layers)-4)]):
            print('Setting as non-trainable',i,layer.name)
            layer.trainable=False

    for i,layer in enumerate(model.layers[(len(model.layers)-4):]):
            print('Setting as trainable',i,layer.name)
            layer.trainable=True


    crop_img_width, crop_img_height = FLAGS.crop_img_width, FLAGS.crop_img_height
    img_width, img_height = FLAGS.img_width, FLAGS.img_height
    # Generate training data with real-time augmentation
    train_datagen = utils.DroneDataGenerator(rotation_range = 0.2,
                                            rescale = 1./255,
                                            width_shift_range = 0.2, 
                                            height_shift_range=0.2)

    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        shuffle = True,
                                                        color_mode='rgb',
                                                        target_size=(img_width, img_height),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = FLAGS.batch_size)

    val_datagen = utils.DroneDataGenerator(rescale = 1./255)

    val_generator = val_datagen.flow_from_directory(FLAGS.val_dir,
                                                    shuffle = True,
                                                    color_mode='rgb',
                                                    target_size=(img_width, img_height),
                                                    crop_size=(crop_img_height, crop_img_width),
                                                    batch_size = FLAGS.batch_size)

def main(argv):
    # Utility main to load flags
    print('started')
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print ('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))
        sys.exit(1)
    print('reached here')
    _main()

if __name__ == "__main__":
    main(sys.argv)
