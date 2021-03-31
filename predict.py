# LOADING THE MAIN LIBRARIES
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from PIL import Image

import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# argparse — Parser for command-line options, arguments and sub-commands
import argparse

# Defining a function to return all the arguments we added to the parser defined below and then return them
def arguments():
    # ArgumentParser will hold the necesssary information to show in the command line
    parser = argparse.ArgumentParser(description='Classifier: names of input flowers images')
    # Add_Arguments: Define how a single command-line argument should be parsed
    # help - A brief description of what the argument does.
    # dest - The name of the attribute to be added to the object returned by parse_args().
    # ArgumentParser generates the value of dest by taking the first long option string and stripping away the initial -- string
    # action - The basic type of action to be taken when this argument is encountered at the command line, default is store
    # required - to make this input essential
    parser.add_argument('--image_path', type = str, help = 'The location of the image on your local machine', required = True)
    parser.add_argument('--top_k', type = int, help = 'The number of most probable flowers distribution', default = '5')
    parser.add_argument('--model_saved', help = 'This is the model used for evaluation', type = str, required = True)
    parser.add_argument('--class_names', help = 'This is the mapping file', type = str, default = 'label_map.json')
    # All the arguments can be later called by using arguments.
    return (parser.parse_args())

# Defining a fucnction to load the model
def load_model(model_name):
    #loading the model using the name obtained from argument
    loaded = tf.keras.models.load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    return (loaded)

# Defining a function to process the image so that it has size suitable to be used with the model loaded
def process_image(image):
    image_size = 224
    image = tf.cast(image, tf.float32)
    # Image size is previously defined as 224
    image = tf.image.resize(image,(image_size, image_size))
    #normalizing so that it is between 0 and 1
    image = image / 255
    image = image.numpy()
    return(image)

# Defining a function that would read the json file to map names to labels
def class_label(filename):
    with open(filename, 'r') as f:
        class_names = json.load(f)
    return(class_names)

# Defining a function to make the prediction
def predict(image_path, model_used, top_k):
    # PIL.Image.open(fp, mode='r', formats=None)
    image = Image.open(image_path)
    image = np.asarray(image)
    #to make the shape as (224,224,3)
    image = process_image(image)
    # numpy.expand_dims(a, axis) and add one extra dimension
    image = np.expand_dims(image, axis=0)
   
    #using the saved model to make the prediction
    classes = 102
    ps = model_used.predict(image)
    label_list = np.arange(classes)
    probabilities = ps[0]
    # sorting the probability descendingly and showing the indices of them
    probabilities_indices = probabilities.argsort()[-top_k:][::-1]
    top_k_probabilities = probabilities[probabilities_indices]
    # Adding 1 so that both indices match
    top_k_labels = label_list[probabilities_indices+1]
    return (top_k_probabilities, top_k_labels)

def main():
    # Loaded
    loaded = load_model(arguments().model_saved)
    # class_names
    class_names = class_label(arguments().class_names)
    # top_k
    top_k  = arguments().top_k
    # image_path
    my_image_path = arguments().image_path
    
    image_path = my_image_path
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    probs, classes_labels = predict(image_path, loaded, top_k)
    labels_names = [class_names[str(label)] for label in classes_labels]
    print('The most probable flowers for this pic are: \n', labels_names)
    print('They have probabilities correspondingly as the following: \n', probs)
    
if __name__ == "__main__":
    main()
    
    