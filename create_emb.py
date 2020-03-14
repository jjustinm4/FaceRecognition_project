#module to create preprocessed encodings and store it as 'embd.pickle'

import facenet
import os
import copy
import pickle
import cv2
import tensorflow as tf
import numpy as np
import align.detect_face

gpu_memory_fraction=1.0
margin=44
image_size=160
model='model/'

def main():

    images = load_images() #load images from disk
    with tf.Graph().as_default():

        with tf.Session() as sess:
            
            dataset = facenet.get_dataset('dataset') #for getting names of people
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            emb=list(zip(labels,emb))
            with open('embd.pickle','wb') as f:
        	    pickle.dump(emb,f)
#function to load images from disk
def load_images():
    dataset = facenet.get_dataset('dataset')
    paths, labels = facenet.get_image_paths_and_labels(dataset)

    img_list = []

    for image in paths:
        img = cv2.imread(image) #reading
        img=facenet.crop(img,False,160) #crop
        prewhitened = facenet.prewhiten(img) #some adjustments
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

if __name__ == '__main__':
    main()