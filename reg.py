import tkinter as tk
from tkinter import simpledialog
import os
import snap
import shutil
import cv2
import tensorflow as tf
import numpy as np
import align.detect_face
import facenet
import pickle

gpu_memory_fraction=1.0
margin=44
image_size=160
model='model/'

def load_camera_image():

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    camera = cv2.VideoCapture(0)
    while(True):
        return_value, original = camera.read()
        cv2.imshow('capture',original)
        if cv2.waitKey(1) == ord('q'):
            break
    camera.release()
    del(camera)

    img=original
    #Let the network do its magic!!
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    img_list = []

    img_size = np.asarray(img.shape)[0:2]
    #Network finds the bounding boxes here
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if len(bounding_boxes) < 1: #condition to check if there is any face
        print("can't detect face")
        cv2.imshow('Image', img)
        if cv2.waitKey(100000) == ord('q'):
            exit(0)
        cv2.destroyAllWindows()
    #bounding box calculations follows
    det = np.squeeze(bounding_boxes[0,0:4])
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    #cropping and aligning
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = cv2.resize(cropped, (image_size, image_size))
    prewhitened = facenet.prewhiten(aligned)
    img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def create_embd():
    #load processed image, its original and bounding boxes from camera image
    images = load_camera_image()
    
## Step 1: Compute the target "encoding/embedding" for the image
    with tf.Graph().as_default():

        with tf.Session() as sess:
      
            # Load the model
            facenet.load_model(model)
    
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb,images[0]
            
def register(answer):
    dirName='dataset\\'+answer
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print(dirName ,  " Created ")
        embd,image=create_embd()
        #load previously stored encodings
        with open('embd.pickle','rb') as f:
            feature_array = pickle.load(f)
        
        dataset = facenet.get_dataset('dataset') #for getting names of people
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        class_names = [ cls.name.replace('_', ' ') for cls in dataset]
        for i,name in enumerate(class_names):
            if name==answer:
                break
        feature_array.append(list((i,embd)))
        with open('embd.pickle','wb') as f:
            pickle.dump(feature_array,f)
        cv2.imwrite(dirName+'reg.png',image)
    else:    
        print(dirName ,  " already exists")
def main():
    application_window = tk.Tk()

    answer = simpledialog.askstring("Input", "What is your name?",
                                    parent=application_window)
    if answer is not None:
        print("Your name is ", answer)
        register(answer)
    else:
        print("You don't have a name?")
        exit(0)
if __name__ == "__main__":
    main()



