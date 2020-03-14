import os
import serial
import facenet
import copy
import pickle
import cv2
import alert
import tensorflow as tf
import numpy as np
import align.detect_face

tf.logging.set_verbosity(tf.logging.ERROR)

gpu_memory_fraction=1.0
margin=44
image_size=160
model='model/'

def main():
    #load processed image, its original and bounding boxes from camera image
    images,orginal,bb = load_camera_image()

    dataset = facenet.get_dataset('dataset') #for getting names of people
    paths, labels = facenet.get_image_paths_and_labels(dataset)
    class_names = [ cls.name.replace('_', ' ') for cls in dataset]
    #load previously stored encodings
    with open('embd.pickle','rb') as f:
        feature_array = pickle.load(f)
    
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
            
        nrof_images = len(feature_array)
        
 ## Step 2: Find the closest encoding ##

        # Initialize "min_dist" to a large value, say 100
        min_dist = 100
        # Loop over the database names and encodings.
        for i in range(nrof_images):
            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(feature_array[i][1,:]-emb[0,:])
            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name.
            if dist<min_dist:
                min_dist = dist
                identity = class_names[feature_array[i][0]]
        if min_dist > 0.95: #check for threshold 
            print("Not in the database.")
            print('Unauthorized Person')
            print ("it's " + str(identity) + ", the distance is " + str(min_dist)) #dont mind this code
            #alert.sendmail() #send mail if unauthorised
            exit(0)
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))
            print('Authorized Person')
            
            cv2.rectangle(orginal, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)    #boxing face in the image

            #plot result id under box
            text_x = bb[0]
            text_y = bb[3] + 20
            cv2.putText(orginal, identity, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
            cv2.imshow("Hi",orginal)
            
            #ser=serial.Serial('/dev/ttyACM0',9600) #open the door
            #ser.write(b'1')

            if cv2.waitKey(100000) == ord('q'):
                #ser=serial.Serial('/dev/ttyACM0',9600) #close the doors
                #ser.write(b'2')
                exit(0)
            cv2.destroyAllWindows()

## function to get image from camera
def load_camera_image():

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    original = cv2.imread('crop.png') #read the image
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
    return images,original,bb

if __name__ == '__main__':
    main()