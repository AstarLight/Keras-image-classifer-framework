## translate ppm image to png image

import cv2
import os

ORIGINAL_TRAIN_PATH = '../BelgiumTSC_Training/Training'
ORIGINAL_TEST_PATH = '../BelgiumTSC_Testing/Testing'

for train_class in os.listdir(ORIGINAL_TRAIN_PATH):
    if not os.path.isdir('../traffic-sign/train/'+train_class):
        os.mkdir('../traffic-sign/train/'+train_class)        
    for pic in os.listdir(ORIGINAL_TRAIN_PATH + '/'+ train_class):
        if not (pic.split('.')[1] == 'ppm'):
            continue
        im = cv2.imread(ORIGINAL_TRAIN_PATH + '/' +train_class+'/'+ pic) 
        name = pic.split('.')[0]
        new_name = name+'.png'
        print (new_name)
        cv2.imwrite('../traffic-sign/train/' + train_class + '/'+ new_name,im)
        

for test_class in os.listdir(ORIGINAL_TEST_PATH):
    if not os.path.isdir('../traffic-sign/test/'+test_class):
        os.mkdir('../traffic-sign/test/'+test_class)        
    for pic in os.listdir(ORIGINAL_TEST_PATH + '/'+ test_class):
        if not (pic.split('.')[1] == 'ppm'):
            continue
        im = cv2.imread(ORIGINAL_TEST_PATH + '/' +test_class+'/'+ pic) 
        name = pic.split('.')[0]
        new_name = name+'.png'
        print (new_name)
        cv2.imwrite('../traffic-sign/test/' + test_class + '/'+ new_name,im)
