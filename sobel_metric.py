from PIL import Image
from IPython.display import Image,display as myImage,display
import numpy as np
import matplotlib.pyplot as plt
import os, glob
import cv2
import math
import warnings
from scipy.spatial import distance
import time
import scipy
import argparse


def calc_blur(img):
    #img = cv2.imread(img) # Read in the image
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    
    x1,x2 = sobely.shape
    return np.sum(magnitude)/(x1*x2)
    
    
def run_images(image_paths):
    im_dict = {}
    for ind,image_path in enumerate(image_paths):        
        if ind % 1000 == 0:
            print('processed ', ind/len(image_paths) ,' percent of data')

        img = cv2.imread(image_path,0)
        blur = calc_blur(img)
        im_dict[image_path] = blur

    im_dict = {k: v for k, v in sorted(im_dict.items(), key=lambda item: item[1])}
    return im_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    
    #image_paths = glob.glob('Test/12000/**/*_rawcolor.jpeg',recursive=True)
    image_paths = glob.glob(args.input_path,recursive=True)
    
    start_t = time.time()
    im_dict = run_images(image_paths)  
    
    with open(args.output_path, 'w') as f: 
        for key, value in im_dict.items(): 
            f.write('%s,%s\n' % (key, value))
    print('Wrote quality scores to file: ',args.output_path)
    print('run time (seconds): ', time.time()-start_t)