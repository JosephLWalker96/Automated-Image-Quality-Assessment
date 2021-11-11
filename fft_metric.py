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

# function that creates a weighted mask where values furthest from the center of the matrix (which correpsonds to higher frequencies) are larger
def get_distance_1(y, x):
    mid_x, mid_y = (scipy.array([6,6]) - 1) / float(2)
    return ((y - mid_y) ** 2 + (x - mid_x) ** 2) ** 0.5


# function that calculates blur using the fast fourier transform
def calc_blur(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Get weight matrix
    rows, cols = img.shape
    x_inds = np.arange(rows)
    y_inds = np.arange(cols)
    matrix = get_distance_1(x_inds[:,None], y_inds)
    
    check = np.isfinite(magnitude_spectrum)
    if not check.all(): # Check if inf or nan values are present 
        check = check.reshape(-1)
        matrix = matrix.reshape(-1)
        magnitude_spectrum = magnitude_spectrum.reshape(-1)
        
        indices = np.where(check == True)
        
        matrix = matrix[indices]
        magnitude_spectrum = magnitude_spectrum[indices]
 
        return sum(np.multiply(matrix,magnitude_spectrum)) / len(matrix)
    else:
        return sum(sum(np.multiply(matrix,magnitude_spectrum))) / (rows*cols)


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
