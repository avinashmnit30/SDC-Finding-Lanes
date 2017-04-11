# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 00:27:24 2018

@author: RushMe
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import random
from moviepy.editor import VideoFileClip
import math


class image_processor(object):
    def __init__(self, img):
        self.img = img
    def grayscale(self):
        '''
        This will convert the RGB image (self.img) to a single channel image (self.gray). 
        '''
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
    def gaussian_blur(self):
        '''
        This reduce the noise in the image (self.gray). 
        '''
        self.blur = cv2.GaussianBlur(self.gray, (self.kernel_size, self.kernel_size), 0)
    def canny_transform(self):
        '''
        This will seperate out the high gradient pixels of the image (self.blur)
        '''
        self.canny = cv2.Canny(self.blur, self.low_threshold, self.high_threshold)
    def ROI_mask(self):
        """
        Applies an image mask (to self.canny).
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        ysize = self.canny.shape[0]
        xsize = self.canny.shape[1]
        vertices = [np.array([ [0, ysize], [xsize/2,(ysize/2)+ 10], [xsize,ysize] ], np.int32)]
        # Defining the image mask
        mask = np.zeros_like(self.canny)   
        # Defining color to fill the mask depending on the number of channels in the image
        if len(self.canny.shape) > 2:
            channel_count = self.canny.shape[2]  
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255    
        # Filling the pixels inside the polygon (formed by "vertices") with the defined color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.masked = cv2.bitwise_and(self.canny, mask)

    def hough_transform(self):
        '''
        This will find out the probable lane lines in the image (self.masked).
        '''
        lines = cv2.HoughLinesP(self.masked, self.rho, self.theta, self.threshold, np.array([]), minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)
        self.hough = np.zeros((self.masked.shape[0], self.masked.shape[1], 3), dtype=np.uint8)
        draw_lines(self.hough, lines)
    def weighted_img(self):
        """
        Merges the initial_img (self.img) with hough_img (self.hough) using the formula: 
        initial_img * α + hough_img * β + λ    
        """
        self.final =  cv2.addWeighted(self.img, self.α, self.hough, self.β, self.λ)
    def find_lanes(self, kernel_size = 5, low_threshold = 50, high_threshold = 150, rho = 1, theta = np.pi/180, threshold = 15, min_line_len = 150, max_line_gap = 75, α=0.8, β=1.0, λ=0.0):
        '''
        kernal_size: Gaussian kernal parameters
        low_threshold, high_threshold: Canny Transform parameters
        rho, theta, threshold, min_line_len, max_line_gap: Hough Transform Parameters
        α, β, λ: Merge Function (cv2.addWeighted) parameters  
        '''
        self.kernel_size = kernel_size  
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.rho = rho # distance resolution in pixels of the Hough grid
        self.theta = theta # angular resolution in radians of the Hough grid
        self.threshold = threshold     # minimum number of votes (intersections in Hough grid cell)
        self.min_line_len = min_line_len #minimum number of pixels making up a line
        self.max_line_gap = max_line_gap    # maximum gap in pixels between connectable line segments
        self.α=α 
        self.β=β 
        self.λ=λ
        self.grayscale()
        self.gaussian_blur()
        self.canny_transform()
        self.ROI_mask()
        self.hough_transform()
        self.weighted_img()
    def save_images(self, path, name):
        plt.imsave(path + '/' + '1_orignal_'+ name , self.img)
        plt.imsave(path + '/' + '2_gray_'+ name , self.gray, cmap='gray')
        plt.imsave(path + '/' + '3_blur_'+ name , self.blur, cmap='gray')
        plt.imsave(path + '/' + '4_canny_'+ name , self.canny, cmap='gray')
        plt.imsave(path + '/' + '5_masked_'+ name , self.masked, cmap='gray')
        plt.imsave(path + '/' + '6_hough_'+ name , self.hough, cmap='gray')
        plt.imsave(path + '/' + '7_final_'+ name , self.final)

    
        
def process_image(image):
    im_object = image_processor(image)
    im_object.find_lanes()
    return(im_object.final)


    
def process_Video(file_name, video_path, save_path):
    """
    Applies the process_image pipeline to the video 
    """
    clip = VideoFileClip(video_path + '/' + file_name)
    outputClip = clip.fl_image(process_image)
    outVideoFile = save_path + '/final_' + file_name
    outputClip.write_videofile(outVideoFile, audio=False)
                
        

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """  
    This function draws of particular color and thickness om the image (img)
    """
    right_slopes = []
    right_intercepts = []
    left_slopes = []
    left_intercepts = []
    left_points_x = []
    left_points_y = []
    right_points_x = []
    right_points_y = []

    y_max = img.shape[0]
    y_min = img.shape[0]

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2-y1)/(x2-x1)
            if slope < 0.0 and slope > -math.inf:
                left_slopes.append(slope) # left line
                left_points_x.append(x1)
                left_points_x.append(x2)
                left_points_y.append(y1)
                left_points_y.append(y2)
                left_intercepts.append(y1 - slope*x1)
            if slope > 0.0 and slope < math.inf:
                right_slopes.append(slope) # right line
                right_points_x.append(x1)
                right_points_x.append(x2)
                right_points_y.append(y1)
                right_points_y.append(y2)
                right_intercepts.append(y1 - slope*x1)
            y_min = min(y1,y2,y_min)
            
    if len(left_slopes) > 0:
        left_slope = np.mean(left_slopes)
        left_intercept = np.mean(left_intercepts)
        x_min_left = int((y_min - left_intercept)/left_slope) 
        x_max_left = int((y_max - left_intercept)/left_slope)
        cv2.line(img, (x_min_left, y_min), (x_max_left, y_max), [255, 0, 0], 8)
    
    if len(right_slopes) > 0:
        right_slope = np.mean(right_slopes)
        right_intercept = np.mean(right_intercepts)
        x_min_right = int((y_min - right_intercept)/right_slope) 
        x_max_right = int((y_max - right_intercept)/right_slope)
        cv2.line(img, (x_min_right, y_min), (x_max_right, y_max), [255, 0, 0], 8)
            
            

# To test the images
img_path = 'test_images'
save_path = 'image_result'
for file in os.listdir(img_path):
    print(file)
    image = mpimg.imread(img_path + '/' + file)
    image = image_processor(image)
    image.find_lanes()
    image.save_images(save_path, file)



# To test the video
video_path = 'test_videos'
save_path = 'video_result'
for file in os.listdir(video_path):
    print(file)
    process_Video(file, video_path, save_path)
    