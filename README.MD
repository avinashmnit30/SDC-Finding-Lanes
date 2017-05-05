# Self Driving Cars: Finding Lane Lines on the Road

---

## Dependencies
* Python 3.5 or later
* Numpy
* OpenCV
* Matplotlib
* Moviepy

## Using the Pipeline on Images
* Add the orignal image to the test_images folder 
* Run the lane_detection.py file 
* The result images are saved in image_result folder

## Using the Pipeline on Videos
* Add the orignal video to the test_videos folder
* Run the lane_detection.py file
* The result videos are saved in video_result folder


## Pipeline
![model](./Pipeline/pipeline_v2jpeg1.jpeg)

As shown in the figure the pipeline consists of 6 steps:-
![Orignal](./test_images/solidWhiteRight.jpg)
### Convert to Grayscale: 
This will convert the image to a single channel for later use by the Canny Transform (Step 3) for finding the gradient of the pixel intensity.
![Gray](./image_result/2_gray_solidWhiteCurve.jpg)
### Apply Gaussian Smoothing (or Gaussian Blur): 
This step filters out the noise in the image.
![Blur](./image_result/3_blur_solidWhiteCurve.jpg)
### Apply Canny Transform: 
This is used to find the edges in the image. An edge is formed by pixels where the gradient of the image intensity sharply changes.
![Canny](./image_result/4_canny_solidWhiteCurve.jpg)
### Find the Region of Interest: 
This step separates out the region of image where the road lanes are possible to occur.
![Masked](./image_result/5_masked_solidWhiteCurve.jpg)
### Apply Hough Transform: 
This step separates out the lines in the image that probably form the lanes.
![Hough](./image_result/6_hough_solidWhiteCurve.jpg)
### Merge with the Original Image: 
This step merges the output of the hough transform to the original image.
![Final](./image_result/2_final_solidWhiteCurve.jpg)

##
![V1](./video_result/final_solidWhiteRight.mp4)

##
[![V2](./test_images/solidWhiteRight.jpg)](./video_result/final_solidWhiteRight.mp4)


## For more information and results
https://wp.me/p9KXJx-T

## Contributers
Avinash Sharma: avinashmnit30@gmail.com

## License and Copyright
© Avinash Sharma

