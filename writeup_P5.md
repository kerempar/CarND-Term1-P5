##Vehicle Detection
###Kerem Par
####kerempar@gmail.com
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_not_vehicle.png =600x600
[image2]: ./output_images/HOG_example.png =800x250
[image3]: ./output_images/sliding_windows.png =600x350
[image4]: ./output_images/sliding_window.png =600x350
[image5]: ./output_images/heatmap.png =600x350
[image6]: ./output_images/labels_map.png =600x350
[image7]: ./output_images/output_bboxes.png =600x350
[image8]: ./output_images/test_images_bboxes.png =600x600
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the method `get_hog_features` in the code cell named `Method for Extracting HOG Features` of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images from the provided dataset. Number of vehicle and non-vehicle images are 8792 and 8968, respectively. It is a balanced dataset where the numbers are roughly equal. Here are some random sample images from both `vehicle` and `non-vehicle` classes:

![alt text][image1]

I extract the HOG features using `skimage.hog()` function. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The code for extracting HOG features from the training dataset with the finalized parameters is in the code cell named `Extract HOG Features for Input Datasets, Define Labels Vector, and Split` of the notebook. I used the following parameters of `colorspace=YUV`, `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)` in the final configuration.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of different colorspace and parameters (orientations, pixels per cell and cells per block) with respect to training time and accuracy. I observed the single channel implementations does not produce accuracies as high as using `ALL` channels. On the other hand, using `orientations=11`, `pixels_per_cell=(16, 16)` reduces feature vector length and still produce the highest accuracies. Here are the results of my explorations: 

|**Configuration**|Colorspace|Orientations|Pixels Per Cell|Cells Per Block|HOG Channel|Feature Vector Length|Feature Extraction Time|Classifier|Training Time|Accuracy|
|:-------------:|:-------------:|:-----:|:-----:|:-----:|:-----:|:-----:|-----:|:-----:|-----:|-----:|
|**1**|	RGB | 9 | 8| 2| 0| 1764|27.55|Linear SVC|4.32|95.55|
|**2**|	RGB | 9 | 8| 2| 1| 1764|27.00|Linear SVC|3.6|96.68|
|**3**|	RGB | 9 | 8| 2| 2| 1764|27.15|Linear SVC|3.82|95.75|
|**4**|	RGB | 9 | 8| 2| ALL| 5292|82.18|Linear SVC|17.11|97.49|
|**5**|	RGB | 11 | 16| 2| ALL| 1188|48.33|Linear SVC|2.56|96.68|
|**6**|	HSV | 9 | 8| 2| 0| 1764|28.51|Linear SVC|5.24|93.05|
|**7**|	HSV | 9 | 8| 2| 1| 1764|28.31|Linear SVC|5.45|91.84|
|**8**|	HSV | 9 | 8| 2| 2| 1764|27.44|Linear SVC|4.01.00|96.42|
|**9**|	HSV | 9 | 8| 2| ALL| 5292|83.04|Linear SVC|7.15|98.00|
|**10**| HSV | 11 | 16| 2| ALL| 1188|49.21|Linear SVC|1.46|97.94|
|**11**| LUV | 9 | 8| 2| 0| 1764|29.49|Linear SVC|3.84|96.31|
|**12**| LUV | 9 | 8| 2| 1| 1764|29.8|Linear SVC|4.11|94.68|
|**13**| LUV | 9 | 8| 2| 2| 1764|29.71|Linear SVC|5.45|92.40|
|**14**| LUV | 9 | 8| 2| ALL| 5292|84.61|Linear SVC|7.84|97.18|
|**15**| LUV | 11 | 16| 2| ALL| 1188|51.18|Linear SVC|1.39|97.58|
|**16**| HLS | 9 | 8| 2| 0| 1764|28.42|Linear SVC|4.91|92.96|
|**17**| HLS | 9 | 8| 2| 1| 1764|28.11|Linear SVC|4.09|96.34|
|**18**| HLS | 9 | 8| 2| 2| 1764|28.72|Linear SVC|6.90|90.54|
|**19**| HLS | 9 | 8| 2| ALL| 5292|82.41|Linear SVC|7.82|97.94|
|**20**| HLS | 11 | 16| 2| ALL| 1188|50.11|Linear SVC|1.22|97.58|
|**21**| YUV | 9 | 8| 2| 0| 1764|27.32|Linear SVC|3.73|96.42|
|**22**| YUV | 9 | 8| 2| 1| 1764|27.78|Linear SVC|4.06|93.41|
|**23**| YUV | 9 | 8| 2| 2| 1764|27.97|Linear SVC|4.43|92.45|
|**24**| YUV | 9 | 8| 2| ALL| 5292|81.98|Linear SVC|6.04|98.42|
|**25**| YUV | 11 | 16| 2| ALL| 1188|53.30|Linear SVC|1.09|98.11|
|**26**| YCrCb | 9 | 8| 2| 0| 1764|27.5|Linear SVC|4.01|96.59|
|**27**| YCrCb | 9 | 8| 2| 1| 1764|28.21|Linear SVC|3.78|94.20|
|**28**| YCrCb | 9 | 8| 2| 2| 1764|27.81|Linear SVC|4.77|93.55|
|**29**| YCrCb | 9 | 8| 2| ALL| 5292|70.94|Linear SVC|6.96|98.51|
|**30**| YCrCb | 11 | 16| 2| ALL| 1188|49.34|Linear SVC|0.96|98.17|


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM which have been shown to work well with the hog features in the code cell named `Training a Classifier (Support Vector Machine)` of the IPython notebook. I more focused on and only used the gradient based features. I splitted tha dataset and used 80% of it as `training set` and 20% of it as `test set`. I achieved a test accuracy of 98.2%.    

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window search is in the method `find_vehicles` in the code cell named `Method for Using Classifier to Detect Cars in an Image` of the notebook. I used the method of extracting HOG features just once for the entire region of interest and subsample that array for each sliding window. Additionally, I have used six runs of sliding window search for each frame with different scales and start and stop positions (all in the lower half of the frames) and combined the outputs becuase the size and position of cars in the image will be different depending on their distance from the camera. I have used the following combinations:

|ystart|ystop|Window size|Number of windows|Scale|
|:----------:|:----------:|:-----:|:-----:|:-----:|
|400|464| 64x64 | 39 | 1.0|
|416|480| 64x64 | 39 | 1.0|
|400|496| 96x96 | 25 | 1.5|
|432|528| 96x96 | 25 | 1.5|
|400|528| 128x128 | 19 | 2.0|
|432|560| 128x128 | 19 | 2.0|
|400|596| 192x192 | 12 | 3.0|
|464|660| 192x192 | 12 | 3.0| 

The following image shows all the search windows drawn on a test image.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales with different start and stop positions using 3-channel HOG features, which provided a nice result. Here is an example of rectangles drawn on a test image after the sliding window search:

![alt text][image4]

Here the resulting bounding boxes are drawn onto the test images after filtering for false positives and combining overlapping bounding boxes:

![alt text][image8]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Rather than performing the heatmap/threshold/label steps only for the current frame's detections, the detections for the past 15 frames are stored and combined and added to the heatmap in the `Vehicle_Detect` class.

#####Here are the detected rectangles on a test image and the corresponding heatmap:

![alt text][image4]
![alt text][image5]

##### Here is the output of `scipy.ndimage.measurements.label()` on the thresholded heatmap:
![alt text][image6]

##### Here the resulting bounding boxes then overlaid on the image:
![alt text][image7]


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the difficulties was to find the right set of parameters used to extract features both to be able to achieve a high accuracy for the classifier and at the same time to keep the efficiency in terms of execution time.

The pipeline would probably fail in cases where types of vehicles which are very different from the ones in the training set are encountered and when lighting and environmental conditions are significantly different.

To make the pipeline more robust, different classifiers like an SVM and a neural network can be used at the same time and the results could be combined. To make it more adaptive to different lighting and environmental conditions, the frames could be preprocessed to get an idea about the conditions and one of the different (possibly separately trained) classifiers can be selected and used for each case.  
