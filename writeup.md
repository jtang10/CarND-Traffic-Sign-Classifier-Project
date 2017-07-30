#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data/img_by_class.jpg "Visualization"
[image2]: ./data/training_examples.jpg "Training examples"
[image3]: ./data/transformations.jpg "Augmentation"
[image4]: ./data/test_img1r.jpg "Traffic Sign 1"
[image5]: ./data/test_img2r.jpg "Traffic Sign 2"
[image6]: ./data/test_img3r.jpg "Traffic Sign 3"
[image7]: ./data/test_img4r.jpg "Traffic Sign 4"
[image8]: ./data/test_img5r.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jtang10/CarND-Traffic-Sign-Classifier-Project)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many images each different
class has. From the plot it is obvious that the number of training data for each labels is not balanced. It
could be a good practice if the fake data could be generated for the labels with less data. The balance training
data would lead to better generalization.

![alt text][image1]


###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Since the color could be a feature for different traffic signs (stop sign is always red whereas the roundabout is always
blue), I didn't convert the images to grayscale. Please see below for an example for the training images.

![alt text][image2]

Then based on what I observed from the original training data, I did several augmentation as the following. The CNN should
recognize the augmented training data with the same label as before.

**1. Perspective transformation:**
    Perspective transformation is similar to zooming in. For more detail please check [OpenCV doc](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#perspective-transformation).

**2. Rotation:**
    Each image is rotated counter-clockwise by 25 degree.

**3. Erosion:**
    The boundaries of the forgound is eroded away partially. Check the [OpenCV doc](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#erosion) for more details.

**4. Flip:**
    The image is flipped left-to-right.

**5. Brightness Adjustment:**
    Some images are too dark while some of the others are too bright. The brightness is fixed to 128.

Please see the following plot for an example of the transformations. 
![alt text][image3]

Then I randomly picked 10000 training data and augmented them. After that there are 84799 training data, 50000 more than before.
After the augmentation, all the training, validation and test data are normalized and shuffled. Note that 
the normalization should only use the mean and standard deviation of the training set rather than those of 
all three sets. This prevents acquiring prior knowledge of the validation and testing dataset. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x64                   |
| Fully connected		| etc.        									|
| Softmax				| etc.        									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

All the hyperparameters:
EPOCHS = 20
BATCH_SIZE = 128
rate = 0.001

Adamoptimizer is used as the lab solution.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
- validation set accuracy of 95%
- test set accuracy of 93.2%

The modified model doesn't really improve the validation set accuracy. The 5% point improvement is mainly from
the augmented training data. 

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    The first architecture I used was the original LeNet given from the lab. It is a good start but only give 89% validataion set accuracy.

* What were some problems with the initial architecture?
    The initial architecture was simple. It could get initial result but hard to improve. The pooling procedure
    would lose some information. Better practice would be using dropout. Also the convolution layer is not wide
    enough to extract more feature.
* How was the architecture adjusted and why was it adjusted? 
    More filters were used in the convolution layers.
* Which parameters were tuned? How were they adjusted and why?
    I tried to adjust the learning rate but did not find much improvement. EPOCHES was increased after larger training set was used and it was quite effective. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
    Convolution layer is quite essential in all computer vision problems. A dropout would improve the data loss from pooling.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because they are not properly cropped as the all the examples shown in the training, validation and testing datasets.
Some of the got the complex background or the stick. Also the shooting angle varies a lot from all downloaded images.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h     		    | Yield   						    			| 
| Bumpy Road     		| Bumpy Road 									|
| 50 km/h				| Stop											|
| Icy Road	      		| End of speed and passing limit 				|
| Roundabout			| Priority Road      							|


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. This compares favorably to the accuracy on the test set of 93.2% accuracy.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .981           			| Yield   									| 
| .018     	    			| Turn Left 								|
| .00001					| Turn Right								|
| .00001	      			| Ahead Only					 				|
| .000002				    | No entry      							|

I won't include all the images test top 5 results because that costs too much labor but not very insightful. However,
I list out the following ways to improve the accuracy:

1. Balance out the trainging dataset so every label can be properly trained.
2. Increase the training set resolution so each image can be more easily identified by nakes eye, which also means more feature (more pixels) for the neural networks to train. 
3. Download proper images. All the images I downloaded were not square. I cropped them and resized them to 32x32x3. As a result, some of them are hard to identify even by human.


