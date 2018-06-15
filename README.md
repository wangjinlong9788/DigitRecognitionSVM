# Digit Recognition with SVM

Use Histogram of Oriented Gradients (HOG) as feature vectors.

The local object appearance and shape within an image can be described by the distribution of intensity gradients or edge directions.

Define E[(x−c)^k] as the moment of k order about x.  

Image moment is a certain particular weighted average (moment) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation.

source:https://en.wikipedia.org/wiki/Image_moment https://blog.csdn.net/keith_bb/article/details/70197104

The function 	retval	=	cv.moments(	array[, binaryImage]) computes moments, up to the 3rd order, of a vector shape or a rasterized shape

The moments are defined as:

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/moments.jpg)

where f(i,j) is the grayscale of figure with M*N and the centroid are

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/ijaverage.jpg)

Central moment can be:

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/centermoments.jpg)

Scale invariants is then:

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/sclaeinvariant.jpg)

The well-known Hu moment Invariants with respect to translation, scale, and rotation can be constructed:

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/invariants.jpg)

dst	=	cv.Sobel(	src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]	)

Example code of using sobel operator: sobel.py

Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/operator.PNG)

Result:

![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/car.jpg)
![image](https://github.com/wangjinlong9788/DigitRecognitionSVM/blob/master/car.PNG)

# Support Vector Machine(SVM)

A Support Vector Machine (SVM) is a discriminative classifier formally defined by a separating hyperplane. 

In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples.

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/separating-lines.png)
 
Our goal is to find the line passing as far as possible from all points of each separable set.

The operation of the SVM algorithm is based on finding the hyperplane that gives the largest minimum distance to the training examples.

The optimal separating hyperplane maximizes the margin of the training data.

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/optimal-hyperplane.png)

The representations of the hyperplane could be

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/hyperplane.PNG)

where x symbolizes the training examples closest to the hyperplane.

In general, the training examples that are closest to the hyperplane are called support vectors. This representation is known as the canonical hyperplane.

The problem of maximizing the margin  is equivalent to the problem of minimizing the function:

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/functionmin.PNG)

This is a problem of Lagrangian optimization that can be solved using Lagrange multipliers to obtain the weight vector  and the bias of the optimal hyperplane.

A distance from the corresponding training sample to their correct decision region could be added to the minimizing the function:

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/svm_basics3.png)

The new optimization problem is

![image](https://github.com/wangjinlong9788/NumberRecognitionSVM/blob/master/newoptimization%20.PNG)
# OpenCV
Need to install  module mlpy and cv2




# Accuracy:93.56%
