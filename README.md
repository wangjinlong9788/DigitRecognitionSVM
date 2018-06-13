# DigitRecognitionSVM

Use Histogram of Oriented Gradients (HOG) as feature vectors.

The local object appearance and shape within an image can be described by the distribution of intensity gradients or edge directions.

Define E[(x−c)^k] as the moment of k order about x.

The function 	retval	=	cv.moments(	array[, binaryImage]  computes moments, up to the 3rd order, of a vector shape or a rasterized shape
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
