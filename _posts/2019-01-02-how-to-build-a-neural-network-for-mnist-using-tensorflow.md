---
title: "How to build a neural network for MNIST using tensorflow"
excerpt_separator: "<!--more-->"
categories:
  - Post Formats
tags:
  - machine learning
  - deep learning
  - image recognition
  - tensorflow
  - mnist
---
# Instroduction
The first step to dive into machine learning is to do experiment on MNIST dataset for most of beginers. MNIST is a dataset of handwriting digits. Beginners can learn fundamental knowledge from creating program to recognize digits, while researchers can develope new algorithmns and models upon it. It is said that it's no guarantee that one algorithmn working well on MNIST works well on the other domain, but it's guaranteed that one algorithmn doesn't work well on the domain if it does'nt work well on MNIST.

In my case, I'm a beginer of machine learning, furthermore digit recognition is one of my production challenge. The problem is about recognizing non-embossed font digits on cards. The cards are fixed size with various background, pure color or images. The digits are arranged in 4x4 grid with fixed font size. However, as you can see, the digit size can not be guaranteed due to two obvious reasons. One is caused by the font itselft. Each digit from 0 to 9 doesn't occupy the same area on the card. The another is caused by the distance of the card and the camera.

Anyway, the first step is to try MNIST since there is no enough images to develop digit recongnition model. By try MNIST, I can learn the fundamental knowledge of develop a model for digit recognition.

# Articles
Before I dive into a thick book, I find some articles by googling to get a quick started.
* [How to build a neural network to recognize handwritten digits with tensorflow](https://www.digitalocean.com/community/tutorials/how-to-build-a-neural-network-to-recognize-handwritten-digits-with-tensorflow)
* [Not another MNIST tutorial with tensorflow](https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow)
* [Visualizing MNIST](https://colah.github.io/posts/2014-10-Visualizing-MNIST/)

The first two lead me to understand how a linear classifier works and how to make such a classifier. The last one tells technologies for dimentional data reduction.

So, what are they?

# Regression and Classification
Before you can apply any machine learning algorithmn, it's critical to tell what kind of task you are going to do first, regression or classification. Both regression and classification tasks are predictive modeling problem. A predictive modeling is a problem of approximating a mapping function from input to output using historical data to predicate upon new data where you don't have answer.
> Classification predictive modeling is the task of approximating a mapping function (f) from input variables (X) to discrete output variables (y).

> Regression predictive modeling is the task of approximating a mapping function (f) from input variables (X) to a continuous output variable (y).

More details you can go to read [Classification vs. Regression in machine learning](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/).

Digit recognition is a classification problem that you'd want to give a label to a given image of a digit. In digit recognition case, image each label is a point in a 10-dimension space, and the whole space is separated by 10 lines. So the problem is modeled as the given image is transformed to a point in the space and the nearest line representing the label that the image should has. Then the problem becomes,
1. how to map the image to the point?
2. What is the line?
3. how to measure the distance?

When we talk about label representation, it's natrual to encode the 10 labels into a vector in which the value of the position representing the label is 1, and others are 0. For example, for a image of digit 3, the vector is `[0,0,0,1,0,0,0,0,0,0]`. This vector can also be interpreted as the 1 means the probability of that the image belongs to the cooresponding label is 100%. Because a image can belong to one category in the case, so the probabilities of others are of cause 0. If the vector is the combination of probabilities of that the image belongs to each category, the line of these points is a kind of probability distribution. All images with the same label should be on the line or near to it. Since we are measuring distance of probability distribution, cross-entropy is one of good choice.

Till now, the classification problem is transformed into a linear regression problem. The model function is y = Wx + b modeling a line and cross-entropy is used to optimize W so that for a given example, the best y is predicted.

# Single Layer Neural Network
[Not another MNIST tutorial with tensorflow](https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow) gives a single layer Neural Network for this problem. An image of 28x28 is flat as a 1x784 vector to be the input layer of the network. Each pixel of the image is a node in the network linking to neuron in the output layer. As discussed previously, there are 10 neuron in the output layer to give a 1x10 vector as output in which each number in a position representing the probability of that the image belongs to that category. Each neuron accepts 784 inputs and give one output. The output is the input multiplying Weight and adding a Bias.

$$
\begin{bmatrix}
   a & b \\
   c & d
\end{bmatrix}
$$

$$\frac{\partial f(y)}{\partial x} = \frac{\partial f}{\partial y} \times \frac{\partial y}{\partial x}$$