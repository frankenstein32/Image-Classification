# Image-Classification
Classifying images of dogs, cats, horses and humans using Support Vector Machine

Our brains make vision seem easy. It doesn't take any effort for humans to tell apart a lion and a jaguar, read a sign, or recognize a human's face. But these are actually hard problems to solve with a computer: they only seem easy because our brains are incredibly good at understanding images.

__Problem Statement__ - Given a DataSet of Images of cats, dogs, horses and humans, Your job is to prepare a classifier that can classify given image and predicts its class.

__DataSet__ 
- Dataset contains 4 folders named as cats, dogs, humans and horses in which images for each category is saved. The images can be of different size so we need to reshape all of the images to one std size.
- Dataset is attached in the repo and do save the dataset in the same folder as that of the code otherwise you need to change the directory of the path in the code to load images from it.


__Classifier__ : The classifier used is of Support vector Machine and I have used One-Vs-One scheme to perform classification between multiple Classes.

__Support Vector Machine__ 
- Support Vector Machines are supervised learning models for classification and regression problems. They can solve linear and non-linear problems and work well for many practical problems.
- The idea of Support Vector Machines is simple: The algorithm creates a line which separates the classes in case e.g. in a classification problem. 
- The goal of the line is to maximizing the margin between the points on either side of the so called decision line. 
- The benefit of this process is, that after the separation, the model can easily guess the target classes (labels) for new cases.

__Essential Libraries__: There are a few libraries that one need to install in order to run the given code.
- Numpy
- keras
- matplotlib

__Screen shots__ : Inorder to run the given code see the below attached Screenshots.

1. Run the Following Command to run the Code

![screenshot from 2018-12-13 19-08-49](https://user-images.githubusercontent.com/34310411/49951550-364f4080-ff20-11e8-9e73-e68a43dca21f.png)


2. Results were as Follows.

![screenshot from 2018-12-13 19-10-48](https://user-images.githubusercontent.com/34310411/49951581-52eb7880-ff20-11e8-830a-9de296e1c5d8.png)
