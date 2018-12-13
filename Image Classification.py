 
### Multiclass Classification Project

# 
# Learning Goals
# - How to read different size images from a directory?
# - How to implement One-vs-one scheme for Multiclass classification?
# - How to use SVM for image classifcation?
#

# Importing Essential Libraries
import numpy as np
import os 
import random
from pathlib import Path
from keras.preprocessing import image
import matplotlib.pyplot as plt

### SVM Classifier ###
class SVM:

    #init method to initialize the values
    def __init__(self,C=1.0):
        self.C = C
        self.W = 0
        self.b = 0
    
    #To calculate the hingeLoss
    def hingeLoss(self,W,b,X,Y):
        loss  = 0.0
        
        loss += .5*np.dot(W,W.T)
        
        m = X.shape[0]
        
        for i in range(m):
            ti = Y[i]*(np.dot(W,X[i].T)+b)
            loss += self.C *max(0,(1-ti))
            
        return loss[0][0]
    
    #To train the model
    def fit(self,X,Y,batch_size=50,learning_rate=0.001,maxItr=500):
        
        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]
        
        n = learning_rate
        c = self.C
        
        #Init the model parameters
        W = np.zeros((1,no_of_features))
        bias = 0
        
        #Initial Loss
        
        #Training from here...
        # Weight and Bias update rule is implemented
        losses = []
        
        for i in range(maxItr):

            #Training Loop
            l = self.hingeLoss(W,bias,X,Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)
            
            #Batch Gradient Descent(Paper) with random shuffling
            for batch_start in range(0,no_of_samples,batch_size):

                #Assume 0 gradient for the batch
                gradw = 0
                gradb = 0
                
                #Iterate over all examples in the mini batch
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        i = ids[j]
                        ti =  Y[i]*(np.dot(W,X[i].T)+bias)
                        
                        if ti>1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c*Y[i]*X[i]
                            gradb += c*Y[i]
                            
                #Gradient for the batch is ready! Update W,B
                W = W - n*W + n*gradw
                bias = bias + n*gradb
                
        #Assigning the values to weights and bias Variable of class 
        self.W = W
        self.b = bias
        return W,bias,losses


### DataSet-Preparation ###

#Settting the path of the folder
p = Path("./Images/")

#Extracting the name of all of the folders in the directory
dirs = p.glob("*")

#Library Created to assign labels
labels_dict = {"cat":0,"dog":1,"horse":2,"human":3}

#To store the image data
image_data = []

#To store the labels of the data
labels = []

#Looping over each folder and exracting the image
for folder_dir in dirs:

    label = str(folder_dir).split("/")[-1][:-1]
    
    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path,target_size=(32,32))  #To load image from the system
        img_array = image.img_to_array(img) # Toconvert the image into array
        image_data.append(img_array)        #Appending the converted array into Image_data
        labels.append(labels_dict[label])   #Appending the corresponding labels
        
""" Uncomment to print the shape of the data prepared 
    print(len(image_data))
    print(len(labels)) """

#Convert the prepared data and labels into numpy array
image_data = np.array(image_data,dtype='float32')/255.0
labels = np.array(labels)

# Zip the image_data and labels to shuffle and then Unzip the data
combined = list(zip(image_data,labels))
random.shuffle(combined)

#Unzip the shuffled data 
image_data[:],labels[:] = zip(*combined)

# Function to visualize the data into images
def drawImg(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()    
    return 

""" Uncomment to see the the first 10 images of 
    the prepared data
    for i in range(10):
        drawImg(image_data[i]) """


### We need to convert data for One-for-One classifcation! ###

#Reshaping the image from (808 x 32 x 32 x 3) to (808 x 3072)
M = image_data.shape[0] 
image_data = image_data.reshape(M,-1)

"""Uncomment to see the shape of the resized data
    print(image_data.shape)
    print(labels.shape)"""

#To store the number of classes
CLASSES = len(np.unique(labels))

""" Uncomment to print the CLASSES variabe
    print(CLASSES)"""

### Preparing data For One Vs One Classification ###

""" Function to Create the dictionary of labels as Keys
    and their corresponding list of images as Values """ 
def classWiseData(x,y):
    data = {}
    
    for i in range(CLASSES):
        data[i] = []
        
    for i in range(x.shape[0]):
        data[y[i]].append(x[i])
    
    for k in data.keys():
        data[k] = np.array(data[k])
        
    return data

"""Funtion which take Data of two classes and combine them into one matrix"""
def getDataPairForSVM(d1,d2):

    l1,l2 = d1.shape[0],d2.shape[0]
    
    samples = l1+l2
    features = d1.shape[1]
    
    #To store the new Combined data
    data_pair = np.zeros((samples,features))
    data_labels = np.zeros((samples,))
    
    data_pair[:l1,:] = d1
    data_pair[l1:,:] = d2
    
    data_labels[:l1] = -1
    data_labels[l1:] = +1
    
    return data_pair,data_labels
    
### Training of NC2 Classifiers

#Instatiating the Object of the SVM class
mySVM = SVM()
""" Funtion to train the SVM classifier and store them in a 
    dictionary having labels as their Keys and trained weights 
    as their Values """
def trainSVMs(x,y):
    
    svm_classifiers = {}
    for i in range(CLASSES):
        svm_classifiers[i] = {}
        for j in range(i+1,CLASSES):
            xpair,ypair = getDataPairForSVM(data[i],data[j])
            wts,b,loss = mySVM.fit(xpair,ypair,learning_rate=0.00001,maxItr=1000)
            svm_classifiers[i][j] = (wts,b)
    
            """ Uncomment to see the plot of Loss in during training of each classifier            
                plt.plot(loss)
                plt.show() """
            
    return svm_classifiers

### Prediction ###

# To make prediction between two Classes
def binaryPredict(x,w,b):

    #Generating Hypothesis 
    z  = np.dot(x,w.T) + b

    #Predicting Class on the basis of Hypothesis
    if z>=0:
        return 1
    else:
        return -1

# Predict Function to make prediction Using every Classifier
def predict(x):
    
    count = np.zeros((CLASSES,))
    
    for i in range(CLASSES):
        for j in range(i+1,CLASSES):
            w,b = svm_classifiers[i][j]

            #Take a majority prediction 
            z = binaryPredict(x,w,b)
            
            if(z==1):
                count[j] += 1
            else:
                count[i] += 1
    
    final_prediction = np.argmax(count)
    return final_prediction

#Function to calculate the Accuracy of the classifier
def accuracy(x,y):
    
    count = 0
    for i in range(x.shape[0]):
        prediction = predict(x[i])

        #Checking Predictions with Labels
        if(prediction==y[i]):
            count += 1
            
    return count/x.shape[0]

#Calls to initiate the Data Preparation and Training pf the classifier
data = classWiseData(image_data,labels)
svm_classifiers = trainSVMs(image_data,labels)

#Checking the accuracy of the Classifier
print("Accuracy of the MySVM model: ",accuracy(image_data,labels))


### SVM Classification using SK-learn ###
from sklearn import svm

#Initializing the Classifier
svm_classifier = svm.SVC(kernel='linear',C=1.0)

#Training the classifier
svm_classifier.fit(image_data,labels)

#Checking the Accuracy of the model
print("Accuracy of the sklearn's SVM model",svm_classifier.score(image_data,labels))

