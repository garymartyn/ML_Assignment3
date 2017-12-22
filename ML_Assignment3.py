
# coding: utf-8

# In[1207]:


# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy import optimize as op
from sklearn.linear_model import LogisticRegression


# In[1221]:


# getting the data from the file
import os  
path = os.getcwd() + '/owlsNumLabels.csv'  
data = pd.read_csv(path, sep="," ,header=None, names=['body-length','wing-length', 'body-width','wing-width' , 'type'])
data.head()


# In[1209]:


labels=["LongEaredOwl","SnowyOwl","BarnOwl"]

# this function returns the correct label for a given input as the labels were changed to [0,1,2] in preprocessing
# 0=LongEaredOwl
# 1=SnowyOwl
# 2=BarnOwl
def returnLabelString(number):
    return labels[number]


# In[1210]:


# data manipulation
LABELS=[0,1,2]
Lambda=0.01 # lambda = the learning rate

numColumns = data.shape[1]

# num of examples
m = data.shape[0]
# number of features
n = numColumns-1
# number of labels
k = len(np.unique(data['type']))

# initialise arrays to all ones, with added column of 1's
X = np.ones((m,n+1))
y = np.array((m,1))

# set values in the y vector
y=data["type"].values

# set values in the X vector, leaving column 0: as all 1's
X[:,1]= data["body-length"].values
X[:,2]= data["wing-length"].values
X[:,3]= data["body-width"].values
X[:,4]= data["wing-width"].values

# #perform normalisation on the data
# ((value - mean)/ stdDeviation)
for j in range(n):
    X[:, j+1] = (X[:, j+1] - X[:,j+1].mean())/(X[:,j+1].std())


# In[1211]:


# the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[1212]:


# function takes in two lists, compares the values and outputs the accuracy at which they are similar
# Parameters predicted= the list of class values/labels that the model produced
# actual= the actual class values/labels of the data
def calculateAccuracy(predicted, actual):
    sum=0
    length=len(predicted)
    for i in range(length):
        if predicted[i]==actual[i]:
            sum=sum+1
    tot=(sum/length)
    return tot # return the accuracy


# In[1213]:


#Logistic regression cost function
test=0
def Cost(theta, X, y):
    m = len(y) # the number of samples input
    hTheta = sigmoid(X.dot(theta)) # probability y=1: given that its parameterised by theta
    tc = np.copy(theta) #tc = theta copy we will be using 
    tc[0] = 0 # we dont regularize theta[0] as its used as a bias term

    return (1/m) * (-y.T.dot(np.log(hTheta)) - (1-y).T.dot(np.log(1-hTheta))) + ((Lambda/(2*m))*np.sum(tc**2))
    # return the value produced by the equation


# In[1214]:


# Gradient Descent 
def Gradient(theta, X, y):
    m=X.shape[0] # number of samples
    n=X.shape[1] # number of features
    theta = theta.reshape((n, 1)) # reshape theta into a n*1 vector mat the matrix multiplication to work
    y = y.reshape((m, 1)) # need to reshape the target vector to a m*1 vector
    hTheta = sigmoid(X.dot(theta)) # # probability y=1: given that its parameterised by theta
    tc = np.copy(theta) # tc= theta copy
    tc[0]=0 # we dont regularize theta[0] as its used as a bias term
    grad=((1/m) * X.T.dot(hTheta-y)) + ((Lambda/m)*tc)# calculate the gradient
    
    return grad # return the gradient


# In[1215]:


import scipy.optimize as opt  
#Optimal theta 
def logisticRegression(X, y, theta):
    result = opt.fmin_tnc(func=Cost, x0=theta, fprime=Gradient, args=(X, y)) 

    return result[0]


# In[1216]:


# funciton which returns the mean of an input list
# parameter = lst (the list you wish to average, this implementation assumes it contains ints or floats)
def getMean(lst):
    return sum(lst)/len(lst)


# In[1217]:


f = open('LogisticOutput.txt','w') # file the output will be wrote to
# specify the number of times you want the model to be tested on the data
numFolds=10
scores=[] # list to keep track of the algorithms average accuarcy
skScores=[] # list to keep track of the sklearn implemenations average accuracy
#runResults=[]

# iterate over the data the specified number of amounts
# in each, calculate the accuracy and output the pr
for j in range(numFolds):
    all_theta = np.zeros((k,n+1))# initialise all_theta to store the theta values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    lr= LogisticRegression().fit(X_train,y_train)
    yhat = lr.predict(X_test)
    skAcc=calculateAccuracy(y_test, yhat)
    skScores.append(skAcc*100)
    #One vs all
    i = 0
    for samp in LABELS:
        value=np.array(y_train==samp,dtype=int)
        # value acts as the temporary y vector, where values are all set to 1 or 0,
        # i.e. 1 for the label being tested, and rest are set to 0
        optimalTheta = logisticRegression(X_train, value, np.zeros((n+1,1)))
        all_theta[i] = optimalTheta
        i=i+1
        
    Probabilities = sigmoid(X_test.dot(all_theta.T)) #probability that eacy is 1, for each label
    predicts = [LABELS[np.argmax(Probabilities[i, :])] for i in range(X_test.shape[0])]# get the prediction(highest value[0,1,2]) for each sample
        
    acc=calculateAccuracy(y_test, predicts)
    scores.append(acc*100)
    
    for i in range(X_test.shape[0]):
        f.write("predicted label: %s - actual label : %s" % (returnLabelString(p[i]),returnLabelString(y_test[i])) +"\n")
    f.write("\n")
    print("Test Accuracy %f " %(acc * 100) +"%")
    
for l in range(numFolds):
    f.write("\n\tAccuracy of fold %d = %f" %(l+1,scores[l]))

# calculate the mean accuracy of the iterations and its range (plus and minus values)
# for the implemented algorithm
mean = getMean(scores)
indexH, highest = max(enumerate(scores), key=operator.itemgetter(1))
indexL, lowest = min(enumerate(scores), key=operator.itemgetter(1))
avgDiff=((highest-lowest) /2)
print("Average Accuracy = %f +/- %f"  %(mean,avgDiff))

# write the results of the implemented model to the .txt file
f.write("\n\t Implemented algorithm")
f.write("\nBest Accuracy was iteration number %d, with %f Accuracy " %(indexH,highest))
f.write("\nWorst Accuracy was iteration number %d, with %f Accuracy " %(indexL,lowest))
f.write("\n"+"Average Accuracy = %f (Percent) +/- %f (Percent)"  %(mean,avgDiff)+"\n")

# compute the average and range of the values for the sklearn model
skMean = getMean(skScores)
indexHsk, highestSK = max(enumerate(skScores), key=operator.itemgetter(1))
indexLsk, lowestSK = min(enumerate(skScores), key=operator.itemgetter(1))
avgDiffSk=((highestSK-lowestSK) /2)
print("Average Accuracy = %f +/- %f"  %(skMean,avgDiffSk))

# output the results of the sklearn logistic regresssion to the .txt file
f.write("\n\t Sklearn algorithm")
f.write("\nBest Accuracy was iteration number %d, with %f Accuracy " %(indexHsk,highestSK))
f.write("\nWorst Accuracy was iteration number %d, with %f Accuracy " %(indexLsk,lowestSK))
f.write("\n"+"Average Accuracy = %f (Percent) +/- %f (Percent)"  %(skMean,avgDiffSk))

f.close()# close the file

