import numpy as np
from matplotlib import pyplot as plt

# initialze train and test set 
dataset= np.load("mnist.npz")
x_train = dataset['x_train']
y_train = dataset['y_train']

x_test = dataset['x_test']
y_test = dataset['y_test']
x_train = x_train.astype(np.float64)
x_test = x_test.astype(np.float64)
x_train/=255
x_test/=255

# Q1 Part A
rows = 5
cols = 10
counter=0
temp = [0]* 10

i=0

#adding subplot for each image
while(counter!=50):
    x_class = y_train[i]
    
    if(temp[x_class]<5):
        plt.subplot(rows, cols, (temp[x_class])*10 + x_class+1)
        temp[x_class]+=1
        plt.imshow(x_train[i], plt.get_cmap('gray'))
        counter+=1
    i+=1
plt.show()


#Q1 Part B

#reshape the 28X28 to 784X1 matrix
x_train = np.reshape(x_train , (60000, 784))
x_test = np.reshape(x_test , (10000, 784))


#find mean and covariances of all classes
class_arrays = [[] for _ in range(10)]
for x,y in zip(x_train , y_train):
    class_arrays[y].append(x)

class_arrays = [np.array(class_samples) for class_samples in class_arrays]
class_mean = [np.mean(class_samples, axis=0) for class_samples in class_arrays]

lambdas = 1e-6 # set lambda value
class_covariance = [np.cov(class_samples,rowvar = False) + lambdas* np.identity(784)  for class_samples in class_arrays]
class_priors =[(len(class_arrays[i]) / x_train.shape[0] ) for i in range(0,10)]


#applying QDA formula
Wi = [0] * 10
wi = [0] * 10
b = [0] * 10

for i in range(0,10):
    Wi[i] =   np.linalg.inv(class_covariance[i])
    wi[i] =   np.dot(np.transpose(class_mean[i]), Wi[i])
    b[i] = (-1/2)*np.log(np.linalg.slogdet(class_covariance[i])[0] ) + np.log(class_priors[i]) - (1/2)* (np.dot(np.dot(np.transpose(class_mean[i]), Wi[i]) , class_mean[i]))
    

#Q1 part C
correct_samples =0
class_accuracy = [0]* 10 
class_test_samples = [0] * 10
    
#iterating through all test samples
for i in range(0, x_test.shape[0]):
    x = x_test[i]
    y = y_test[i]
    class_test_samples[y]+=1
    predicted_class =0
    max_logarithmic_likelihood = float("-inf")
    
    #finding max likelihood for a sample and predicting its class
    for j in range(0 ,10):
        likelihood = -(1/2) * (np.dot(np.dot(np.transpose(x) , Wi[j]), x) - (2) * (np.dot(wi[j], x)) ) + b[j] 
        if(likelihood >= max_logarithmic_likelihood) :
            predicted_class = j
            max_logarithmic_likelihood = likelihood
    if(predicted_class==y):
        correct_samples+=1
        class_accuracy[y] +=1
        
#total accuracy  
accuracy = correct_samples / x_test.shape[0] 
print("Total accuracy:" , accuracy)

#accuracy for each class
for i in range(0,10):
    print("Class", i , " accuracy:" ,class_accuracy[i]/(class_test_samples[i]))