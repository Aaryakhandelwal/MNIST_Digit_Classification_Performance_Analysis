import numpy as np
from matplotlib import pyplot as plt

# initialze train and test set 
X= np.load("mnist.npz")
x_train = X['x_train']
y_train = X['y_train']

x_test = X['x_test']
y_test = X['y_test']


#reshaping the x_train and x_test
x_train=np.reshape(x_train,(60000,784)) 
x_test=np.reshape(x_test,(10000,784)) 
X=np.empty((1000,784))
class_mapping=[]
samples_in_each_class=[0 for i in range(10)]


# taking 100 samples from each class
i=0
z=0
while sum(samples_in_each_class)<1000:    
    if(samples_in_each_class[y_train[i]]<100):
        samples_in_each_class[y_train[i]]+=1
        X[z]=x_train[i]
        class_mapping.append(y_train[i])
        z+=1
    i+=1

X=np.transpose(X)

#centralizing all the samples by subtracting mean
mean = np.mean(X, axis=1)
# for i in range(784):
#     for j in range(1000):
#         X[i][j]-=mean[i]
        


#find covariance matrix, corresponding eigen values and eigenvectors    
covariance_matrix = np.dot(X, X.T) / (X.shape[1]-1)


eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

#MSE is calculated
U = eigenvectors
Y = np.dot(U.T , X)
X_recon = np.dot(U,Y)
mse =0
for i in range(784):
    for j in range(1000):
        mse+= (X[i][j] - X_recon[i][j])**2
        
        
        
#corresponding to different p there is plotting of the 5 samples from each class and accuracies related to the p 
def plot_with_p (U, p , mean , class_mapping , x_test , y_test):
    Up = U[:, :p]
    Yp = np.dot(Up.T,X)
    Xp = np.dot(Up,Yp)
    for i in range(784):
        for j in range(1000):
            Xp[i][j]+=mean[i]
            
   
    Xp = np.transpose(Xp)
    Xp = np.reshape(Xp, (1000,28,28))
    
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
            plt.imshow(Xp[i], plt.get_cmap('gray'))
            counter+=1
        i+=1
    plt.show()
    
    Yp = np.transpose(Yp)
    class_arrays = [[] for _ in range(10)]
    for x,y in zip(Yp , class_mapping):
        class_arrays[y].append(x)

    class_arrays = [np.array(class_samples) for class_samples in class_arrays]
    class_mean = [np.mean(class_samples, axis=0) for class_samples in class_arrays]

    lambdas = 1e-6 # set lambda value
    class_covariance = [np.cov(class_samples,rowvar = False) + lambdas* np.identity(p)  for class_samples in class_arrays]
    class_priors =[(len(class_arrays[i]) / Yp.shape[0] ) for i in range(0,10)]

    Wi = [0] * 10
    wi = [0] * 10
    b = [0] * 10

    for i in range(0,10):
        Wi[i] =   np.linalg.inv(class_covariance[i])
        wi[i] =   np.dot(np.transpose(class_mean[i]), Wi[i])
        b[i] = (-1/2)*np.log(np.linalg.slogdet(class_covariance[i])[1]) + np.log(class_priors[i]) - (1/2)* (np.dot(np.dot(np.transpose(class_mean[i]), Wi[i]) , class_mean[i]))
    
    correct_samples =0
    class_accuracy = [0]* 10 
    class_test_samples = [0] * 10
    
   
      

    #iterating through all test samples
    for i in range(0, x_test.shape[0]):
        x = x_test[i]
        Y = np.dot(Up.T,x)
        y = y_test[i]
        class_test_samples[y]+=1
        predicted_class =0
        max_logarithmic_likelihood = float("-inf")
        
        #finding max likelihood for a sample and predicting its class
        for j in range(0 ,10):
            likelihood = -(1/2) * (np.dot(np.dot(np.transpose(Y) , Wi[j]), Y) - (2) * (np.dot(wi[j], Y)) ) + b[j] 
            if(likelihood >= max_logarithmic_likelihood) :
                predicted_class = j
                max_logarithmic_likelihood = likelihood
        if(predicted_class==y):
            correct_samples+=1
            class_accuracy[y] +=1
            
    #total accuracy  
    accuracy = correct_samples / x_test.shape[0] 
    print("For p= ", p)
    print("Total accuracy:" , accuracy)

    #accuracy for each class
    for i in range(0,10):
        print("Class", i , " accuracy:" ,class_accuracy[i]/(class_test_samples[i]))
        
        
        
#plotting for different p= 5 , 10 and 15
        
plot_with_p(U,5,mean,class_mapping , x_test , y_test)
plot_with_p(U,10,mean,class_mapping , x_test , y_test)
plot_with_p(U,20,mean,class_mapping , x_test , y_test)



