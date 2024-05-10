
import numpy as np


# function for predicting classes for all test samples given a subset of x_train
def find_right_ginni_index(d, x_region, y_region, midpoint):
    x_region2 = []
    y_region2= []
    pclasses_region2 = [0]*3
    total_samples_region = 0

    for j in range(0,x_region.shape[0]):
        if (x_region[j][d]>midpoint):
            x_region2.append(x_region[j])
            y_region2.append(y_region[j])
            pclasses_region2[y_region[j]] +=1
            total_samples_region+=1

    for p in range(0,3):
        pclasses_region2[p]/= total_samples_region

    # convert into np arrays
    x_region2 = np.array(x_region2)
    y_region2 = np.array(y_region2)


    ginni_index_region2 = 0
    for p in range(0,3):
        ginni_index_region2+= pclasses_region2[p] * (1-pclasses_region2[p])
    return ginni_index_region2, total_samples_region, x_region2, y_region2

def find_left_ginni_index(d, x_region, y_region, midpoint):
    x_region1 = []
    y_region1= []
    pclasses_region1 = [0]*3
    total_samples_region=0

    for j in range(0,x_region.shape[0]):
        if (x_region[j][d]<=midpoint):
            x_region1.append(x_region[j])
            y_region1.append(y_region[j])
            pclasses_region1[y_region[j]] +=1
            total_samples_region+=1

    for p in range(0,3):
        pclasses_region1[p]/= total_samples_region

    # convert into np arrays
    x_region1 = np.array(x_region1)
    y_region1 = np.array(y_region1)


    ginni_index_region1 = 0
    for p in range(0,3):
        ginni_index_region1+= pclasses_region1[p] * (1-pclasses_region1[p])
    return ginni_index_region1, total_samples_region, x_region1, y_region1




def find_total_ginni_index (d, x_region , y_region):
    midpoint = 0
    for j in range(0, x_region.shape[0]):
        midpoint += x_region[j][d]
    midpoint /= x_region.shape[0]

    ginni_index_region1, total_samples_region1, x_region1, y_region1 = find_left_ginni_index(d, x_region, y_region, midpoint)
    ginni_index_region2, total_samples_region2, x_region2, y_region2 = find_right_ginni_index(d, x_region, y_region, midpoint)

    total_ginni_index = ((ginni_index_region1 * total_samples_region1 ) + (ginni_index_region2 * total_samples_region2)) / x_region.shape[0]
    return total_ginni_index, midpoint


# calculates the class_accuracies for each dataset and gives the predicted classes for all test samples
def decision_trees(X, Y, X_test, Y_test):
    X=X.T
    mean = np.mean(X, axis=1,keepdims=True)
    X=X-mean


    X_test=X_test.T
    X_test=X_test-mean

    covariance_matrix = np.dot(X, X.T) / (X.shape[0]-1)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]


    U = eigenvectors

    U10 = U[:, :10]
    Y10 = np.dot(U10.T,X)
    X_test =np.dot(U10.T,X_test)

    X = Y10.T
    X_test = X_test.T

    # Split 1
    min_ginni_index = 1
    dimension_split1 = 0
    midpoint_split1 = 0
    for i in range (0, X.shape[1]):
        gd,midpoint = find_total_ginni_index(i,X,Y)
        # print(gd)
        if(min_ginni_index > gd):
            min_ginni_index = gd
            dimension_split1 = i
            midpoint_split1 = midpoint
    # print("Min ginni index for split1:" , min_ginni_index)


    # Split 2

    midpoint = 0
    for j in range(0, X.shape[0]):
        midpoint += X[j][dimension_split1]
    midpoint /= X.shape[0]

    x_region1 = []
    x_region2 = []

    y_region1= []
    y_region2= []


    for j in range(0,X.shape[0]):
        if (X[j][dimension_split1]>midpoint):
            x_region2.append(X[j])
            y_region2.append(Y[j])
        else:
            x_region1.append(X[j])
            y_region1.append(Y[j])

    x_region1 = np.array(x_region1)
    y_region1 = np.array(y_region1)
    x_region2 = np.array(x_region2)
    y_region2 = np.array(y_region2)

    min_ginni_index_region1 = 1
    dimension_split2_region1 = 0
    midpoint_split2_region1 =0
    for i in range (0, X.shape[1]):
        gd, midpoint = find_total_ginni_index(i,x_region1,y_region1)
        # print(gd)
        if(min_ginni_index_region1 > gd):
            min_ginni_index_region1 = gd
            dimension_split2_region1 = i
            midpoint_split2_region1 = midpoint

    # print("Min ginni index for region1:" , min_ginni_index_region1)



    min_ginni_index_region2 = 1
    dimension_split2_region2 = 0
    midpoint_split2_region2 = 0
    for i in range (0, X.shape[1]):
        gd,midpoint = find_total_ginni_index(i,x_region2,y_region2)
        # print(gd)
        if(min_ginni_index_region2 > gd):
            min_ginni_index_region2 = gd
            dimension_split2_region2 = i
            midpoint_split2_region2 = midpoint

    # print("Min ginni index for region2:", min_ginni_index_region2)

    split1 = [dimension_split1, midpoint_split1]
    if (min_ginni_index_region1 < min_ginni_index_region2):
        split2= [dimension_split2_region1, midpoint_split2_region1]
    else:
        split2= [dimension_split2_region2, midpoint_split2_region2]

    if (split1[0] == split2[0]):
        if (split1[1] > split2[1]):
            temp = split1[1]
            split1[1]= split2[1]
            split2[1] = temp


    #  find class of all samples in the test set belonging to class 0, 1, 2
    region_samples = np.full((4,3),0)
    for i in range(0,X.shape[0]):
        y = Y[i]
        x_d1 = X[i][split1[0]]
        x_d2 = X[i][split2[0]]

        if (x_d1 < split1[1]):
            region_samples[1][y] +=1
        else:
            if (x_d2 < split2[1]):
                region_samples[2][y] +=1
            else:
                region_samples[3][y] +=1


    # print(region_samples)

    region_predictions = np.full((4,1),0)
    for i in range(1, 4):
        max_class=0
        max_samples =0
        for j in range(0,3):
            if (region_samples[i][j] > max_samples):
                max_samples = region_samples[i][j]
                max_class= j
        region_predictions[i]= max_class


    # print(region_predictions)

    #  calculating accuracies

    class_samples = [0] * 3
    correct_predicted_class_samples = [0] * 3
    total_correct_predicted_samples =0

    predicted_class_for_samples = [0] * X_test.shape[0]

    for i in range(0, X_test.shape[0]):
        x = X_test[i]
        y = Y_test[i]
        x_d1 = x[split1[0]]
        x_d2 = x[split2[0]]
        class_samples[y] +=1

        if (x_d1< split1[1]):
            predicted_class = region_predictions[1]
        else:
            if (x_d2 < split2[1]):
                predicted_class = region_predictions[2]
            else:
                predicted_class = region_predictions[3]
        predicted_class_for_samples[i] = predicted_class
        if (predicted_class == y):
            correct_predicted_class_samples[y] +=1
            total_correct_predicted_samples +=1
    class_accuracies= [0]* 3
    for i in range (0,3):
        class_accuracies[i] = correct_predicted_class_samples[i]/class_samples[i]

    # for i in range (0,3):
        # print("Class ",i, " accuracy: ",class_accuracies[i])

    total_accuracy = total_correct_predicted_samples/ X_test.shape[0]
    # print("Total Accuracy:", total_accuracy)

    return predicted_class_for_samples



# initialze train and test set
X= np.load("mnist.npz")
x_train = X['x_train']
y_train = X['y_train']

x_test = X['x_test']
y_test = X['y_test']

x_train=np.reshape(x_train,(60000,784))
x_test=np.reshape(x_test,(10000,784))

X = []
Y = []
for i in range (0,x_train.shape[0]):
    x = x_train[i]
    y = y_train[i]
    if (y==0 or y==1 or y==2):
        X.append(x)
        Y.append(y)

X = np.array(X)
Y = np.array(Y)

X_test = []
Y_test = []
for i in range (0,x_test.shape[0]):
    x = x_test[i]
    y = y_test[i]
    if (y==0 or y==1 or y==2):
        X_test.append(x)
        Y_test.append(y)

X_test = np.array(X_test)
Y_test = np.array(Y_test)


# randomly splits the original dataset into 5 different datasets
predicted_class_for_bags = []
total_samples = X.shape[0]
for i in range(0,5):
    indices = np.random.choice(total_samples, size = total_samples, replace = True)
    x = X[indices]
    y = Y[indices]
    predicted_class_for_bags.append(decision_trees(x, y, X_test, Y_test))



# major voting algorithm - calculating class accuracies
class_samples = [0] * 3
correct_predicted_class_samples = [0] * 3
total_correct_predicted_samples = 0


for i in range(0, X_test.shape[0]):
    y = Y_test[i]
    class_samples[y]+=1
    predictions = {0:0 , 1:0, 2:0}
    for j in range (0,5):
        predictions[predicted_class_for_bags[j][i][0]]+=1
    predicted_class = max(predictions, key=predictions.get)
    if (predicted_class == y):
        correct_predicted_class_samples[y] +=1
        total_correct_predicted_samples +=1

class_accuracies= [0]* 3
for i in range (0,3):
    class_accuracies[i] = correct_predicted_class_samples[i]/class_samples[i]

for i in range (0,3):
    print("Class ",i, " accuracy: ",class_accuracies[i])

total_accuracy = total_correct_predicted_samples/ X_test.shape[0]
print("Total Accuracy:", total_accuracy)




