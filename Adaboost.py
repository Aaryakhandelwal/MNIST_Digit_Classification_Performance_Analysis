# %%
import numpy as np
import random
import math
from matplotlib import pyplot as plt

# %%

# initialze train and test set
X= np.load("mnist.npz")
x_train = X['x_train']
y_train = X['y_train']

x_test = X['x_test']
y_test = X['y_test']

x_train=np.reshape(x_train,(60000,784))
x_test=np.reshape(x_test,(10000,784))

x_new_train = []
y_new_train = []

# considering only 2 classes and changing labels to -1 and 1
for i in range(x_train.shape[0]):
    if (y_train[i] == 0):
        x_new_train.append(x_train[i])
        y_new_train.append(-1)
    elif (y_train[i] == 1):
        x_new_train.append(x_train[i])
        y_new_train.append(1)
x_train = np.array(x_new_train)
y_train = np.array(y_new_train)

x_new_test = []
y_new_test = []
for i in range(x_test.shape[0]):
    if (y_test[i] == 0):
        x_new_test.append(x_test[i])
        y_new_test.append(-1)
    elif (y_test[i] == 1):
        x_new_test.append(x_test[i])
        y_new_test.append(1)


# print(x_train.shape)
# print(y_train.shape)

# finding val set
c1=0
c2=0
x_val = []
y_val = []
x_new_train = []
y_new_train = []
for i in range(x_train.shape[0]):
    y = y_train[i]
    if (y==-1 and c1<1000):
        x_val.append(x_train[i])
        y_val.append(y)
        c1+=1
    elif (y==1 and c2<1000):
        x_val.append(x_train[i])
        y_val.append(y)
        c2+=1
    else:
        x_new_train.append(x_train[i])
        y_new_train.append(y)


x_val = np.array(x_val)
y_val = np.array(y_val)

x_train = np.array(x_new_train)
y_train = np.array(y_new_train)

x_test = np.array(x_new_train)
y_test = np.array(y_new_train)

# print(x_val.shape)
# print(y_val.shape)
# print(x_train.shape)
# print(y_train.shape)





# %%
# pca

X = x_train
Y = y_train

X=X.T
mean = np.mean(X, axis=1,keepdims=True)
X=X-mean

x_val=x_val.T
x_val=x_val-mean

x_test = x_test.T
x_test = x_test - mean

# find covariance matrix, corresponding eigen values and eigenvectors
covariance_matrix = np.dot(X, X.T) / (X.shape[0]-1)

eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
U = eigenvectors

# find pca of the x_train, x_val and x_test
U5 = U[:, :5]
Y5 = np.dot(U5.T,X)
x_val =np.dot(U5.T,x_val)
x_test = np.dot(U5.T, x_test)

X = Y5.T
x_val = x_val.T
x_test = x_test.T

# print(X.shape)
# print(Y.shape)
# print(x_val.shape)
# print(y_val.shape)
# print(x_test.shape)
# print(y_test.shape)



# %%

predictions = []
alphas = []
weights = [1/X.shape[0]] * X.shape[0]
accuracies = []


midpoints_all = []
sorted_train_all = []
for d in range(X.shape[1]):

    # sorting the train set for a dimension and taking random 1000 midpoints
    sorted_indices = np.argsort(X[:, d])

    sorted_X = X[sorted_indices]
    sorted_Y = Y[sorted_indices]

    sorted_X = sorted_X.T

    midpoints = []
    for i in range(sorted_X.shape[1] -1):
        a = sorted_X[d][i]
        b = sorted_X[d][i+1]
        midpoint = (a+b) / 2
        midpoints.append(midpoint)
    midpoints = np.random.choice(midpoints, size=1000, replace=False)
    midpoints_all.append( midpoints)
    sorted_train_all.append([sorted_X, sorted_Y])


def iteration(label):
    # creates a decison tree with the updated weights and gives the accuracy over the val set

    final_miss_classified_indices = []
    min_miss_classified_weight = float("inf")
    mid = 0
    dimension = 0
    final_predicted_class_1 = 0
    final_predicted_class_2 = 0

    for d in range(X.shape[1]):

        # sorting the train set for a dimension and taking random 1000 midpoints
        # sorted_indices = np.argsort(X[:, d])

        # sorted_X = X[sorted_indices]
        # sorted_Y = Y[sorted_indices]

        # sorted_X = sorted_X.T

        # midpoints = []
        # for i in range(sorted_X.shape[1] -1):
        #     a = sorted_X[d][i]
        #     b = sorted_X[d][i+1]
        #     midpoint = (a+b) / 2
        #     midpoints.append(midpoint)

        # midpoints = np.random.choice(midpoints, size=1000, replace=False)
        sorted_X = sorted_train_all[d][0]
        sorted_Y = sorted_train_all[d][1]
        for i in range (1000):
            miss_classified_indices = []
            midpoint = midpoints_all[d][i]
            minus_ones = 0
            ones = 0
            pos = 0

            # predicting class for the region 1 based on majority voting
            for j in range (sorted_X.shape[1]):
                if (sorted_X[d][j] <= midpoint):
                    if (sorted_Y[j] == -1):
                        minus_ones +=1
                    else:
                        ones +=1
                else:
                    pos = j
                    break

            predicted_class_1 , predicted_class_2 = 0, 0
            if (minus_ones > ones):
                predicted_class_1 = -1
                predicted_class_2 = 1
            else:
                predicted_class_1  = 1
                predicted_class_2 = -1

            # finding miss classified weight for a prediction

            miss_classified_weight = 0

            for j in range (sorted_X.shape[1]):
                if (j < pos and sorted_Y[j] != predicted_class_1):
                    miss_classified_indices.append(j)
                    miss_classified_weight += weights[j]
                elif (j >= pos and sorted_Y[j] != predicted_class_2):
                    miss_classified_indices.append(j)
                    miss_classified_weight += weights[j]
            miss_classified_weight /= sum(weights)

            # finding the decison tree with minimum miss classified weight
            if (min_miss_classified_weight > miss_classified_weight):
                min_miss_classified_weight = miss_classified_weight
                mid = midpoint
                dimension = d
                final_predicted_class_1 = predicted_class_1
                final_predicted_class_2 = predicted_class_2
                final_miss_classified_indices = miss_classified_indices

    # updating the weights based on miss classification
    loss = min_miss_classified_weight
    alpha =  np.log((1-loss) / loss)
    # print(len(final_miss_classified_indices))
    for i in range(len(final_miss_classified_indices)):
        idx = final_miss_classified_indices[i]
        weights[idx] *= np.exp(alpha)

    alphas.append(alpha)
    predictions.append([dimension, mid, final_predicted_class_1, final_predicted_class_2])

    # finding accuracy on val set
    correct_predicted_samples = 0
    accuracy = 0
    for i in range(x_val.shape[0]):
        y = y_val[i]
        prediction = 0

        for j in range (0, len(alphas)):
            d = predictions[j][0]
            m = predictions[j][1]
            p1 = predictions[j][2]
            p2 = predictions[j][3]

            if (x_val[i][d] <=m):
                p = p1
            else:
                p = p2
            prediction+= alphas[j]*p

        if (prediction < 0):
            prediction = -1
        else:
            prediction = 1

        if (prediction == y):
            correct_predicted_samples+=1

    # print("Correct predicted samples:", correct_predicted_samples)
    accuracy =correct_predicted_samples / x_val.shape[0]
    accuracies.append(accuracy)

    print(f"Accuracy of decision tree in iteration {label}: " , accuracy )
    return accuracy


# parameters required for the most accurate decision tree


max_accuracy = 0
no_of_decision_trees = []
best_tree_index = 0
# iterating and creating 300 decision trees

for i in range(300):
    no_of_decision_trees.append(i+1)
    accuracy = iteration(i+1)
    print(f"Dimension: {predictions[-1][0]}, Midpoint: {predictions[-1][1]}, PredictionClass1: {predictions[-1][2]}, PredictionsClass2: {predictions[-1][3]}, Alpha: {alphas[-1]}")
    if (accuracy > max_accuracy ):
        max_accuracy = accuracy
        best_tree_index = i

# print(max_accuracy, best_decision_tree_dimension, best_decision_tree_midpoint, best_decision_tree_predicted_class_1, best_decision_tree_predicted_class_2)



# %%
# plotting the graph for accuracies over val set for each decision stump


plt.plot(no_of_decision_trees, accuracies)

plt.title('Plot of accuracy on val set VS number of trees')
plt.xlabel('Number of trees')
plt.ylabel('Accuracies')


plt.show()

# print(f"Accuracy over val set: {max_accuracy}")
# print(f"Parameters of most accurate decision tree: \n Dimension: {best_decision_tree_dimension} \n Midpoint for split: {best_decision_tree_midpoint} \n Predicted class 1: {best_decision_tree_predicted_class_1} \n Predicted class 2: {best_decision_tree_predicted_class_2}")



# %%
# finding accuracy over the test set

correct_predicted_samples = 0
for i in range(x_test.shape[0]):
    y = y_test[i]
    prediction = 0
    for j in range (0, best_tree_index+1):
        d = predictions[j][0]
        m = predictions[j][1]
        p1 = predictions[j][2]
        p2 = predictions[j][3]

        if (x_test[i][d] <=m):
            p = p1
        else:
            p = p2
        prediction+= alphas[j]*p
    prediction = np.sign(prediction)
    if (prediction == y):
        correct_predicted_samples+=1

    accuracy =correct_predicted_samples/ x_test.shape[0]

print(f"Accuracy over the test set: {accuracy}")






