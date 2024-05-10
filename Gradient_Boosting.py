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

# find pca of the x_train and x_test
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

def iteration(label):
    # creates a decision tree wiht updated labels and finds the min MSE for the val set

    min_SSR = float("inf")
    final_midpoint = 0
    final_predicted_class_1 = 0
    final_predicted_class_2 = 0
    final_dimension = 0


    for d in range(X.shape[1]):
        # sorting the train depending on the dimension and taking 1000 random midpoints
        sorted_indices = np.argsort(X[:, d])

        sorted_X = X[sorted_indices]
        sorted_Y = [labels[i] for i in sorted_indices]

        sorted_X = sorted_X.T

        midpoints = []
        for i in range(sorted_X.shape[1] -1):
            a = sorted_X[d][i]
            b = sorted_X[d][i+1]
            midpoint = (a+b) / 2
            midpoints.append(midpoint)

        midpoints = np.random.choice(midpoints, size=1000, replace=False)
        for i in range(1000):
            midpoint = midpoints[i]
            predicted_class_1 = 0
            no_of_samples_class1 = 0
            predicted_class_2 = 0
            no_of_samples_class2 = 0

            # finding predictions by taking mean of the labels for the region

            for j in range(sorted_X.shape[1]):
                if (sorted_X[d][j] <= midpoint):

                    predicted_class_1 += sorted_Y[j]
                    no_of_samples_class1 +=1
                else:
                    predicted_class_2 += sorted_Y[j]
                    no_of_samples_class2 +=1
            predicted_class_1 /= no_of_samples_class1
            predicted_class_2 /= no_of_samples_class2

            # calculate SSR
            ssr = 0
            for j in range(sorted_X.shape[1]):
                if (sorted_X[d][j] <= midpoint):
                    ssr += (predicted_class_1 - sorted_Y[j])**2
                else:
                    ssr += (predicted_class_2 - sorted_Y[j])**2

            if (ssr < min_SSR):

                min_SSR = ssr
                final_midpoint = midpoint
                final_predicted_class_1 = predicted_class_1
                final_predicted_class_2 = predicted_class_2
                final_dimension = d

    predictions.append([final_dimension, final_midpoint, final_predicted_class_1, final_predicted_class_2])

    # calculating mse
    mse = 0
    for i in range(x_val.shape[0]):
        y = y_val[i]
        prediction = 0
        for j in range (0, len(predictions)):
            d = predictions[j][0]
            m = predictions[j][1]
            p1 = predictions[j][2]
            p2 = predictions[j][3]

            if (x_val[i][d] <=m):
                p = p1
            else:
                p = p2
            prediction+= 0.01 *p
        mse += (y - prediction) ** 2

    mse /= x_val.shape[0]
    print(f"MSE for iteration {label}: {mse}")
    return mse, final_dimension, final_midpoint, final_predicted_class_1, final_predicted_class_2


# paramters of the decision tree with the min MSE
min_mse = float("inf")
best_decision_tree_dimension = 0
best_decision_tree_midpoint = 0
best_decision_tree_predicted_class_1 = 0
best_decision_tree_predicted_class_2 = 0
best_tree_index = 0

mses = []
no_of_decision_trees =[]


labels = []
for i in range(Y.shape[0]):
    labels.append(Y[i])


for i in range (300):
    mse, dimension, midpoint, predicted_class_1, predicted_class_2 = iteration(i+1)
    no_of_decision_trees.append(i+1)
    mses.append(mse)

    # updating the labels based on the predictions
    for j in range(X.shape[0]):
        x = X[j][dimension]
        if (x <= midpoint):
            labels[j] = (labels[j] - (0.01 * (predicted_class_1)))
        else:
            labels[j] = (labels[j] - (0.01 * (predicted_class_2)))


    if (min_mse > mse):
        min_mse = mse
        best_tree_index = i





# %%
# plotting the graph

temp = mses
plt.plot(no_of_decision_trees, mses)

plt.title('Plot of MSE on val set VS number of trees')
plt.xlabel('Number of trees')
plt.ylabel('MSE')

plt.show()

# print(best_decision_tree_dimension, best_decision_tree_predicted_class_1 , best_decision_tree_predicted_class_2 , best_decision_tree_midpoint , min_mse )
# print(f"Min MSE over the val set: {min_mse}")


# %%
# finding the MSE over the tests set
mse = 0
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
        prediction+= 0.01 * p
    mse += (y-prediction)**2

mse /= x_test.shape[0]

print(f"MSE over the test set: {mse}")
print(predictions[best_tree_index])



# %%
