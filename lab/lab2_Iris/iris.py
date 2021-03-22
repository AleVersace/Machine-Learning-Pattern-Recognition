# IRIS dataset. Load, visualize and compute basic statistics.
# Dataset contains 150 samples of iris flowers belonging to 3 different families
# with 50 samples for each class.
# Each sample has 4 attributes. Record dataset:
# sepal lenght(cm),sepal width(cm),petal length(cm),petal width(cm),familyName

import numpy as np
import matplotlib.pyplot as plt

# Loading dataset with 2d-array shape 4x150 and 1d-array 150 containing class labels
filename = 'lab/lab2_Iris/iris/iris.csv'

def load(dataset):
    attr_list = []
    classes = []
    with open(dataset, 'r') as f:
        for line in f.readlines():
            l = line.split(",")
            name = l.pop()
            classes.append(name[0:-1])
            attr_list.append(np.array([float(i) for i in l], dtype=np.float32))
    
    return np.concatenate(attr_list, 0).reshape(150, 4).T, np.array(classes)


attr, classes = load(filename)

# Create masks to filter data by label 
M0 = (classes == classes[0])
M1 = (classes == classes[50])
M2 = (classes == classes[100])
# Sepal length
sepal_l_0 = attr[0,M0]
sepal_l_1 = attr[0,M1]
sepal_l_2 = attr[0,M2]
# Sepal width
sepal_w_0 = attr[1,M0]
sepal_w_1 = attr[1,M1]
sepal_w_2 = attr[1,M2]
# Petal length
petal_l_0 = attr[2,M0]
petal_l_1 = attr[2,M1]
petal_l_2 = attr[2,M2]
# Petal width
petal_w_0 = attr[3,M0]
petal_w_1 = attr[3,M1]
petal_w_2 = attr[3,M2]


### Pairs of values with scatter plots

def histogram(f0, f1, f2, label, classes, b=10, d=True, a=1, edgec="black"):
    fig, ax = plt.subplots()
    ax.hist(f0, bins=b, density=d, label=classes[0], alpha=a, edgecolor=edgec)
    ax.hist(f1, bins=b, density=d, label=classes[50], alpha=a, edgecolor=edgec)
    ax.hist(f2, bins=b, density=d, label=classes[100], alpha=a, edgecolor=edgec)
    ax.set_xlabel(label)
    ax.legend()
    plt.show()

def scatter(f0, f1, f2, s0, s1, s2, xlabel, ylabel, classes):
    ig, ax = plt.subplots()
    ax.scatter(f0, s0, label=classes[0])
    ax.scatter(f1, s1, label=classes[50])
    ax.scatter(f2, s2, label=classes[100])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()

# Just Sepal length
histogram(sepal_l_0, sepal_l_1, sepal_l_2, "Sepal length", classes, a=0.5)

# Sepal length x Sepal width
scatter(sepal_l_0, sepal_l_1, sepal_l_2, sepal_w_0, sepal_w_1, sepal_w_2, "Sepal length", "Sepal width", classes)

# Sepal length x Petal length
scatter(sepal_l_0, sepal_l_1, sepal_l_2, petal_l_0, petal_l_1, petal_l_2, "Sepal length", "Petal length", classes)

# Sepal length x Petal width
scatter(sepal_l_0, sepal_l_1, sepal_l_2, petal_w_0, petal_w_1, petal_w_2, "Sepal length", "Petal width", classes)

# Sepal width x Sepal length
scatter(sepal_w_0, sepal_w_1, sepal_w_2, sepal_l_0, sepal_l_1, sepal_l_2, "Sepal width", "Sepal length", classes)

# Just Sepal width
histogram(sepal_w_0, sepal_w_1, sepal_w_2, "Sepal width", classes, a=0.5)

# Sepal width x Petal length
scatter(sepal_w_0, sepal_w_1, sepal_w_2, petal_l_0, petal_l_1, petal_l_2, "Sepal width", "Petal length", classes)

# Sepal width x Petal width
scatter(sepal_w_0, sepal_w_1, sepal_w_2, petal_w_0, petal_w_1, petal_w_2, "Sepal width", "Sepal width", classes)

# Petal length x Sepal length
scatter(petal_l_0, petal_l_1, petal_l_2, sepal_l_0, sepal_l_1, sepal_l_2, "Petal length", "Sepal length", classes)

# Petal length x Sepal width
scatter(petal_l_0, petal_l_1, petal_l_2, sepal_w_0, sepal_w_1, sepal_w_2, "Petal length", "Sepal width", classes)

# Just Petal length
histogram(petal_l_0, petal_l_1, petal_l_2, "Petal length", classes, a=0.5)

# Petal length x Petal width
scatter(petal_l_0, petal_l_1, petal_l_2, petal_w_0, petal_w_1, petal_w_2, "Petal length", "Petal width", classes)

# Petal width x Sepal length
scatter(petal_w_0, petal_w_1, petal_w_2, sepal_l_0, sepal_l_1, sepal_l_2, "Petal width", "Sepal length", classes)

# Petal width x Sepal width
scatter(petal_w_0, petal_w_1, petal_w_2, sepal_w_0, sepal_w_1, sepal_w_2, "Petal width", "Sepal width", classes)

# Petal width x Petal length
scatter(petal_w_0, petal_w_1, petal_w_2, petal_l_0, petal_l_1, petal_l_2, "Petal width", "Petal length", classes)

# Just Petal width
histogram(petal_w_0, petal_w_1, petal_w_2, "Petal width", classes, a=0.5)


# Statistics
print("\nMean over columns of the entire dataset:")
mu = attr.mean(1)   # Mean with numpy over columns (1), over rows(0)
print(mu)           # 1d-array with shape 1x4
print(mu.shape)


# Broadcasting  (Getting centered data)
print("\nRemoving mean of the dataset from the dataset:")   # We need the mean vector to be compatible with the dataset so we reshape it to a column vector
DC = attr - attr.mean(1).reshape((attr.shape[0], 1))    # 4x150 - 4x1 so the broadcasting is possible
print(DC)

sepal_l_0_c = DC[0, M0]
sepal_l_1_c = DC[0, M1]
sepal_l_2_c = DC[0, M2]
sepal_w_0_c = DC[1, M0]
sepal_w_1_c = DC[1, M1]
sepal_w_2_c = DC[1, M2]
petal_l_0_c = DC[2, M0]
petal_l_1_c = DC[2, M1]
petal_l_2_c = DC[2, M2]
petal_w_0_c = DC[3, M0]
petal_w_1_c = DC[3, M1]
petal_w_2_c = DC[3, M2]

# Plotting the centered data:
print("\nPlotting the centered data... (Just 1 hist and 1 scatter)")

# Just Sepal length centered
histogram(sepal_l_0_c, sepal_l_1_c, sepal_l_2_c, "Sepal length centered", classes, a=0.5)

# Sepal length x Sepal width
scatter(sepal_l_0_c, sepal_l_1_c, sepal_l_2_c, sepal_w_0_c, sepal_w_1_c, sepal_w_2_c, "Sepal length", "Sepal width", classes)