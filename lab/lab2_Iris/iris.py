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

# Just Sepal length
fig, ax = plt.subplots()
ax.hist(sepal_l_0, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(sepal_l_1, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(sepal_l_2, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Sepal length")
ax.legend()
plt.show()

# Sepal length x Sepal width
ig, ax = plt.subplots()
ax.scatter(sepal_l_0, sepal_w_0, label=classes[0])
ax.scatter(sepal_l_1, sepal_w_1, label=classes[50])
ax.scatter(sepal_l_2, sepal_w_2, label=classes[100])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Sepal width")
ax.legend()
plt.show()

# Sepal length x Petal length
ig, ax = plt.subplots()
ax.scatter(sepal_l_0, petal_l_0, label=classes[0])
ax.scatter(sepal_l_1, petal_l_1, label=classes[50])
ax.scatter(sepal_l_2, petal_l_2, label=classes[100])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal length")
ax.legend()
plt.show()

# Sepal length x Petal width
ig, ax = plt.subplots()
ax.scatter(sepal_l_0, petal_w_0, label=classes[0])
ax.scatter(sepal_l_1, petal_w_1, label=classes[50])
ax.scatter(sepal_l_2, petal_w_2, label=classes[100])
ax.set_xlabel("Sepal length")
ax.set_ylabel("Petal width")
ax.legend()
plt.show()

# Sepal width x Sepal length
ig, ax = plt.subplots()
ax.scatter(sepal_w_0, sepal_l_0, label=classes[0])
ax.scatter(sepal_w_1, sepal_l_1, label=classes[50])
ax.scatter(sepal_w_2, sepal_l_2, label=classes[100])
ax.set_xlabel("Sepal width")
ax.set_ylabel("Sepal length")
ax.legend()
plt.show()

# Just Sepal width
fig, ax = plt.subplots()
ax.hist(sepal_w_0, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(sepal_w_1, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(sepal_w_2, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Sepal width")
ax.legend()
plt.show()

# Sepal width x Petal length
ig, ax = plt.subplots()
ax.scatter(sepal_w_0, petal_l_0, label=classes[0])
ax.scatter(sepal_w_1, petal_l_1, label=classes[50])
ax.scatter(sepal_w_2, petal_l_2, label=classes[100])
ax.set_xlabel("Sepal width")
ax.set_ylabel("Petal length")
ax.legend()
plt.show()

# Sepal width x Petal width
ig, ax = plt.subplots()
ax.scatter(sepal_w_0, petal_w_0, label=classes[0])
ax.scatter(sepal_w_1, petal_w_1, label=classes[50])
ax.scatter(sepal_w_2, petal_w_2, label=classes[100])
ax.set_xlabel("Sepal width")
ax.set_ylabel("Petal width")
ax.legend()
plt.show()

# Petal length x Sepal length
ig, ax = plt.subplots()
ax.scatter(petal_l_0, sepal_l_0, label=classes[0])
ax.scatter(petal_l_1, sepal_l_1, label=classes[50])
ax.scatter(petal_l_2, sepal_l_2, label=classes[100])
ax.set_xlabel("Petal length")
ax.set_ylabel("Sepal length")
ax.legend()
plt.show()

# Petal length x Sepal width
ig, ax = plt.subplots()
ax.scatter(petal_l_0, sepal_w_0, label=classes[0])
ax.scatter(petal_l_1, sepal_w_1, label=classes[50])
ax.scatter(petal_l_2, sepal_w_2, label=classes[100])
ax.set_xlabel("Petal length")
ax.set_ylabel("Sepal width")
ax.legend()
plt.show()

# Just Petal length
fig, ax = plt.subplots()
ax.hist(petal_l_0, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(petal_l_1, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(petal_l_2, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Petal length")
ax.legend()
plt.show()

# Petal length x Petal width
ig, ax = plt.subplots()
ax.scatter(petal_l_0, petal_w_0, label=classes[0])
ax.scatter(petal_l_1, petal_w_1, label=classes[50])
ax.scatter(petal_l_2, petal_w_2, label=classes[100])
ax.set_xlabel("Petal length")
ax.set_ylabel("Petal width")
ax.legend()
plt.show()

# Petal width x Sepal length
ig, ax = plt.subplots()
ax.scatter(petal_w_0, sepal_l_0, label=classes[0])
ax.scatter(petal_w_1, sepal_l_1, label=classes[50])
ax.scatter(petal_w_2, sepal_l_2, label=classes[100])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.legend()
plt.show()

# Petal width x Sepal width
ig, ax = plt.subplots()
ax.scatter(petal_w_0, sepal_w_0, label=classes[0])
ax.scatter(petal_w_1, sepal_w_1, label=classes[50])
ax.scatter(petal_w_2, sepal_w_2, label=classes[100])
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal width")
ax.legend()
plt.show()

# Petal width x Petal length
ig, ax = plt.subplots()
ax.scatter(petal_w_0, petal_l_0, label=classes[0])
ax.scatter(petal_w_1, petal_l_1, label=classes[50])
ax.scatter(petal_w_2, petal_l_2, label=classes[100])
ax.set_xlabel("Petal width")
ax.set_ylabel("Petal length")
ax.legend()
plt.show()

# Just Petal width
fig, ax = plt.subplots()
ax.hist(petal_w_0, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(petal_w_1, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(petal_w_2, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Petal width")
ax.legend()
plt.show()


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
print("\nPlotting the centered data...")

# Just Sepal length centered
fig, ax = plt.subplots()
ax.hist(sepal_l_0_c, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(sepal_l_1_c, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(sepal_l_2_c, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Sepal length centered")
ax.legend()
plt.show()

# Just Sepal width centered
fig, ax = plt.subplots()
ax.hist(sepal_w_0_c, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(sepal_w_1_c, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(sepal_w_2_c, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Sepal width centered")
ax.legend()
plt.show()

# Just Petal length centered
fig, ax = plt.subplots()
ax.hist(petal_l_0_c, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(petal_l_1_c, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(petal_l_2_c, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Petal length centered")
ax.legend()
plt.show()

# Just Petal width centered
fig, ax = plt.subplots()
ax.hist(petal_w_0_c, bins=10, density=True, label=classes[0], alpha=0.5, edgecolor="black")
ax.hist(petal_w_1_c, bins=10, density=True, label=classes[50], alpha=0.5, edgecolor="black")
ax.hist(petal_w_2_c, bins=10, density=True, label=classes[100], alpha=0.5, edgecolor="black")
ax.set_xlabel("Petal width centered")
ax.legend()
plt.show()
