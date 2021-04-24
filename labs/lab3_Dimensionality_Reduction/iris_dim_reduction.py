# Lab on Dimensionality Reduction applied on Iris Dataset
# PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis)

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as splin

# Dimension of sample's classes
S = 50

# Take an array and reshape it as a column array
def vcol(v):
    return v.reshape((v.size, 1))

# Take an array and reshape it as row array
def vrow(v):
    return v.reshape((1, v.size))

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
    classes.pop()
    return np.concatenate(attr_list, 0).reshape(150, 4).T, np.array(classes)


D, classes = load(filename)

# Statistics
print("\nMean over columns of the entire dataset:")
mu = D.mean(1)   # Mean with numpy over columns (1), over rows(0)
mu = vcol(mu)
print("Mean array {}".format(mu))           # 1d-array with shape 1x4

#####
# PCA
#####

# Calculate Covariance Matrix C = 1/N DC * DC'
# C = 1/DC.shape[1] * np.dot(DC, DC.T)
# print("\nCovariance Matrix {}".format(C))

# Calculate eigenvalues and eigenvectors 
# For a square matrix we can use np.linalg.eig
# In this case because the covariance matrix is symmetric we can use linalg.eigh
# which returns the eigenvalues and the relative eigenvectors sorted from smallest to largest
# s, U = np.linalg.eigh(C)

# Reverse the order of the eigenvectors and take frist m columns (highest eigenvalues corrispondence)
# P = U[:, ::-1][:, 0:m]

# Since the Covariance Matrix is semi-definite positive we can use SVD decomposition
# C = USV' whenre V'=U' in this case
# U, s, Vh = np.linalg.svd(C)
# P = U[:, 0:m]     This time eigvectors are already sorted correctly

# Apply the projection to a single point x or a matrix of samples D
# y = np.dot(P.T, x)
# y = np.dot(P.T, D)

# Scatter plot function
def scatter(f0, f1, f2, s0, s1, s2, classes, xlabel="", ylabel=""):
    ig, ax = plt.subplots()
    ax.scatter(f0, s0, label=classes[S*0])
    ax.scatter(f1, s1, label=classes[S])
    ax.scatter(f2, s2, label=classes[S*2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


# Compute Covariance Matrix removing mean
def covMatrix(Data):
    DC = Data - vcol(Data.mean(1))
    C = (1/DC.shape[1]) * np.dot(DC, DC.T)
    return C

# Projection matrix using PCA
def pcaProjMatrix(Data, m):
    C = covMatrix(Data)

    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    # or    Different plot because of different values of autovectors in SVD
    # U, s, Vh = np.linalg.svd(C)
    # P = U[:, 0:m]

    # Projected points (matrix)
    DP = np.dot(P.T, Data)
    return DP

m = 2
DP = pcaProjMatrix(D, m)
scatter(DP[0][0:S], DP[0][S:S*2], DP[0][S*2:S*3], DP[1][0:S], DP[1][S:S*2], DP[1][S*2:S*3], classes)


#####
# LDA
#####

# Compute LDA transformation matrix W

# Sb between class covariance matrix

# Sw within class covariance matrix can be computer as weighted sum of covariance matrices of each class Sw,c of c-th class
# Sw,c can be computed as PCA: remove class mean from same class samples and compute C

# Create masks to filter data by label 
M = []
M.append((classes == classes[0]))
M.append((classes == classes[S]))
M.append((classes == classes[S*2]))


# Within class covariance matrix
# Covariance Matrixes same class samples list
def withinClassCovMatrix(D, L, M):
    D0 = D[:, M[0]]
    D1 = D[:, M[1]]
    D2 = D[:, M[2]]
    # Compute classes means over columns of the dataset matrix
    mu0 = vcol(D0.mean(axis=1))
    mu1 = vcol(D1.mean(axis=1))
    mu2 = vcol(D2.mean(axis=1))
    n0 = D0.shape[1]
    n1 = D1.shape[1]
    n2 = D2.shape[1]
    # Compute within covariance matrix for each class
    Sw0 = (1/n0)*np.dot(D0-mu0, (D0-mu0).T)
    Sw1 = (1/n1)*np.dot(D1-mu1, (D1-mu1).T)
    Sw2 = (1/n2)*np.dot(D2-mu2, (D2-mu2).T)
    return (1/(n0+n1+n2))*(n0*Sw0+n1*Sw1+n2*Sw2)



# Between class covariance matrix
# Dataset, Matrixes same class samples list, column means
def betweenClassCovM(D, Dd, mu):
    Sb = 0
    for m in Dd:
        diff = (vcol(m.mean(1)) - mu)
        Sb += S*diff*diff.T
    return Sb/(S*3)

Dd = []
Dd.append(D[:, M[0]])
Dd.append(D[:, M[1]])
Dd.append(D[:, M[2]])
Sb = betweenClassCovM(D, Dd, mu)
Sw = withinClassCovMatrix(D, classes, M)
print(Sb)
print(Sw)

# Generalized eigenvalue problem Sb w = l Sw w
# Projection Matrix using LDA generalized eigenvalue problem
def ldaProjMatrix(Data, Sb, Sw, m):

    s, U = splin.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]

    DP = np.dot(W.T, Data)
    return DP

DP = ldaProjMatrix(D, Sb, Sw, m)
scatter(DP[0][0:S], DP[0][S:S*2], DP[0][S*2:S*3], DP[1][0:S], DP[1][S:S*2], DP[1][S*2:S*3], classes)

