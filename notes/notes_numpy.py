import numpy as np

b = np.array([1, 2, 3])
print(b)
print(b.shape)

b = np.array([[1,2,3], [4,5,6]])
print(b)
print("Array shape: {}".format(b.shape))
print("Array size: {}".format(b.size))
print("Array ndim: {}".format(b.ndim))
print("Array data type: {}".format(b.dtype))

b = np.array([1,2,3,4], dtype=np.float64)
print(b)
a = np.array(b)
print(a)

a = np.zeros((2,5), dtype=np.float32)
print(a)
a = np.ones(3)
print(a)
a = np.arange(10)
print(a)
a = np.eye(4)
print(a)

a = np.linspace(0, 5, 10)   # Evenly spaced values
print(a)

# Array operations
print("\nArray operations:")
x = np.array([[1,2,3], [4,5,6]])
y = np.array([[2,2,2], [3,3,3]])
print(x+y)
print(x*y)

# Matrix operations
print("\nMatrix operations:")
x = np.array([[1,2], [3,4], [5,6]])
y = np.array([[1,2,3], [4,5,6]])
print(np.dot(x, y))

# Manipulation
print("\nManipulation:")
x = np.arange(3)
print(x.reshape((1, x.size)))
print(x.reshape((x.size, 1)))
x = np.arange(12).reshape(2,2,3)
print(x)
print(x.ravel())    # Not a row vector!!
print(x.ravel().shape)

x = np.arange(3).reshape(3, 1)
print(x.T)  # Traspose
print(x.sum())
print(x.max())

x = np.array([[1,2,3], [4,5,6]])
print(x.sum(axis = 0))  # column-wise
print(x.sum(axis = 1))  # row-wise


# Slicing
print("\nSlicing:")
print(x[0]) # First row
print(x[0:2])   # 0 included 2 excluded
print(x[:, 1:3])    # Columns again 1 included 3 excluded
idx = np.array([0,0,1])
print(x[idx, :])    # Row 0, row 0, row 1
print(x[:, idx])    # Column 0, column 0, column 1

# Boolean Arrays
print("\nBoolean Arrays:")
x = np.array([0,1,1,1,0,0,1,0], dtype=bool)
print(x)
x = np.array([[1,2,3], [4,5,6]])
m = x > 3
print(m)    # Result comparations matrix

# Sharing data arrays
print("\nSharing data arrays")
x = np.zeros(6)
print(x)
y = x[:3]
print(y)
y[:] = 3
print(x)
print(y)
print(x.flags.owndata)
print(y.flags.owndata)  # Nope


# Broadcasting: numpy let us apply elementwise operations
# to arrays with different shapes
# It will prepend 1s on the smallest sized data
# Axes with shape 1 are treated as if they have the same dimension of the larger array
# so it will replicate the smallest one a number of time needed to be compatible with the larger one.
print("\nBroadcasting:")
# TODO:Update this topic with examples

# Matrix A = NxM compatible with NxM, Nx1, 1xM, 1x1


# Linear Algebra
print("\nLinear Algebra:")
x = np.arange(16).reshape(4,4)
print(np.linalg.eig(x))