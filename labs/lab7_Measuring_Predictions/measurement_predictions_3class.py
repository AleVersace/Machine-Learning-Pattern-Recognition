import numpy as np
import scipy.special

ll = np.load('data/commedia_ll.npy')
labels = np.load('data/commedia_labels.npy')
ll += np.log(1/3)
ll -= scipy.special.logsumexp(ll, axis=0)
predicted = ll.argmax(axis=0)

###
# Compute Confusion Matrix (3 classes hell, pur, heaven)
###
p0 = (predicted == 0)
p1 = (predicted == 1)
p2 = (predicted == 2)
l0 = (labels == 0)
l1 = (labels == 1)
l2 = (labels == 2)
confMatr = np.zeros((3,3))
confMatr[0][0] = (p0*l0).sum()
confMatr[0][1] = (p0*l1).sum()
confMatr[0][2] = (p0*l2).sum()
confMatr[1][0] = (p1*l0).sum()
confMatr[1][1] = (p1*l1).sum()
confMatr[1][2] = (p1*l2).sum()
confMatr[2][0] = (p2*l0).sum()
confMatr[2][1] = (p2*l1).sum()
confMatr[2][2] = (p2*l2).sum()
print(confMatr)
