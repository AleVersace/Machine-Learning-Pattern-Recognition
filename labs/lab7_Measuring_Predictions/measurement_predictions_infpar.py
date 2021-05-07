import numpy as np
import scipy.special


llr = np.load('data/commedia_llr_infpar.npy')
labels = np.load('data/commedia_labels_infpar.npy')


""" Compute confusion matrix for binary task
@param triple: (prior class probability, cost of false negative, cost of false positive)
@param llr: log likelihood ratios
@param labels: actual samples' labels to compute confusion matrix 
"""
def confusionBinaryMatr(triple, llr, labels):
    Ht = llr > -np.log(triple[0]*triple[1] / ((1-triple[0]) * triple[2]))
    Hf = llr <= -np.log(triple[0]*triple[1] / ((1-triple[0]) * triple[2]))
    l1 = (labels == 1)
    l0 = (labels == 0)
    confMatr = np.zeros((2, 2))
    confMatr[0][0] = (Hf*l0).sum()
    confMatr[0][1] = (Hf*l1).sum()
    confMatr[1][0] = (Ht*l0).sum()
    confMatr[1][1] = (Ht*l1).sum()
    print(confMatr)
    
confusionBinaryMatr((0.5, 1, 1), llr, labels)
confusionBinaryMatr((0.8, 1, 1), llr, labels)
confusionBinaryMatr((0.5, 10, 1), llr, labels)
confusionBinaryMatr((0.8, 1, 10), llr, labels)
