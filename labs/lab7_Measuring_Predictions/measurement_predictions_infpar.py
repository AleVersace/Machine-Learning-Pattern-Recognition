import numpy as np
import scipy.special


llr = np.load('data/commedia_llr_infpar.npy')
labels = np.load('data/commedia_labels_infpar.npy')


""" Compute confusion matrix for binary task
@param triple: (prior class probability, cost of false negative, cost of false positive)
@param llr: log likelihood ratios
@param labels: actual samples' labels to compute confusion matrix
@return confMatr: 2x2 numpy confusion Matrix 
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
    return confMatr

cm1 = confusionBinaryMatr((0.5, 1, 1), llr, labels)
cm2 = confusionBinaryMatr((0.8, 1, 1), llr, labels)
cm3 = confusionBinaryMatr((0.5, 10, 1), llr, labels)
cm4 = confusionBinaryMatr((0.8, 1, 10), llr, labels)

#####
## Binary Task: Evaluation
#####

"""
Computes the un-normalized bayes risk of using the classifier R useful to compare different systems
@param triple: (prior class probability, cost of false negative, cost of false positive)
@param confMatr: 2x2 binary class confusion matrix
@return DCFu: empirical bayes risk of using the classifier on the evaluation set
"""
def bayesRisk(triple, confMatr):
    FNR = confMatr[0][1] / (confMatr[0][1] + confMatr[1][1])
    FPR = confMatr[1][0] / (confMatr[1][0] + confMatr[0][0])
    DCFu = round(triple[0] * triple[1] * FNR + (1 - triple[0]) * triple[2] * FPR, 3)
    print(DCFu)
    return DCFu

DCFu1 = bayesRisk((0.5, 1, 1), cm1)
DCFu2 = bayesRisk((0.8, 1, 1), cm2)
DCFu3 = bayesRisk((0.5, 10, 1), cm3)
DCFu4 = bayesRisk((0.8, 1, 10), cm4)
print('\n')

"""
Computes the normalized bayes risk (Detection Cost Function binary task) to compare classifier with an optimal system that doesn't use the test data
(ie: returns always 1 or always 0) -> min(p1 * cfn, (1-p1) * cfp)
If DCF is < 1 the system if useful, otherwise is even harmful

@param triple: (prior class probability, cost of false negative, cost of false positive)
@param DCFu: un-normalize detection cost function
@return DCF: normalize detection cost function
"""
def normalizedDCF(triple, DCFu):
    m = min(triple[0] * triple[1], (1 - triple[0]) * triple[2])
    DCF = round(DCFu/m, 3)
    print(DCF)
    return DCF

DCF1 = normalizedDCF((0.5, 1, 1), DCFu1)
DCF2 = normalizedDCF((0.8, 1, 1), DCFu2)
DCF3 = normalizedDCF((0.5, 10, 1), DCFu3)
DCF4 = normalizedDCF((0.8, 1, 10), DCFu4)
