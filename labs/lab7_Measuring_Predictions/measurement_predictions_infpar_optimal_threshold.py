import numpy as np
import scipy.special


""" Compute confusion matrix for binary task
@param triple: (prior class probability, cost of false negative, cost of false positive)
@param llr: log likelihood ratios
@param labels: actual samples' labels to compute confusion matrix
@return confMatr: 2x2 numpy confusion Matrix 
"""
def confusionBinaryMatr(score, llr, labels):
    Ht = llr > score
    Hf = llr <= score
    l1 = (labels == 1)
    l0 = (labels == 0)
    confMatr = np.zeros((2, 2))
    confMatr[0][0] = (Hf*l0).sum()
    confMatr[0][1] = (Hf*l1).sum()
    confMatr[1][0] = (Ht*l0).sum()
    confMatr[1][1] = (Ht*l1).sum()
    return confMatr


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
    return DCFu


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
    return DCF


"""
Compute the minimum detection costs for 4 different application (prior, cfn, cfp) using an optimal threshold computation over increasing order scores.
"""
if __name__ == '__main__':
    llr = np.load('data/commedia_llr_infpar.npy')
    labels = np.load('data/commedia_labels_infpar.npy')
    scores = llr.copy()
    scores.sort()

    min1 = 2
    min2 = 2
    min3 = 2
    min4 = 2
    t1 = (0.5, 1, 1)
    t2 = (0.8, 1, 1)
    t3 = (0.5, 10, 1)
    t4 = (0.8, 1, 10)
    for s in scores:
        cm1 = confusionBinaryMatr(s, llr, labels)
        cm2 = confusionBinaryMatr(s, llr, labels)
        cm3 = confusionBinaryMatr(s, llr, labels)
        cm4 = confusionBinaryMatr(s, llr, labels)
        DCFu1 = bayesRisk(t1, cm1)
        DCFu2 = bayesRisk(t2, cm2)
        DCFu3 = bayesRisk(t3, cm3)
        DCFu4 = bayesRisk(t4, cm4)
        DCF1 = normalizedDCF(t1, DCFu1)
        DCF2 = normalizedDCF(t2, DCFu2)
        DCF3 = normalizedDCF(t3, DCFu3)
        DCF4 = normalizedDCF(t4, DCFu4)
        if DCF1 < min1:
            min1 = DCF1
        if DCF2 < min2:
            min2 = DCF2
        if DCF3 < min3:
            min3 = DCF3
        if DCF4 < min4:
            min4 = DCF4

    print("{}: {}".format(t1, min1))
    print("{}: {}".format(t2, min2))
    print("{}: {}".format(t3, min3))
    print("{}: {}".format(t4, min4))