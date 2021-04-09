import sklearn.datasets
import numpy as np
import scipy

# Loading Iris Dataset
def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

# Covariance Matrix
def covM(x, MUc):
    return 1/x.shape[1] * np.dot((x - MUc), (x - MUc).T)

# Take an array and reshape it as a column array
def vcol(v):
    return v.reshape((v.size, 1))

#####
# Gaussian Models
#####

# Split the dataset in training set and evaluation set
def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

#####
# Multivariate Gaussian Classifier
#####

def logMVG(x, mu, C):
    return - x.shape[0]/2 * np.log(2*np.pi) - 1/2 * (np.linalg.slogdet(C)[1]) - 1/2*((np.dot((x - mu).T, np.linalg.inv(C))).T * (x - mu)).sum(axis=0)


if __name__=='__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    DTR1 = DTR[:, (LTR == L[0])]
    DTR2 = DTR[:, (LTR == L[50])]
    DTR3 = DTR[:, (LTR == L[100])]
    MUc1 = vcol(DTR1.mean(axis=1))
    MUc2 = vcol(DTR2.mean(axis=1))
    MUc3 = vcol(DTR3.mean(axis=1))
    Cc1 = covM(DTR1, MUc1)
    Cc2 = covM(DTR2, MUc2)
    Cc3 = covM(DTR3, MUc3)

    # Density estimation on test samples given each class (likelihoods) 
    d1 = np.exp(logMVG(DTE, MUc1, Cc1))
    d2 = np.exp(logMVG(DTE, MUc2, Cc2))
    d3 = np.exp(logMVG(DTE, MUc3, Cc3))
    
    # Score Matrix S where S[i,j] is the score of sample j given class i
    S = np.vstack((d1, d2, d3))

    ##
    # Class posterior probabilities combining S with prior probabilities Pc = (1/3)
    ##
    # Joint Probabilities
    Sjoint = S / 3 # 1/3 is the prior probability of a class (3 class all equals)
    # Marginal Prob
    MarginalP = Sjoint.sum(axis=0)
    # Class Posterior Probabilities = Joint P / Marginal Prob
    SPost = Sjoint / MarginalP
    # Predicted Label
    PredictedClass = SPost.argmax(axis=0)
    print("\nPredicted classes: {}".format(PredictedClass))

    ##
    # Model Accuracy (Number of correct estimations on the evaluation set)
    ##
    acc = ((PredictedClass == LTE).sum()) / (LTE.shape[0])
    err = 1 - acc
    print("\nAccuracy: {}, Error: {}".format(acc, err))


    ###
    # Working with log densities is less problematic (numerical issues) and we obtain the same result
    ###
    print("\nUsing log densities:")

    ld1 = logMVG(DTE, MUc1, Cc1)
    ld2 = logMVG(DTE, MUc2, Cc2)
    ld3 = logMVG(DTE, MUc3, Cc3)
    lS = np.vstack((d1, d2, d3))
    LSjoint = lS + np.log(1/3)

    LMarginalP = scipy.special.logsumexp(LSjoint, axis=0)     # More robust
    # or
    #maxLc = LSjoint.max(axis=0)
    #LMarginalp = maxLc + np.log(np.exp(LSjoint - maxLc).sum(axis=0))    # Same thing as the library
    logSPost = LSjoint - LMarginalP

    lPredicted = logSPost.argmax(axis=0)
    print(lPredicted)
    lacc = ((lPredicted == LTE).sum()) / (LTE.shape[0])
    lerr = 1 - lacc
    print("\nAccuracy: {}, Error: {}".format(lacc, lerr))



