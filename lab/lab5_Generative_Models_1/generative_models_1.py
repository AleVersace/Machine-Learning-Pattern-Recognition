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


# Uses log-MVG probability process to estimate labels and prints results
def computeAndStats(ld1, ld2, ld3):
    lS = np.vstack((ld1, ld2, ld3))
    LSjoint = lS + np.log(1/3)
    LMarginalP = scipy.special.logsumexp(LSjoint, axis=0)
    logSPost = LSjoint - LMarginalP
    lPredicted = logSPost.argmax(axis=0)
    print("\nTied Covariance Gaussian Classifier:\nPredicted Classes:\n{}".format(lPredicted))
    lacc = ((lPredicted == LTE).sum()) / (LTE.shape[0])
    lerr = 1 - lacc
    print("\nAccuracy: {}, Error: {}".format(lacc, lerr))


# Function that uses a MVG and a log-MVG approach
def GaussianClassifiers(DTR, LTR, DTE, LTE, D, L):

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
    print("\nMVG Classifier:\nPredicted classes: {}".format(PredictedClass))

    ##
    # Model Accuracy (Number of correct estimations on the evaluation set)
    ##
    acc = ((PredictedClass == LTE).sum()) / (LTE.shape[0])
    err = 1 - acc
    print("\nAccuracy: {}, Error: {}".format(acc, err))


    #####
    ## Working with log-MVG densities is less problematic (numerical issues) and we obtain the same result
    #####

    ld1 = logMVG(DTE, MUc1, Cc1)
    ld2 = logMVG(DTE, MUc2, Cc2)
    ld3 = logMVG(DTE, MUc3, Cc3)
    lS = np.vstack((ld1, ld2, ld3))
    LSjoint = lS + np.log(1/3)

    LMarginalP = scipy.special.logsumexp(LSjoint, axis=0)     # More robust
    # or
    #maxLc = LSjoint.max(axis=0)
    #LMarginalp = maxLc + np.log(np.exp(LSjoint - maxLc).sum(axis=0))    # Same thing as the library
    logSPost = LSjoint - LMarginalP

    lPredicted = logSPost.argmax(axis=0)
    print("\nlog-MVG Classifier:\nPredicted Classes:\n{}".format(lPredicted))
    lacc = ((lPredicted == LTE).sum()) / (LTE.shape[0])
    lerr = 1 - lacc
    print("\nAccuracy: {}, Error: {}".format(lacc, lerr))


    #####
    ## Naive Bayes Gaussian Classifier (deducted from MVG)
    #####

    NBCov1 = Cc1*np.eye(Cc1.shape[0])
    NBCov2 = Cc2*np.eye(Cc2.shape[0])
    NBCov3 = Cc3*np.eye(Cc3.shape[0])
    ld1 = logMVG(DTE, MUc1, NBCov1)
    ld2 = logMVG(DTE, MUc2, NBCov2)
    ld3 = logMVG(DTE, MUc3, NBCov3)
    computeAndStats(ld1, ld2, ld3)



    #####
    ## Tied Covariance Gaussian Classifier (deducted from MVG)
    #####
    WithinCov = 1/DTR.shape[1] * (DTR1.shape[1]*Cc1 + DTR2.shape[1]*Cc2 + DTR3.shape[1]*Cc3)   # Within Class cov matrix deducted from MVG (Same of LDA)
    ld1 = logMVG(DTE, MUc1, WithinCov)
    ld2 = logMVG(DTE, MUc2, WithinCov)
    ld3 = logMVG(DTE, MUc3, WithinCov)
    computeAndStats(ld1, ld2, ld3)



if __name__=='__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    GaussianClassifiers(DTR, LTR, DTE, LTE, D, L)