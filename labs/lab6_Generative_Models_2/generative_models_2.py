# Not working properly. Results are different from what is expected.

from data.load import load_data, split_data
import numpy as np
import scipy.special

# Create a set with word contained at least once in the 3 training set classes (union of words)
def unionWords(lInf_train, lPur_train, lPar_train):
    commW = set()
    for tercet in lInf_train:
        for w in tercet.split(" "):
            commW.add(w)
    for tercet in lPur_train:
        for w in tercet.split(" "):
            commW.add(w)
    for tercet in lPar_train:
        for w in tercet.split(" "):
            commW.add(w)
    return commW

# Create dictionary with words occurrencies
def dictOccurrencies(D, words, eps):
    d = {}

    for w in words: # set eps as start occurrency for each word contained in the whole training set
        d[w] = eps 
    
    for tercets in D:   # update occurrency based on class word occurrency
        for w in tercets.split(" "):
            if w in d.keys():
                d[w] += 1
            else:
                d[w] = eps + 1
    return d

# Compute each cantica ML parameters = (#Occ of word j in class C) / (Total words in class C)
def mlParameters(d):
    mlParams = {i[0]: (np.log(i[1]) - np.log(sum(d.values()))) for i in d.items()}
    return mlParams

# Compute evaluation on D list of tercets
def evalDictOccurrencies(mlP1, mlP2, mlP3, D, eps, label):
    d = {}
    i = 0
    for tercet in D:   # for each tercet evaluate
        for w in tercet.split(" "):
            if w in d.keys():
                d[w] += 1
            else:
                d[w] = eps + 1
        
        # class-conditionals on a single tercet
        Sp1 = []
        for s in d.items():
            if s[0] in mlP1.keys():
                Sp1.append(s[1]*mlP1[s[0]])
        Sp1 = sum(Sp1)
        Sp2 = []
        for s in d.items():
            if s[0] in mlP2.keys():
                Sp2.append(s[1]*mlP2[s[0]])
        Sp2 = sum(Sp2)
        Sp3 = []
        for s in d.items():
            if s[0] in mlP3.keys():
                Sp3.append(s[1]*mlP3[s[0]])
        Sp3 = sum(Sp3)
        St = np.vstack((Sp1, Sp2, Sp3))

        if i == 0:
            S = St
            i = 1
        S = np.hstack((S, St))
    return S

# Evaluate tercets based on 3 classes (3 cantiche)
def evalTercets3(mlP1, mlP2, mlP3, eval, eps, label):
    logS = evalDictOccurrencies(mlP1, mlP2, mlP3, eval, eps, label)

    ##
    # ScoreMatrix
    ##

    logS = logS + np.log(1/3)
    logSPost = logS - scipy.special.logsumexp(logS, axis = 0)
    logSPost = np.exp(logSPost)
    lPredicted = logSPost.argmax(axis=0)
    print(lPredicted.shape)
    print("\nMultinomial model Classifier:\nPredicted Classes:\n{}".format(lPredicted))
    lacc = ((lPredicted == label).sum()) / (len(eval))
    lerr = 1 - lacc
    print("\nAccuracy: {}, Error: {}".format(lacc, lerr))

    return lPredicted
    


if __name__=='__main__':
    
    # Load the tercets and split the lists in training and test lists
    eps = 0.001
    label = [0, 1, 2]
    lInf, lPur, lPar = load_data()

    lInf_train, lInf_evaluation = split_data(lInf, 4)
    lPur_train, lPur_evaluation = split_data(lPur, 4)
    lPar_train, lPar_evaluation = split_data(lPar, 4)

    words = unionWords(lInf_train, lPur_train, lPar_train)

    # compute words occurrencies (dicts)
    dInf = dictOccurrencies(lInf_train, words, eps)
    dPur = dictOccurrencies(lPur_train, words, eps)
    dPar = dictOccurrencies(lPar_train, words, eps)

    # lists of ml parameters
    mlInf = mlParameters(dInf)
    mlPur = mlParameters(dPur)
    mlPar = mlParameters(dPar)

    # Evaluate sets
    S = evalTercets3(mlInf, mlPur, mlPar, lInf_evaluation, eps, label[0])
    S = evalTercets3(mlInf, mlPur, mlPar, lPur_evaluation, eps, label[1])
    S = evalTercets3(mlInf, mlPur, mlPar, lPar_evaluation, eps, label[2])

    print(len(lInf_evaluation))
    print(len(lPur_evaluation))
    print(len(lPar_evaluation))
