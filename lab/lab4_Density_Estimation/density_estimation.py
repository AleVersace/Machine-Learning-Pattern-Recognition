import numpy as np 
import matplotlib.pyplot as plt

dataset = "lab/lab4_Density_Estimation/Data/XGau.npy"

#####
#   Gaussian density estimation
#####


# Histogram function
def histogram(f0, label, classes, b=10, d=True, a=1, edgec="black"):
    fig, ax = plt.subplots()
    ax.hist(f0, bins=b, density=d, label=classes, alpha=a, edgecolor=edgec)
    ax.set_xlabel(label)
    ax.legend()
    plt.show()


# Compute Normal Density
# params: (1d array input, mean, variance)
# returns: 1d array output y, where yi is a Normal(xi| mean, variance)
def GAU_pdf(x, mu, var):    # Using broadcasting
    return 1/(np.sqrt(2*np.pi*var)) * np.exp(-(x-mu)**2/(2*var))

# Load dataset (10000, 1)
XGAU = np.load(dataset)
histogram(XGAU, "label", "label1", b=50)


# Try to calculate Normal density of an array
XPlot = np.linspace(-8, 12, 10000)
plt.figure()
plt.plot(XPlot, GAU_pdf(XPlot, 1.0, 2.0))
plt.show()


# Compute Likelihood L(mu, var) for the dataset XGAU
ll_samples = GAU_pdf(XGAU, 1., 2.)
likelihood = ll_samples.prod()  # This will give us 0 because every yi, Im(xi) of a probability distribution is less than 1 
print(likelihood)               # Multipling them, if the dataset il large we will get 0


# We must work in log scale to calculate the likelihood
# In log scale the lo-likelihood is defined as sum of all yi log-NormalDensity
def GAU_logpdf(x, mu, v):
    return -1/2 * np.log(2*np.pi) - 1/2 * np.log(v) - (x - mu)**2 / (2*v)
    

ll_samples = GAU_logpdf(XGAU, 1., 2.)
log_likelihood = np.sum(ll_samples)
print(log_likelihood)
ll_samples1 = GAU_logpdf(XGAU, 5., 20.)
log_likelihood1 = np.sum(ll_samples1)
print(log_likelihood1)
ll_samples2 = GAU_logpdf(XGAU, 50., 100.)
log_likelihood2 = np.sum(ll_samples2)
print(log_likelihood2)


# Finally analyse the dataset with Maximum Likelihood
def loglikelihood(x, mML, vML):
    y = GAU_logpdf(x, mML, vML)
    return np.sum(y)

print("\nAnalyse dataset XGAU:")
muML = XGAU.mean()
vML = XGAU.var()
ll = loglikelihood(XGAU, muML, vML)
print(ll)
print(muML)
print(vML)

plt.figure()
plt.hist(XGAU, bins=50, density=True, alpha=1, edgecolor="black")
plt.plot(XPlot, np.exp(GAU_logpdf(XPlot, muML, vML)))
plt.show()



#####
#   Multvariate Gaussian
#####
print("\nMultivariate Gaussian:")

def logpdf_GAU_ND(x, mu, C):
    return - x.shape[0]/2 * np.log(2*np.pi) - 1/2 * (np.linalg.slogdet(C)[1]) - 1/2*((np.dot((x - mu).T, np.linalg.inv(C))).T * (x - mu)).sum(axis=0)

XND = np.load("lab/lab4_Density_Estimation/Solution/XND.npy")
mu = np.load("lab/lab4_Density_Estimation/Solution/muND.npy")
C = np.load("lab/lab4_Density_Estimation/Solution/CND.npy")
pdfSol = np.load("lab/lab4_Density_Estimation/Solution/llND.npy")
pdfGau = logpdf_GAU_ND(XND, mu, C)
print(np.abs(pdfSol - pdfGau).mean())

