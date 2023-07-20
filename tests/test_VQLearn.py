import os
import numpy as np
from sklearn import datasets
import NeuralVQ as nvq
import matplotlib.pyplot as plt

## Load data 
# Iris 
#data = datasets.load_iris().data 
#labels = datasets.load_iris().target

# worms2, subset 
data, labels = nvq.load_worms2(N=5000, seed=123)

N = data.shape[0]
d = data.shape[1]


## Generate random starting prototypes in range of data 
M = int(np.round(5 * np.sqrt(N))) # number of desired prototypes 
W0 = nvq.randinit_W0_NG(data = data, M = M, method = 'rand_global_rng', seed = 123)


## Set Neural Gas parameters 
# Learning params
rho0 = np.sqrt(M) # initial neighborhood radius
rho_anneal = 0.95 # anneal rho(t) = 0.95 * rho(t-1)
rho_min = 0.75 # minimum value of rho, to prevent devolving into k-Means
eta_min = 0.01 # minimum value of neighborhood updates, to expedite calculation
# Convergence params, learning stops when any of the respective monitoring quantities is >= their supplied value 
n_epochs = 100 
conv_delBMU = 0.01
conv_delMQE = 0.00

## Learning 
# This uses OMP in parallel. Can change number of threads used for calculation, e.g.: os.environ["OMP_NUM_THREADS"] = "5"
# Initialize learner class 
vql = nvq.VQLearn(X=data, W=W0, rho0=rho0, rho_anneal=rho_anneal, rho_min=rho_min, eta_min=eta_min, verbosity=2, XL = labels)
# Learn 
vql.learn(n_epochs = n_epochs, conv_delBMU=conv_delBMU, conv_delMQE=conv_delMQE)

## View learn histories
plt.subplot(1,2,1)
plt.step(vql.LearnHist.Epoch, vql.LearnHist.MQE)
plt.xlabel('Epoch'); plt.ylabel('MQE')
plt.subplot(1,2,2)
plt.step(vql.LearnHist.Epoch, vql.LearnHist.delBMU)
plt.xlabel('Epoch'); plt.ylabel('delBMU'); 
plt.suptitle('Learn History')
plt.show()


## Recall 
# A full recall is performed at the end of learning, stored as member of learn object 
vql.Recall 
# Can also perform one manually, with data & learned prototypes 
rec = nvq.VQRecall(nBMU = 2, nAnnoyTrees=50)
rec.Recall(X=data, W=vql.W(), XL = labels)
# Check that they're the same 
vql.Recall.BMU == rec.BMU

## Visualize prototypes in data cloud 
# Extract learned prototyeps 
W = vql.W()
# Plot data + initial + learned prototypes
plotdim1 = 0; plotdim2 = 1
plt.scatter(x=data[:,plotdim1], y=data[:,plotdim2], c='gray', s=1)
plt.scatter(x=W0[:,plotdim1], y=W0[:,plotdim2], c='dodgerblue', s=4, marker='d') # initial prototypes 
plt.scatter(x=W[:,plotdim1], y=W[:,plotdim2], c='firebrick', s=4, marker='s') # learned prototypes 
plt.show()

