import os
import numpy as np
from sklearn import datasets
import NeuralVQ as nvq
import matplotlib.pyplot as plt

# Load iris 
data = datasets.load_iris().data 
labels = datasets.load_iris().target

N = data.shape[0]
d = data.shape[1]

# Generate random starting prototypes in range of data 
M = 10 # number of desired prototypes 
W0 = np.random.uniform(low=data.min(), high=data.max(), size=(M,d))

# Set Neural Gas learning parameters 
n_epochs = 100 
rho0 = np.sqrt(M)
rho_anneal = 0.95 
rho_min = 0.75 
min_h = 0.01 

# Initialize learner class 
learn = nvq.VQLearn(X=data, W=W0, rho0=rho0, rho_anneal=rho_anneal, rho_min=rho_min, min_h=min_h)
# This uses OMP in parallel. Can change number of threads used for calculation. 
os.environ["OMP_NUM_THREADS"] = "5"
learn.train(n_epochs = 100)

# Extract learned prototyeps 
W = learn.W()

# Plot pairwise data + prototypes
plotdim1 = 0; plotdim2 = 1
plt.scatter(x=data[:,plotdim1], y=data[:,plotdim2], c='black', s=2)
plt.scatter(x=W0[:,plotdim1], y=W0[:,plotdim2], c='blue', s=4, marker='s')
plt.scatter(x=W[:,plotdim1], y=W[:,plotdim2], c='red', s=4, marker='s')
plt.show()

## Recall 
rec = nvq.VQRecall(nBMU = 2, nAnnoyTrees=50)
rec.Recall(X=data, W=W, XL = labels)

