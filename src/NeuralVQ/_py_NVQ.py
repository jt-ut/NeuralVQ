import numpy as np 

def randinit_W0_NG(data, M, method='rand_dim_rng', seed=None):
    np.random.seed(seed)
    N = data.shape[0]
    d = data.shape[1]

    if method=='rand_dim_rng':
        W0 = np.zeros((M,d))
        for i in range(d):
            W0[:,i] = np.random.uniform(low=data[:,i].min(), high=data[:,i].max(), size=(M,))
    elif method=='rand_global_rng':
        W0 = np.random.uniform(low=data.min(), high=data.max(), size=(M,d))
    elif method=='rand_sample':
        W0 = data[np.random.choice(N, 2, replace=False), :]
    else:
        raise ValueError("init must be one of {'rand_dim_rng','rand_global_rng','rand_sample'}")
    
    return W0


def load_worms2(N = 10000, seed = None):
    # Package data loading instructions taken from here: 
    # https://kiwidamien.github.io/making-a-python-package-vi-including-data-files.html

    import pkg_resources
    import numpy as np 
    """
    Return a subsample of the worms2 dataset, with cluster labels
    """

    # Check requested sample size, should be <= 105,600
    if(N > 105600): raise ValueError("N must be <= 105600")
    
    # This is a stream-like object. If you want the actual info, call stream.read()
    stream = pkg_resources.resource_stream(__name__, 'data/worms2_N105600_data.csv')
    X = np.loadtxt(stream, delimiter=',')
    stream = pkg_resources.resource_stream(__name__, 'data/worms2_N105600_labels.csv')
    XL = np.loadtxt(stream, delimiter=',').astype('int')

    # If requested sample size = 105,600 (max), return 
    if(N == 105600): return X, XL

    # Otherwise, sub-sample 
    if seed is not None: np.random.seed(seed)
    idx = np.random.choice(X.shape[0], size=int(N), replace=False)
    X = X[idx,:]
    XL = XL[idx]

    return X, XL