import pkg_resources
import numpy as np 

def load_worms2(N = 10000, seed = None):
    # Package data loading instructions taken from here: 
    # https://kiwidamien.github.io/making-a-python-package-vi-including-data-files.html
    """
    The `worms2 dataset <http://cs.uef.fi/sipu/datasets/FastDensityPeaks.pdf>`_
    
    Returns a subsample of the worms2 dataset, of desired size, with associated cluster labels. 

    Parameters
    ----------
    N : int
      Integer denoting the requested sub-sample size, must be :math:`\leq 105,600`
    seed : int 
      Optional (default = ``None``). If given, passed to ``numpy.random.seed``. 

    Returns
    -------
    X : `numpy.ndarray, dtype=float64, shape=(N,2)`
      Sample, with invidivual observations in rows 
    XL : `numpy.ndarray, dtype=int64, shape=(N,)
      Cluster labels associated with each observation in the returned sample. 
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