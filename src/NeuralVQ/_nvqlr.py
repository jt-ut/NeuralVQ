import numpy as np 
import warnings
import NeuralVQ._nvqlr_cpp as cppnvq
import os
from scipy.sparse import coo_array

## Prototype initializer via random sampling 
def randinit_W0_NG(data, M, method='rand_dim_rng', seed=None):
    """
    Prototype initialization for Neural Gas via random sampling. 

    Parameters
    ----------
    data : `numpy.ndarray, shape=(N,d)`
      The data matrix which will be learned
    M : `int`
      Number of prototype vectors to be initialized. 
    method : `string`
      Must be one of {'rand_dim_rng','rand_global_rng','rand_sample'}. 
      `rand_dim_rng` initializes prototypes uniformly in the range of the data, by dimension.
      `rand_global_rng` initializes prototypes uniformly in the global range of the data. 
      `rand_sample` initializes prototypes to ``M`` randomly selected data vectors. 
    seed : `int`
      Optional (default=``None``) seed controlling random sampling. If given, passed to ``numpy.random.seed``. 
    
    Returns
    -------
    `numpy.ndarray, shape=(M,d)`
    """
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






class VQRecall():
    """
    Vector Quantizer Recall Class

    A class to perform Recall of a dataset through a Vector Quantizer (possibly in parallel), and store all products resulting from the forward / backward mappings. 
    The `Annoy library <https://github.com/spotify/annoy>`_ is used for fast distance calculation and nearest / neighbor search.  
    """


    #### Constructor 
    def __init__(self, nBMU = 2, nAnnoyTrees = 50):
        """"""
        ## Input Checks   
        if nBMU < 2:
            raise ValueError("nBMU must be >= 2")
        
        ## Assign  
        self.nBMU = int(nBMU)
        """
        Number of BMUs calculated during recall. 

        Set at class instantiation, must be :math:`\geq 2`.
        """
        self.nAnnoyTrees = int(nAnnoyTrees)
        """
        Number of trees used for approximate nearest neighbor search by Annoy. 
        
        Set at class instantiation.
        """
        self.__cppvqr = cppnvq.VQRecall(nBMU = self.nBMU, nAnnoyTrees = self.nAnnoyTrees)
        self.__flag_recall = False



    #### Recall Method 
    def Recall(self, X, W, XL = None, BMU_only = False, n_jobs = None):
        """
        Performs recall of data through the Vector Quantizer. 

        Parameters
        ----------
        X : `numpy.ndarray, dtype=float64, order=C` 
          The data matrix to be recalled, of dimension ``N x d`` (assumes data vectors in rows). 
          This array must have `dtype = 'float64'` and be arranged in C-contiguous order; 
          it will be cast as such if either are violated. 
        W : `numpy.ndarray, dtype=float64, order=C`
          The prototype matrix used for recall, of dimension ``M x d`` (assumed prototype vectors in rows). 
          This array must be arranged in C-contiguous order, and will be cast as such if given with Fortran ordering. 
        XL : `array-like, dtype=int` 
          Optional (default=``None``) vector of labels associated with each data vector. 
          If given, quantities related to the labeled data mapping will be computed. 
          Labels must be integer-encoded, which can be achieved via, e.g., ``numpy.unique(return_inverse=True)``.
        BMU_only : bool 
          Optional flag (default=``False``) indicating whether all recall products should be computed. 
          Set to ``True`` to compute only the BMU and Quantization Error (saves a small amount of time). 
        n_jobs : int or ``None``
          Optional (default = ``None``) number of OMP threads used for Recall. 
          If supplied as a positive integer, the system environmennt variable ``OMP_NUM_THREADS`` will be set to ``n_jobs``. 

        Returns
        -------
        None, internal containers are populated during call. 
        """

        ## 0. Input Checks 
        # Cast X & W, if necessary 
        if not X.flags['C_CONTIGUOUS'] or not X.dtype=='float64': 
            X = X.astype('float64', order='C')
            warnings.warn("X cast to C-ordered float64")
        if not W.flags['C_CONTIGUOUS'] or not W.dtype=='float64': 
            W = W.astype('float64', order='C')
            warnings.warn("X cast to C-ordered float64")
        # Save dimensions 
        self.N = X.shape[0]
        """
        Number of data vectors which were recalled. 
        
        Set during each call to ``Recall`` method. 
        """
        self.d = X.shape[1]
        """
        Size (dimension) of data & prototype vectors. 
        
        Set during each call to ``Recall`` method. 
        """
        self.M = W.shape[0]
        """
        Number of VQ prototypes used for recall. 
        
        Set during each call to ``Recall`` method. 
        """
        if W.shape[1] != self.d: raise ValueError("ncol(X) != ncol(W)")
        # Ensure labels are integers 
        if XL is not None and XL.dtype != 'int': raise ValueError("XL must be of type 'int'")
        # Set OMP threads, if given 
        if n_jobs is not None: os.environ["OMP_NUM_THREADS"] = str(int(n_jobs))


        ## 1. Perform Recall, using internal c++ class 
        if XL is None:
            self.__cppvqr.Recall(X=X, W=W, BMU_only=BMU_only)
        else:
            self.__cppvqr.Recall(X=X, W=W, XL=XL, BMU_only=BMU_only)
        self.__flag_recall = True


    ## Return BMUs 
    def BMU(self):
        """
        BMUs of each data vector.

        Returns
        -------
        `numpy.ndarray, dtype = int, shape = (N, nBMU)`
          Element `(i,j)` contains the `j-th` BMU of the `i-th` data vector.
          BMUs are identified by their row index in the prototype matrix `W`. 
        """
        if not self.__flag_recall: raise RuntimeError("Call self.Recall before accessing BMU")
        return np.array(self.__cppvqr.BMU, dtype='int').reshape((self.N, self.nBMU), order='F')
    
    ## Return QEs 
    def QE(self):
        """
        Quantization Error of each data vector.

        Returns
        -------
        `numpy.ndarray, dtype = float64, shape = (N, nBMU)`
          Element `(i,j)` contains the Quantization Error made when quantizing data vector `i` by it's `j-th` BMU. 
        """
        if not self.__flag_recall: raise RuntimeError("Call self.Recall before accessing QE")
        return np.array(self.__cppvqr.QE, dtype='float64').reshape((self.N, self.nBMU), order='F')


    ## Return Receptive Field
    def RF(self):
        """
        Receptive Fields of each prototype.

        Returns
        -------
        `list (length M) of lists`
          Element `i` contains the indices of data vectors for whom prototype `i` is BMU. 
        """
        return self.__cppvqr.RF
    

    ## Return Receptive Field
    def RF_Size(self):
        """
        Size of each prototype's Receptive Field. 

        Returns
        -------
        `numpy.ndarray, dtype = int, shape = (M,)`
          Element `i` contains the number of data vectors for whom prototype `i` is BMU. 
        """
        return np.array(self.__cppvqr.RF_Size, dtype='int').reshape(self.M)
    

    ## Return CADJ 
    def CADJ(self):
        """
        The CADJ matrix.

        Returns
        -------
        `scipy.sparse._arrays.coo_array, dtype=int, shape=(M,M)`
          Element `(i,j)` contains the number of data vectors for whom prototype `i` is BMU1 (first-ranking), 
          and prototype `j` is BMU2 (second-ranking). 
        """
        return coo_array((self.__cppvqr.CADJ, (self.__cppvqr.CADJi, self.__cppvqr.CADJj)), shape=(self.M,self.M))
    

    ## Return RFL_Dist 
    def RFL_Dist(self):
        """
        The Receptive Field Label Distribution

        Returns
        -------
        `list (length M)` of `dicts`
          Element `i` of the list contains a dictionary whose keys are the unique set of labels supplied via `XL`, 
          and whose values correspond to the counts of data with those labels in the Receptive Field of prototype `i`. 
        """
        return self.__cppvqr.RFL_Dist



    ## Return RFL
    def RFL(self):
        """
        The Winning Receptive Field Label 

        Returns
        -------
        `numpy.ndarray, dtype=int, shape=(M,)`
          Element `i` contains the winning plurality label of the data in the Receptive Field of prototype `i`. 
          This corresponds to the key with the largest associated value in `RFL_Dist[i]`. 
        """
        return np.array(self.__cppvqr.RFL)
    

    ## Return RFL_Purity
    def RFL_Purity(self):
        """
        The Receptive Field Label Purity 

        Returns
        -------
        `numpy.ndarray, dtype=float64, shape=(M,)`
          Element `i` contains the Purity of the Receptive Field of prototype `i`. 
          Purity is defined as 1 minus the `Hellinger Distance <https://en.wikipedia.org/wiki/Hellinger_distance>`_ between the ideal and observed label distributions of Receptive Field `i`. 
          The ideal distribution posits all data in `RF[i]` have the same label. 
        """
        return np.array(self.__cppvqr.RFL_Purity)
    

    ## Return RFL_Purity_UOA 
    def RFL_Purity_UOA(self):
        """
        The Unweighted Average of Receptive Field Purities 

        Returns
        -------
        `float`
          Unweighted average of ``RFL_Purity``
        """
        return self.__cppvqr.RFL_Purity_UOA
    

    ## Return RFL_Purity_WOA 
    def RFL_Purity_WOA(self):
        """
        The Weighted Average of Receptive Field Purities 

        Returns
        -------
        `float`
          Average of ``RFL_Purity``, using ``RF_Size`` as weights. 
        """
        return self.__cppvqr.RFL_Purity_WOA





        







            
        

       



