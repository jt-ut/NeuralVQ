Usage
=====

.. code-block:: python
    
    import NeuralVQ as nvq 
    
    # Load example data + labels 
    X, XL = nvq.load_worms2(N = 5000, seed=123)

    # Initialize 100 random prototypes 
    W0 = nvq.randinit_W0_NG(data = X, M=100, method='rand_sample', seed=123)

    # Perform a recall 
    rec = nvq.VQRecall(nBMU=2)
    rec.Recall(X=X, W=W0, XL=XL)

    # Inspect some of the recall products 
    print(rec.BMU())
    print(rec.QE())
    print(rec.RFL())







