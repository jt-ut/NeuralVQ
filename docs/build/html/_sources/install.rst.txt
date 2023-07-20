Install
=======

To install on Stampede: 

.. code-block:: bash

    # Load an appropriate Python configuration. 
    # Requirements: a gcc compiler, and a python3 version that supports pybind11
    module load gcc/9.1.0
    module load python3 
    
    # Upgrade scipy & pybind11 
    python3 -m pip install scipy --user --upgrade
    python3 -m pip install pybind11 --upgrade --user

    # Clone repository from Git, enter it
    git clone https://github.com/jt-ut/NeuralVQ.git
    
    # Enter cloned directory, install with verbose output 
    cd NeuralVQ
    python3 -m pip install . -vvv --user


     


