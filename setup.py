from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages
import platform 

## Custom C++ build function taken from: 
# https://shwina.github.io/custom-compiler-linker-extensions/
# class custom_build_ext(build_ext):
#     def build_extensions(self):
#         # Override the compiler executables. Importantly, this
#         # removes the "default" compiler flags that would
#         # otherwise get passed on to to the compiler, i.e.,
#         # distutils.sysconfig.get_var("CFLAGS").
#         self.compiler.set_executable("compiler_so", "/usr/local/opt/llvm/bin/clang++")
#         self.compiler.set_executable("compiler_cxx", "/usr/local/opt/llvm/bin/clang++")
#         self.compiler.set_executable("linker_so", "/usr/local/opt/llvm/bin/clang++")
#         build_ext.build_extensions(self)


## Define C++ module as a PyBind11 Extension 
# We need to specify extra build flags due to C++ source requirements
# extra_compile_args = ['-std=c++11', '-fopenmp', '-fPIC', '-Wall', '-g', '-O3', "-arch", "x86_64"]
# Depending on the user's system, need extra link flags 
if platform.system()=='Darwin':
    extra_compile_args = ['-std=c++11', '-Xpreprocessor', '-fopenmp']
    #extra_link_args = ["-fopenmp", '-dynamiclib', '-undefined', 'dynamic_lookup']
    extra_link_args = ['-lomp']
elif platform.system()=='Linux':
    extra_compile_args = ['-std=c++11', '-fopenmp']
    extra_link_args = ['-fopenmp']


# C++ build directive 
ext_module = Pybind11Extension(
    # Name of exposed module 
    'NeuralVQ._cpp_NVQ',
    # List C++ source files containing PyBind11 bindings, 
    # either all globbed, 
    #[str(fname) for fname in Path('src').glob('*.cpp')],
    # or specific ones 
    ['src/NeuralVQ/bindings.cpp'], # Source files  
    # Location of required headers for C++ source code 
    #include_dirs=['NeuralVQ/_cpp_NVQ/include'],
    include_dirs=['/usr/local/Cellar/libomp'],
    # Build flags 
    extra_compile_args = extra_compile_args,
    extra_link_args = extra_link_args
)


# Python build directive 
setup(
    # Admin info 
    name='NeuralVQ',
    version=0.1,
    author='Josh Taylor',
    author_email='joshtaylor@utexas.edu',
    description='Neural Vector Quantizater Learning & Recall based on ANNoy Library',
    # List Python modules to be included in package 
    py_modules = ['_py_NVQ'], 
    # List external (C++) modules to be included, with their build instructions
    ext_modules=[ext_module],
    #cmdclass={"build_ext": custom_build_ext}, #Default build instruction via: cmdclass={"build_ext": build_ext},
    
    #packages = find_packages(), 
    #packages=['NeuralVQ'], 
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={'': ['data/*.csv']},
)

