from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

class custom_build_ext(build_ext):
    def build_extensions(self):
        # Override the compiler executables. Importantly, this
        # removes the "default" compiler flags that would
        # otherwise get passed on to to the compiler, i.e.,
        # distutils.sysconfig.get_var("CFLAGS").
        self.compiler.set_executable("compiler_so", "/usr/local/opt/llvm/bin/clang++")
        self.compiler.set_executable("compiler_cxx", "/usr/local/opt/llvm/bin/clang++")
        self.compiler.set_executable("linker_so", "/usr/local/opt/llvm/bin/clang++")
        build_ext.build_extensions(self)


ext_module = Pybind11Extension(
    'NeuralVQ',
    #[str(fname) for fname in Path('src').glob('*.cpp')],
    ['src/bindings.cpp'],
    include_dirs=['include'],
    #library_dirs=['/usr/local/opt/libomp/lib'],
    extra_compile_args=['-std=c++11', '-fopenmp', '-fPIC', '-Wall', '-g', '-O3', "-arch", "x86_64"],
    extra_link_args=["-fopenmp", '-dynamiclib', '-undefined', 'dynamic_lookup']
)


 
setup(
    name='NeuralVQ',
    version=0.1,
    author='Josh Taylor',
    author_email='joshtaylor@utexas.edu',
    #description='VQ Recall based on ANNoy Library',
    ext_modules=[ext_module],
    #cmdclass={"build_ext": build_ext},
    cmdclass={"build_ext": custom_build_ext}
)
