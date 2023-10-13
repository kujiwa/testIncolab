from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import  os
#python setup.py clean --all install --user
setup(
    name='CudaDemo',
    packages=find_packages(),
    version='0.1.0',
    author='Yuxue Yang',
    ext_modules=[
        CUDAExtension(
            'sumMatrix', # operator name
            ['cuda_ext_1.cu',]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)