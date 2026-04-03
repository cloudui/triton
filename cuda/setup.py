from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension(
            "cuda_kernels",
            ["bindings.cu", "softmax.cu", "softmax_triton.cu", "rmsnorm.cu"],
            extra_compile_args={"nvcc": ["-ccbin", "/usr/bin/gcc"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
