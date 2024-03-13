from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='cuda_spike_tools',
    ext_modules=[cpp_extension.CUDAExtension(
        name='cuda_spike_tools',
        sources=['cuda_spike_tools.cpp', 'get_output_spike_times.cu', 'evaluate_derivatives.cu'],
        # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-Wall']}
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
