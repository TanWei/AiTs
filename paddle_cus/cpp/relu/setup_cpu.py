from paddle.utils.cpp_extension import CppExtension, setup

setup(
    name='custom_setup_ops',
    ext_modules=CppExtension(
        sources=['relu_cpu_fp32.cc']
    )
)

