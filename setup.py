import os
import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    ext_modules=[
        CppExtension(
            name="torchac._backend",
            sources=[
                "torchac/csrc/backend.cpp"
            ],
            extra_compile_args={
                "cxx": ["/std:c++20", "/Zc:__cplusplus"] if sys.platform == "win32" else ["-std=c++20"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)
