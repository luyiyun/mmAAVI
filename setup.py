from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "_cy_utils",
        ["./src/mmAAVI/extension/*.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]
setup(
    name="cython extention utils",
    ext_modules=cythonize(extensions, annotate=True),
)
