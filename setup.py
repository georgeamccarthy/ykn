
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ykn",
    version="1.0",
    author="George A. McCarthy",
    author_email="georgeamccarthy@outlook.com",
    description="Computes the Compton Y-parameter including Klein-Nishina effects.",
    long_description = long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/georgeamccarthy/ykn",
    packages=setuptools.find_packages(),
    license="Apache2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
