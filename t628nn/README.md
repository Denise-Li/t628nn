# Optimising Convolutional Neural Networks Inference on Low-Powered GPUs
This repository contains the accompanying code to the Optimising Convolutional Neural Networks Inference on Low-Powered GPUs paper published in 2019. The repository is laid out in a way that allows simple adaptation of the code as well as one-click reproduction and verification of the results we published.

The published paper is a condensement of my Bachelors and Masters projects. As such, I have included both the works in this repository in case someone is interested in a more in-depth explanation of the work done. You can find the Bachelors project in the [BachelorsProject.pdf](BachelorsProject.pdf) file and the Masters project in the [MastersProject.pdf](MastersProject.pdf) file.

## Setting up the Environment
To run the code, you need an environment that satisfies the following conditions:

- Has Python 3 installed and available on the Path.

- Python 3 has *pyopencl*, *numpy* and *pytest* modules installed.

- A path to the compiled .cpp files from the [clblastsimulations](clblastsimulations) folder is available in the *BLAST_SIMULATOR_PATH* environment variable.

- The [implementation](implementation) folder is available on the PYTHONPATH.

## Reproducing the Results

To reproduce the CLBlast LeNet performance results from Figure 3 and Table 2 of the paper ([Masters project](MastersProject.pdf) Figure 5.5 and Table 5.3) run the [/implementation/simulator/lenet.py](implementation/simulator/lenet.py) file.

To reproduce the CLBlast VGG16 performance results from Figure 4 of the paper ([Masters project](MastersProject.pdf) Figure 5.7 and Table A1) run the [/implementation/simulator/vgg16.py](implementation/simulator/vgg16.py) file.

To reproduce the MortonGEMM performance results from Figure 7 and Figure 8 ([Masters project](MastersProject.pdf) Figure 7.5 and Table 7.3) run the [/implementation/vgg16/vgg16_convolutional_part.py](implementation/vgg16/vgg16_convolutional_part.py) and the [/implementation/vgg16/vgg16_fully_connected_part.py](implementation/vgg16/vgg16_fully_connected_part.py) files.

## Main Contribution

The main contribution of our work was the MortonGEMM based convolution. This can be found in the [ConvolutionalLayer.py](implementation/cl_layers/ConvolutionalLayer.py). If you are looking to extend this work or make use of it, this is probably the file you're looking for.

The MortonGEMM matrix multiplication itself (not convolution) can be found in the [morton_gemm_op.py](implementation/cl_operations/morton_gemm_op.py).

## Correctness and Testing

In order to make sure the published code is correct, the entire system was continuously tested during development. At the beginning of the project, all operations that we were going to implement in OpenCL were first implemented in Python with Numpy. Each of these Python implementations had a corresponding hand-written unit test. These implementations and unit tests can be found in the [sanity_ckeck.py](implementation/sanity_check.py) file.

To run tests against sanity_check.py, navigate to the [implementation](implementation/) directory and run `pytest sanity_check.py`.

All subsequently implemented OpenCL operations were tested against these Python-based implementations. For example, if you open up the [max_pool_op.py](implementation/cl_operations/max_pool_op.py) file, you can see that it implements Max-Pooling in OpenCL and then tests it against the Python implementation from sanity_check.py.

The neural network layers themselves (in [cl_layers](cl_layers/)) are composed of multiple OpenCL operations. The layers are tested against the operations, the operations are tested against the Nympy implementations, which are in turn tested by the hand-written unit tests.
