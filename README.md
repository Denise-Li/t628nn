Optimising Convolutional Neural Networks





Setting up the Environment


To run the code, you need an environment that satisfies the following conditions:

The main public libraries include pyopencl, numpy and pytest which are requeired for access GPU and performing operations. 
They can be installed via pip install.
1. Operate system: Linux.
2. Platform anaconda.
3. file type: .py and .cpp.
4. programming language: python3.




Testing


The code for the test time has been uploaded to convolutionallayer.py(it is in implementation file) 
also can be run under vgg16 file(also in implementation file).

The two lines of code that are most useful for testing time are:

1.with Timer("the name which you want to split"):

2.cl.wait_for_events([evt/ze])

Write these two lines of code on each layer for time testing and then record, statistics these will get 
the time table with each layer.
