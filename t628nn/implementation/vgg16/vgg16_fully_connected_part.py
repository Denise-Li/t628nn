from cl_util.tools import get_cl, create_buffer, Timer
from vgg16.vgg16_layers import VGG16FullyConnectedLayer


"""
Execution of the fully-connected layers of VGG16 using our custom GEMV implementation. Running this file should
reproduce the results seen Figure 7.5 and Table 7.4 of the Masters Project pdf, as well as in Figure 8 of
the published paper.
"""


dev, ctx, q = get_cl()

configs = [("fc1",), ("fc2",), ("fc3",)]
layers = [VGG16FullyConnectedLayer(ctx, 7*7*512, 4096), VGG16FullyConnectedLayer(ctx, 4096, 4096), VGG16FullyConnectedLayer(ctx, 4096, 1000)]

xnp, xcl = create_buffer(ctx, (512*7*7, 1))

X = xcl

for c, l in zip(configs, layers):
    with Timer(c[0]):
        X, _ = l(q, X)
