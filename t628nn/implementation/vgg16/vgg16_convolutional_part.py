import numpy as np
import pyopencl as cl
from sanity_check import insert
from cl_util.tools import get_cl, create_buffer, Timer
from cl_operations import ZeroOp
from vgg16.vgg16_layers import VGG16ConvolutionalLayer, VGG16MaxPoolingLayer


"""
Morton GEMM & im2col convolution execution of convolutional layers of VGG16. Running this file should reproduce
the results seen Figure 7.4 and Table 7.3 of the Masters Project pdf, as well as in Figure 7 and Figure 8 of
the published paper.
"""


configs = [
    ("conv1-1", (1, 8, 224, 224), (1, 64, 224, 224)),
    ("conv1-2", (1, 64, 224, 224), (1, 64, 224, 224)),
    ("mp1", (1, 64, 224, 224), (1, 64, 112, 112)),
    ("conv2-1", (1, 64, 112, 112), (1, 128, 112, 112)),
    ("conv2-2", (1, 128, 112, 112), (1, 128, 112, 112)),
    ("mp2", (1, 128, 112, 112), (1, 128, 56, 56)),
    ("conv3-1", (1, 128, 56, 56), (1, 256, 56, 56)),
    ("conv3-2", (1, 256, 56, 56), (1, 256, 56, 56)),
    ("conv3-3", (1, 256, 56, 56), (1, 256, 56, 56)),
    ("mp3", (1, 256, 56, 56), (1, 256, 28, 28)),
    ("conv4-1", (1, 256, 28, 28), (1, 512, 28, 28)),
    ("conv4-2", (1, 512, 28, 28), (1, 512, 28, 28)),
    ("conv4-3", (1, 512, 28, 28), (1, 512, 28, 28)),
    ("mp4", (1, 512, 28, 28), (1, 512, 14, 14)),
    ("conv5-1", (1, 512, 14, 14), (1, 512, 14, 14)),
    ("conv5-2", (1, 512, 14, 14), (1, 512, 14, 14)),
    ("conv5-3", (1, 512, 14, 14), (1, 512, 14, 14)),
    ("mp5", (1, 512, 14, 14), (1, 512, 7, 7))
]

dev, ctx, q = get_cl()

layers = list()

for name, inputs, outputs in configs:
    if "conv" in name:
        layers.append(VGG16ConvolutionalLayer(ctx, inputs, outputs))
    else:
        layers.append(VGG16MaxPoolingLayer(ctx, inputs, outputs))

for l in layers:
    print(l.get_output_size())

maximum = max(l.get_output_size() for l in layers)
zeroer = ZeroOp(ctx, maximum)

print(maximum)

buff1np, buff1 = create_buffer(ctx, (maximum,))
buff2np, buff2 = create_buffer(ctx, (maximum,))

X = np.random.random_sample(configs[0][1]).astype(np.float32)

res = insert(X, buff1np)
cl.wait_for_events([cl.enqueue_copy(q, buff1, res)])

with Timer("Full"):
    for l, c in zip(layers, configs):
        with Timer(c[0]):
            buff1, buff2 = l(q, zeroer, buff1, buff2)
