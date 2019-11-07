from cl_util.tools import create_buffer
from sanity_check import to_morton_left
from cl_layers import ConvolutionalLayer, MaxPoolLayer, FullyConnectedLayer


"""
This file contains abstractions over the OpenCL layer implementations in the cl_layers directory. It creates layers,
prepopulates their internal parameters with small random values and creates the corresponding layer form the cl_layers
directory. This makes running simulations easier.
"""


class VGG16ConvolutionalLayer:
    def __init__(self, ctx, shape, out_shape):
        xn, xd, xh, xw = shape
        yn, yd, yh, yw = out_shape
        assert xn == yn and xh == yh and xw == yw

        self._layer = ConvolutionalLayer(ctx, shape, (yd, xd, 3, 3), (1, 1), True)

        fnp, fcl = create_buffer(ctx, (yd, xd, 3, 3))
        fnp = to_morton_left(fnp.reshape(yd, xd*3*3)).copy()
        self._filters = fnp, fcl

        self._biases = create_buffer(ctx, (1, yd, 1, 1))

    def __call__(self, q, zeroer, buff1, buff2):
        return self._layer(q, zeroer, buff1, self._filters[1], self._biases[1], buff2)

    def get_output_size(self):
        return self._layer.get_output_size()


class VGG16MaxPoolingLayer:
    def __init__(self, ctx, shape, out_shape):
        xn, xd, xh, xw = shape
        yn, yd, yh, yw = out_shape
        assert xn == yn and xd == yd and xh == 2*yh and xw == 2*yw
        self._layer = MaxPoolLayer(ctx, shape, (2, 2), (2, 2))

    def __call__(self, q, zeroer, buff1, buff2):
        return self._layer(q, zeroer, buff1, buff2)

    def get_output_size(self):
        return self._layer.get_output_size()


class VGG16FullyConnectedLayer:
    def __init__(self, ctx, input_dims, output_dims):

        wnp, wcl = create_buffer(ctx, (output_dims, input_dims))
        bnp, bcl = create_buffer(ctx, (output_dims, 1))

        self._layer = FullyConnectedLayer(ctx, input_dims, output_dims)

        ynp, ycl = create_buffer(ctx, (self._layer.get_output_size(), 1))

        self._params = wnp, wcl, bnp, bcl, ynp, ycl

    def __call__(self, q, X):
        wnp, wcl, bnp, bcl, ynp, ycl = self._params

        return self._layer(q, X, wcl, bcl, ycl)

    def get_output_size(self):
        return self._layer.get_output_size()
