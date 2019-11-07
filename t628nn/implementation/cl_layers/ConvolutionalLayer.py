import numpy as np
import pyopencl as cl
from sanity_check import convolve, my_round, check_equal, insert, to_morton_left
from cl_operations import *
from cl_util.tools import get_cl, create_buffer，Timer


class ConvolutionalLayer:
    def __init__(self, ctx, xshape, fshape, strides, preserve=False):
        xn, xd, xh, xw = xshape
        yd, fs, fh, fw = fshape
        assert fs == xd

        assert fw*fh*xd % 8 == 0
        assert yd % 8 == 0

        sh, sw = strides
        if preserve:
            assert sh == 1
            assert sw == 1
            assert fh % 2 == 1
            assert fw % 2 == 1
            xpad = xhp, xhpt, xwp, xwpt = fh-1, fw//2, fw-1, fw//2
        else:
            xpad = xhp, xhpt, xwp, xwpt = 0, 0, 0, 0


        self._addPadding = PaddingOp(ctx, xshape, (0, 0, 0, 0), xpad)
        yh, yw = (xh + xhp - fh) // sh + 1, (xw +xwp - fw) // sw + 1

        self._params = xn, xd, xh, xw, yh, yw, xhp, xhpt, xwp, xwpt, fh, fw, sh, sw

        xn, xd, xh, xw = self._addPadding.get_output_shape()
        xdr = my_round(xd, 8)
        self._im2col = Im2ColOp(ctx, (xn, xdr, xh, xw), fshape[2:], strides, "MORTON", True)

        (ih, iw) = self._im2col.get_output_shape()
        self._morton_gemm = MortonGEMM(ctx, yd, ih, iw)

        iwp = iw - yh*yw
        self._padding_remover = PaddingOp(ctx, (1, 1, ih, iw-iwp), (0, 0, iwp, 0), (0, 0, 0, 0))

    def __call__(self, q, zeroer, buff1, fcl, bcl, buff2):
        xn, xd, xh, xw, yh, yw, xhp, xhpt, xwp, xwpt, fh, fw, sh, sw = self._params
        evts = list()

        ze1, _, _ = zeroer(q, buff2)
        evts.append(ze1)

        evt1, _, _ = self._addPadding(q, buff1, buff2)
        evts.append(evt1)

        if xn*xd*xh*xw > self._im2col.get_output_size():
            ze2, _, _ = zeroer(q, buff1)
            evts.append(ze2)

        evt2, _, _ = self._im2col(q, buff2, buff1)
        evts.append(evt2)

        if self._addPadding.get_output_size() > self._morton_gemm.get_output_size():
            ze3, _, _ = zeroer(q, buff2)
            evts.append(ze3)

        evt3, _, _ = self._morton_gemm(q, fcl, buff1, bcl, buff2)
        evts.append(evt3)

        ze4, _, _ = zeroer(q, buff1)
        evts.append(ze4)

        with Timer("padding2"):
            evt4, _, _ = self._padding_remover(q, buff2, buff1)
            evts.append(evt4)
            cl.wait_for_events([evt4])

        cl.wait_for_events(evts)
        return buff1, buff2

    def get_output_size(self):
        return max(l.get_output_size() for l in [self._addPadding, self._im2col, self._morton_gemm, self._padding_remover])


def test_convolution():

    dev, ctx, q = get_cl()

    xnp, xcl = create_buffer(ctx, (1, 16, 28, 30))
    fnp, fcl = create_buffer(ctx, (32, 16, 3, 3))
    bnp, bcl = create_buffer(ctx, (1, 32, 1, 1))

    fnpmorton = to_morton_left(fnp.reshape((32,3*3*16))).copy()
    cl.enqueue_copy(q, fcl, fnpmorton)
    q.flush()

    buff1np, buff1cl = create_buffer(ctx, (10000000,))
    buff2np, buff2cl = create_buffer(ctx, (10000000,))

    buff1np = insert(xnp, buff1np)
    cl.enqueue_copy(q, buff1cl, buff1np)
    q.flush()

    layer = ConvolutionalLayer(ctx, (1, 16, 28, 30), fnp.shape, (1, 1), True)
    zeroer = ZeroOp(ctx, 10000000)

    rescl, _ = layer(q, zeroer, buff1cl, fcl, bcl, buff2cl)

    assert 124416 == layer.get_output_size()

    padded = np.zeros((1, 16, 30, 32))
    padded[:, :, 1:-1, 1:-1] = xnp[:, :16, :28, :30]

    expected = convolve(padded, fnp, (1, 1)) + bnp
    expected *= expected > 0
    expected = expected.reshape((1, 32, 28, 30))

    cl.enqueue_copy(q, buff1np, rescl)
    q.flush()
    actual = buff1np[:1*32*30*28].reshape((1, 32, 28, 30))

    assert check_equal(expected, actual)
