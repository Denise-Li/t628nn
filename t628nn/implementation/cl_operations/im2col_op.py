import pyopencl as cl
from sanity_check import check_equal, im2col, from_morton_right, my_round
from cl_util.tools import zero_buffer, get_cl, create_buffer, definitions


code = """
    __kernel void im2col(__global const float* inputs, __global float* outputs){
    size_t Yx = get_global_id(0);
    size_t Yy = get_global_id(1);
    size_t Xz = get_global_id(2);
    for(size_t Fy = 0; Fy < FH; Fy++){
        for(size_t Fx = 0; Fx < FW; Fx++){
            size_t Ix = Yy * YW + Yx;
            size_t Iy = Xz * FW * FH + Fy * FW + Fx;

            size_t z = XW * XH * Xz;
            size_t y = XW * (Yy * SH + Fy);
            size_t x = (Yx * SW + Fx);
            #ifdef MORTON
                uint ix = (Ix/2)*2*IH + (Iy/4)*4*2 + ((Ix & 0b1)*4) + ( Iy & 0b11 );
                outputs[ix] = inputs[z + y + x];
            #endif
            #ifdef NCHW
                outputs[Iy*IW + Ix] = inputs[z + y + x];
            #endif
        }
    }
}
"""


class Im2ColOp:
    def __init__(self, ctx, x_shape, filters, strides, layout="NCHW", buffered=False):
        Xn, Xd, Xh, Xw = x_shape
        Fh, Fw = filters
        Sh, Sw = strides
        assert Xn == 1
        Yh, Yw = (Xh - Fh) // Sh + 1, (Xw - Fw) // Sw + 1
        Ih, Iw = Fw * Fh * Xd, Yw * Yh

        if layout == "MORTON":
            assert Ih % 8 == 0
            if buffered:
                Iw = my_round(Iw, 32)
            else:
                assert Iw % 32 == 0

        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build(
            list(definitions("XD XH XW FH FW SH SW YH YW IH IW".split(' '), (Xd, Xh, Xw, Fh, Fw, Sh, Sw, Yh, Yw, Ih, Iw))) + ["-D", "%s=1"%layout]
        )

        self._params = Xn, Xd, Xh, Xw, Fh, Fw, Sh, Sw, Yh, Yw, Ih, Iw

    def get_output_size(self):
        Xn, Xd, Xh, Xw, Fh, Fw, Sh, Sw, Yh, Yw, Ih, Iw = self._params
        return Ih*Iw

    def get_output_shape(self):
        Xn, Xd, Xh, Xw, Fh, Fw, Sh, Sw, Yh, Yw, Ih, Iw = self._params
        return Ih, Iw

    def __call__(self, q, X, I=None):
        Xn, Xd, Xh, Xw, Fh, Fw, Sh, Sw, Yh, Yw, Ih, Iw = self._params
        inp = None
        if I is None:
            inp, I = zero_buffer(self._ctx, (Ih, Iw))

        kernel = self._prog.im2col
        kernel.set_args(X, I)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (Yw, Yh, Xd), None)

        return evt, inp, I


def test_im2col_op():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (1, 64, 25, 20))
    expected = im2col(xnp, (3, 2), (2, 1))

    op = Im2ColOp(ctx, xnp.shape, (3, 2), (2, 1))
    evt, inp, icl = op(q, xcl)
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, inp, icl)])
    assert check_equal(expected, inp)

    tnp, tcl = create_buffer(ctx, (1, op.get_output_size()))
    evt, returned_np, tcl = op(q, xcl, tcl)
    assert returned_np is None
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, tnp, tcl)])
    assert check_equal(expected.flatten(), tnp.flatten())


def test_im2col_op_morton():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (1, 64, 33, 65))
    expected = im2col(xnp, (2, 2), (1, 1))

    op = Im2ColOp(ctx, xnp.shape, (2, 2), (1, 1), "MORTON")
    evt, inp, icl = op(q, xcl)
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, inp, icl)])
    assert check_equal(expected, from_morton_right(inp, expected.shape))
