import pyopencl as cl
from sanity_check import check_equal, max_pool
from cl_util.tools import zero_buffer, get_cl, definitions2, create_buffer


code = """
    #if defined(cl_khr_fp64)
    #  pragma OPENCL EXTENSION cl_khr_fp64: enable
    #elif defined(cl_amd_fp64)
    #  pragma OPENCL EXTENSION cl_amd_fp64: enable
    #else
    #  error double precision is not supported
    #endif
    __kernel void max_pool(__global const float* inputs, __global float* outputs){
        size_t row = get_global_id(0);         // Which column of the feature this kernel computes
        size_t col = get_global_id(1);         // Which row of the feature this kernel it computes
        size_t feature = get_global_id(2);

        float cumulative = inputs[XW*XH*feature + XW*row*SH + col*SW];
        for(uint y = 0; y < FH; y++){
            for(uint x = 0; x < FW; x++){
                float next = inputs[
                        XW*XH*feature  + // Start of feature
                        XW*(row*SH+y)  + // Start of feature row
                        (col*SW + x)     // Feature column
                    ];
                cumulative = next > cumulative ? next : cumulative;
            }
        }
        outputs[
            YW*YH*feature      + // Start of feature
            YW*row + // Start of feature row
            col
        ] = cumulative;
    }
"""


class MaxPoolOp:
    def __init__(self, ctx, shape, filters, strides):
        n, d, xh, xw = shape
        fh, fw = filters
        sh, sw = strides

        yh, yw = (xh - fh) // sh + 1, (xw - fw) // sw + 1

        assert n == 1

        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build(
            definitions2(n=n, d=d, xh=xh, xw=xw, yh=yh, yw=yw, sh=sh, sw=sw, fh=fh, fw=fw)
        )

        self._params = n, d, xh, xw, yh, yw, sh, sw, fh, fw

    def get_output_size(self):
        n, d, xh, xw, yh, yw, sh, sw, fh, fw = self._params
        return n*d*yh*yw

    def __call__(self, q, X, Y=None):
        n, d, xh, xw, yh, yw, sh, sw, fh, fw = self._params
        ynp = None
        if Y is None:
            ynp, Y = zero_buffer(self._ctx, (n, d, yh, yw))

        kernel = self._prog.max_pool
        kernel.set_args(X, Y)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (yh, yw, d), None)

        return evt, ynp, Y


def test_max_pooling_op():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (1, 32, 30, 30))

    expected = max_pool(xnp, (2, 3), (1, 2))

    op = MaxPoolOp(ctx, (1, 32, 30, 30), (2, 3), (1, 2))
    evt, ynp, ycl = op(q, xcl)
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, ynp, ycl)])
    print(ynp[0, 0, :6, :6])
    print(expected[0, 0, :6, :6])
    assert check_equal(expected, ynp)

