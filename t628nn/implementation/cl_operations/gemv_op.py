import numpy as np
import pyopencl as cl
from sanity_check import check_equal
from cl_util.tools import zero_buffer, get_cl, create_buffer


code = """
__kernel void gemv_vectorized(__global const float4* matrix, __global const float4* input,  __global float* output, __global const float* biases){
    size_t index = get_global_id(0);
    
    float cumulative = 0.0f;
    
    for(size_t i = 0; i < K; i++){
        cumulative += dot(
            matrix[K*index + i], input[i]
        );
    }
    cumulative += biases[index];
    output[index] = cumulative > 0 ? cumulative : 0;
    
}
"""


class GEMVOp:
    def __init__(self, ctx, input_dimensionality, output_dimensionality):
        assert input_dimensionality%4 == 0
        self._params = input_dimensionality, output_dimensionality

        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build(
            ["-D", "K=%s"%(input_dimensionality//4)]
        )

    def get_output_size(self):
        input_dimensionality, output_dimensionality = self._params
        return output_dimensionality

    def get_output_shape(self):
        input_dimensionality, output_dimensionality = self._params
        return output_dimensionality, 1

    def __call__(self, q, X, W, biases, Y=None):
        input_dimensionality, output_dimensionality = self._params
        ynp = None
        if Y is None:
            ynp, Y = zero_buffer(self._ctx, self.get_output_shape())

        kernel = self._prog.gemv_vectorized
        kernel.set_args(W, X, Y, biases)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (output_dimensionality, ), None)

        return evt, ynp, Y


def test_gemv_op():
    dev, ctx, q = get_cl()
    xnp, xcl = create_buffer(ctx, (500, 1))
    wnp, wcl = create_buffer(ctx, (256, 500))
    bnp, bcl = create_buffer(ctx, (256, 1))

    expected = np.dot(wnp, xnp) + bnp
    expected *= expected > 0

    op = GEMVOp(ctx, 500, 256)
    evt, ynp, ycl = op(q, xcl, wcl, bcl)
    cl.wait_for_events([evt])
    cl.wait_for_events([cl.enqueue_copy(q, ynp, ycl)])
    assert check_equal(expected, ynp)
