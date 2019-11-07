import numpy as np
import pyopencl as cl
from sanity_check import check_equal, to_morton_right, to_morton_left
from cl_util.tools import zero_buffer, get_cl, definitions2, create_buffer, Timer


code = """
    __kernel void
        matmul (
            global float4 * const A ,
            global float4 * const B ,
            global float * C, global float* biases)
        {
        uint row = get_global_id (0);
        uint col = get_global_id (1);
        float4 ab = (float4) 0.0;
        for (uint i = 0; i < HK; i += 2)
        {
            float4 a0 = A[ row * HK + i ];
            float4 a1 = A[ row * HK + i + 1];
            float4 b0 = B[ col * HK + i ];
            float4 b1 = B[ col * HK + i + 1];
            ab += (float4)( dot (a0 , b0 ), dot (a1 , b0 ),
            dot (a0 , b1 ), dot (a1 , b1 ));
        }
        ab += (float4)(biases[row*2], biases[row*2+1], biases[row*2], biases[row*2+1]);
        # ifdef RELU
            ab = max(ab, (float4)(0.0f, 0.0f, 0.0f, 0.0f));
        # endif
        uint ix = N*row*2 + col*2;
        C[ ix ] = ab.s0;
        C[ ix + 1] = ab.s2;
        C[ ix + N] = ab.s1;
        C[ ix + N + 1] = ab.s3;
    }
"""


class MortonGEMM:
    def __init__(self, ctx, m, k, n):
        assert m % 8 == 0
        assert k % 8 == 0
        assert n % 32 == 0
        self._m, self._k, self._n = m, k, n
        self._ctx = ctx
        self._prog = cl.Program(ctx, code).build(definitions2(m=m, n=n, k=k, hk=k//2, qk=k//4, relu=1))

    def get_output_size(self):
        return self._m * self._n

    def __call__(self, q, a, b, biases, c=None):
        cnp = None
        if c is None:
            cnp, c = zero_buffer(self._ctx, (self._m, self._n))
        kernel = self._prog.matmul
        kernel.set_args(a, b, c, biases)
        evt = cl.enqueue_nd_range_kernel(q, kernel, (self._m//2, self._n//2), (4, 16))

        return evt, cnp, c


def test_mortongemm():
    M, K, N = 512, 1024, 256

    dev, ctx, q = get_cl()

    anp, acl = create_buffer(ctx, (M, K))
    bnp, bcl = create_buffer(ctx, (K, N))
    onp, ocl = create_buffer(ctx, (M, 1))

    amorton = to_morton_left(anp).copy()
    bmorton = to_morton_right(bnp).copy()

    cl.enqueue_copy(q, acl, amorton)
    cl.enqueue_copy(q, bcl, bmorton)
    q.flush()

    op = MortonGEMM(ctx, M, K, N)
    with Timer("Morton GEMM"):
        evt, cnp, ccl = op(q, acl, bcl, ocl)
        cl.wait_for_events([evt])

    cl.enqueue_copy(q, cnp, ccl)
    q.flush()

    sanity = np.dot(anp, bnp) + onp
    sanity *= sanity > 0

    assert check_equal(cnp, sanity, epsilon=0.0003)
