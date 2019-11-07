import pyopencl as cl
from cl_operations import GEMVOp


class FullyConnectedLayer:
    def __init__(self, ctx, input_dims, output_dims):
        self._params = input_dims, output_dims
        assert input_dims%4 == 0

        self._gemv = GEMVOp(ctx, input_dims, output_dims)

    def __call__(self, q, buff1, wcl, bcl, buff2):

        evt1, _, _ = self._gemv(q, buff1, wcl, bcl, buff2)
        cl.wait_for_events([evt1])

        return buff2, buff1

    def get_output_size(self):
        return self._gemv.get_output_size()
