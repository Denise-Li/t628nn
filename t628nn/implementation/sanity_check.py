import numpy as np


"""
This file contains simple Numpy implementations, complementary to each operation we implement in OpenCL and execute
on-GPU. We use these implementations to verify the results produced by our GPU implementation. Each of these operations
has a corresponding hand-written unit test in this file, as to ensure that the Numpy implementations work.

Tests can be run by navigating to this file in a terminal and running `pytest sanity_check.py` .
"""


def check_equal(a, b, epsilon=0.00001):
    """
    Function checks whether two matrices are equal up to a certain degree of accuracy. Also checks that
    the shapes match.

    :param a: Numpy matrix
    :param b: Numpy matrix
    :param epsilon: Maximum allowed sum of squared differences
    :return: True if the matrices are equal, false otherwise
    """
    if a.shape != b.shape:
        print("Shape mismatch!")
        return False
    err = a-b
    err = err * err
    err = err.sum()
    if err > epsilon:
        print("Error of %s"%err)
        return False
    return True


def test_check_equal():
    """ Test of the check_equal function """
    assert check_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert check_equal(np.array([[1], [2], [3]]), np.array([[1], [2], [3]]))
    assert not check_equal(np.array([1, 2, 3]), np.array([[1], [2], [3]]))
    assert not check_equal(np.array([1, 2, 3]), np.array([1, 5, 3]))


def nchw_2_nhwc(mat):
    """
    Function rearranges a 4-dimensional numpy matrix from the (amount of inputs, channels, height, width) shape to
    the (amount of inputs, height, width, channels) shape.
    :param mat: NCHW-arranges matrix
    :return: HNWC-arranged matrix
    """
    return mat.swapaxes(1, 3).swapaxes(1, 2)


def nhwc_2_nchw(mat):
    """
    Function rearranges a 4-dimensional numpy matrix from the (amount of inputs, height, width, channels) layout to
    the (amount of inputs, channels, height, width) layout.
    :param mat: HNWC-layout matrix
    :return: NCHW-layout matrix
    """
    return mat.swapaxes(1, 2).swapaxes(1, 3)


def test_nchw_2_nhwc():
    """ Test of the nchw_2_nhwc function """
    a = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    b = np.array([[[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]]])
    assert check_equal(b, nchw_2_nhwc(a))


def test_nhwc_2_nchw():
    """ Test of the nhwc_2_nchw function """
    a = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    b = np.array([[[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]]])
    assert check_equal(a, nhwc_2_nchw(b))


def im2col(X, filters, strides):
    """
    Function implements the im2col operation.
    :param X: Input matrix in NCHW layout.
    :param filters: Tuple describing the shape of the convolution kernel, eg. (3, 3) for VGG16 convolution.
    :param strides: Tuple describing the strides the filter will take along the height and width. Usually (1, 1).
    :return: Columnized matrix (look up the im2col transformation)
    """
    fh, fw = filters
    sh, sw = strides
    xn, xc, xh, xw = X.shape
    assert xn == 1
    yn, yc, yh, yw = xn, xc, (xh - fh)//sh + 1, (xw - fw)//sw + 1
    I = np.zeros((fw*fh*yc, yw*yh))
    i_col = 0
    for y in range(yh):
        for x in range(yw):
            I[:, i_col:i_col+1] = X[0, :, y*sh:y*sh+fh, x*sw:x*sw+fw].reshape((-1, 1))
            i_col += 1

    return I


def test_im2col():
    """ Test of the im2col function """
    image = np.array([[
        [[1,  2,  3,  4, 5], [6,  7,  8,  9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    ]])

    cols = np.array([
        [1, 2, 3, 6, 7, 8, 16, 17, 18, 21, 22, 23],
        [3, 4, 5, 8, 9, 10, 18, 19, 20, 23, 24, 25],
        [6, 7, 8, 11, 12, 13, 21, 22, 23, 26, 27, 28],
        [8, 9, 10, 13, 14, 15, 23, 24, 25, 28, 29, 30]
    ]).T

    assert check_equal(cols, im2col(image, (2, 3), (1, 2)))


def convolve(X, filters, strides):
    """
    Function implements convolution on multiple 3-dimensional inputs. It makes use of the im2col function.
    :param X: Matrix in NCHW layout
    :param filters: 4-dimensional filter matrix of shape (output channels, input channels, filder height, filter width)
    :param strides: Stride along height and width, (1, 1) for VGG16
    :return: Result of convolution
    """
    sh, sw = strides
    xn, xc, xh, xw = X.shape
    yc, xd, fh, fw = filters.shape
    assert xd == xc
    yh, yw = (xh - fh)//sh + 1, (xw - fw)//sw + 1
    filters = filters.reshape((yc, -1))
    I = im2col(X, (fh, fw), (sh, sw))
    return np.dot(filters, I).reshape((xn, yc, yh, yw))


def test_convolve():
    """ Test of the convolve function """
    image = np.array([[
        [[1,  2,  3,  4, 5], [6,  7,  8,  9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    ]])

    filters = np.array([[[[-1, 1], [1, -1]], [[-1, -1], [1, 1]]], [[[1, 0], [1, 0]], [[0, 1], [0, 1]]]])

    expected = np.array([[[[10, 10], [10, 10]], [[46, 58], [66, 78]]]])

    assert check_equal(expected, convolve(image, filters, (1, 3)))


def max_pool(X, filters, strides):
    """
    Function performs max-pooling.

    :param X: Input matrix in NCHW layout
    :param filters: Tuple describing the shape of the area to be max-pooled (height, width), (2, 2) for VGG16
    :param strides: Tuple describing the strides pooling takes along the height and width
    :return: Max-pooling result
    """
    fh, fw = filters
    sh, sw = strides
    xn, xc, xh, xw = X.shape
    yh, yw = (xh - fh)//sh + 1, (xw - fw)//sw + 1
    Y = np.zeros((xn, xc, yh, yw))
    for n in range(xn):
        for c in range(xc):
            for y in range(yh):
                for x in range(yw):
                    Y[n, c, y, x] = X[n, c, y*sh:y*sh+fh, x*sw:x*sw+fw].max()
    return Y


def test_max_pool():
    """ Test of the max_pool function """
    image = np.array([[
        [[1,  2,  3,  4, 5], [6,  7,  8,  9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    ]])

    expected = np.array([
        [[[7, 10], [12, 15]], [[22, 25], [27, 30]]]])

    assert check_equal(expected, max_pool(image, (2, 2), (1, 3)))

    expected = np.array([
        [[[8, 10], [13, 15]], [[23, 25], [28, 30]]]])

    assert check_equal(expected, max_pool(image, (2, 3), (1, 2)))


def to_morton_left(mat):
    """
    Method rearranges a matrix from row-major layout to the layout depicted on the left in Figure 3.1 in the Masters
    Project pdf, as well as the left in Figure 5 of the published paper.
    :param mat: Row Major layout matrix
    :return: R_2_4_R layout matrix
    """
    h, w = mat.shape
    assert h % 2 == 0
    assert w % 8 == 0
    return mat.reshape((h//2, 2, w//4, 4)).swapaxes(1, 2).flatten()


def from_morton_left(mat, shape):
    """
    Method rearranges a matrix from the layout depicted on the left in Figure 3.1 in the Masters Project pdf, as well
    as the left in Figure 5 of the published paper row-major, back to row-major layout.
    :param mat: R_2_4_R layout matrix
    :param shape: Required shape of the row-major matrix. The code cannot infer this from the R_2_4_R layout matrix.
    :return: Row Major layout matrix
    """
    h, w = shape
    assert h % 2 == 0
    assert w % 8 == 0
    return mat.reshape(h//2, w//4, 2, 4).swapaxes(1, 2).reshape((h, w))


def test_to_morton_left():
    """ Test of the to_morton_left function """
    a = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32]
    ])

    b = np.array([1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 21, 22, 23, 24, 29, 30, 31, 32])

    assert check_equal(b.flatten(), to_morton_left(a))


def test_from_morton_left():
    """ Test of the from_morton_left function """
    a = np.array([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32]
    ])

    b = np.array([1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20, 25, 26, 27, 28, 21, 22, 23, 24, 29, 30, 31, 32])

    assert check_equal(a, from_morton_left(b, a.shape))


def to_morton_right(mat):
    """
    Method rearranges a matrix from row-major layout to the layout depicted on the right in Figure 3.1 in the Masters
    Project pdf, as well as the right in Figure 5 of the published paper.
    :param mat: Row Major layout matrix
    :return: C_4_2_C layout matrix
    """
    h, w = mat.shape
    assert h % 8 == 0
    assert w % 2 == 0
    return mat.T.reshape((w//2, 2, h//4, 4)).swapaxes(1, 2).flatten()


def from_morton_right(mat, shape):
    """
    Method rearranges a matrix from the layout depicted on the right in Figure 3.1 in the Masters Project pdf, as well
    as the right in Figure 5 of the published paper row-major, back to row-major layout.
    :param mat: C_4_2_C layout matrix
    :param shape: Required shape of the row-major matrix. The code cannot infer this from the C_4_2_C layout matrix.
    :return: Row Major layout matrix
    """
    h, w = shape
    assert h % 8 == 0
    assert w % 2 == 0
    return mat.reshape(w//2, h//4, 2, 4).swapaxes(1, 2).reshape((w, h)).T


def test_to_morton_right():
    """ Test of the to_morton_right function """
    a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32]
    ])

    b = np.array([1, 5, 9, 13, 2, 6, 10, 14, 17, 21, 25, 29, 18, 22, 26, 30, 3, 7, 11, 15, 4, 8, 12, 16, 19, 23, 27, 31, 20, 24, 28, 32])

    assert check_equal(b.flatten(), to_morton_right(a))


def test_from_morton_right():
    """ Test of the from_morton_right function """
    a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
        [25, 26, 27, 28],
        [29, 30, 31, 32]
    ])

    b = np.array([1, 5, 9, 13, 2, 6, 10, 14, 17, 21, 25, 29, 18, 22, 26, 30, 3, 7, 11, 15, 4, 8, 12, 16, 19, 23, 27, 31, 20, 24, 28, 32])

    assert check_equal(a, from_morton_right(b, a.shape))


def my_round(x, base):
    """ Function rounds the number x up to the nearest multiple of base. """
    return x + (base - (x % base)) % base


def test_my_round():
    """ Test of the my_round function """
    assert 20 == my_round(17, 5)
    assert 15 == my_round(15, 5)


def insert(mat, buffer):
    """
    Function inserts the data from mat into the provided buffer. This operation is necessary for dealing with the way
    Numpy manages memory. If we wanted to copy the resulting buffer onto the GPU, we need to do all of this explicitly,
    (especially the copy() call in the return statement), since Numpy is an abstraction over the memory management and
    may internally not actually move any data anywhere (which would mess with the OpenCL copying data onto the GPU).

    A similar Numpy view effect can be observed by doing a=a.T for some numpy array 'a' and then copying 'a' onto
    the GPU. The non-transposed version of 'a' will end up getting copied. Numpy + OpenCL can exhibit strange
    behaviours, resulting in odd looking functions like this one.
    """
    size = 1
    for s in mat.shape:
        size *= s
    buffer[:size] = mat.reshape((1, -1))[:]
    return buffer.copy()


def test_insert():
    """ Test of the insert function """
    a = np.array([
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 6],
                [7, 8]
            ]
        ],
        [
            [
                [9, 10],
                [11, 12]
            ],
            [
                [13, 14],
                [15, 16]
            ]
        ]
    ])
    buff = np.zeros(20, np.float32)
    res = insert(a, buff)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0]).astype(np.float32)
    assert check_equal(expected, res)
