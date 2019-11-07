import json
from simulator import blast_fully_connected_single, blast_conv_single


"""
CLBlast simulation of VGG16 execution. Running this file should reproduce the results seen in
Figure 5.7 and Table A.1 of the Honours Project pdf, and Figure 4 of the published paper.
"""


print(json.dumps({
    "conv1-1": blast_conv_single(3, 226, 226, 64, 3, 3, 1, 1),
    "conv1-2": blast_conv_single(64, 226, 226, 64, 3, 3, 1, 1),
    "conv2-1": blast_conv_single(64, 114, 114, 128, 3, 3, 1, 1),
    "conv2-2": blast_conv_single(128, 114, 114, 128, 3, 3, 1, 1),
    "conv3-1": blast_conv_single(128, 58, 58, 256, 3, 3, 1, 1),
    "conv3-2": blast_conv_single(256, 58, 58, 256, 3, 3, 1, 1),
    "conv3-3": blast_conv_single(256, 58, 58, 256, 3, 3, 1, 1),
    "conv4-1": blast_conv_single(256, 30, 30, 512, 3, 3, 1, 1),
    "conv4-2": blast_conv_single(512, 30, 30, 512, 3, 3, 1, 1),
    "conv4-3": blast_conv_single(512, 30, 30, 512, 3, 3, 1, 1),
    "conv5-1": blast_conv_single(512, 16, 16, 512, 3, 3, 1, 1),
    "conv5-2": blast_conv_single(512, 16, 16, 512, 3, 3, 1, 1),
    "conv5-3": blast_conv_single(512, 16, 16, 512, 3, 3, 1, 1),
    "FC1": blast_fully_connected_single(4096, 7 * 7 * 512),
    "FC2": blast_fully_connected_single(4096, 4096),
    "FC3": blast_fully_connected_single(1000, 4096)
}, indent=4))
