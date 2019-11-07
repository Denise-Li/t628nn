import json
from simulator import blast_fully_connected, blast_conv, blast_subsampling


"""
CLBlast simulation of LeNet execution. Running this file should reproduce the results seen in
Figure 5.5 and Table 5.3 of the Masters Project pdf, as well as in Figure 3 and Table 2 of the published paper.
"""


INPUTS = 100


print(json.dumps({
    "C1": blast_conv(INPUTS, 1, 32, 32, 6, 5, 5, 1, 1),
    "S2": blast_subsampling(INPUTS, 6, 28, 28, 2, 2, 2, 2),
    "C3": blast_conv(INPUTS, 6, 14, 14, 16, 5, 5, 1, 1),
    "S4": blast_subsampling(INPUTS, 16, 10, 10, 2, 2, 2, 2),
    "F5": blast_fully_connected(INPUTS, 16 * 5 * 5, 120),
    "F6": blast_fully_connected(INPUTS, 120, 84)
}, indent=4))
