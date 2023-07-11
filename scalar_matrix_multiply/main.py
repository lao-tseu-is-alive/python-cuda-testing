#!/usr/bin/python
###############################
# image_manipulation.py
# testing cupy to multiply an array by a scalar with gpu
# https://documen.tician.de/pycuda/
###############################
import numpy as np
import cupy as cp
from timeit import timeit

line = 1024
col = 1024
num_repeats = 1000
a_input = np.random.randn(line, col)
a_result_numpy = a_input * 2.0
# testing numpy array multiply by scalar
timer_numpy = timeit("a_input * 2.0", globals=globals(), number=num_repeats)

# testing of gpu array multiply by scalar using PyCuda
gpuArray = cp.array(a_input)
a_result_gpu = gpuArray * 2.0
timer_gpu_multiply = timeit("gpuArray * 2.0", globals=globals(), number=num_repeats)
# end of gpu array multiply by scalar using PyCuda
print("Matrix mult by scalar in GPU                       : {:.8f}".format(timer_gpu_multiply))
# print("Matrix mult by scalar in GPU (Total with transfer) : {:.8f}".format(timer_gpu))
print("Matrix mult by scalar in numpy                     : {:.8f}".format(timer_numpy))
print("To check all is good let's subtract the two matrix it should return a zero one")
print(cp.asnumpy(a_result_gpu) - a_result_numpy)
