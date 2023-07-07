#!/usr/bin/python
###############################
# main.py
# testing pycuda to multiply an array by a scalar with gpu
# https://documen.tician.de/pycuda/
###############################
import scalar_matrix_multiply as sm
import numpy
import pycuda.driver as cuda
from timeit import timeit

line = 24
col = 24
num_repeats = 100
func_double_it = sm.mod.get_function("double_it")
a_input = sm.make_random_numpy_array(line, col)
a_result_numpy = a_input * 2.0
# testing numpy array multiply by scalar
timer_numpy = timeit("a_input * 2.0", globals=globals(), number=num_repeats)

# testing of gpu array multiply by scalar using PyCuda
gpuArray = sm.transfer_array_in_gpu(a_input)
timer_gpu_multiply = timeit("func_double_it(gpuArray, block=(24, 24, 1))", globals=globals(), number=num_repeats)
# func_double_it(gpuArray, block=(32, 32, 1))
a_result_gpu_timer = numpy.empty_like(a_input)
cuda.memcpy_dtoh(a_result_gpu_timer, gpuArray)
# end of gpu array multiply by scalar using PyCuda
print("Matrix mult by scalar in GPU                       : {:.8f}".format(timer_gpu_multiply))
# print("Matrix mult by scalar in GPU (Total with transfer) : {:.8f}".format(timer_gpu))
print("Matrix mult by scalar in numpy                     : {:.8f}".format(timer_numpy))
print(a_result_gpu_timer / a_result_numpy)
