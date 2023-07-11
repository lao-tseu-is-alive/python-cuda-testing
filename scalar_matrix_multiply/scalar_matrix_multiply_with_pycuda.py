###############################
# scalar_matrix_multiply_with_pycuda.py
# testing pycuda to multiply an array by a scalar with gpu
# https://documen.tician.de/pycuda/
###############################
import sys
import numpy
import pycuda
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from timeit import default_timer as timer

mod = SourceModule("""
  __global__ void double_it(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*12;
    a[idx] *= 2;
  }
  """)


def print_versions():
    print(f'Python:\t {sys.version}')
    print(f'Numpy :\t {numpy.version.version}')
    print(f'PyCuda:\t {pycuda.VERSION_TEXT}')
    print(f'Cuda  :\t {".".join(map(str, cuda.get_version()))}')
    print(f'Driver:\t {cuda.get_driver_version()}')


def make_random_numpy_array(lines, cols):
    a = numpy.random.randn(lines, cols)
    a = a.astype(numpy.float32)
    return a


def transfer_array_in_gpu(numpy_array):
    a_gpu = cuda.mem_alloc(numpy_array.nbytes)
    cuda.memcpy_htod(a_gpu, numpy_array)
    return a_gpu


def get_gpu_array_ref(numpy_array):
    return gpuarray.to_gpu(numpy_array)


if __name__ == '__main__':
    print_versions()
    line = 12
    col = 12
    func_double_it = mod.get_function("double_it")
    a_input = make_random_numpy_array(line, col)

    # start of numpy array multiply by scalar
    start_numpy = timer()
    a_result_numpy = a_input * 2.0
    end_numpy = timer()
    timer_numpy = end_numpy - start_numpy
    # end of gpu array multiply by scalar using PyCuda

    # start of gpu array multiply by scalar using PyCuda
    start_gpu = timer()
    gpuArray = transfer_array_in_gpu(a_input)
    start_gpu_multiply = timer()
    func_double_it(gpuArray, block=(12, 12, 1))
    end_gpu_multiply = timer()
    timer_gpu_multiply = end_gpu_multiply - start_gpu_multiply
    a_result_gpu = numpy.empty_like(a_input)
    cuda.memcpy_dtoh(a_result_gpu, gpuArray)
    end_gpu = timer()
    timer_gpu = end_gpu - start_gpu
    # end of gpu array multiply by scalar using PyCuda

    start_gpu_direct = timer()
    gpu_a_ref = gpuarray.to_gpu(a_input)
    a_result_gpu_direct = (2*gpu_a_ref).get()
    end_gpu_direct = timer()
    timer_gpu_direct = end_gpu_direct - start_gpu_direct
    print("comparing  numpy array result with gpu ")
    print(a_result_numpy - a_result_gpu)
    print(a_result_numpy - a_result_gpu_direct)
    print("Matrix mult by scalar in GPU                       : {:.8f}".format(timer_gpu_multiply))
    print("Matrix mult by scalar in GPU (Total with transfer) : {:.8f}".format(timer_gpu))
    print("Matrix mult by scalar in GPU direct                : {:.8f}".format(timer_gpu_direct))
    print("Matrix mult by scalar in numpy                     : {:.8f}".format(timer_numpy))
    print("### compare arrays :")

