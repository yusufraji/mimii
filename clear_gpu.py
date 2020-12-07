from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
cuda.select_device(0)
cuda.close()