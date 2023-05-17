# cuda_moments
Computation of binned statistical moments with CUDA, CuPy and CFFI.

__Create a Gaussian distribution of particles each with a bin index:__</br>
`python create_distribution.py`
- this saves a .npz file to `/inputs`

__Run CPU benchmark using CFFI:__</br>
`python c_cpu.py`</br>
CPU: AMD Ryzen Threadripper 2970WX 24-Core Processor

__Run GPU benchmark using CuPy:__</br>
`python c_gpu.py`</br>
GPU: NVIDIA TITAN V

- these save a .npz file with the slice moments in `/outputs`

__Compare output moments:__</br>
`python compare_moments.py`

The plot below shows the speedup achieved on the GPU compared to the single CPU execution, as a function of the block size and the number of particles in the distribution.
<p align="center">
<img src="https://github.com/pkicsiny/cuda_moments/blob/master/images/gpu_speedup.png" width="600">
</p>
