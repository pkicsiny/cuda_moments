import numpy as np
import cupy as cp
import os
import time
import argparse
import cffi

# specify distribution
n_m = int(1e6)
n_s = 100

# load input
loaded = np.load(f"inputs/random_nm_{n_m}_ns_{n_s}.npz")
x  = cp.array( loaded[ "x"] )
px = cp.array( loaded["px"] )
y  = cp.array( loaded[ "y"] )
py = cp.array( loaded["py"] )
z  = cp.array( loaded[ "z"] )
d  = cp.array( loaded[ "d"] )
weight = cp.array( loaded["weight"] )  # int, len=n_m, num charges repr. by each macropart
slices = cp.array( loaded["slices"], dtype=np.float64)  # int, len=n_m, slice idx of each macropart
slice_moments = cp.zeros(n_s*(1+6+10+6+10),dtype=np.float64)  # num parts, 6 first, 10 second order sums+moments

# load kernels
with open('kernels/compute_moments_gpu.h', 'r') as file:
    cuda_str = file.read()
module = cp.RawModule(code=cuda_str)
            
kernel_1 = module.get_function("compute_moments_1")
kernel_2 = module.get_function("compute_moments_2")

#x  = cp.ones(n_m)
#px = cp.ones(n_m)
#y  = cp.ones(n_m)
#py = cp.ones(n_m)
#z  = cp.ones(n_m)
#d  = cp.ones(n_m)

# num block x blocksize bust be at least the number of macroparts
blocksize = 1  # max 1024; cannot be arbitrary bc I use it in kernel
n_blocks = int(np.ceil(n_m / blocksize))  # can be arbitrary
print(f"n_blocks: {n_blocks}, blocksize: {blocksize}")

# compute moments
start = time.time()
kernel_1(grid=(n_blocks,), block=(blocksize,), args=(x, px, y, py, z, d, slices, slice_moments, np.int64(n_m), np.int64(n_s)), shared_mem=n_s*8*17)  # shared mem size in byte (double=8byte)
cp.cuda.stream.get_current_stream().synchronize()
kernel_2(grid=(1,), block=(n_s,), args=(slice_moments, np.int64(n_s), np.int64(weight.get()[0])))
cp.cuda.stream.get_current_stream().synchronize()
end = time.time()
print(f"Elapsed walltime: {end - start} [s]")

# print
slice_moments_cpu = slice_moments.get()
np.set_printoptions(formatter={'float': '{: .6e},'.format})
print("counts:",    slice_moments_cpu[      :   n_s])
print("sum(x): ",   slice_moments_cpu[   n_s: 2*n_s])
print("sum(px):",   slice_moments_cpu[ 2*n_s: 3*n_s])
print("sum(y): ",   slice_moments_cpu[ 3*n_s: 4*n_s])
print("sum(py):",   slice_moments_cpu[ 4*n_s: 5*n_s])
print("sum(z): ",   slice_moments_cpu[ 5*n_s: 6*n_s])
print("sum(d): ",   slice_moments_cpu[ 6*n_s: 7*n_s])
print("sum(xx):  ", slice_moments_cpu[ 7*n_s: 8*n_s])
print("sum(xpx): ", slice_moments_cpu[ 8*n_s: 9*n_s])
print("sum(xy):  ", slice_moments_cpu[ 9*n_s:10*n_s])
print("sum(xpy): ", slice_moments_cpu[10*n_s:11*n_s])
print("sum(pxpx):", slice_moments_cpu[11*n_s:12*n_s])
print("sum(pxy): ", slice_moments_cpu[12*n_s:13*n_s])
print("sum(pxpy):", slice_moments_cpu[13*n_s:14*n_s])
print("sum(yy):  ", slice_moments_cpu[14*n_s:15*n_s])
print("sum(ypy): ", slice_moments_cpu[15*n_s:16*n_s])
print("sum(pypy):", slice_moments_cpu[16*n_s:17*n_s])
print("E(x): ",     slice_moments_cpu[17*n_s:18*n_s])
print("E(px):",     slice_moments_cpu[18*n_s:19*n_s])
print("E(y): ",     slice_moments_cpu[19*n_s:20*n_s])
print("E(py):",     slice_moments_cpu[20*n_s:21*n_s])
print("E(z): ",     slice_moments_cpu[21*n_s:22*n_s])
print("E(d): ",     slice_moments_cpu[22*n_s:23*n_s])
print("Cov(xx):  ", slice_moments_cpu[23*n_s:24*n_s])
print("Cov(xpx): ", slice_moments_cpu[24*n_s:25*n_s])
print("Cov(xy):  ", slice_moments_cpu[25*n_s:26*n_s])
print("Cov(xpy): ", slice_moments_cpu[26*n_s:27*n_s])
print("Cov(pxpx):", slice_moments_cpu[27*n_s:28*n_s])
print("Cov(pxy) :", slice_moments_cpu[28*n_s:29*n_s])
print("Cov(pxpy):", slice_moments_cpu[29*n_s:30*n_s])
print("Cov(yy):  ", slice_moments_cpu[30*n_s:31*n_s])
print("Cov(ypy): ", slice_moments_cpu[31*n_s:32*n_s])
print("Cov(pypy):", slice_moments_cpu[32*n_s:      ])

# save for comparison with cpu kernel
out_dir = "outputs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.savez_compressed(os.path.join(out_dir, f"gpu_moments_nm_{n_m}_ns_{n_s}"),
counts   = slice_moments_cpu[      :   n_s],
e_x      = slice_moments_cpu[17*n_s:18*n_s],
e_px     = slice_moments_cpu[18*n_s:19*n_s],
e_y      = slice_moments_cpu[19*n_s:20*n_s],
e_py     = slice_moments_cpu[20*n_s:21*n_s],
e_z      = slice_moments_cpu[21*n_s:22*n_s],
e_d      = slice_moments_cpu[22*n_s:23*n_s],
cov_xx   = slice_moments_cpu[23*n_s:24*n_s],
cov_xpx  = slice_moments_cpu[24*n_s:25*n_s],
cov_xy   = slice_moments_cpu[25*n_s:26*n_s],
cov_xpy  = slice_moments_cpu[26*n_s:27*n_s],
cov_pxpx = slice_moments_cpu[27*n_s:28*n_s],
cov_pxy  = slice_moments_cpu[28*n_s:29*n_s],
cov_pxpy = slice_moments_cpu[29*n_s:30*n_s],
cov_yy   = slice_moments_cpu[30*n_s:31*n_s],
cov_ypy  = slice_moments_cpu[31*n_s:32*n_s],
cov_pypy = slice_moments_cpu[32*n_s:      ],
)
