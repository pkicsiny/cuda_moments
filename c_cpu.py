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
x  = loaded["x"]
px = loaded["px"]
y  = loaded["y"]
py = loaded["py"]
z  = loaded["z"]
d  = loaded["d"]
weight = loaded["weight"]
slices = loaded["slices"]
slice_moments = np.zeros(n_s*(1+6+10),dtype=np.float64)  # num parts, 6 first, 10 second moments
print(f"[c_cpu.py] nm: {n_m}, ns: {n_s}\n")

# create ffi library
ffi_interface = cffi.FFI()
ffi_interface.cdef(
    """
    void compute_slice_moments(double* x, double* px, double* y, double* py, double* zeta, double* delta, int64_t* particles_slice, double* moments, int n_part, int n_slices, int64_t* weight);
    """
)
ffi_interface.set_source("_compute_moments", 
        """ #include "kernels/compute_moments_cpu.h" """
)
ffi_interface.compile(verbose=True)

from _compute_moments import ffi, lib

#x  = np.ones(n_m)
#px = np.ones(n_m)
#y  = np.ones(n_m)
#py = np.ones(n_m)
#z  = np.ones(n_m)
#d  = np.ones(n_m)

# typecast input arrays
x_cffi  = ffi.cast( "double *", ffi.from_buffer(x))
px_cffi = ffi.cast( "double *", ffi.from_buffer(px))
y_cffi  = ffi.cast( "double *", ffi.from_buffer(y))
py_cffi = ffi.cast( "double *", ffi.from_buffer(py))
z_cffi  = ffi.cast( "double *", ffi.from_buffer(z))
d_cffi  = ffi.cast( "double *", ffi.from_buffer(d))
slices_cffi  = ffi.cast( "int64_t *", ffi.from_buffer(slices))
moments_cffi = ffi.cast( "double *", ffi.from_buffer(slice_moments))
weight_cffi  = ffi.cast( "int64_t *", ffi.from_buffer(weight))
print(x[slices==0].mean(), x[slices==1].mean())

# compute moments
start = time.time()
#for i in range(100):
moments_cffi = ffi.cast( "double *", ffi.from_buffer(slice_moments))
lib.compute_slice_moments(x_cffi, px_cffi, y_cffi, py_cffi, z_cffi, d_cffi, slices_cffi, moments_cffi, n_m, n_s, weight_cffi)
end = time.time()
print(f"Elapsed walltime: {end - start} [s]")

# print
slice_moments_result =  np.array(ffi.unpack(moments_cffi,len(slice_moments)))
np.set_printoptions(formatter={'float': '{: .6e},'.format})
print("counts:",    slice_moments_result[      :   n_s]) 
print("E(x): ",     slice_moments_result[   n_s: 2*n_s])
print("E(px):",     slice_moments_result[ 2*n_s: 3*n_s])
print("E(y): ",     slice_moments_result[ 3*n_s: 4*n_s])
print("E(py):",     slice_moments_result[ 4*n_s: 5*n_s])
print("E(z): ",     slice_moments_result[ 5*n_s: 6*n_s])
print("E(d): ",     slice_moments_result[ 6*n_s: 7*n_s])
print("Cov(xx):  ", slice_moments_result[ 7*n_s: 8*n_s])
print("Cov(xpx): ", slice_moments_result[ 8*n_s: 9*n_s])
print("Cov(xy):  ", slice_moments_result[ 9*n_s:10*n_s])
print("Cov(xpy): ", slice_moments_result[10*n_s:11*n_s])
print("Cov(pxpx):", slice_moments_result[11*n_s:12*n_s])
print("Cov(pxy) :", slice_moments_result[12*n_s:13*n_s])
print("Cov(pxpy):", slice_moments_result[13*n_s:14*n_s])
print("Cov(yy):  ", slice_moments_result[14*n_s:15*n_s])
print("Cov(ypy): ", slice_moments_result[15*n_s:16*n_s])
print("Cov(pypy):", slice_moments_result[16*n_s:      ])

# save for comparison with gpu kernel
out_dir = "outputs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
np.savez_compressed(os.path.join(out_dir, f"cpu_moments_nm_{n_m}_ns_{n_s}"),
counts   = slice_moments_result[      :   n_s], 
e_x      = slice_moments_result[   n_s: 2*n_s],
e_px     = slice_moments_result[ 2*n_s: 3*n_s],
e_y      = slice_moments_result[ 3*n_s: 4*n_s],
e_py     = slice_moments_result[ 4*n_s: 5*n_s],
e_z      = slice_moments_result[ 5*n_s: 6*n_s],
e_d      = slice_moments_result[ 6*n_s: 7*n_s],
cov_xx   = slice_moments_result[ 7*n_s: 8*n_s],
cov_xpx  = slice_moments_result[ 8*n_s: 9*n_s],
cov_xy   = slice_moments_result[ 9*n_s:10*n_s],
cov_xpy  = slice_moments_result[10*n_s:11*n_s],
cov_pxpx = slice_moments_result[11*n_s:12*n_s],
cov_pxy  = slice_moments_result[12*n_s:13*n_s],
cov_pxpy = slice_moments_result[13*n_s:14*n_s],
cov_yy   = slice_moments_result[14*n_s:15*n_s],
cov_ypy  = slice_moments_result[15*n_s:16*n_s],
cov_pypy = slice_moments_result[16*n_s:      ],
)
