import numpy as np
import cupy as cp
import os
import time
import argparse
import cffi

# scan distributions and blocksize
n_m_arr = np.array([int(1e4), int(1e5), int(5e5), int(1e6), int(3e6), int(1e7)], dtype=int)
blocksize_arr = np.array([1, 16, 32, 64, 128, 256, 512, 1024], dtype=int)
n_s = 100
np.set_printoptions(formatter={'float': '{: .6e},'.format})

out_dir = "outputs"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for n_m in n_m_arr:
    for blocksize in blocksize_arr:

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
        print(f"[c_cpu.py] =============== nm: {n_m}, ns: {n_s} =======================\n")

        # load kernels
        with open('kernels/compute_moments_gpu.h', 'r') as file:
            cuda_str = file.read()
        module = cp.RawModule(code=cuda_str)
                    
        kernel_1 = module.get_function("compute_moments_1")
        kernel_2 = module.get_function("compute_moments_2")

        # num block x blocksize bust be at lease the number of macroparts
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
        
        # save for comparison
        slice_moments_cpu = slice_moments.get()
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

        # load moments
        cpu_moments = np.load(os.path.join(out_dir, f"cpu_moments_nm_{n_m}_ns_{n_s}.npz"))
        gpu_moments = np.load(os.path.join(out_dir, f"gpu_moments_nm_{n_m}_ns_{n_s}.npz"))
        cpu_e_x  = cpu_moments["e_x"]
        cpu_e_px = cpu_moments["e_px"]
        cpu_e_y  = cpu_moments["e_y"]
        cpu_e_py = cpu_moments["e_py"]
        cpu_e_z  = cpu_moments["e_z"]
        cpu_e_d  = cpu_moments["e_d"]
        cpu_cov_xx   = cpu_moments["cov_xx"]
        cpu_cov_xpx  = cpu_moments["cov_xpx"]
        cpu_cov_xy   = cpu_moments["cov_xy"]
        cpu_cov_xpy  = cpu_moments["cov_xpy"]
        cpu_cov_pxpx = cpu_moments["cov_pxpx"]
        cpu_cov_pxy  = cpu_moments["cov_pxy"]
        cpu_cov_pxpy = cpu_moments["cov_pxpy"]
        cpu_cov_yy   = cpu_moments["cov_yy"]
        cpu_cov_ypy  = cpu_moments["cov_ypy"]
        cpu_cov_pypy = cpu_moments["cov_pypy"]
        
        gpu_e_x  = gpu_moments["e_x"]
        gpu_e_px = gpu_moments["e_px"]
        gpu_e_y  = gpu_moments["e_y"]
        gpu_e_py = gpu_moments["e_py"]
        gpu_e_z  = gpu_moments["e_z"]
        gpu_e_d  = gpu_moments["e_d"]
        gpu_cov_xx   = gpu_moments["cov_xx"]
        gpu_cov_xpx  = gpu_moments["cov_xpx"]
        gpu_cov_xy   = gpu_moments["cov_xy"]
        gpu_cov_xpy  = gpu_moments["cov_xpy"]
        gpu_cov_pxpx = gpu_moments["cov_pxpx"]
        gpu_cov_pxy  = gpu_moments["cov_pxy"]
        gpu_cov_pxpy = gpu_moments["cov_pxpy"]
        gpu_cov_yy   = gpu_moments["cov_yy"]
        gpu_cov_ypy  = gpu_moments["cov_ypy"]
        gpu_cov_pypy = gpu_moments["cov_pypy"]
        
        print(f"[compare_moments.py] nm: {n_m}, ns: {n_s}\n")
        
        # calculate differences
        diff_e_x      = np.abs(gpu_e_x      - cpu_e_x     ) 
        diff_e_px     = np.abs(gpu_e_px     - cpu_e_px    )  
        diff_e_y      = np.abs(gpu_e_y      - cpu_e_y     ) 
        diff_e_py     = np.abs(gpu_e_py     - cpu_e_py    ) 
        diff_e_z      = np.abs(gpu_e_z      - cpu_e_z     ) 
        diff_e_d      = np.abs(gpu_e_d      - cpu_e_d     )
        diff_cov_xx   = np.abs(gpu_cov_xx   - cpu_cov_xx  ) 
        diff_cov_xpx  = np.abs(gpu_cov_xpx  - cpu_cov_xpx ) 
        diff_cov_xy   = np.abs(gpu_cov_xy   - cpu_cov_xy  ) 
        diff_cov_xpy  = np.abs(gpu_cov_xpy  - cpu_cov_xpy ) 
        diff_cov_pxpx = np.abs(gpu_cov_pxpx - cpu_cov_pxpx) 
        diff_cov_pxy  = np.abs(gpu_cov_pxy  - cpu_cov_pxy ) 
        diff_cov_pxpy = np.abs(gpu_cov_pxpy - cpu_cov_pxpy) 
        diff_cov_yy   = np.abs(gpu_cov_yy   - cpu_cov_yy  ) 
        diff_cov_ypy  = np.abs(gpu_cov_ypy  - cpu_cov_ypy ) 
        diff_cov_pypy = np.abs(gpu_cov_pypy - cpu_cov_pypy) 
        
        print(f"abs_e_x     : sum: {np.sum( diff_e_x     ):.6e}, mean: {np.mean( diff_e_x     ):.6e}, max: {np.max( diff_e_x     ):.6e}") 
        print(f"abs_e_px    : sum: {np.sum( diff_e_px    ):.6e}, mean: {np.mean( diff_e_px    ):.6e}, max: {np.max( diff_e_px    ):.6e}")
        print(f"abs_e_y     : sum: {np.sum( diff_e_y     ):.6e}, mean: {np.mean( diff_e_y     ):.6e}, max: {np.max( diff_e_y     ):.6e}")
        print(f"abs_e_py    : sum: {np.sum( diff_e_py    ):.6e}, mean: {np.mean( diff_e_py    ):.6e}, max: {np.max( diff_e_py    ):.6e}")
        print(f"abs_e_z     : sum: {np.sum( diff_e_z     ):.6e}, mean: {np.mean( diff_e_z     ):.6e}, max: {np.max( diff_e_z     ):.6e}")
        print(f"abs_e_d     : sum: {np.sum( diff_e_d     ):.6e}, mean: {np.mean( diff_e_d     ):.6e}, max: {np.max( diff_e_d     ):.6e}")
        print(f"abs_cov_xx  : sum: {np.sum( diff_cov_xx  ):.6e}, mean: {np.mean( diff_cov_xx  ):.6e}, max: {np.max( diff_cov_xx  ):.6e}")
        print(f"abs_cov_xpx : sum: {np.sum( diff_cov_xpx ):.6e}, mean: {np.mean( diff_cov_xpx ):.6e}, max: {np.max( diff_cov_xpx ):.6e}")
        print(f"abs_cov_xy  : sum: {np.sum( diff_cov_xy  ):.6e}, mean: {np.mean( diff_cov_xy  ):.6e}, max: {np.max( diff_cov_xy  ):.6e}")
        print(f"abs_cov_xpy : sum: {np.sum( diff_cov_xpy ):.6e}, mean: {np.mean( diff_cov_xpy ):.6e}, max: {np.max( diff_cov_xpy ):.6e}")
        print(f"abs_cov_pxpx: sum: {np.sum( diff_cov_pxpx):.6e}, mean: {np.mean( diff_cov_pxpx):.6e}, max: {np.max( diff_cov_pxpx):.6e}")
        print(f"abs_cov_pxy : sum: {np.sum( diff_cov_pxy ):.6e}, mean: {np.mean( diff_cov_pxy ):.6e}, max: {np.max( diff_cov_pxy ):.6e}")
        print(f"abs_cov_pxpy: sum: {np.sum( diff_cov_pxpy):.6e}, mean: {np.mean( diff_cov_pxpy):.6e}, max: {np.max( diff_cov_pxpy):.6e}")
        print(f"abs_cov_yy  : sum: {np.sum( diff_cov_yy  ):.6e}, mean: {np.mean( diff_cov_yy  ):.6e}, max: {np.max( diff_cov_yy  ):.6e}")
        print(f"abs_cov_ypy : sum: {np.sum( diff_cov_ypy ):.6e}, mean: {np.mean( diff_cov_ypy ):.6e}, max: {np.max( diff_cov_ypy ):.6e}")
        print(f"abs_cov_pypy: sum: {np.sum( diff_cov_pypy):.6e}, mean: {np.mean( diff_cov_pypy):.6e}, max: {np.max( diff_cov_pypy):.6e}")

        # 1e-15 numerical noise
        rel_abs_e_x      = diff_e_x     / np.abs(cpu_e_x      )
        rel_abs_e_px     = diff_e_px    / np.abs(cpu_e_px     )
        rel_abs_e_y      = diff_e_y     / np.abs(cpu_e_y      )
        rel_abs_e_py     = diff_e_py    / np.abs(cpu_e_py     )
        rel_abs_e_z      = diff_e_z     / np.abs(cpu_e_z      )
        rel_abs_e_d      = diff_e_d     / np.abs(cpu_e_d      )
        rel_abs_cov_xx   = diff_cov_xx  / np.abs(cpu_cov_xx   )
        rel_abs_cov_xpx  = diff_cov_xpx / np.abs(cpu_cov_xpx  )
        rel_abs_cov_xy   = diff_cov_xy  / np.abs(cpu_cov_xy   )
        rel_abs_cov_xpy  = diff_cov_xpy / np.abs(cpu_cov_xpy  )
        rel_abs_cov_pxpx = diff_cov_pxpx/ np.abs(cpu_cov_pxpx )
        rel_abs_cov_pxy  = diff_cov_pxy / np.abs(cpu_cov_pxy  )
        rel_abs_cov_pxpy = diff_cov_pxpy/ np.abs(cpu_cov_pxpy )
        rel_abs_cov_yy   = diff_cov_yy  / np.abs(cpu_cov_yy   )
        rel_abs_cov_ypy  = diff_cov_ypy / np.abs(cpu_cov_ypy  )
        rel_abs_cov_pypy = diff_cov_pypy/ np.abs(cpu_cov_pypy )
        print("----------------------------------------------")
        print(f"rel_e_x     : sum: {np.sum(rel_abs_e_x     ):.6e}, mean: {np.mean(rel_abs_e_x     ):.6e}, max: {np.max(rel_abs_e_x     ):.6e}") 
        print(f"rel_e_px    : sum: {np.sum(rel_abs_e_px    ):.6e}, mean: {np.mean(rel_abs_e_px    ):.6e}, max: {np.max(rel_abs_e_px    ):.6e}")
        print(f"rel_e_y     : sum: {np.sum(rel_abs_e_y     ):.6e}, mean: {np.mean(rel_abs_e_y     ):.6e}, max: {np.max(rel_abs_e_y     ):.6e}")
        print(f"rel_e_py    : sum: {np.sum(rel_abs_e_py    ):.6e}, mean: {np.mean(rel_abs_e_py    ):.6e}, max: {np.max(rel_abs_e_py    ):.6e}")
        print(f"rel_e_z     : sum: {np.sum(rel_abs_e_z     ):.6e}, mean: {np.mean(rel_abs_e_z     ):.6e}, max: {np.max(rel_abs_e_z     ):.6e}")
        print(f"rel_e_d     : sum: {np.sum(rel_abs_e_d     ):.6e}, mean: {np.mean(rel_abs_e_d     ):.6e}, max: {np.max(rel_abs_e_d     ):.6e}")
        print(f"rel_cov_xx  : sum: {np.sum(rel_abs_cov_xx  ):.6e}, mean: {np.mean(rel_abs_cov_xx  ):.6e}, max: {np.max(rel_abs_cov_xx  ):.6e}")
        print(f"rel_cov_xpx : sum: {np.sum(rel_abs_cov_xpx ):.6e}, mean: {np.mean(rel_abs_cov_xpx ):.6e}, max: {np.max(rel_abs_cov_xpx ):.6e}")
        print(f"rel_cov_xy  : sum: {np.sum(rel_abs_cov_xy  ):.6e}, mean: {np.mean(rel_abs_cov_xy  ):.6e}, max: {np.max(rel_abs_cov_xy  ):.6e}")
        print(f"rel_cov_xpy : sum: {np.sum(rel_abs_cov_xpy ):.6e}, mean: {np.mean(rel_abs_cov_xpy ):.6e}, max: {np.max(rel_abs_cov_xpy ):.6e}")
        print(f"rel_cov_pxpx: sum: {np.sum(rel_abs_cov_pxpx):.6e}, mean: {np.mean(rel_abs_cov_pxpx):.6e}, max: {np.max(rel_abs_cov_pxpx):.6e}")
        print(f"rel_cov_pxy : sum: {np.sum(rel_abs_cov_pxy ):.6e}, mean: {np.mean(rel_abs_cov_pxy ):.6e}, max: {np.max(rel_abs_cov_pxy ):.6e}")
        print(f"rel_cov_pxpy: sum: {np.sum(rel_abs_cov_pxpy):.6e}, mean: {np.mean(rel_abs_cov_pxpy):.6e}, max: {np.max(rel_abs_cov_pxpy):.6e}")
        print(f"rel_cov_yy  : sum: {np.sum(rel_abs_cov_yy  ):.6e}, mean: {np.mean(rel_abs_cov_yy  ):.6e}, max: {np.max(rel_abs_cov_yy  ):.6e}")
        print(f"rel_cov_ypy : sum: {np.sum(rel_abs_cov_ypy ):.6e}, mean: {np.mean(rel_abs_cov_ypy ):.6e}, max: {np.max(rel_abs_cov_ypy ):.6e}")
        print(f"rel_cov_pypy: sum: {np.sum(rel_abs_cov_pypy):.6e}, mean: {np.mean(rel_abs_cov_pypy):.6e}, max: {np.max(rel_abs_cov_pypy):.6e}")
