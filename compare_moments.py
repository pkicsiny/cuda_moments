import numpy as np

# specify distribution
n_m = int(1e6)
n_s = 100

# load moments
cpu_moments = np.load(f"outputs/cpu_moments_nm_{n_m}_ns_{n_s}.npz")
gpu_moments = np.load(f"outputs/gpu_moments_nm_{n_m}_ns_{n_s}.npz")
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
