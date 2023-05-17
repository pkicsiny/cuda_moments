import numpy as np
import os

# number of particles
n_m = int(1e5)

# number of bins
n_s = 100

print(f"[create_random_input.py] nm: {n_m}, ns: {n_s}\n")

# uniform charge slicing
slices = np.array([np.random.choice(n_s) for i in range(n_m)], dtype=np.int64)
weight = np.ones_like(slices, dtype=np.int64) * int(2e4)

sigma_x         = 1e-6
sigma_y         = 2e-9
sigma_z_tot     = 5e-3
sigma_px        = 2e-6
sigma_py        = 4e-9
sigma_delta_tot = 1e-2
x        = sigma_x         * np.random.randn(n_m)
y        = sigma_y         * np.random.randn(n_m)
z        = sigma_z_tot     * np.random.randn(n_m)
px       = sigma_px        * np.random.randn(n_m)
py       = sigma_py        * np.random.randn(n_m)
d        = sigma_delta_tot * np.random.randn(n_m)

input_dir = "inputs"
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

np.savez_compressed(os.path.join(input_dir, f"random_nm_{n_m}_ns_{n_s}"),
                   x =x,
                   px=px,
                   y =y,
                   py=py,
                   z =z,
                   d =d,
                   weight =weight,
                   slices =slices,
)
