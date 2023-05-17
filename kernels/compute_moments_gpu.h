extern "C" {
__global__ void compute_moments_1(double* x_in, double* px_in, double* y_in, double* py_in, double* zeta_in, double* delta_in,
	           	double* s_in, double* m_out, const int n_macroparts, const int n_slices) {

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

//        if (gid >= n_macroparts) return;
	// init shared memory used for partial sums
	extern __shared__ double sdata[];  // len n_slices * 17 (count + 6 x sum(xi), 10 x sum(xi*xy))
        int full_pass = (int)(17*n_slices / blockDim.x);
        int residual = (17*n_slices)%blockDim.x;
        for (int i=0; i<full_pass; i++){
          sdata[i*blockDim.x + tid] = 0.0;
	}
        if (tid < residual){
	  sdata[full_pass*blockDim.x+tid] = 0.0;
	}
	__syncthreads();

	if (gid < n_macroparts){
  	  // this could be one long 1D contiguous array
          int    s_i     =     s_in[gid];
	  double x_i     =     x_in[gid];
  	  double px_i    =    px_in[gid];
  	  double y_i     =     y_in[gid];
  	  double py_i    =    py_in[gid];
  	  double zeta_i  =  zeta_in[gid];
  	  double delta_i = delta_in[gid];
  	  //printf("gid: %d, x_i: %g\n", gid, x_i);

          // count
	  atomicAdd(&sdata[s_i], 1);

	  // sum(xi)
  	  atomicAdd(&sdata[  n_slices+s_i],     x_i);
  	  atomicAdd(&sdata[2*n_slices+s_i],    px_i);
  	  atomicAdd(&sdata[3*n_slices+s_i],     y_i);
  	  atomicAdd(&sdata[4*n_slices+s_i],    py_i);
  	  atomicAdd(&sdata[5*n_slices+s_i],  zeta_i);
  	  atomicAdd(&sdata[6*n_slices+s_i], delta_i);
  	  
  	  // sum(xi*xj)
  	  atomicAdd(&sdata[ 7*n_slices+s_i],     x_i*x_i);
  	  atomicAdd(&sdata[ 8*n_slices+s_i],    x_i*px_i);
  	  atomicAdd(&sdata[ 9*n_slices+s_i],     x_i*y_i);
  	  atomicAdd(&sdata[10*n_slices+s_i],    x_i*py_i);
  	  atomicAdd(&sdata[11*n_slices+s_i],   px_i*px_i);
  	  atomicAdd(&sdata[12*n_slices+s_i],    px_i*y_i);
  	  atomicAdd(&sdata[13*n_slices+s_i],   px_i*py_i);
  	  atomicAdd(&sdata[14*n_slices+s_i],     y_i*y_i);
  	  atomicAdd(&sdata[15*n_slices+s_i],    y_i*py_i);
  	  atomicAdd(&sdata[16*n_slices+s_i],   py_i*py_i);
        }
	__syncthreads();
        
	// write count and first and second order partial sums from shared to global mem
        for (int i=0; i<full_pass; i++){
              //printf("on block %d thread %d: adding in fullpass i:[%d/%d]: index=%d value=%g\n", blockIdx.x, tid, i+1, full_pass, i*blockDim.x + tid, sdata[i*blockDim.x + tid]);
              atomicAdd(&m_out[i*blockDim.x + tid], sdata[i*blockDim.x + tid]);
	}
	if (tid < residual){
              atomicAdd(&m_out[full_pass*blockDim.x + tid], sdata[full_pass*blockDim.x + tid]);
	}
	//printf("block %d, thread %d: m_out[33]: %g\n", blockIdx.x, tid, m_out[33]);
}
__global__ void compute_moments_2(double* m_out, const int n_slices, const int weight) {

	// for this n_slices threads are enough
	// one thread computes moments of one slice                                                                                                                     
        unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	// compute first and second moments in global memory (1 count + 6 1st order sums + 10 2nd order sums + 6 1st order moments + 10 2nd order moments = 33*n_slices)

        // first order moments E(xi)
    	m_out[17*n_slices + gid] = m_out[  n_slices + gid] / m_out[gid];  // E(x)
        m_out[18*n_slices + gid] = m_out[2*n_slices + gid] / m_out[gid];  // E(px)
        m_out[19*n_slices + gid] = m_out[3*n_slices + gid] / m_out[gid];  // E(y)
        m_out[20*n_slices + gid] = m_out[4*n_slices + gid] / m_out[gid];  // E(py)
        m_out[21*n_slices + gid] = m_out[5*n_slices + gid] / m_out[gid];  // E(zeta)
        m_out[22*n_slices + gid] = m_out[6*n_slices + gid] / m_out[gid];  // E(delta)

        // second order momemnts E(xi*xj)-E(xi)E(xj)
        m_out[23*n_slices + gid] = m_out[ 7*n_slices + gid] / m_out[gid] - m_out[17*n_slices + gid] * m_out[17*n_slices + gid];  // Sigma xx
        m_out[24*n_slices + gid] = m_out[ 8*n_slices + gid] / m_out[gid] - m_out[17*n_slices + gid] * m_out[18*n_slices + gid];  // Cov xpx
        m_out[25*n_slices + gid] = m_out[ 9*n_slices + gid] / m_out[gid] - m_out[17*n_slices + gid] * m_out[19*n_slices + gid];  // Cov xy
        m_out[26*n_slices + gid] = m_out[10*n_slices + gid] / m_out[gid] - m_out[17*n_slices + gid] * m_out[20*n_slices + gid];  // Cov xpy
        m_out[27*n_slices + gid] = m_out[11*n_slices + gid] / m_out[gid] - m_out[18*n_slices + gid] * m_out[18*n_slices + gid];  // Sigma pxpx
        m_out[28*n_slices + gid] = m_out[12*n_slices + gid] / m_out[gid] - m_out[18*n_slices + gid] * m_out[19*n_slices + gid];  // Cov pxy
        m_out[29*n_slices + gid] = m_out[13*n_slices + gid] / m_out[gid] - m_out[18*n_slices + gid] * m_out[20*n_slices + gid];  // Cov pxpy
        m_out[30*n_slices + gid] = m_out[14*n_slices + gid] / m_out[gid] - m_out[19*n_slices + gid] * m_out[19*n_slices + gid];  // Sigma yy
        m_out[31*n_slices + gid] = m_out[15*n_slices + gid] / m_out[gid] - m_out[19*n_slices + gid] * m_out[20*n_slices + gid];  // Cov ypy
        m_out[32*n_slices + gid] = m_out[16*n_slices + gid] / m_out[gid] - m_out[20*n_slices + gid] * m_out[20*n_slices + gid];  // Sigma pypy

	m_out[gid] *= weight;  // scale from macroparticle to real charge

}
}
