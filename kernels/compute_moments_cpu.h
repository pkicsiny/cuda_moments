////////////////////////////////////////
// this file is credited to X. Buffat //
////////////////////////////////////////

void compute_slice_moments(double* x, double* px, double* y, double* py, double* zeta, double* delta, int64_t* particles_slice, double* moments, int n_part, int n_slices, int64_t* weight) {
    int n_first_moments = 7;
    int n_second_moments = 10;
    int n_moments = n_first_moments+n_second_moments;

    // init results array
    for(int i = 0;i<n_slices*n_moments;++i) {
        moments[i] = 0.0;
    }

    //// first order moments ////

    // this array is initted on every thread
    double tmpSliceM[n_slices*n_first_moments];
    for(int i = 0;i<n_slices*n_first_moments;++i) {
        tmpSliceM[i] = 0.0;
    }

    // sum of data chunk on each thread for every var
    for(int i = 0;i<n_part;++i) {
        int i_slice = particles_slice[i];
        if(i_slice >= 0 && i_slice < n_slices){
            tmpSliceM[i_slice] += 1.0;
            tmpSliceM[n_slices + i_slice] += x[i];
            tmpSliceM[2*n_slices + i_slice] += px[i];
            tmpSliceM[3*n_slices + i_slice] += y[i];
            tmpSliceM[4*n_slices + i_slice] += py[i];
            tmpSliceM[5*n_slices + i_slice] += zeta[i];
            tmpSliceM[6*n_slices + i_slice] += delta[i];
        }
    }

    // reduction into results array, this is not possible on the GPU
    for (int i_slice = 0;i_slice<n_slices;++i_slice) {
        for(int j = 0;j<n_first_moments;++j){
            moments[j*n_slices + i_slice] += tmpSliceM[j*n_slices + i_slice];
        }
    }

    // mean
    for (int i_slice = 0;i_slice<n_slices;++i_slice) {
        for(int j = 1;j<n_first_moments;++j){
            moments[j*n_slices+i_slice] /= moments[i_slice];
        }
    }
    //////// second order moments ///////////////

    // result array on every thread
    double tmpSliceM2[n_slices*n_second_moments];
    for(int i = 0;i<n_slices*n_second_moments;++i) {
        tmpSliceM2[i] = 0.0;
    }

    // sum of data chunk on each thread for every var
    for(int i = 0;i<n_part;++i) {
        int i_slice = particles_slice[i];
        if(i_slice >=0 && i_slice < n_slices){
            tmpSliceM2[i_slice] += x[i]*x[i]; //Sigma_11
            tmpSliceM2[n_slices + i_slice] += x[i]*px[i]; //Sigma_12
            tmpSliceM2[2*n_slices + i_slice] += x[i]*y[i]; //Sigma_13
            tmpSliceM2[3*n_slices + i_slice] += x[i]*py[i]; //Sigma_14
            tmpSliceM2[4*n_slices + i_slice] += px[i]*px[i]; //Sigma_22
            tmpSliceM2[5*n_slices + i_slice] += px[i]*y[i]; //Sigma_23
            tmpSliceM2[6*n_slices + i_slice] += px[i]*py[i]; //Sigma_24
            tmpSliceM2[7*n_slices + i_slice] += y[i]*y[i]; //Sigma_33
            tmpSliceM2[8*n_slices + i_slice] += y[i]*py[i]; //Sigma_34
            tmpSliceM2[9*n_slices + i_slice] += py[i]*py[i]; //Sigma_44
        }
    }

    // reduction into results array, this is not possible on the GPU
    for (int i_slice = 0;i_slice<n_slices;++i_slice) {
        for(int j = 0;j<n_second_moments;++j) {
            moments[(n_first_moments+j)*n_slices+i_slice] += tmpSliceM2[j*n_slices+i_slice];
        }
    }

    for(int i_slice = 0;i_slice<n_slices;++i_slice) {
     
	// E(xy)
	for(int j = n_first_moments;j<n_moments;++j){
            moments[j*n_slices+i_slice] /= moments[i_slice];
        }

	// - E(x)E(y)
        moments[7*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[n_slices+i_slice]; //Sigma_11
        moments[8*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[2*n_slices+i_slice]; //Sigma_12
        moments[9*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_13
        moments[10*n_slices + i_slice] -= moments[n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_14
        moments[11*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[2*n_slices+i_slice]; //Sigma_22
        moments[12*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_23
        moments[13*n_slices + i_slice] -= moments[2*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_24
        moments[14*n_slices + i_slice] -= moments[3*n_slices+i_slice]*moments[3*n_slices+i_slice]; //Sigma_33
        moments[15*n_slices + i_slice] -= moments[3*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_34
        moments[16*n_slices + i_slice] -= moments[4*n_slices+i_slice]*moments[4*n_slices+i_slice]; //Sigma_44
	moments[i_slice] *= weight[0];  // added to scale num_macroparts_per_slice to real charge
    }
}
