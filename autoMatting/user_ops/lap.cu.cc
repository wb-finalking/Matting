#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "matting.h"
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

__global__ void Kernel(double *dev_input, int *dev_rows, int *dev_cols, double *dev_vals,int* dev_height, int* dev_width)  
{  
	int h = blockId.x;
	int w = threadId.x;
	double epsilon = 0.00001;
	while(h < *dev_height)
	{
		while(w < *dev_width)
		{
			int tid = h * width + w;
			// Calculate win_inds
			Eigen::MatrixXd win_inds(1, 9);
			win_inds(0,0) = (*dev_height) * (w)   + (h)   + 1;
			win_inds(0,1) = (*dev_height) * (w)   + (h+1) + 1;
			win_inds(0,2) = (*dev_height) * (w)   + (h+2) + 1;
			win_inds(0,3) = (*dev_height) * (w+1) + (h)   + 1;
			win_inds(0,4) = (*dev_height) * (w+1) + (h+1) + 1;
			win_inds(0,5) = (*dev_height) * (w+1) + (h+2) + 1;
			win_inds(0,6) = (*dev_height) * (w+2) + (h)   + 1;
			win_inds(0,7) = (*dev_height) * (w+2) + (h+1) + 1;
			win_inds(0,8) = (*dev_height) * (w+2) + (h+2) + 1;
			
			// Calculate winI, which is a 9 by 3 matrix
			// The values on each row are values of a pixel on 3 channels
			// in the window whose center is (i, j)
			// Each colomn as one kind of color depth of winI
			Eigen::MatrixXd winI(9,3);
			winI(0,0) = dev_input[3*( h   *(*dev_width+2)+w)];
			winI(1,0) = dev_input[3*((h+1)*(*dev_width+2)+w)];
			winI(1,0) = dev_input[3*((h+2)*(*dev_width+2)+w)];
			winI(0,0) = dev_input[3*( h   *(*dev_width+2)+w+1)];
			winI(1,0) = dev_input[3*((h+1)*(*dev_width+2)+w+1)];
			winI(1,0) = dev_input[3*((h+2)*(*dev_width+2)+w+1)];
			winI(0,0) = dev_input[3*( h   *(*dev_width+2)+w+2)];
			winI(1,0) = dev_input[3*((h+1)*(*dev_width+2)+w+2)];
			winI(1,0) = dev_input[3*((h+2)*(*dev_width+2)+w+2)];
			
			winI(0,1) = dev_input[3*( h   *(*dev_width+2)+w)  +1];
			winI(1,1) = dev_input[3*((h+1)*(*dev_width+2)+w)  +1];
			winI(1,1) = dev_input[3*((h+2)*(*dev_width+2)+w)  +1];
			winI(0,1) = dev_input[3*( h   *(*dev_width+2)+w+1)+1];
			winI(1,1) = dev_input[3*((h+1)*(*dev_width+2)+w+1)+1];
			winI(1,1) = dev_input[3*((h+2)*(*dev_width+2)+w+1)+1];
			winI(0,1) = dev_input[3*( h   *(*dev_width+2)+w+2)+1];
			winI(1,1) = dev_input[3*((h+1)*(*dev_width+2)+w+2)+1];
			winI(1,1) = dev_input[3*((h+2)*(*dev_width+2)+w+2)+1];
			
			winI(0,2) = dev_input[3*( h   *(*dev_width+2)+w)  +2];
			winI(1,2) = dev_input[3*((h+1)*(*dev_width+2)+w)  +2];
			winI(1,2) = dev_input[3*((h+2)*(*dev_width+2)+w)  +2];
			winI(0,2) = dev_input[3*( h   *(*dev_width+2)+w+1)+2];
			winI(1,2) = dev_input[3*((h+1)*(*dev_width+2)+w+1)+2];
			winI(1,2) = dev_input[3*((h+2)*(*dev_width+2)+w+1)+2];
			winI(0,2) = dev_input[3*( h   *(*dev_width+2)+w+2)+2];
			winI(1,2) = dev_input[3*((h+1)*(*dev_width+2)+w+2)+2];
			winI(1,2) = dev_input[3*((h+2)*(*dev_width+2)+w+2)+2];
			
			// Calculate mean value of matrix, which is 3 by 1
			Eigen::MatrixXd win_mu(1,3);
			double sum1 = 0;
			double sum2 = 0;
			double sum3 = 0;

			for (int i = 0; i < 9; i++) {
				sum1 += winI(i,0);
				sum2 += winI(i,1);
				sum3 += winI(i,2);
			}

			win_mu(0,0) = sum1 / 9;
			win_mu(0,1) = sum2 / 9;
			win_mu(0,2) = sum3 / 9;
			
			// Calculate the covariance matrix
			// Cov = E(X^2) - E(X)^2
			Eigen::MatrixXd expection_xx = winI.transpose() * winI / 9;
			Eigen::MatrixXd expection_x = win_mu.transpose() * win_mu;
			Eigen::MatrixXd covariance = expection_xx - expection_x;

			// Calculate (Cov + epsilon / |Wk| * I_3)^(-1)
			Eigen::MatrixXd eye_c = Eigen::MatrixXd::Identity(3,3);
			Eigen::MatrixXd before_inv = covariance + epsilon / 9 * eye_c;
			Eigen::MatrixXd win_var = before_inv.inverse();

			// Calculate Ii - Mu_k and Ij - Mu_k, which are 9 by 3 matrix
			Eigen::MatrixXd IiMinusMuk = winI - win_mu.replicate(9,1);
			Eigen::MatrixXd IjMinusMuk = IiMinusMuk.transpose();

			// Calcualte the part on the right hand side of Kronecker delta
			// which is the matrix with size of 9 by 9, and then put them in one column
			Eigen::MatrixXd eyeMatrix = Eigen::MatrixXd::Identity(9, 9);
			Eigen::MatrixXd tvals = eyeMatrix - (1 + IiMinusMuk * win_var * IjMinusMuk) / 9;
			tvals = tvals.resize(1, 81);

			Eigen::MatrixXd repe_col = win_inds.transpose().replicate(1,9).resize(1,81);
			Eigen::MatrixXd repe_row = win_inds.replicate(9,1).resize(1, 81);
			Eigen::MatrixXd putInRow = tvals.reshape(1, 81);

			int start = (h * (*dev_width) + w) * 81;
			for (int i = 0; i < 81; i++) {
				dev_rows[start+i] = repe_row(0, i);
				dev_cols[start+i] = repe_col(0, i);
				dev_vals[start+i] = tvals(0, i);
			}

			
			w = w + 500;
		}
		h = h + 500;
	}	
	
}

extern "C"  
SpMat getLaplacianMatrix<GPUDevice>(const GPUDevice& d, int win_size, double epsilon, cv::Mat input)
{  
    double *dev_input = 0;  
    int *dev_rows = 0;  
    int *dev_cols = 0;
	double *dev_vals = 0;
	int dev_height, dev_width;
    cudaError_t cudaStatus;  
	
	int len = 0;
	//int win_size = 3;

	int neb_size = pow(win_size * 2 + 1, 2); // Size of window
	int neb_size_square = pow(neb_size, 2);
	int height = input.size().height;
	int width = input.size().width;
	int img_size = height * width;
	int tlen = (height - 2 * win_size) * (width - 2 * win_size);
	//tlen -= sum(consts_map_sub)[0];
	tlen *= neb_size_square;
	
	cv::Mat row_inds = cv::Mat::zeros(tlen, 1, CV_32S);
	cv::Mat col_inds = cv::Mat::zeros(tlen, 1, CV_32S);
	cv::Mat vals = cv::Mat::zeros(tlen, 1, CV_64F);
  
    // Allocate GPU buffers for three vectors (two input, one output)    .  
    cudaStatus = cudaMalloc((void**)&dev_input, img_size * 3 * sizeof(double));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }  
	cudaStatus = cudaMalloc((void**)&dev_rows, tlen * sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }
	cudaStatus = cudaMalloc((void**)&dev_cols, tlen * sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }
	cudaStatus = cudaMalloc((void**)&dev_vals, tlen * sizeof(double));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }
	cudaStatus = cudaMalloc((void**)&dev_height, sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }
	cudaStatus = cudaMalloc((void**)&dev_width,  sizeof(int));  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMalloc failed!");  
        goto Error;  
    }
  
  
    // Copy input vectors from host memory to GPU buffers.  
    cudaStatus = cudaMemcpy(dev_input, input.data(), size * 3 * sizeof(double), cudaMemcpyHostToDevice);  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaMemcpy failed!");  
        goto Error;  
    }
	int blockNum = height-2*win_size+1;
	int threadNum = width-2*win_size+1;
	cudaStatus = cudaMemcpy(dev_height, &blockNum, sizeof(int), cudaMemcpyHostToDevice); 
	cudaStatus = cudaMemcpy(dev_width, &threadNum, sizeof(int), cudaMemcpyHostToDevice); 
	
	// Launch a kernel on the GPU with one thread for each element.
	dim3 grid(500, 1, 1), block(500, 1, 1);
    Kernel<<<grid, block>>>(dev_c, dev_a, dev_b); 
  
    // Check for any errors launching the kernel  
    cudaStatus = cudaGetLastError();  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));  
        goto Error;  
    }  
      
    // cudaDeviceSynchronize waits for the kernel to finish, and returns  
    // any errors encountered during the launch.  
    cudaStatus = cudaDeviceSynchronize();  
    if (cudaStatus != cudaSuccess) {  
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);  
        goto Error;  
    }  
  
    // Copy output vector from GPU buffer to host memory.  
    cudaStatus = cudaMemcpy(row_inds.data(), dev_rows, tlen * sizeof(int), cudaMemcpyDeviceToHost);  
	cudaStatus = cudaMemcpy(col_inds.data(), dev_cols, tlen * sizeof(int), cudaMemcpyDeviceToHost);  
	cudaStatus = cudaMemcpy(vals.data(), dev_vals, tlen * sizeof(double), cudaMemcpyDeviceToHost); 
	
	// Convert row_inds, col_inds, vals to a tripletList
	// Then convert tripletList into a sparse matrix
	SpMat A(img_size, img_size);
	std::vector<Trip> tripletList;
	tripletList.reserve(len);
	for (int i = 0; i < len; i++) {
		tripletList.push_back(Trip(row_inds.at<int>(i, 0) - 1,col_inds.at<int>(i, 0) - 1,vals.at<double>(i, 0)));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());
	
Error:  
    cudaFree(dev_input);  
    cudaFree(dev_rows);  
    cudaFree(dev_cols);
	cudaFree(dev_vals);	
	cudaFree(dev_height);
	cudaFree(dev_width);
    
	//  A = A.transpose() * A;
    return A;  
}  