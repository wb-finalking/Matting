#define EIGEN_USE_MKL_ALL 

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>
#include <Eigen/PardisoSupport>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> Trip;
typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;
//using namespace Eigen;


/*
Status MattingShape(shape_inference::InferenceContext* c) {
  c->set_output(0, c->input(0));
  return Status::OK();
}
*/
	
REGISTER_OP("Matting")
    .Input("i_in: T")
	.Input("f_b_in: T")
	.Input("lambda_in: T")
    .Output("a_out: T")
    .Attr("T: {float, double}")
	.Attr(GetConvnetDataFormatAttrString());
/*	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });*/

template <typename Device, typename T>
class MattingOp : public OpKernel {
	public:
	explicit MattingOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {

		string data_format;
    	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    	OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                	errors::InvalidArgument("Invalid data format"));

	}

	void Compute(tensorflow::OpKernelContext* context) override
	{
	
		const Tensor& I = context->input(0);
		auto I_=I.flat<T>().data();
		const Tensor& FB = context->input(1);
		auto FB_=FB.flat<T>().data();
		//const Tensor& B = context->input(2);
		//auto B_=B.flat<T>().data();
		const Tensor& Lambda = context->input(2);
		auto Lambda_=Lambda.flat<T>();
		 
		//allocate output_tensor space
		const int32 batch_raw = GetTensorDim(I, data_format_, 'N');
   		OP_REQUIRES(context,FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                	errors::InvalidArgument("batch is too large"));
    	const int batch = static_cast<int>(batch_raw);

		const int32 input_rows_raw = GetTensorDim(I, data_format_, 'H');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input rows too large"));
		const int input_rows = static_cast<int>(input_rows_raw);

		const int32 input_cols_raw = GetTensorDim(I, data_format_, 'W');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input cols too large"));
		const int input_cols = static_cast<int>(input_cols_raw);
		
		TensorShape out_shape({batch, input_rows, input_cols});
        	//ShapeFromFormat(data_format_, batch, input_rows, input_cols, 1);
		tensorflow::Tensor* output_tensor = nullptr;
		//OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(), &output_tensor));
		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));

	
		//matting
		auto out=output_tensor->flat<T>().data();
		
		int step=input_cols*input_rows;
		//std::vector<cv::Mat> input_channels(3);
		//input_channels[0] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_));
		//input_channels[1] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_)+step);
		//input_channels[2] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_)+2*step);
		
		for(int i=0;i<batch;i++)
		{
			cv::Mat input(cv::Size(input_cols,input_rows),CV_32FC3,const_cast<T*>(I_)+step*i*3);
			cv::Mat input_fb(cv::Size(input_cols,input_rows),CV_32FC3,const_cast<T*>(FB_)+step*i*3);

			input.convertTo(input,CV_64FC3);	
			input_fb.convertTo(input_fb,CV_64FC3);	

			std::vector<cv::Mat> fb_channels(3);
			split(input_fb, fb_channels);
			//cv::Mat input_b(cv::Size(input_cols,input_rows),CV_64FC1,const_cast<T*>(B_)+step*i);
			T lambda =Lambda_(i);
			//input.convertTo(input,CV_8UC3);
			//lambda = 80;
			cv::Mat alpha=getAlpha(lambda, 1, 0.00001, input, fb_channels[0], fb_channels[2]);
			alpha.convertTo(alpha,CV_32FC1);				
			memcpy(out+step*i,alpha.data,sizeof(T)*alpha.rows*alpha.cols);

		}
		//merge(input_channels,input);

		//------- test
		/*
		cv::Mat output=input.clone();
		for(int i=20;i<100;i++)
			for(int j=0;j<I.dim_size(0);j++)
			{
				output.at<T>(i,j)=output.at<T>(i+20,j);
				//output.at<T>(i,j,1)=output.at<T>(i+20,j,0);
				//output.at<T>(i,j,2)=output.at<T>(i+20,j,0);
			}
		std::vector<cv::Mat> out_channels(3);
		split(output, out_channels);

		memcpy(out,out_channels[0].data,sizeof(float)*output.rows*output.cols);
		memcpy(out+step,out_channels[1].data,sizeof(float)*output.rows*output.cols);
		memcpy(out+2*step,out_channels[2].data,sizeof(float)*output.rows*output.cols);
		*/

		//cv::Mat alpha=performMatting(lambda, 1, 0.00001, input, input_m);		
		//cv::Mat alpha=input_m;
		//alpha=alpha*255;
		//alpha.convertTo(alpha,CV_8UC1);
		//cv::imwrite("tf_alpha.png", alpha);		
		
		//memcpy(out,alpha.data,sizeof(double)*alpha.rows*alpha.cols);
		//output.at<T>(0,0)=0;
 	}
	/*
	cv::Mat performMatting(double lambda, int win_size, double epsilon, cv::Mat input, cv::Mat input_m)
	{
		cv::Mat consts_map; // 0-1 values where 1 means pixel scribbled
		cv::Mat consts_vals; // The original value of scribbled pixel
		int height = input.size().height;
		int width = input.size().width;

		// Find the scribbled pixels
		consts_map=(input_m>0.98)+(input_m<0.02);
		consts_map=consts_map/255;

		// Calculate consts_vals
		consts_map.convertTo(consts_map,CV_64FC1);
		consts_vals = input_m;
		consts_vals = consts_vals.mul(consts_map);

		// Function to get Alpha by natural matting
		cv::Mat alpha = getAlpha(lambda, win_size, epsilon, input, consts_map, consts_vals);
		//cv::Mat alpha=consts_vals;
		//alpha.convertTo(alpha,CV_32FC1);

		return alpha;
	}
	*/
	cv::Mat getAlpha(T lambda, int win_size, double epsilon, cv::Mat input, cv::Mat input_f, cv::Mat input_b)
	{
		int height = input.size().height;
		int width = input.size().width;
		int img_size = height * width;

		// Solve the equation x = (A + lambda * D) \ (lambda * consts_vals(:));
		// To make it clear, let left * x = right
		// left = A + lambda * D and right = lambda * consts_vals(:)

		// Calculation of left side(A + lambda * D)
		SpMat A = getLaplacianMatrix(win_size, epsilon, input);
		std::cout << "getLaplacianMatrix..." << std::endl;


		cv::Mat input_f_trans = input_f.t();
		input_f_trans = input_f_trans.reshape(img_size, 1);
		cv::Mat input_b_trans = input_b.t();
		input_b_trans = input_b_trans.reshape(img_size, 1);
		SpMat D(img_size, img_size);
		for (int i = 0; i < img_size; i++) {
			D.coeffRef(i, i) = input_f_trans.at<double>(0, i) + input_b_trans.at<double>(0, i);
		}
		SpMat left = A + lambda * D;

		// Calculation of right side (lambda * consts_vals(:))
		//cv::Mat consts_vals_in_a_col;
		//cv::Mat transpo = consts_vals.t();
		//consts_vals_in_a_col = transpo.reshape(1, img_size);
		Eigen::VectorXd right(img_size);
		for (int i = 0; i < img_size; i++) {
			//right(i) = lambda * consts_vals_in_a_col.at<double>(i, 0);
			right(i) = lambda * input_f_trans.at<double>(0, i);
		}

		int* innerPointer = left.innerIndexPtr();
		int* outerPointer = left.outerIndexPtr();
		double* valuePointer = left.valuePtr();
		double* rightPointer = right.data();

		int size = input.size().width * input.size().height;

		//SparseMatrixEquationSolver sparseMatrixEquationSolver(outerPointer, innerPointer, valuePointer, rightPointer, size);
		//double* alphaArray = sparseMatrixEquationSolver.solveEquation();
		
		Eigen::VectorXd x;
		/*
		Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double> >  BCGST;
		BCGST.preconditioner().setDroptol(.00001);
		BCGST.compute(left);
		x = BCGST.solve(right);
		*/
	
		
		//Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
		//cg.compute(left);
		//x = cg.solve(right);
		
		
		Eigen::PardisoLU<SpMat> cg;
		cg.compute(left);
		x = cg.solve(right);
		//x = lambda * x;
	
		//Eigen::GMRES<SpMat, Eigen::IdentityPreconditioner> gmres;
		//gmres.compute(left);
		//x = gmres.solve(right);

		std::cout << "solve..." << std::endl;
		Eigen::Map<Eigen::MatrixXd> fullDmap(x.data(), input.size().width, input.size().height);
		cv::Mat alpha(fullDmap.rows(), fullDmap.cols(), CV_64FC1, fullDmap.data());
		alpha = alpha.t();
		//alpha = alpha * 255;
		//cv::imshow("Display window", alpha);
		//cv::waitKey(0);

		return alpha;
	}
	
	Eigen::SparseMatrix<double> getLaplacianMatrix(int win_size, double epsilon, cv::Mat input)
	{
		int len = 0;
		//int win_size = 3;
		cv::Mat repe_col;
		cv::Mat repe_row;

		int neb_size = pow(win_size * 2 + 1, 2); // Size of window
		int neb_size_square = pow(neb_size, 2);
		int height = input.size().height;
		int width = input.size().width;
		int img_size = height * width;

		// Store the index of M
		cv::Mat indsM = cv::Mat::zeros(height, width, CV_32S);
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				indsM.at<int>(j, i) = height * i + j + 1;
			}
		}

		// consts_map_sub = consts_map[win_size: height-(win_size+1), width-(winsize+1)]
		//cv::Mat consts_map_sub; // consts_map with margin excluded
		//consts_map_sub = consts_map.rowRange(win_size, height - (win_size + 1));
		//consts_map_sub = consts_map_sub.colRange(win_size, width - (win_size + 1));

		int tlen = (height - 2 * win_size) * (width - 2 * win_size);
		//tlen -= sum(consts_map_sub)[0];
		tlen *= neb_size_square;

		cv::Mat row_inds = cv::Mat::zeros(tlen, 1, CV_32S);
		cv::Mat col_inds = cv::Mat::zeros(tlen, 1, CV_32S);
		cv::Mat vals = cv::Mat::zeros(tlen, 1, CV_64F);

		// Iterate on all window center
		for (int j = win_size; j < width - win_size; j++) {
			for (int i = win_size; i < height - win_size; i++) {

				// Skip if the current pixel is scribbled
				//if ((int)consts_map.at<double>(i, j) == 1) {
				//	continue;
				//}

				// Calculate win_inds, which is a 1 by 9 matrix
				// The value is the index of all element in the window whose center is (i, j)
				cv::Mat win_inds = cv::Mat::zeros(1, 9, CV_64F);
				cv::Mat temp = indsM.rowRange(i - win_size, i + win_size + 1);
				temp = temp.colRange(j - win_size, j + win_size + 1);
				win_inds.at<double>(0, 0) = double(temp.at<int>(0, 0));
				win_inds.at<double>(0, 1) = double(temp.at<int>(1, 0));
				win_inds.at<double>(0, 2) = double(temp.at<int>(2, 0));
				win_inds.at<double>(0, 3) = double(temp.at<int>(0, 1));
				win_inds.at<double>(0, 4) = double(temp.at<int>(1, 1));
				win_inds.at<double>(0, 5) = double(temp.at<int>(2, 1));
				win_inds.at<double>(0, 6) = double(temp.at<int>(0, 2));
				win_inds.at<double>(0, 7) = double(temp.at<int>(1, 2));
				win_inds.at<double>(0, 8) = double(temp.at<int>(2, 2));

				// Calculate winI, which is a 9 by 3 matrix
				// The values on each row are values of a pixel on 3 channels
				// in the window whose center is (i, j)
				// Each colomn as one kind of color depth of winI
				cv::Mat winI = input.rowRange(i - win_size, i + win_size + 1);
				winI = winI.colRange(j - win_size, j + win_size + 1);
				cv::Mat winI_temp = cv::Mat::zeros(9, 3, CV_64F);
				std::vector<cv::Mat> channels(3);
				split(winI, channels);
				cv::Mat ch1 = channels[0];
				cv::Mat ch2 = channels[1];
				cv::Mat ch3 = channels[2];

				winI_temp.at<double>(0, 0) = ch1.at<double>(0, 0);
				winI_temp.at<double>(1, 0) = ch1.at<double>(1, 0);
				winI_temp.at<double>(2, 0) = ch1.at<double>(2, 0);
				winI_temp.at<double>(3, 0) = ch1.at<double>(0, 1);
				winI_temp.at<double>(4, 0) = ch1.at<double>(1, 1);
				winI_temp.at<double>(5, 0) = ch1.at<double>(2, 1);
				winI_temp.at<double>(6, 0) = ch1.at<double>(0, 2);
				winI_temp.at<double>(7, 0) = ch1.at<double>(1, 2);
				winI_temp.at<double>(8, 0) = ch1.at<double>(2, 2);

				winI_temp.at<double>(0, 1) = ch2.at<double>(0, 0);
				winI_temp.at<double>(1, 1) = ch2.at<double>(1, 0);
				winI_temp.at<double>(2, 1) = ch2.at<double>(2, 0);
				winI_temp.at<double>(3, 1) = ch2.at<double>(0, 1);
				winI_temp.at<double>(4, 1) = ch2.at<double>(1, 1);
				winI_temp.at<double>(5, 1) = ch2.at<double>(2, 1);
				winI_temp.at<double>(6, 1) = ch2.at<double>(0, 2);
				winI_temp.at<double>(7, 1) = ch2.at<double>(1, 2);
				winI_temp.at<double>(8, 1) = ch2.at<double>(2, 2);

				winI_temp.at<double>(0, 2) = ch3.at<double>(0, 0);
				winI_temp.at<double>(1, 2) = ch3.at<double>(1, 0);
				winI_temp.at<double>(2, 2) = ch3.at<double>(2, 0);
				winI_temp.at<double>(3, 2) = ch3.at<double>(0, 1);
				winI_temp.at<double>(4, 2) = ch3.at<double>(1, 1);
				winI_temp.at<double>(5, 2) = ch3.at<double>(2, 1);
				winI_temp.at<double>(6, 2) = ch3.at<double>(0, 2);
				winI_temp.at<double>(7, 2) = ch3.at<double>(1, 2);
				winI_temp.at<double>(8, 2) = ch3.at<double>(2, 2);

				winI = winI_temp / 255;
				//winI = winI_temp;

				// Calculate mean value of matrix, which is 3 by 1
				cv::Mat win_mu = cv::Mat::zeros(1, 3, CV_64F);
				double sum1 = 0;
				double sum2 = 0;
				double sum3 = 0;

				for (int i = 0; i < neb_size; i++) {
					sum1 += winI.at<double>(i, 0);
					sum2 += winI.at<double>(i, 1);
					sum3 += winI.at<double>(i, 2);
				}

				win_mu.at<double>(0, 0) = sum1 / neb_size;
				win_mu.at<double>(0, 1) = sum2 / neb_size;
				win_mu.at<double>(0, 2) = sum3 / neb_size;

				// Calculate the covariance matrix
				// Cov = E(X^2) - E(X)^2
				cv::Mat expection_xx = winI.t() * winI / neb_size;
				cv::Mat expection_x = win_mu.t() * win_mu;
				cv::Mat covariance = expection_xx - expection_x;

				// Calculate (Cov + epsilon / |Wk| * I_3)^(-1)
				cv::Mat eye_c = cv::Mat::eye(input.channels(), input.channels(), CV_64F);
				cv::Mat before_inv = covariance + epsilon / neb_size * eye_c;
				cv::Mat win_var = before_inv.inv();

				// Calculate Ii - Mu_k and Ij - Mu_k, which are 9 by 3 matrix
				cv::Mat IiMinusMuk = winI - repeat(win_mu, neb_size, 1);
				cv::Mat IjMinusMuk = IiMinusMuk.t();

				// Calcualte the part on the right hand side of Kronecker delta
				// which is the matrix with size of 9 by 9, and then put them in one column
				cv::Mat eyeMatrix = cv::Mat::eye(neb_size, neb_size, CV_64F);
				cv::Mat tvals = eyeMatrix - (1 + IiMinusMuk * win_var * IjMinusMuk) / neb_size;
				tvals = tvals.reshape(0, neb_size_square);

				repe_col = repeat(win_inds.t(), 1, neb_size).reshape(0, neb_size_square);
				repe_row = repeat(win_inds, neb_size, 1).reshape(0, neb_size_square);
				cv::Mat putInRow = tvals.reshape(0, neb_size_square);

				for (int i = len; i < neb_size_square + len; i++) {
					row_inds.at<int>(i, 0) = repe_row.at<double>(i - len, 0);
					col_inds.at<int>(i, 0) = repe_col.at<double>(i - len, 0);
					vals.at<double>(i, 0) = tvals.at<double>(i - len, 0);
				}

				len = len + neb_size_square;
			}
		}

		// Convert row_inds, col_inds, vals to a tripletList
		// Then convert tripletList into a sparse matrix
		SpMat A(img_size, img_size);
		std::vector<Trip> tripletList;
		tripletList.reserve(len);
		for (int i = 0; i < len; i++) {
			tripletList.push_back(Trip(row_inds.at<int>(i, 0) - 1,col_inds.at<int>(i, 0) - 1,vals.at<double>(i, 0)));
		}
		A.setFromTriplets(tripletList.begin(), tripletList.end());

		//  A = A.transpose() * A;
		return A;
	}

private:
	TensorFormat data_format_;
	
};

//REGISTER_KERNEL_BUILDER(Name("Matting").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"), MattingOp<CPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("Matting").Device(DEVICE_CPU), MattingOp);

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Matting").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MattingOp<CPUDevice, T>);
	
REGISTER_CPU(float)
//REGISTER_CPU(float)

//---------------------------- grad---------------------------
REGISTER_OP("MattingGrad")
    .Input("i_in: T")
	.Input("f_b_in: T")
	.Input("lamb_in: T")
	.Input("grad_in: T")
	.Output("f_b_grad_out: T")
	.Output("lamb_grad_out: T")
    .Attr("T: {float, double}")
	.Attr(GetConvnetDataFormatAttrString());
/*	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });*/


template <typename Device, typename T>
class MattingGradOp : public OpKernel {
	public:
	explicit MattingGradOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {

		string data_format;
    	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    	OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                	errors::InvalidArgument("Invalid data format"));

	}

	void Compute(tensorflow::OpKernelContext* context) override
	{
	
		const Tensor& I = context->input(0);
		auto I_=I.flat<T>().data();
		const Tensor& FB = context->input(1);
		auto FB_=FB.flat<T>().data();
		//const Tensor& B = context->input(2);
		//auto B_=B.flat<T>().data();
		const Tensor& Lambda = context->input(2);
		auto Lambda_=Lambda.flat<T>();
		const Tensor& Grad_in = context->input(3);
		auto Grad_in_=Grad_in.flat<T>().data();
		//std::cout<<Grad_in.shape()<<std::endl;
		
		//std::cout << Grad_in.dim_size(0)<<" "<<
			//<<Grad_in.dim_size(1)<<" "<<
			//<<Grad_in.dim_size(2)<<std::endl;
		//allocate output_tensor space
		const int32 batch_raw = GetTensorDim(I, data_format_, 'N');
   		OP_REQUIRES(context,FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                	errors::InvalidArgument("batch is too large"));
    	const int batch = static_cast<int>(batch_raw);

		const int32 input_rows_raw = GetTensorDim(I, data_format_, 'H');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input rows too large"));
		const int input_rows = static_cast<int>(input_rows_raw);

		const int32 input_cols_raw = GetTensorDim(I, data_format_, 'W');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input cols too large"));
		const int input_cols = static_cast<int>(input_cols_raw);
		
		TensorShape out_shape({batch, input_rows, input_cols, 3});
        	//ShapeFromFormat(data_format_, batch, input_rows, input_cols, 1);
		tensorflow::Tensor* FB_output_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &FB_output_tensor));
		//tensorflow::Tensor* B_output_tensor = nullptr;
		//OP_REQUIRES_OK(context, context->allocate_output(1, out_shape, &B_output_tensor));
		
		TensorShape out_shape2({1});
        	//ShapeFromFormat(data_format_, batch, 1, 1, 1);
		tensorflow::Tensor* Lamb_output_tensor = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(1, out_shape2, &Lamb_output_tensor));

	
		//matting
		auto FB_output=FB_output_tensor->flat<T>().data();
		//auto B_output=B_output_tensor->flat<T>().data();
		auto Lamb_output=Lamb_output_tensor->flat<T>();
		
		int step=input_cols*input_rows;
		//std::vector<cv::Mat> input_channels(3);
		//input_channels[0] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_));
		//input_channels[1] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_)+step);
		//input_channels[2] = cv::Mat(cv::Size(I.dim_size(0),I.dim_size(1)),CV_32FC1,const_cast<T*>(I_)+2*step);
		T sum_Lamb_grad=0;
		for(int i=0;i<batch;i++)
		{
			cv::Mat input(cv::Size(input_cols,input_rows),CV_32FC3,const_cast<T*>(I_)+step*i*3);
			cv::Mat input_fb(cv::Size(input_cols,input_rows),CV_32FC3,const_cast<T*>(FB_)+step*i*3);
			//cv::Mat input_b(cv::Size(input_cols,input_rows),CV_64FC1,const_cast<T*>(B_)+step*i);
			cv::Mat grad(cv::Size(input_cols,input_rows),CV_32FC1,const_cast<T*>(Grad_in_)+step*i);
			T lambda =Lambda_(0);

			input.convertTo(input,CV_64FC3);	
			input_fb.convertTo(input_fb,CV_64FC3);
			grad.convertTo(grad,CV_64FC1);
			
			std::vector<cv::Mat> fb_channels(3);
			split(input_fb, fb_channels);
			std::cout<<"input_fb:"<<input_fb.rowRange(0,10)<<std::endl;

			cv::Mat F_grad;
			cv::Mat B_grad;
			cv::Mat FB_grad = cv::Mat(cv::Size(input_cols,input_rows),CV_64FC3,cv::Scalar(0));;
			std::vector<cv::Mat> grad_channels(3);
			T Lamb_grad;
			
			//const int size0=B.dim_size(0);
			//const int size1=B.dim_size(1);
			//const int size2=B.dim_size(2);
			//std::cout << size0<<size1<<size2<<std::endl;
			//F_grad = cv::Mat(cv::Size(input_cols,input_rows),CV_64FC1,cv::Scalar(0));
			//B_grad = cv::Mat(cv::Size(input_cols,input_rows),CV_64FC1,cv::Scalar(0));
			//Lamb_grad=0;
			//lambda = 80;
			getGrad(input, fb_channels[0], fb_channels[2], lambda, grad, grad_channels[0], grad_channels[2], Lamb_grad);
			grad_channels[1] = cv::Mat(cv::Size(input_cols,input_rows),CV_64FC1,cv::Scalar(0));
			merge(grad_channels,FB_grad);
			std::cout<<"grad_FB:"<<FB_grad.rowRange(0,10)<<std::endl;
			//std::cout<<"fb_channels:"<<fb_channels[0].rowRange(0,10)<<std::endl;

			FB_grad.convertTo(FB_grad,CV_32FC3);
			memcpy(FB_output+step*i*3,FB_grad.data,sizeof(T)*step*3);
			//memcpy(B_output+step*i,B_grad.data,sizeof(T)*step);
			sum_Lamb_grad += Lamb_grad;
			//Lamb_output(i) = Lamb_grad;
		}
		Lamb_output(0) = sum_Lamb_grad/batch;
 	}

	void getGrad(cv::Mat input, cv::Mat input_f, cv::Mat input_b, T lambda, cv::Mat grad, cv::Mat& F_grad, cv::Mat& B_grad, T& Lamb_grad)
	{
		int height = input.size().height;
		int width = input.size().width;
		int img_size = height * width;
	
		cv::Mat grad_trans = grad.t();
		grad_trans = grad_trans.reshape(img_size, 1);
		Eigen::VectorXd grad_v(img_size);
		for (int i = 0; i < img_size; i++) {
			grad_v(i) = grad_trans.at<double>(0, i);
		}

		SpMat Laplace = getLaplacianMatrix(1, 0.00001, input);

		
		cv::Mat input_f_trans = input_f.t();
		input_f_trans = input_f_trans.reshape(img_size, 1);
		SpMat F_diag(img_size, img_size);
		for (int i = 0; i < img_size; i++) {
			F_diag.coeffRef(i, i) = input_f_trans.at<double>(0, i);
		}

		cv::Mat input_b_trans = input_b.t();
		input_b_trans = input_b_trans.reshape(img_size, 1);
		SpMat B_diag(img_size, img_size);
		for (int i = 0; i < img_size; i++) {
			B_diag.coeffRef(i, i) = input_b_trans.at<double>(0, i);
		}

		SpMat D = lambda * F_diag + lambda * B_diag + Laplace;
		
		Eigen::VectorXd x;
		Eigen::VectorXd right(img_size);
		
		Eigen::PardisoLU<SpMat> cg;
		cg.compute(D);
		
		// df/dB
		for (int i = 0; i < img_size; i++) {
			right(i) = input_f_trans.at<double>(0, i);
		}
		x = cg.solve(right);
		
		SpMat tmp(img_size, img_size);
		for (int i = 0; i < img_size; i++) {
			tmp.coeffRef(i, i) = x[i];
		}
		right = tmp * grad_v;
		x = cg.solve(x);
		x = -lambda * lambda * x;
		//x = x.dot()

		Eigen::Map<Eigen::MatrixXd> fullDmap(x.data(), input.size().width, input.size().height);
		B_grad = cv::Mat(fullDmap.rows(), fullDmap.cols(), CV_64FC1, fullDmap.data());
		B_grad = B_grad.t();
		

		// df/dF
		x = cg.solve(grad_v);
		x = x * lambda;
		fullDmap = Eigen::Map<Eigen::MatrixXd>(x.data(), input.size().width, input.size().height);
		F_grad = cv::Mat(fullDmap.rows(), fullDmap.cols(), CV_64FC1, fullDmap.data());
		F_grad = F_grad.t();
		//std::cout<<"F_grad:"<<F_grad.rowRange(0,10)<<std::endl;
		F_grad = F_grad + B_grad;

		// df/dlamb
		for (int i = 0; i < img_size; i++) {
			right(i) = input_f_trans.at<double>(0, i);
		}
		x = cg.solve(right);
		right = (F_diag + B_diag) * x;
		x = cg.solve(right);
		x = -lambda * x;
	
		for (int i = 0; i < img_size; i++) {
			right(i) = input_f_trans.at<double>(0, i);
		}
		x = x + cg.solve(right);
		Lamb_grad = x.dot(grad_v);//img_size;
		//std::cout<<"lambda_grad: "<<Lamb_grad<<std::endl;
		//std::cout<<"grad: "<<grad.rowRange(0,10)<<std::endl;
	}

	Eigen::SparseMatrix<double> getLaplacianMatrix(int win_size, double epsilon, cv::Mat input)
	{
		int len = 0;
		//int win_size = 3;
		cv::Mat repe_col;
		cv::Mat repe_row;

		int neb_size = pow(win_size * 2 + 1, 2); // Size of window
		int neb_size_square = pow(neb_size, 2);
		int height = input.size().height;
		int width = input.size().width;
		int img_size = height * width;

		// Store the index of M
		cv::Mat indsM = cv::Mat::zeros(height, width, CV_32S);
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				indsM.at<int>(j, i) = height * i + j + 1;
			}
		}

		// consts_map_sub = consts_map[win_size: height-(win_size+1), width-(winsize+1)]
		//cv::Mat consts_map_sub; // consts_map with margin excluded
		//consts_map_sub = consts_map.rowRange(win_size, height - (win_size + 1));
		//consts_map_sub = consts_map_sub.colRange(win_size, width - (win_size + 1));

		int tlen = (height - 2 * win_size) * (width - 2 * win_size);
		//tlen -= sum(consts_map_sub)[0];
		tlen *= neb_size_square;

		cv::Mat row_inds = cv::Mat::zeros(tlen, 1, CV_32S);
		cv::Mat col_inds = cv::Mat::zeros(tlen, 1, CV_32S);
		cv::Mat vals = cv::Mat::zeros(tlen, 1, CV_64F);

		// Iterate on all window center
		for (int j = win_size; j < width - win_size; j++) {
			for (int i = win_size; i < height - win_size; i++) {

				// Calculate win_inds, which is a 1 by 9 matrix
				// The value is the index of all element in the window whose center is (i, j)
				cv::Mat win_inds = cv::Mat::zeros(1, 9, CV_64F);
				cv::Mat temp = indsM.rowRange(i - win_size, i + win_size + 1);
				temp = temp.colRange(j - win_size, j + win_size + 1);
				win_inds.at<double>(0, 0) = double(temp.at<int>(0, 0));
				win_inds.at<double>(0, 1) = double(temp.at<int>(1, 0));
				win_inds.at<double>(0, 2) = double(temp.at<int>(2, 0));
				win_inds.at<double>(0, 3) = double(temp.at<int>(0, 1));
				win_inds.at<double>(0, 4) = double(temp.at<int>(1, 1));
				win_inds.at<double>(0, 5) = double(temp.at<int>(2, 1));
				win_inds.at<double>(0, 6) = double(temp.at<int>(0, 2));
				win_inds.at<double>(0, 7) = double(temp.at<int>(1, 2));
				win_inds.at<double>(0, 8) = double(temp.at<int>(2, 2));

				// Calculate winI, which is a 9 by 3 matrix
				// The values on each row are values of a pixel on 3 channels
				// in the window whose center is (i, j)
				// Each colomn as one kind of color depth of winI
				cv::Mat winI = input.rowRange(i - win_size, i + win_size + 1);
				winI = winI.colRange(j - win_size, j + win_size + 1);
				cv::Mat winI_temp = cv::Mat::zeros(9, 3, CV_64F);
				std::vector<cv::Mat> channels(3);
				split(winI, channels);
				cv::Mat ch1 = channels[0];
				cv::Mat ch2 = channels[1];
				cv::Mat ch3 = channels[2];

				winI_temp.at<double>(0, 0) = ch1.at<double>(0, 0);
				winI_temp.at<double>(1, 0) = ch1.at<double>(1, 0);
				winI_temp.at<double>(2, 0) = ch1.at<double>(2, 0);
				winI_temp.at<double>(3, 0) = ch1.at<double>(0, 1);
				winI_temp.at<double>(4, 0) = ch1.at<double>(1, 1);
				winI_temp.at<double>(5, 0) = ch1.at<double>(2, 1);
				winI_temp.at<double>(6, 0) = ch1.at<double>(0, 2);
				winI_temp.at<double>(7, 0) = ch1.at<double>(1, 2);
				winI_temp.at<double>(8, 0) = ch1.at<double>(2, 2);

				winI_temp.at<double>(0, 1) = ch2.at<double>(0, 0);
				winI_temp.at<double>(1, 1) = ch2.at<double>(1, 0);
				winI_temp.at<double>(2, 1) = ch2.at<double>(2, 0);
				winI_temp.at<double>(3, 1) = ch2.at<double>(0, 1);
				winI_temp.at<double>(4, 1) = ch2.at<double>(1, 1);
				winI_temp.at<double>(5, 1) = ch2.at<double>(2, 1);
				winI_temp.at<double>(6, 1) = ch2.at<double>(0, 2);
				winI_temp.at<double>(7, 1) = ch2.at<double>(1, 2);
				winI_temp.at<double>(8, 1) = ch2.at<double>(2, 2);

				winI_temp.at<double>(0, 2) = ch3.at<double>(0, 0);
				winI_temp.at<double>(1, 2) = ch3.at<double>(1, 0);
				winI_temp.at<double>(2, 2) = ch3.at<double>(2, 0);
				winI_temp.at<double>(3, 2) = ch3.at<double>(0, 1);
				winI_temp.at<double>(4, 2) = ch3.at<double>(1, 1);
				winI_temp.at<double>(5, 2) = ch3.at<double>(2, 1);
				winI_temp.at<double>(6, 2) = ch3.at<double>(0, 2);
				winI_temp.at<double>(7, 2) = ch3.at<double>(1, 2);
				winI_temp.at<double>(8, 2) = ch3.at<double>(2, 2);

				winI = winI_temp / 255;
				//winI = winI_temp;

				// Calculate mean value of matrix, which is 3 by 1
				cv::Mat win_mu = cv::Mat::zeros(1, 3, CV_64F);
				double sum1 = 0;
				double sum2 = 0;
				double sum3 = 0;

				for (int i = 0; i < neb_size; i++) {
					sum1 += winI.at<double>(i, 0);
					sum2 += winI.at<double>(i, 1);
					sum3 += winI.at<double>(i, 2);
				}

				win_mu.at<double>(0, 0) = sum1 / neb_size;
				win_mu.at<double>(0, 1) = sum2 / neb_size;
				win_mu.at<double>(0, 2) = sum3 / neb_size;

				// Calculate the covariance matrix
				// Cov = E(X^2) - E(X)^2
				cv::Mat expection_xx = winI.t() * winI / neb_size;
				cv::Mat expection_x = win_mu.t() * win_mu;
				cv::Mat covariance = expection_xx - expection_x;

				// Calculate (Cov + epsilon / |Wk| * I_3)^(-1)
				cv::Mat eye_c = cv::Mat::eye(input.channels(), input.channels(), CV_64F);
				cv::Mat before_inv = covariance + epsilon / neb_size * eye_c;
				cv::Mat win_var = before_inv.inv();

				// Calculate Ii - Mu_k and Ij - Mu_k, which are 9 by 3 matrix
				cv::Mat IiMinusMuk = winI - repeat(win_mu, neb_size, 1);
				cv::Mat IjMinusMuk = IiMinusMuk.t();

				// Calcualte the part on the right hand side of Kronecker delta
				// which is the matrix with size of 9 by 9, and then put them in one column
				cv::Mat eyeMatrix = cv::Mat::eye(neb_size, neb_size, CV_64F);
				cv::Mat tvals = eyeMatrix - (1 + IiMinusMuk * win_var * IjMinusMuk) / neb_size;
				tvals = tvals.reshape(0, neb_size_square);

				repe_col = repeat(win_inds.t(), 1, neb_size).reshape(0, neb_size_square);
				repe_row = repeat(win_inds, neb_size, 1).reshape(0, neb_size_square);
				cv::Mat putInRow = tvals.reshape(0, neb_size_square);

				for (int i = len; i < neb_size_square + len; i++) {
					row_inds.at<int>(i, 0) = repe_row.at<double>(i - len, 0);
					col_inds.at<int>(i, 0) = repe_col.at<double>(i - len, 0);
					vals.at<double>(i, 0) = tvals.at<double>(i - len, 0);
				}

				len = len + neb_size_square;
			}
		}

		// Convert row_inds, col_inds, vals to a tripletList
		// Then convert tripletList into a sparse matrix
		SpMat A(img_size, img_size);
		std::vector<Trip> tripletList;
		tripletList.reserve(len);
		for (int i = 0; i < len; i++) {
			tripletList.push_back(Trip(row_inds.at<int>(i, 0) - 1,col_inds.at<int>(i, 0) - 1,vals.at<double>(i, 0)));
		}
		A.setFromTriplets(tripletList.begin(), tripletList.end());

		//  A = A.transpose() * A;
		return A;
	}

private:
	TensorFormat data_format_;
	
};

//REGISTER_KERNEL_BUILDER(Name("Matting").Device(tensorflow::DEVICE_CPU).TypeConstraint<float>("T"), MattingOp<CPUDevice, float>);
//REGISTER_KERNEL_BUILDER(Name("Matting").Device(DEVICE_CPU), MattingOp);

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("MattingGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MattingGradOp<CPUDevice, T>);
	
REGISTER_CPU(float)
//REGISTER_CPU(float)
