#define EIGEN_USE_MKL_ALL 

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

typedef Eigen::ThreadPoolDevice CPUDevice;

using namespace tensorflow;

REGISTER_OP("MattingLoss")
    .Input("alpha_in: T")
	.Input("ground_in: T")
    .Output("a_out: T")
    .Attr("T: {float, double}")
	.Attr(GetConvnetDataFormatAttrString());


template <typename Device, typename T>
class MattingLossOp : public OpKernel {
	public:
	explicit MattingLossOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {

		string data_format;
    	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    	OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                	errors::InvalidArgument("Invalid data format"));

	}

	void Compute(tensorflow::OpKernelContext* context) override
	{
	
		const Tensor& A = context->input(0);
		auto A_=A.flat<T>().data();
		const Tensor& G = context->input(1);
		auto G_=G.flat<T>().data();
		 
		//allocate output_tensor space
		const int32 batch_raw = GetTensorDim(A, data_format_, 'N');
   		OP_REQUIRES(context,FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                	errors::InvalidArgument("batch is too large"));
    	const int batch = static_cast<int>(batch_raw);

		const int32 input_rows_raw = GetTensorDim(A, data_format_, 'H');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input rows too large"));
		const int input_rows = static_cast<int>(input_rows_raw);

		const int32 input_cols_raw = GetTensorDim(A, data_format_, 'W');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input cols too large"));
		const int input_cols = static_cast<int>(input_cols_raw);
		
		//std::vector<int> v(1);
		//v.push_back(batch);
		//gtl::ArraySlice<int> dim_sizes(v);
		//TensorShape out_shape(dim_sizes);
		TensorShape out_shape({batch});
		//TensorShape out_shape =
        	//ShapeFromFormat(data_format_, batch, -1, -1, -1);
		tensorflow::Tensor* output_tensor = nullptr;
		//OP_REQUIRES_OK(context, context->allocate_output(0, tensorflow::TensorShape(), &output_tensor));
		OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));

	
		// get loss
		auto out=output_tensor->flat<T>();
		
		int step=input_cols*input_rows;
		
		for(int i=0;i<batch;i++)
		{
			cv::Mat alpha(cv::Size(input_cols,input_rows),CV_32FC1,const_cast<T*>(A_)+step*i);
			cv::Mat ground(cv::Size(input_cols,input_rows),CV_32FC1,const_cast<T*>(G_)+step*i);			
			
			T loss=getLoss(alpha, ground);
			out(i) = loss;		
			//memcpy(out+i,&loss,sizeof(T));

		}

	}

	T getLoss(cv::Mat& alpha, cv::Mat& ground)
	{
		int height = alpha.size().height;
		int width = alpha.size().width;
		int img_size = height * width;
		
		//get weight mat
		cv::Mat W(cv::Size(ground.size().height,ground.size().width),CV_32FC1,cv::Scalar(0));

		cv::Mat tmp1 = ground > 0.98;
		tmp1 = tmp1 / 255;
		tmp1.convertTo(tmp1,CV_32FC1);	
		T w1 = cv::sum(tmp1)[0] / img_size;
		//tmp1 = tmp1 * w1;
		W = W + tmp1 * w1;

		cv::Mat tmp0 = ground < 0.02;
		tmp0 = tmp0 / 255;
		tmp0.convertTo(tmp0,CV_32FC1);
		T w0 = cv::sum(tmp0)[0] / img_size;
		//tmp0 = tmp0 * w0;
		W = W + tmp0 * w0;

		cv::Mat tmp = 1 - (tmp0 + tmp1);
		T w = 1 - w1 - w0;
		//tmp = tmp * w0;
		W = W + tmp * w;		

		//get loss
		cv::Mat d=(alpha - ground).mul(W);
		float num = img_size;
		T loss = cv::trace(d.t() * d)[0];
	
		return loss;
	}

private:
	TensorFormat data_format_;
	
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("MattingLoss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MattingLossOp<CPUDevice, T>);
	
REGISTER_CPU(float)

REGISTER_OP("MattingLossGrad")
    .Input("alpha_in: T")
	.Input("ground_in: T")
	.Input("grad_in: T")
    .Output("grad_out: T")
    .Attr("T: {float, double}")
	.Attr(GetConvnetDataFormatAttrString());


template <typename Device, typename T>
class MattingLossGradOp : public OpKernel {
	public:
	explicit MattingLossGradOp(tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {

		string data_format;
    	OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    	OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                	errors::InvalidArgument("Invalid data format"));

	}

	void Compute(tensorflow::OpKernelContext* context) override
	{
	
		const Tensor& A = context->input(0);
		auto A_=A.flat<T>().data();
		const Tensor& G = context->input(1);
		auto G_=G.flat<T>().data();
		const Tensor& Grad_in = context->input(2);
		auto Grad_in_=Grad_in.flat<T>();
		 
		//allocate output_tensor space
		const int32 batch_raw = GetTensorDim(A, data_format_, 'N');
   		OP_REQUIRES(context,FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                	errors::InvalidArgument("batch is too large"));
    	const int batch = static_cast<int>(batch_raw);

		const int32 input_rows_raw = GetTensorDim(A, data_format_, 'H');
		OP_REQUIRES(
		    context,
		    FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
		    errors::InvalidArgument("Input rows too large"));
		const int input_rows = static_cast<int>(input_rows_raw);

		const int32 input_cols_raw = GetTensorDim(A, data_format_, 'W');
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

	
		// get loss
		auto out=output_tensor->flat<T>().data();
		
		int step=input_cols*input_rows;
		
		for(int i=0;i<batch;i++)
		{
			cv::Mat alpha(cv::Size(input_cols,input_rows),CV_32FC1,const_cast<T*>(A_)+step*i);
			cv::Mat ground(cv::Size(input_cols,input_rows),CV_32FC1,const_cast<T*>(G_)+step*i);	
			T grad_in = Grad_in_(i);
			
			cv::Mat grad_out=getGrad(alpha, ground, grad_in);
			//std::cout<<"grad_out:"<<grad_out.rowRange(0,10)<<std::endl;
			grad_out = grad_out * grad_in;
			
			//std::cout<<"grad_in:"<<grad_in<<std::endl;
			memcpy(out+step*i,grad_out.data,sizeof(T)*step);

		}

	}

	cv::Mat getGrad(cv::Mat& alpha, cv::Mat& ground, T grad_in)
	{
		int height = alpha.size().height;
		int width = alpha.size().width;
		int img_size = height * width;
		
		//get weight mat
		cv::Mat W(cv::Size(ground.size().width,ground.size().height),CV_32FC1,cv::Scalar(0));

		cv::Mat tmp1 = ground > 0.98;
		tmp1 = tmp1 / 255;
		tmp1.convertTo(tmp1,CV_32FC1);	
		T w1 = cv::sum(tmp1)[0] / img_size;
		//tmp1 = tmp1 * w1;
		W = W + tmp1 * w1 * w1;

		cv::Mat tmp0 = ground < 0.02;
		tmp0 = tmp0 / 255;
		tmp0.convertTo(tmp0,CV_32FC1);	
		T w0 = cv::sum(tmp0)[0] / img_size;
		//tmp0 = tmp0 * w0;
		W = W + tmp0 * w0 * w0;

		cv::Mat tmp = 1 - (tmp0 + tmp1);
		T w = 1 - w1 - w0;
		//tmp = tmp * w0;
		W = W + tmp * w * w;		

		//get loss
		cv::Mat d=(alpha - ground).mul(W * 2);
		//cv::Mat d=alpha - ground;
		//std::cout<<"d:"<<d.rowRange(60,70)<<std::endl;
		//std::cout<<"alpha:"<<alpha.rowRange(60,70)<<std::endl;
		//std::cout<<"ground:"<<ground.rowRange(60,70)<<std::endl;

		return d;
	}

private:
	TensorFormat data_format_;
	
};

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("MattingLossGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MattingLossGradOp<CPUDevice, T>);
	
REGISTER_CPU(float)
