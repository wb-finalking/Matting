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
using GPUDevice = Eigen::GpuDevice;

using namespace tensorflow;

template <typename Device>
SpMat getLaplacianMatrix(const Device& d, int win_size, double epsilon, cv::Mat input);

SpMat getLaplacianMatrix<GPUDevice>(const GPUDevice& d, int win_size, double epsilon, cv::Mat input);

