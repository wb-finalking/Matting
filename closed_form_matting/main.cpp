#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<Eigen/SparseLU>
#include<Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <Eigen/Dense>
//#include <unsupported/Eigen/IterativeSolvers>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

using namespace Eigen;

Eigen::SparseMatrix<double> getLaplacianMatrix(int win_size, double epsilon, cv::Mat input, cv::Mat consts_map)
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
	cv::Mat consts_map_sub; // consts_map with margin excluded
	consts_map_sub = consts_map.rowRange(win_size, height - (win_size + 1));
	consts_map_sub = consts_map_sub.colRange(win_size, width - (win_size + 1));

	int tlen = (height - 2 * win_size) * (width - 2 * win_size);
	tlen -= sum(consts_map_sub)[0];
	tlen *= neb_size_square;

	cv::Mat row_inds = cv::Mat::zeros(tlen, 1, CV_32S);
	cv::Mat col_inds = cv::Mat::zeros(tlen, 1, CV_32S);
	cv::Mat vals = cv::Mat::zeros(tlen, 1, CV_64F);

	// Iterate on all window center
	for (int j = win_size; j < width - win_size; j++) {
		for (int i = win_size; i < height - win_size; i++) {

			// Skip if the current pixel is scribbled
			if ((int)consts_map.at<char>(i, j) == 1) {
				continue;
			}

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

			winI_temp.at<double>(0, 0) = ch1.at<uchar>(0, 0);
			winI_temp.at<double>(1, 0) = ch1.at<uchar>(1, 0);
			winI_temp.at<double>(2, 0) = ch1.at<uchar>(2, 0);
			winI_temp.at<double>(3, 0) = ch1.at<uchar>(0, 1);
			winI_temp.at<double>(4, 0) = ch1.at<uchar>(1, 1);
			winI_temp.at<double>(5, 0) = ch1.at<uchar>(2, 1);
			winI_temp.at<double>(6, 0) = ch1.at<uchar>(0, 2);
			winI_temp.at<double>(7, 0) = ch1.at<uchar>(1, 2);
			winI_temp.at<double>(8, 0) = ch1.at<uchar>(2, 2);

			winI_temp.at<double>(0, 1) = ch2.at<uchar>(0, 0);
			winI_temp.at<double>(1, 1) = ch2.at<uchar>(1, 0);
			winI_temp.at<double>(2, 1) = ch2.at<uchar>(2, 0);
			winI_temp.at<double>(3, 1) = ch2.at<uchar>(0, 1);
			winI_temp.at<double>(4, 1) = ch2.at<uchar>(1, 1);
			winI_temp.at<double>(5, 1) = ch2.at<uchar>(2, 1);
			winI_temp.at<double>(6, 1) = ch2.at<uchar>(0, 2);
			winI_temp.at<double>(7, 1) = ch2.at<uchar>(1, 2);
			winI_temp.at<double>(8, 1) = ch2.at<uchar>(2, 2);

			winI_temp.at<double>(0, 2) = ch3.at<uchar>(0, 0);
			winI_temp.at<double>(1, 2) = ch3.at<uchar>(1, 0);
			winI_temp.at<double>(2, 2) = ch3.at<uchar>(2, 0);
			winI_temp.at<double>(3, 2) = ch3.at<uchar>(0, 1);
			winI_temp.at<double>(4, 2) = ch3.at<uchar>(1, 1);
			winI_temp.at<double>(5, 2) = ch3.at<uchar>(2, 1);
			winI_temp.at<double>(6, 2) = ch3.at<uchar>(0, 2);
			winI_temp.at<double>(7, 2) = ch3.at<uchar>(1, 2);
			winI_temp.at<double>(8, 2) = ch3.at<uchar>(2, 2);

			winI = winI_temp / 255;

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
	std::vector<T> tripletList;
	tripletList.reserve(len);
	for (int i = 0; i < len; i++) {
		tripletList.push_back(
			T(row_inds.at<int>(i, 0) - 1,
			col_inds.at<int>(i, 0) - 1,
			vals.at<double>(i, 0)));
	}
	A.setFromTriplets(tripletList.begin(), tripletList.end());

	//  A = A.transpose() * A;
	return A;
}

cv::Mat getAlpha(double lambda, int win_size, double epsilon, cv::Mat input, cv::Mat consts_map, cv::Mat consts_vals)
{
	int height = input.size().height;
	int width = input.size().width;
	int img_size = height * width;

	// Solve the equation x = (A + lambda * D) \ (lambda * consts_vals(:));
	// To make it clear, let left * x = right
	// left = A + lambda * D and right = lambda * consts_vals(:)

	// Calculation of left side(A + lambda * D)
	SpMat A = getLaplacianMatrix(win_size, epsilon, input, consts_map);
	std::cout << "getLaplacianMatrix..." << std::endl;


	cv::Mat consts_map_trans = consts_map.t();
	SpMat D(img_size, img_size);
	for (int i = 0; i < img_size; i++) {
		D.coeffRef(i, i) = (int)consts_map_trans.at<char>(0, i);
	}
	SpMat left = A + lambda * D;

	// Calculation of right side (lambda * consts_vals(:))
	cv::Mat consts_vals_in_a_col;
	cv::Mat transpo = consts_vals.t();
	consts_vals_in_a_col = transpo.reshape(1, img_size);
	Eigen::VectorXd right(img_size);
	for (int i = 0; i < img_size; i++) {
		right(i) = lambda * consts_vals_in_a_col.at<char>(i, 0);
	}

	int* innerPointer = left.innerIndexPtr();
	int* outerPointer = left.outerIndexPtr();
	double* valuePointer = left.valuePtr();
	double* rightPointer = right.data();

	int size = input.size().width * input.size().height;

	//SparseMatrixEquationSolver sparseMatrixEquationSolver(outerPointer, innerPointer, valuePointer, rightPointer, size);
	//double* alphaArray = sparseMatrixEquationSolver.solveEquation();
	VectorXd x;
	/*BiCGSTAB<SpMat, Eigen::IncompleteLUT<double> >  BCGST;
	BCGST.preconditioner().setDroptol(.001);
	BCGST.compute(left);
	x = BCGST.solve(right);*/

	Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
	cg.compute(left);
	x = cg.solve(right);

	//Eigen::GMRES<SpMat, Eigen::IdentityPreconditioner> gmres;
	//gmres.compute(left);
	//x = gmres.solve(right);
	std::cout << "solve..." << std::endl;
	Map<MatrixXd> fullDmap(x.data(), input.size().width, input.size().height);
	cv::Mat alpha(fullDmap.rows(), fullDmap.cols(), CV_64FC1, fullDmap.data());
	alpha = alpha.t();
	//alpha = alpha * 255;
	//cv::imshow("Display window", alpha);
	//cv::waitKey(0);

	return alpha;
}

cv::Mat performMatting(double lambda, int win_size, double epsilon, double threshold, cv::Mat input, cv::Mat input_m)
{
	cv::Mat temp; // The difference between origin image and scribbled image
	cv::Mat consts_map; // 0-1 values where 1 means pixel scribbled
	cv::Mat consts_vals; // The original value of scribbled pixel
	int height = input.size().height;
	int width = input.size().width;

	// Find the scribbled pixels
	temp = abs(input - input_m);

	cv::Mat ch1, ch2, ch3;

	cv::Mat ch1_final, ch2_final, ch3_final;
	std::vector<cv::Mat> channels(3), channelsFinal(3);

	// Calculate consts_maps
	split(temp, channels);
	split(input_m, channelsFinal);
	ch1 = channels[0];
	ch2 = channels[1];
	ch3 = channels[2];
	ch1_final = channelsFinal[0];
	ch2_final = channelsFinal[1];
	ch3_final = channelsFinal[2];
	consts_map = (ch1 + ch2 + ch3) > threshold; //get scribbled pixels
	consts_map = consts_map / 255;
	//std::cout << "alpha " << consts_map.row(0) << std::endl;
	//cv::imshow("Display window", consts_map);
	//cv::waitKey(0);

	// Calculate consts_vals
	ch1_final = ch1_final.mul(consts_map);
	ch2_final = ch2_final.mul(consts_map);
	ch3_final = ch3_final.mul(consts_map);
	consts_vals = ch1_final / 255;

	// Function to get Alpha by natural matting
	cv::Mat alpha = getAlpha(lambda, win_size, epsilon, input, consts_map, consts_vals);
	return alpha;
}

int main()
{
	double lambda = 100; // Weight of scribbled piexel obedience
	int win_size = 1; // The distance between center and border
	double epsilon = 0.00001;
	double thresholdForScribble = 0.001;

	std::string path_prefix = "F:/project/image-mattingC++/code";
	std::string img_path = path_prefix + "/bmp/dandelion/dandelion.bmp";// "/bmp/dandelion/dandelion.bmp";
	std::string img_m_path = path_prefix + "/bmp/dandelion/dandelion_m.bmp"; //"/bmp/dandelion/dandelion_m.bmp";
	cv::Mat img = cv::imread(img_path);
	cv::Mat img_m = cv::imread(img_m_path);
	cv::Mat alpha = performMatting(lambda, win_size, epsilon, thresholdForScribble, img, img_m);

	//std::cout <<"image "<< img.row(0) << std::endl;
	//std::cout <<"alpha "<< alpha.row(0) << std::endl;
	cv::imshow("alpha", alpha);
	cv::waitKey(0);

	/*std::cout << img.type() << std::endl;
	int matrix[2][2]{{ 1, 2 }, { 3, 4 }};
	cv::Mat tmp(cv::Size(2, 2), CV_8UC1, matrix);
	matrix[0][0] = 2;
	std::cout << tmp.type()<< std::endl;
	std::cout <<( tmp > 2) << std::endl;*/

	return 0;
}