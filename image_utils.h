#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core/eigen.hpp>

std::vector<cv::Mat> splitImage(const cv::Mat & image, int M, int N );
std::vector<cv::Mat> splitImagePair(const cv::Mat& image);

cv::Mat loadImage(std::string path);

cv::Mat& rgb2bgr(cv::Mat& img);
cv::Mat& bgr2rgb(cv::Mat& img);
Eigen::Tensor<float, 3, Eigen::RowMajor> cvToTensor(cv::Mat& img);

cv::Mat loadSingleImageAsRGB(std::string path);
std::pair<cv::Mat, cv::Mat> loadImagePairAsRGB(std::string path);

   
template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;
template<typename Scalar,int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank, Eigen::RowMajor> &tensor,const sizeType rows,const sizeType cols);
template<typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims);
Eigen::Tensor<float, 3, Eigen::RowMajor>  scaleRow(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input_tensor, const Eigen::Matrix3f& scale_matrix, int height, int width, int row);
Eigen::Tensor<float, 3, Eigen::RowMajor>  scaleTensor(const Eigen::Tensor<float, 3, Eigen::RowMajor>& input_tensor, const Eigen::Matrix3f& scale_matrix, int height, int width);

