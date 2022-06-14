#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <opencv2/core/eigen.hpp>
#include "image_utils.h"
using namespace cv;


std::vector<cv::Mat> splitImage(const cv::Mat & image, int M, int N )
{
  // All images should be the same size ...
  int width  = image.cols / M;
  int height = image.rows / N;
  // ... except for the Mth column and the Nth row
  int width_last_column = width  + ( image.cols % width  );
  int height_last_row   = height + ( image.rows % height );

  std::vector<cv::Mat> result;

  for( int i = 0; i < N; ++i )
  {
    for( int j = 0; j < M; ++j )
    {
      // Compute the region to crop from
      cv::Rect roi( width  * j,
                    height * i,
                    ( j == ( M - 1 ) ) ? width_last_column : width,
                    ( i == ( N - 1 ) ) ? height_last_row   : height );

      result.push_back( image( roi ) );
    }
  }

  return result;
}

std::vector<cv::Mat> splitImagePair(const cv::Mat& image) {
  return splitImage(image, 2, 1);
}

Mat loadImage(std::string path) {
    Mat image;
    image = imread( path, 1 );
    if ( !image.data )
    {
        printf("No image data \n");
    }
    return image;
}

Mat& rgb2bgr(Mat& img) {
  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  return img;
}

Mat& bgr2rgb(Mat& img) {
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  return img;
}

Eigen::Tensor<float, 3, Eigen::RowMajor> cvToTensor(Mat& img) {
  Eigen::Tensor<float, 3, Eigen::RowMajor> a_tensor(img.rows, img.cols, 3);
    cv::cv2eigen(img, a_tensor);
    return a_tensor;
}
Eigen::Tensor<float, 3, Eigen::ColMajor> cvToColMajorTensor(Mat& img) {
  Eigen::Tensor<float, 3, Eigen::ColMajor> a_tensor(img.rows, img.cols, 3);
    cv::cv2eigen(img, a_tensor);
    return a_tensor;
}

cv::Mat loadSingleImageAsRGB(std::string path) {
  cv::Mat img = loadImage(path);
  return bgr2rgb(img);
}

std::pair<cv::Mat, cv::Mat> loadImagePairAsRGB(std::string path) {
  std::pair<cv::Mat, cv::Mat> ret_pair;


  auto imgs = splitImagePair(loadImage(path));

  ret_pair.first = bgr2rgb(imgs.at(0));
  ret_pair.second = bgr2rgb(imgs.at(1));

  return ret_pair;

}

   
template<typename T>
using  MatrixType = Eigen::Matrix<T,Eigen::Dynamic, Eigen::Dynamic>;

template<typename Scalar,int rank, typename sizeType>
auto Tensor_to_Matrix(const Eigen::Tensor<Scalar,rank, Eigen::ColMajor> &tensor,const sizeType rows,const sizeType cols)
{
    return Eigen::Map<const MatrixType<Scalar>> (tensor.data(), rows,cols);
}



template<typename Scalar, typename... Dims>
auto Matrix_to_Tensor(const MatrixType<Scalar> &matrix, Dims... dims)
{
    constexpr int rank = sizeof... (Dims);
    return Eigen::TensorMap<Eigen::Tensor<const Scalar, rank, Eigen::ColMajor>>(matrix.data(), {dims...});
}


Eigen::Tensor<float, 3, Eigen::ColMajor>  scaleRow(const Eigen::Tensor<float, 3, Eigen::ColMajor>& input_tensor, const Eigen::Matrix3f& scale_matrix, int height, int width, int row) {
  std::array<long, 3> offset = {row,0,0};
    std::array<long, 3> extent = {1,width,3};
    std::array<long, 2> shape = {width,3};

    // // Slice a row
    Eigen::Tensor<float, 2, Eigen::ColMajor> row_tensor = input_tensor.slice(offset, extent).reshape(shape);

    // // Convert to matrix and transpose 
    Eigen::MatrixXf matrix = Tensor_to_Matrix(row_tensor, width, 3);

    // // Multiply
    Eigen::MatrixXf result = matrix * scale_matrix.transpose();

    // // Cast back to rank-3 tensor
    return Matrix_to_Tensor(result, 1, width, 3);
}

Eigen::Tensor<float, 3, Eigen::ColMajor>  scaleTensorByRow(const Eigen::Tensor<float, 3, Eigen::ColMajor>& input_tensor, const Eigen::Matrix3f& scale_matrix, int height, int width) {

  Eigen::Tensor<float, 3, Eigen::ColMajor>  first_row = scaleRow(input_tensor, scale_matrix,  height,  width, 0);
  for(int i =1; i < height; i++) {
    // Scaling the ith row
    Eigen::Tensor<float, 3, Eigen::ColMajor> ith_row = scaleRow(input_tensor, scale_matrix,  height,  width, i);
    Eigen::Tensor<float, 3, Eigen::ColMajor> intermediate = first_row.concatenate(ith_row, 0);
    first_row = intermediate;
  }
  return first_row;
}

Eigen::Tensor<float, 3, Eigen::ColMajor>  scaleWholeTensor(const Eigen::Tensor<float, 3, Eigen::ColMajor>& input_tensor, const Eigen::Matrix3f& scale_matrix, int height, int width) {
  std::array<long, 2> shape = {width,3};
  std::array<long, 2> shape2 = {height * width, 3};
  
  // // Slice a row
  Eigen::Tensor<float, 2, Eigen::ColMajor> row_tensor = input_tensor.reshape(shape2);
  // // Convert to matrix and transpose 
  Eigen::MatrixXf matrix = Tensor_to_Matrix(row_tensor, height * width, 3);

  // // Multiply
  Eigen::MatrixXf result = matrix * scale_matrix.transpose();

  // // Cast back to rank-3 tensor
  Eigen::Tensor<float, 3, Eigen::ColMajor> scaled_left = Matrix_to_Tensor(result, height,  width, 3);
  return scaled_left;
}

