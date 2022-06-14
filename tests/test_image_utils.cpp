#include "../image_utils.h"


int main() {

  Eigen::setNbThreads(8);
  Eigen::Matrix3f scale_matrix_left_;
  scale_matrix_left_ << 1, 0, 0,
                        0, 1, 0,
                        0, 0, 0;
  cv::Mat image(128, 128, CV_8UC3, cv::Scalar(255, 64, 128));
  cv::imwrite("BEFORE.png", image);
  
  image = bgr2rgb(image);

  auto start_time = std::chrono::high_resolution_clock::now();
  Eigen::Tensor<float, 3, Eigen::ColMajor> tensor1 = cvToColMajorTensor(image);


  Eigen::Tensor<float, 3, Eigen::ColMajor>  input_tensor = tensor1;
  Eigen::Matrix3f scale_matrix = scale_matrix_left_;
  int height = tensor1.dimensions()[0];
  int width = tensor1.dimensions()[1];

  
  Eigen::Tensor<float, 3, Eigen::ColMajor> scaled_left = scaleWholeTensor(tensor1, scale_matrix_left_, height, width);
  // std::cout << scaled_left << std::endl;
  cv::eigen2cv(scaled_left, image);
  image = rgb2bgr(image);
  cv::imwrite("AFTER1.png", image);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto time = end_time - start_time;

  std::cout << "Scaled " << image.rows << " by " << image.cols << " image using method 1 in " << 
    time/std::chrono::milliseconds(1) << "ms" << std::endl;

  scaled_left = scaleTensorByRow(tensor1, scale_matrix_left_, height, width);
  // std::cout << scaled_left << std::endl;
  cv::eigen2cv(scaled_left, image);
  image = rgb2bgr(image);
  cv::imwrite("AFTER2.png", image);

  end_time = std::chrono::high_resolution_clock::now();
  time = end_time - start_time;

  std::cout << "Scaled " << image.rows << " by " << image.cols << " image using method 2 in " << 
    time/std::chrono::milliseconds(1) << "ms" << std::endl;


  return 0;
}