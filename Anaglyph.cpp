#include "Anaglyph.h"
#include "image_utils.h"


cv::Mat& Anaglyph::scaleAndMerge(cv::Mat& img1, cv::Mat& img2) const {


    Eigen::Tensor<float, 3, Eigen::RowMajor> tensor1 = cvToTensor(img1);
    Eigen::Tensor<float, 3, Eigen::RowMajor> tensor2 = cvToTensor(img2);

    Eigen::Tensor<float, 3, Eigen::RowMajor> scaled_left = scaleTensor(tensor1, scale_matrix_left_, tensor1.dimensions()[0],tensor1.dimensions()[1]);
    Eigen::Tensor<float, 3, Eigen::RowMajor> scaled_right = scaleTensor(tensor2, scale_matrix_right_, tensor1.dimensions()[0],tensor1.dimensions()[1]);
    Eigen::Tensor<float, 3, Eigen::RowMajor> merged = scaled_left + scaled_right;

    cv::eigen2cv(merged, img1);
    img1 = rgb2bgr(img1);
    return img1;
}

TrueAnaglyph::TrueAnaglyph() {

        scale_matrix_left_ << 0.299, 0.587, 0.114,
                        0, 0, 0,
                        0, 0, 0;


        scale_matrix_right_ << 0, 0, 0,
                        0, 0, 0,
                        0.299, 0.587, 0.114;
}


GrayAnaglyph::GrayAnaglyph() {

        scale_matrix_left_ << 0.299, 0.587, 0.114,
                        0, 0, 0,
                        0, 0, 0;


        scale_matrix_right_ << 0, 0, 0,
                        0.299, 0.587, 0.114,
                        0.299, 0.587, 0.114;
}


ColorAnaglyph::ColorAnaglyph() {

        scale_matrix_left_ << 1, 0, 0,
                        0, 0, 0,
                        0, 0, 0;


        scale_matrix_right_ << 0, 0, 0,
                        0, 1, 0,
                        0, 0, 1;
}

HalfColorAnaglyph::HalfColorAnaglyph() {

        scale_matrix_left_ << 0.299, 0.587, 0.114,
                        0, 0, 0,
                        0, 0, 0;


        scale_matrix_right_ << 0, 0, 0,
                        0, 1, 0,
                        0, 0, 1;
}
