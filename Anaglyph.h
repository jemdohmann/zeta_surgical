
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Anaglyph {
    public:
        cv::Mat& scaleAndMerge(cv::Mat& img1, cv::Mat& img2) const;

    protected:
        Eigen::Matrix3f scale_matrix_left_;
        Eigen::Matrix3f scale_matrix_right_;
};

class TrueAnaglyph : public Anaglyph {
    public:
        TrueAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};

class GrayAnaglyph : public Anaglyph {
    public:
        GrayAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};


class ColorAnaglyph : public Anaglyph {
    public:
        ColorAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};


class HalfColorAnaglyph : public Anaglyph {
    public:
        HalfColorAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};


class ThreeDTVAnaglyph : public Anaglyph {
    public:
        ThreeDTVAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};

class DuBoisAnaglyph : public Anaglyph {
    public:
        DuBoisAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};

class RoscoluxAnaglyph : public Anaglyph {
    public:
        RoscoluxAnaglyph();
        Eigen::Matrix3f& getLeft(){return scale_matrix_left_;};
        Eigen::Matrix3f& getRight(){return scale_matrix_right_;};

};