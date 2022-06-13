
#include <opencv2/opencv.hpp>
#include "Anaglyph.h"

class ImagePair {
    public :
     ImagePair(std::string path1, std::string path2);
     cv::Mat mergeImages(const Anaglyph& anaglyph);

    private:
        cv::Mat left_image_;
        cv::Mat right_image_;
};