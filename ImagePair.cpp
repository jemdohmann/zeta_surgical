#include "ImagePair.h"
#include "image_utils.h"


ImagePair::ImagePair(std::string path1, std::string path2) {
    if(path2.size() > 0) {
        left_image_ = loadSingleImageAsRGB(path1);
        right_image_ = loadSingleImageAsRGB(path2);
    } else {
        auto imgs = loadImagePairAsRGB(path1);
        left_image_ = imgs.first;
        right_image_ = imgs.second;
    }  
}

cv::Mat ImagePair::mergeImages(const Anaglyph& anaglyph) {
    return anaglyph.scaleAndMerge(left_image_, right_image_);
}