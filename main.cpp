#include <iostream>
#include "ImagePair.h"
#include <chrono>
#include <thread>

Anaglyph make_anaglyph(std::string id) {
    if(id == "color") {
        return ColorAnaglyph();
    } else if(id == "true") {
        return TrueAnaglyph();
    } else if(id == "halfcolor") {
        return HalfColorAnaglyph();
    } else if(id == "gray") {
        return GrayAnaglyph();
    } else if(id == "3dtv") {
        return ThreeDTVAnaglyph();
    } else if(id == "dubois") {
        return DuBoisAnaglyph();
    } else if(id == "roscoulx") {
        return RoscoluxAnaglyph();
    } else {
        throw "Unkown anaglyph ID, valid IDs are: 'color', 'true','halfcolor', 'gray', '3dtv', 'dubois', 'roscoulx'";
    }
}
int main(int argc, char *argv[])
{   
    Eigen::initParallel();
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string anaglyph_name = "";
    std::string path1 = "";
    std::string path2 = "";
    if(argc == 4) {
        anaglyph_name = argv[1];
        path1 = argv[2];
        path2 = argv[3];
        
    } else if(argc == 3) {
        anaglyph_name = argv[1];
        path1 = argv[2];
    } else {
        throw "Incorrect number of arguments. Expected anaglyph name, and at least one image path.!";
    }
    const auto processor_count = std::thread::hardware_concurrency();
    Eigen::setNbThreads(processor_count);
    std::cout << "Merging " << path1 << " " << path2 << " with Anaglyph: " << anaglyph_name <<  " on "<<  Eigen::nbThreads() << " processors " << std::endl;
    
    ImagePair img_pair(path1, path2);

    Anaglyph a = make_anaglyph(anaglyph_name);
    cv::Mat merged = img_pair.mergeImages(a);
    cv::imwrite("Image.png", merged);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;

    std::cout << "compiled " << merged.rows << " by " << merged.cols << " Anaglyph in " <<
    time/std::chrono::milliseconds(1) << "ms" << std::endl;
}

