#ifndef CASTOR_FaceDetector_hpp
#define CASTOR_FaceDetector_hpp

#include <string>
#include <vector>
#include <cv.h>
#include "castor.hpp"

class FaceDetector {
public:
    struct load_error : virtual CastorError {};

    FaceDetector();
    void detect(const cv::Mat& grayImg, std::vector<cv::Rect>& faces);
private:
    cv::CascadeClassifier m_classifier;
};

#endif
