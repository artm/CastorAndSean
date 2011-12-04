#ifndef CASTOR_FaceDetector_hpp
#define CASTOR_FaceDetector_hpp

#include <string>
#include <vector>
#include <boost/exception/all.hpp>
#include <cv.h>

class FaceDetector {
public:
    struct exception_base: virtual std::exception, virtual boost::exception {};
    struct load_error : virtual exception_base {};

    FaceDetector();
    void detect(const cv::Mat& grayImg, std::vector<cv::Rect>& faces);
private:
    cv::CascadeClassifier m_classifier;
};

#endif
