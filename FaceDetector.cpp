#include "FaceDetector.hpp"
#include <gflags/gflags.h>

DEFINE_string(detector_file, "", "face classifier cascade to load");
DEFINE_double(detector_scaleFactor, 1.1,
        "how much the image size is reduced at each image scale");
DEFINE_int32(detector_minNeighbors, 3,
        "how many neighbors should each candiate rectangle have to retain it");
DEFINE_int32(detector_minSize, 0,
        "minimum possible object size. Don't look for faces smaller than that");

FaceDetector::FaceDetector()
{
    if (FLAGS_detector_file.length() > 0) {
        if (!m_classifier.load(FLAGS_detector_file)) {
            throw load_error()
                << boost::errinfo_file_name(FLAGS_detector_file);
        }
    }
}

void FaceDetector::detect(const cv::Mat& grayImg, std::vector<cv::Rect>& faces)
{
    faces.clear();
    m_classifier.detectMultiScale(
            grayImg,
            faces,
            FLAGS_detector_scaleFactor,
            FLAGS_detector_minNeighbors,
            0,
            cv::Size(FLAGS_detector_minSize, FLAGS_detector_minSize));
}
