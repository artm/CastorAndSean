#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <highgui.h>
#include <gflags/gflags.h>

#include "castor.hpp"
#include "FaceDetector.hpp"

namespace fs = boost::filesystem;

DEFINE_int32(cutout_size,100, "width/height of a normalized face square");

void doImport()
{
    FaceDetector detector;
    fs::directory_iterator incoming(incomingDir), dirEnd;

    int totalCnt=0,facesCnt=0,failedCnt=0;

    for(;incoming != dirEnd; ++incoming, ++totalCnt) {
        fs::path inPath = (*incoming).path();
        std::string path = inPath.native();
        cv::Mat input = cv::imread(path, 0); // force grayscale

        std::vector<cv::Rect> faces;
        detector.detect(input, faces);
        if (faces.size() > 0) {
            // cut out all found faces
            facesCnt += faces.size();
            for(int i = 0; i<faces.size(); ++i) {
                // cut out the face
                cv::Mat face(input,faces[i]);
                cv::resize(face, face,
                        cv::Size(FLAGS_cutout_size,FLAGS_cutout_size));
                // normalize brightness / increase contrast
                cv::equalizeHist(face, face);
                // save
                std::string seedName = inPath.stem().native();
                if (faces.size() > 1) {
                    seedName += boost::str(boost::format("_%d") % (i+1));
                }
                cv::imwrite((seedDir / (seedName + ".png")).native(),face);
            }
            // move image to orig/
            fs::rename( inPath, origDir / inPath.filename() );
        } else {
            std::cout << "No faces found in " << path << "\n";
            failedCnt++;
        }
    }
    std::cout << "\nProcessed " << totalCnt << " images. Found "
        << facesCnt << " faces. No faces were found in "
        << failedCnt << " images.\n";
}

