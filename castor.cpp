#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/errinfo_file_name.hpp>
#include <boost/format.hpp>
#include <gflags/gflags.h>
#include <highgui.h>

#include "FaceDetector.hpp"

namespace fs = boost::filesystem;

DEFINE_int32(cutout_size,100,"width/height of a normalized face square");

int main(int argc, char* argv[])
{
    std::string usage("Usage: ");
    usage += std::string(argv[0]) + " DATADIR ";
    google::SetUsageMessage(usage);
    google::ParseCommandLineFlags(&argc, &argv, true);

    if (argc!=2) {
        std::cerr << google::ProgramUsage() << "\n";
        return 1;
    }

    fs::path
        datadir(argv[1]),
        incomingDir(datadir / "new"),
        origDir(datadir / "orig"),
        seedDir(datadir / "seed");

    if (!fs::is_directory(datadir)) {
        std::cerr << datadir << " not found or isn't a directory.\n";
        return 1;
    }

    if (!fs::is_directory(incomingDir)
            || !fs::is_directory(origDir)
            || !fs::is_directory(seedDir)) {
        std::cerr << "Data directory " << datadir << " should contain subdirectories "
            << incomingDir.filename() << ", "
            << origDir.filename() << " and "
            << seedDir.filename() << ".\n";
        return 1;
    }

    try {
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
    } catch (FaceDetector::load_error& e) {
        std::cerr
            << "Error loading classifier cascade for face detector: "
            << *boost::get_error_info< boost::errinfo_file_name >(e) << "\n";
        return 1;
    }

    return 0;
}
