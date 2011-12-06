#include "castor.hpp"

#include <iostream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <gflags/gflags.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/format.hpp>

DEFINE_int32(pca_seedsCount,64,"How many seed images to use for PCA");
DEFINE_int32(pca_maxComponents,0,"How many components to keep");
DEFINE_bool(pca_hobbit,false,"Project seeds to eigenspace and back");
DECLARE_int32(cutout_size);

namespace fs = boost::filesystem;

static boost::mt19937 gen;
int rnd(int base, int len)
{
    boost::uniform_int<> dist(base, len-1);
    return dist(gen);
}

void doPca()
{
    fs::path seedDir = inputDir("seed"), eigenDir = outputDir("eigen"), hobbitDir = outputDir("hobbit");
    fs::directory_iterator seedIter(seedDir), dirEnd;
    std::vector<std::string> seedList;
    for(;seedIter != dirEnd; ++seedIter) {
        if ((*seedIter).path().extension() == ".png")
            seedList.push_back((*seedIter).path().native());
    }

    int idx[FLAGS_pca_seedsCount];
    cv::Mat pcaInput(FLAGS_pca_seedsCount,
            FLAGS_cutout_size*FLAGS_cutout_size,
            CV_32FC1);
    for(int i = 0; i<FLAGS_pca_seedsCount; ++i) {
        do {
            idx[i] = rnd(0,seedList.size());
        } while( std::find(idx,idx+i,idx[i])!=idx+i );

        std::string path = seedList[idx[i]];
        std::cout << "loading " << path << "\n";
        cv::Mat seed = cv::imread(path, 0); // grayscale
        if (seed.rows != FLAGS_cutout_size || seed.cols != FLAGS_cutout_size) {
            // FIXME proper error signaling
            std::cout << boost::format("Expected size: %1%x%1%\n") % FLAGS_cutout_size
                << boost::format("Actual size: %1%x%2%\n") % seed.cols % seed.rows;
            CV_Error(0,"Incompatible seed image size");
        }
        cv::Mat row = pcaInput.row(i);
        seed.reshape(0,1).convertTo(row, CV_32FC1);
    }
    std::cout << "Calculating eigenfaces, this may take a while...";
    std::cout.flush();
    cv::PCA pca(pcaInput, cv::Mat(), CV_PCA_DATA_AS_ROW, FLAGS_pca_maxComponents);
    std::cout << " done\n";

    // save stuff
    // save pca as yaml
    cv::FileStorage storage( eigenDir % "pca.yml", cv::FileStorage::WRITE);
    storage << "eigenvectors" << pca.eigenvectors;
    storage << "eigenvalues" << pca.eigenvalues;
    storage << "mean" << pca.mean;
    storage.release();

    cv::imwrite( eigenDir % "mean.png", pca.mean.reshape(0,FLAGS_cutout_size));
    for(int i = 0; i<pca.eigenvectors.rows; ++i) {
        cv::Mat eigenface = pca.eigenvectors.row(i).reshape(0,FLAGS_cutout_size);
        double min, max;
        cv::minMaxLoc(eigenface, &min, &max);
        double scale = 255.0 / (max - min), offset = - scale * min;
        cv::Mat e8;
        eigenface.convertTo(e8, CV_8UC1, scale, offset);
        cv::equalizeHist(e8, e8);
        std::string path = eigenDir + boost::format("eigen%03d.png") % (i+1);
        std::cout << "Saving " << path << "\n";
        cv::imwrite(path, e8);
    }

    if (FLAGS_pca_hobbit) {
        cv::Mat there = pca.project(pcaInput);
        cv::Mat back = pca.backProject(there);
        for(int i = 0; i < back.rows; ++i) {
            cv::Mat hobbit = back.row(i).reshape(0,FLAGS_cutout_size);
            std::string hobbit_path = hobbitDir + boost::format("hobbit%03d.png") % (i+1);
            cv::imwrite(hobbit_path,hobbit);
        }
    }
}

