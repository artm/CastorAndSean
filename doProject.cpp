#include "castor.hpp"
#include <boost/foreach.hpp>

void doProject()
{
    fs::path seedDir = inputDir("seed"), eigenDir = inputDir("eigen"),
        projectDir = outputDir("projection");

    // load pca
    cv::PCA pca;
    {
        cv::FileStorage storage( eigenDir % "pca.yml", cv::FileStorage::READ);
        storage["eigenvectors"] >> pca.eigenvectors;
        storage["eigenvalues"] >> pca.eigenvalues;
        storage["mean"] >> pca.mean;
    }

    // iterate over seed and collect image filenames
    fs::directory_iterator seedIter(seedDir), dirEnd;
    std::vector<std::string> seedList, electList;
    for(;seedIter != dirEnd; ++seedIter) {
        if ((*seedIter).path().extension() == ".png")
            seedList.push_back((*seedIter).path().native());
    }

    // load compatible seeds
    cv::Mat input( seedList.size(), pca.eigenvectors.cols, CV_32FC1 );
    int i = 0;
    BOOST_FOREACH(std::string path, seedList) {
        cv::Mat seed = cv::imread(path, 0);
        if (seed.rows*seed.cols != pca.eigenvectors.cols) {
            std::cerr
                << boost::format("%1% has incompatible dimensions (%2%x%3%), expected w*h==%4%\n")
                % path % seed.cols % seed.rows % pca.eigenvectors.cols;
            continue;
        }
        cv::Mat row = input.row(i);
        seed.reshape(0,1).convertTo(row, CV_32FC1);
        electList.push_back(fs::path(path).stem().native());
        i++;
    }
    if (i<input.rows)
        input = input.rowRange(0,i-1);

    // project
    std::cout << boost::format("Projecting %d images to eigenspace...") % i;
    std::cout.flush();
    cv::Mat projection = pca.project(input);
    std::cout << "done\n";

    // save the results
    {
        cv::FileStorage yml( projectDir % "projection.yml", cv::FileStorage::WRITE );
        yml << "seednames" << "[" << electList << "]";
        yml << "projection" << projection;
    }
}
