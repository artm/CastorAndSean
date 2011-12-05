#include <string>
#include <map>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/errinfo_file_name.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>
#include <gflags/gflags.h>

#include "castor.hpp"
#include "FaceDetector.hpp"

namespace fs = boost::filesystem;
using namespace boost::assign;

DEFINE_string(mode,"import", "operation mode (import|pca)");

fs::path datadir, incomingDir, origDir, seedDir;

typedef void (*DoFun)();
typedef std::map<std::string, DoFun> ModeMap;
ModeMap mode_map = map_list_of
("import", doImport)
("pca", doPca);

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

    datadir = fs::path(argv[1]);
    incomingDir = datadir / "new";
    origDir = datadir / "orig";
    seedDir = datadir / "seed";

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
        if (mode_map.count(FLAGS_mode) > 0)
            mode_map[FLAGS_mode]();
        else {
            std::cerr << "Unknown operation mode: " << FLAGS_mode << "\n"
                << "Mode should be one of:\n";
            BOOST_FOREACH(const ModeMap::value_type pair, mode_map) {
                std::cerr << "  " << pair.first << "\n";
            }
            return 1;
        }
    } catch (FaceDetector::load_error& e) {
        std::cerr
            << "Error loading classifier cascade for face detector: "
            << *boost::get_error_info< boost::errinfo_file_name >(e) << "\n";
        return 1;
    }

    return 0;
}
