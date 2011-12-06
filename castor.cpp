#include <map>
#include <iostream>
#include <boost/exception/get_error_info.hpp>
#include <boost/exception/errinfo_file_name.hpp>
#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/assign/list_of.hpp>

#include "castor.hpp"
#include "FaceDetector.hpp"

using namespace boost::assign;

DEFINE_string(mode,"import", "operation mode (import|pca|project)");

fs::path datadir;

typedef void (*DoFun)();
typedef std::map<std::string, DoFun> ModeMap;
ModeMap mode_map = map_list_of
("import", doImport)
("pca", doPca)
("project", doProject);

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
    if (!fs::is_directory(datadir)) {
        std::cerr << datadir << " not found or isn't a directory.\n";
        return 1;
    }

    try {
        if (mode_map.count(FLAGS_mode) > 0) {
            mode_map[FLAGS_mode]();
            return 0;
        } else {
            std::cerr << "Unknown operation mode: " << FLAGS_mode << "\n"
                << "Mode should be one of:\n";
            BOOST_FOREACH(const ModeMap::value_type pair, mode_map) {
                std::cerr << "  " << pair.first << "\n";
            }
        }
    } catch (FaceDetector::load_error& e) {
        std::cerr
            << "Error loading classifier cascade for face detector: "
            << *boost::get_error_info< boost::errinfo_file_name >(e) << "\n";
    } catch (NoInputDirectory& e) {
        std::cerr
            << "Input directory '"
            << *boost::get_error_info< boost::errinfo_file_name >(e)
            << "' not found\n";
    }

    return 1;
}

// throw if doesn't exist
fs::path inputDir(const std::string& dirname)
{
    fs::path result = datadir / dirname;
    if (fs::exists(result))
        return result;
    else
        throw NoInputDirectory(result.native());
}

// create if doesn't exist
fs::path outputDir(const std::string& dirname)
{
    fs::path result = datadir / dirname;
    if (!fs::exists(result)) {
        fs::create_directory(result);
    }
    return result;
}

