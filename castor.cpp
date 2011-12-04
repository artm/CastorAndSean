#include <iostream>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>

namespace fs = boost::filesystem;

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

    return 0;
}
