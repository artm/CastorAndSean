#ifndef CASTOR_HPP
#define CASTOR_HPP

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/exception/all.hpp>
#include <string>
#include <gflags/gflags.h>
#include <cv.h>
#include <highgui.h>

namespace fs = boost::filesystem;

void doImport();
void doPca();
void doProject();

struct CastorError : virtual std::exception, virtual boost::exception {};
struct NoInputDirectory : virtual CastorError {
    NoInputDirectory(const std::string& dirname) {
        (*this) << boost::errinfo_file_name(dirname);
    }
};

// throw if doesn't exist
fs::path inputDir(const std::string& dirname);

// create if doesn't exist
fs::path outputDir(const std::string& dirname);

inline std::string operator% (const fs::path& a, const fs::path& b)
{ return (a / b).native(); }
inline std::string operator+ (const fs::path& a, const boost::format& b)
{ return a % b.str(); }

#endif
