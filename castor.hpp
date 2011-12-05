#ifndef CASTOR_HPP
#define CASTOR_HPP

#include <boost/filesystem.hpp>

extern boost::filesystem::path datadir, incomingDir, origDir, seedDir;

void doImport();
void doPca();

#endif
