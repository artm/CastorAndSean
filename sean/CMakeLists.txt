FILE(GLOB SEED_PNG ../../data/seed/*.png)
SET(CLASSIFIER haarcascade_frontalface_alt.xml)
ADD_EXECUTABLE(sean MACOSX_BUNDLE
  ../../data/projection/projection.yml
  ../../data/eigen/pca.yml
  ${CLASSIFIER}
  ${SEED_PNG}
  sean.cpp)
SET_TARGET_PROPERTIES(sean
  PROPERTIES
  COMPILE_FLAGS "-fvisibility-inlines-hidden"
  OSX_ARCHITECTURES i386
  MACOSX_BUNDLE true
  MACOSX_BUNDLE_BUNDLE_NAME "Sean"
  MACOSX_BUNDLE_INFO_STRING "Sean Archer"
  MACOSX_BUNDLE_BUNDLE_VERSION "1"
  MACOSX_BUNDLE_LONG_VERSION_STRING "1"
  MACOSX_BUNDLE_SHORT_VERSION_STRING "1"
  MACOSX_BUNDLE_ICON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/sean.icns"
  MACOSX_BUNDLE_GUI_IDENTIFIER "nl.v2.sean"
  MACOSX_BUNDLE_COPYRIGHT "V2_Lab")

# this list is taken from an xcode project generated by tinder
FIND_AND_LINK_LIBRARY(Accelerate sean)
FIND_AND_LINK_LIBRARY(AudioToolbox sean)
FIND_AND_LINK_LIBRARY(AudioUnit sean)
FIND_AND_LINK_LIBRARY(CoreAudio sean)
FIND_AND_LINK_LIBRARY(QuickTime sean)
FIND_AND_LINK_LIBRARY(QTKit sean)
FIND_AND_LINK_LIBRARY(CoreVideo sean)
FIND_AND_LINK_LIBRARY(Carbon sean)
FIND_AND_LINK_LIBRARY(OpenGL sean)
FIND_AND_LINK_LIBRARY(Cocoa sean)
FIND_AND_LINK_LIBRARY(AppKit sean)
FIND_AND_LINK_LIBRARY(CoreData sean)
FIND_AND_LINK_LIBRARY(Foundation sean)

# cinder needs some help (PATHS)
FIND_LIBRARY(cinder_LIBRARY cinder PATHS ${CINDER_ROOT}/lib)
FIND_AND_LINK_LIBRARY(cinder sean)
FIND_PATH(cinder_INCLUDE cinder/app/AppBasic.h PATHS ${CINDER_ROOT}/include)
REQUIRE_HEADER(cinder_INCLUDE cinder/app/AppBasic.h)

# need 32-bt versions of everything
INCLUDE_DIRECTORIES(${CINDER_ROOT}/blocks/opencv/include)
FILE(GLOB CINDER_OPENCV32 ${CINDER_ROOT}/blocks/opencv/lib/macosx/*.a)
TARGET_LINK_LIBRARIES(sean ${CINDER_OPENCV32})

FIND_PACKAGE(Boost REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#TARGET_LINK_LIBRARIES(sean ${Boost_LIBRARIES})

# copy mac resources...
SET_SOURCE_FILES_PROPERTIES(
  ../../data/projection/projection.yml
  ../../data/eigen/pca.yml
  ${CLASSIFIER}
  PROPERTIES
  MACOSX_PACKAGE_LOCATION Resources)

SET_SOURCE_FILES_PROPERTIES(
  ${SEED_PNG}
  PROPERTIES
  MACOSX_PACKAGE_LOCATION Resources/seed)


