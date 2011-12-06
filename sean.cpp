#include <cinder/app/AppBasic.h>
#include <cinder/gl/gl.h>
#include <cv.h>
#include <vector>
#include <string>

using namespace ci;
using namespace ci::app;

class Sean : public AppBasic {
    void setup();
    void update() {}
    void draw();
};

void Sean::setup()
{
    // load the projected
    std::vector<std::string> seedNames;
    cv::Mat projection;
    {
        cv::FileStorage yml(
                loadResource("projection.yml")->getFilePath().native(),
                cv::FileStorage::READ );
        yml["seednames"] >> seedNames;
        yml["projection"] >> projection;
    }
}

void Sean::draw()
{
    gl::clear( Color(0,0,0) );
}

CINDER_APP_BASIC( Sean, RendererGl )
