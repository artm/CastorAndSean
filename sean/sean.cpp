#include <cinder/app/AppBasic.h>
#include <cinder/gl/gl.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Vbo.h>
#include <cinder/Vector.h>
#include <cinder/Camera.h>
#include <cinder/Arcball.h>
#include <cinder/params/Params.h>

#include <CinderOpenCV.h>

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace ci;
using namespace ci::app;


class Sean : public AppBasic {
    gl::Texture m_texture;
    std::vector<Vec3f> m_positions;

    CameraPersp m_cam;
    params::InterfaceGl m_gui;
    Arcball m_aball;
    void update() {}

    void setup()
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

        Surface8u texture(2048,2048,false);
        int i = 0, fpl = 2048/64; // faces per line
        BOOST_FOREACH(std::string path, seedNames) {
            path = "seed/" + path + ".png";
            Surface face( loadImage( loadResource(path) ) );
            texture.copyFrom(
                    face,
                    face.getBounds(),
                    Vec2i( i%fpl*64,i/fpl*64 ));
            m_positions.push_back( Vec3f(
                    projection.at<float>(i,0),
                    projection.at<float>(i,1),
                    projection.at<float>(i,2)));
            i++;
        }
        gl::Texture::Format fmt;
        fmt.enableMipmapping();
        fmt.setMinFilter(GL_LINEAR_MIPMAP_NEAREST);
        m_texture = gl::Texture( texture, fmt );

        m_cam.setPerspective( 60.0f, getWindowAspectRatio(), 1.0f, 5000.0f );
        //m_gui = params::InterfaceGl( "Sean Archer", Vec2i( 225, 200 ) );

        GLfloat fogColor[4]= {0,0,0,1};
        glFogi(GL_FOG_MODE, GL_EXP);
        glFogfv(GL_FOG_COLOR, fogColor);
        glFogf(GL_FOG_DENSITY, 5e-4);
        glFogf(GL_FOG_START, 500);
        glFogf(GL_FOG_END, 2000);
        glEnable(GL_FOG);

        gl::enableAlphaBlending();
        // sort by distance from origin
        std::sort(m_positions.begin(), m_positions.end(), boost::bind(&Sean::further, this, _1, _2));
    }

    bool further(Vec3f a, Vec3f b) {
        return a.distanceSquared(m_cam.getEyePoint()) > b.distanceSquared(m_cam.getEyePoint());
    }

    void draw()
    {
        gl::clear( Color(0,0,0) );
        gl::color( 1, 1, 0.9, .9);
        //glEnable( GL_DEPTH_TEST );

        gl::setMatrices( m_cam );

        Vec3f mRight, mUp;
        m_cam.getBillboardVectors(&mRight, &mUp);

        m_texture.enableAndBind();
        int fpl = 2048/64;
        float texScale = 1.0f/fpl;
        for(int i=0; i<m_positions.size(); ++i) {
            glMatrixMode(GL_TEXTURE);
            glLoadIdentity();
            glScalef(texScale,texScale,1);
            glTranslatef(i%fpl,i/fpl,0);
            glMatrixMode(GL_MODELVIEW);
            float sz = 100;
            gl::drawBillboard( m_positions[i], Vec2f(sz,sz), 0, mRight, mUp);
        }
        m_texture.unbind();

        params::InterfaceGl::draw();
    }

    void resize(ResizeEvent e)
    {
        m_cam.setAspectRatio(e.getAspectRatio());
        m_aball.setWindowSize(getWindowSize());
        m_aball.setCenter(getWindowSize()/2);
        m_aball.setRadius( std::min(getWindowWidth(),getWindowHeight()) );
    }

    void mouseDown(MouseEvent e)
    {
        m_aball.setQuat(m_cam.getOrientation());
        m_aball.mouseDown(e.getPos());
    }

    void mouseDrag(MouseEvent e)
    {
        m_aball.mouseDrag(e.getPos());
        m_cam.setOrientation(m_aball.getQuat());
    }
};
CINDER_APP_BASIC( Sean, RendererGl )
