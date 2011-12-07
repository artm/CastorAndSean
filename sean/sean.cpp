#include <cinder/app/AppBasic.h>
#include <cinder/gl/gl.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Vbo.h>
#include <cinder/Vector.h>
#include <cinder/Camera.h>
#include <cinder/Capture.h>
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

    Capture m_capture;
    gl::Texture m_RTFace;
    Vec3f m_RTFacePos;
    bool m_haveRTFace;

    cv::CascadeClassifier m_classifier;
    cv::PCA m_pca;

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
        {
            cv::FileStorage yml(
                    loadResource("pca.yml")->getFilePath().native(),
                    cv::FileStorage::READ );
            yml["eigenvectors"] >> m_pca.eigenvectors;
            yml["eigenvalues"] >> m_pca.eigenvalues;
            yml["mean"] >> m_pca.mean;

            // well, actually we only care for the first three...
            m_pca.eigenvectors = m_pca.eigenvectors.rowRange(0,2);
            m_pca.eigenvalues = m_pca.eigenvalues.rowRange(0,2);
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

        m_cam.setPerspective( 90.0f, getWindowAspectRatio(), 10.0f, 5000.0f );
        //m_gui = params::InterfaceGl( "Sean Archer", Vec2i( 225, 200 ) );

        GLfloat fogColor[4]= {0,0,0,1};
        glFogi(GL_FOG_MODE, GL_EXP);
        glFogfv(GL_FOG_COLOR, fogColor);
        glFogf(GL_FOG_DENSITY, 5e-4);
        glEnable(GL_FOG);

        gl::enableAlphaBlending();
        // sort by distance from origin
        std::sort(m_positions.begin(), m_positions.end(), boost::bind(&Sean::further, this, _1, _2));

        std::vector< Capture::DeviceRef > devices = Capture::getDevices(true);
        if (devices.size()>0) {
            m_capture = Capture(320,240,devices[0]);
            m_capture.start();
        }
        m_haveRTFace = false;
        m_RTFacePos = Vec3f(0,0,0);
        m_classifier.load(
                loadResource("haarcascade_frontalface_alt.xml")
                ->getFilePath().native());
        m_RTFace = gl::Texture(64,64);
    }

    bool further(Vec3f a, Vec3f b) {
        return a.distanceSquared(m_cam.getEyePoint()) > b.distanceSquared(m_cam.getEyePoint());
    }

    void update() {
        if (m_capture.isCapturing() && m_capture.checkNewFrame()) {
            // capture a frame
            Surface8u input = m_capture.getSurface();
            cv::Mat frame = toOcvRef(input);
            cv::cvtColor(frame, frame, CV_RGB2GRAY);

            // detect a face
            std::vector<cv::Rect> faces;
            m_classifier.detectMultiScale( frame, faces, 1.3, 3, 0, cv::Size(10,10));
            if (faces.size() > 0) {
                frame = cv::Mat(frame, faces[0]);
                cv::equalizeHist(frame, frame);
                cv::resize(frame,frame,cv::Size(64,64));
                Surface8u face( fromOcv(frame));
                m_RTFace.update( face, m_RTFace.getBounds() );
                m_haveRTFace = true;
                // project to eigen space
                cv::Mat proj = m_pca.project( frame.reshape(0,1) );
                Vec3f vproj = Vec3f(
                        proj.at<float>(0,0),
                        proj.at<float>(0,1),
                        proj.at<float>(0,2));

                m_RTFacePos = m_RTFacePos.lerp(0.03, vproj);
            } else {
                m_haveRTFace = false;
            }

        }

    }

    void draw()
    {
        float sz = 150;

        gl::clear( Color(0,0,0) );

        if (m_RTFacePos.length() > 1) {
            Vec3f dir = m_cam.getViewDirection();
            Vec3f dir1 = m_RTFacePos.normalized() * dir.length();
            m_cam.setViewDirection(dir.lerp(.1, dir1));
        }

        gl::setMatrices( m_cam );
        Vec3f right, up;
        m_cam.getBillboardVectors(&right, &up);


        if (m_RTFacePos.length() > m_cam.getNearClip()) {
            gl::color( 1, 1, 0.9, 1);
            glMatrixMode(GL_TEXTURE);
            glLoadIdentity();
            glMatrixMode(GL_MODELVIEW);
            m_RTFace.enableAndBind();
            gl::drawBillboard( m_RTFacePos, Vec2f(sz,sz), 0, right, up);
            m_RTFace.unbind();
        }

        gl::color( 1, 1, 0.9, .4);

        m_texture.enableAndBind();
        int fpl = 2048/64;
        float texScale = 1.0f/fpl;
        for(int i=0; i<m_positions.size(); ++i) {
            glMatrixMode(GL_TEXTURE);
            glLoadIdentity();
            glScalef(texScale,texScale,1);
            glTranslatef(i%fpl,i/fpl,0);
            glMatrixMode(GL_MODELVIEW);
            gl::drawBillboard( m_positions[i], Vec2f(sz,sz), 0, right, up);
        }
        m_texture.unbind();

        params::InterfaceGl::draw();
    }

    void resize(ResizeEvent e)
    {
        m_cam.setAspectRatio(e.getAspectRatio());
    }
};
CINDER_APP_BASIC( Sean, RendererGl )
