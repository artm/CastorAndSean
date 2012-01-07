#include "Animator.hpp"
#include "PersistentParams.hpp"

#include <cinder/app/AppBasic.h>
#include <cinder/gl/gl.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Vbo.h>
#include <cinder/Vector.h>
#include <cinder/Camera.h>
#include <cinder/Capture.h>
#include <cinder/Arcball.h>
#include <cinder/Rand.h>
#include <cinder/Easing.h>

#include <CinderOpenCV.h>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>

#include <vector>
#include <string>
#include <iostream>


using namespace ci;
using namespace ci::app;

using namespace boost;
using namespace boost::adaptors;


class Sean : public AppBasic {
    virtual ~Sean() {
        PersistentParams::save();
    }

    void setup() {
        // load the projected
        std::vector<std::string> seedNames;
        { // load projection
            cv::FileStorage yml(
                    loadResource("projection.yml")->getFilePath().native(),
                    cv::FileStorage::READ );
            yml["seednames"] >> seedNames;
            yml["projection"] >> m_projection;
        }
        { // load pca
            cv::FileStorage yml(
                    loadResource("pca.yml")->getFilePath().native(),
                    cv::FileStorage::READ );
            yml["eigenvectors"] >> m_pca.eigenvectors;
            yml["eigenvalues"] >> m_pca.eigenvalues;
            yml["mean"] >> m_pca.mean;
        }

        Rand::randomize();
        // decide which three axes to use
        m_chosenAxes = 0;
        shuffleAxes();

        m_interp = 0.0;

        for(int i = 0; i<m_projection.cols; i++) {
            cv::Scalar mean, stddev;
            cv::meanStdDev(m_projection.col(i),mean,stddev);
            m_mean.push_back(mean[0]);
            m_stddev.push_back(stddev[0]);
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
            m_positions[0].push_back( studentized(m_projection, i) );
            m_positions[1].push_back(Vec3f::zero());
            m_order.push_back(i);
            i++;
        }
        gl::Texture::Format fmt;
        fmt.enableMipmapping();
        fmt.setMinFilter(GL_LINEAR_MIPMAP_NEAREST);
        m_texture = gl::Texture( texture, fmt );

        m_cam.setPerspective( 90.0f, getWindowAspectRatio(), 0.01f, 5.0f );

        PersistentParams::load( std::string(getenv("HOME")) + "/.seanrc" );
        m_gui = PersistentParams( "Sean Archer", Vec2i( 250, 250 ) );

        m_gui.addPersistentParam("Flight inertia", &m_flightInertia, 0.1,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_gui.addPersistentParam("Follow inertia", &m_followInertia, 0.5,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_gui.addPersistentParam("Rotate inertia", &m_rotateInertia, 0.1,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_gui.addPersistentParam("Target distance", &m_targetDistance, 0.3,
                "min=0.01 max=5 step=0.01 precision=2");

        m_gui.addPersistentParam("Portrait size", &m_portraitSize, 0.1,
                "min=0.01 max=1.0 step=0.01 precision=2");

        m_gui.addPersistentParam("Reorganization time", &m_interpTime, 2,
                "min=0.5 max=20 step=0.1 precision=1");

        m_gui.addButton("Other axes", boost::bind(&Sean::otherAxes, this), "key=space");
        m_gui.addButton("Toggle fullscreen", boost::bind(&Sean::toggleFullscreen, this),
                "key=f");

        GLfloat fogColor[4]= {0,0,0,1};
        glFogi(GL_FOG_MODE, GL_EXP);
        glFogfv(GL_FOG_COLOR, fogColor);
        glFogf(GL_FOG_DENSITY, 5e-4);
        glEnable(GL_FOG);

        glEnable (GL_DEPTH_TEST);

        gl::enableAlphaBlending();
        std::vector< Capture::DeviceRef > devices = Capture::getDevices(true);
        if (devices.size()>0) {
            m_capture = Capture(320,240,devices[0]);
            m_capture.start();
        }
        m_RTFacePos = Vec3f(0,0,0);
        m_RTProj = cv::Mat::zeros(1,m_pca.eigenvectors.rows,CV_32F);
        m_classifier.load(
                loadResource("haarcascade_frontalface_alt.xml")
                ->getFilePath().native());
        m_RTFace = gl::Texture(64,64);
    }

    bool further(int a, int b) {
        Vec3f eye = m_cam.getEyePoint();
        return p(a).distanceSquared(eye)
            > p(b).distanceSquared(eye);
    }

    void update() {
        m_animator.update();

        if (m_capture.isCapturing() && m_capture.checkNewFrame()) {
            // capture a frame
            Surface8u input = m_capture.getSurface();

            try {
                cv::Mat frame = toOcvRef(input);
                cv::cvtColor(frame, frame, CV_RGB2GRAY);

                // detect a face
                std::vector<cv::Rect> faces;
                m_classifier.detectMultiScale( frame, faces, 1.3, 3, 0,
                        cv::Size(10,10));
                if (faces.size() > 0) {
                    frame = cv::Mat(frame, faces[0]);
                    cv::equalizeHist(frame, frame);
                    cv::resize(frame,frame,cv::Size(64,64));
                    Surface8u face( fromOcv(frame));
                    m_RTFace.update( face, m_RTFace.getBounds() );
                    // project to eigen space
                    m_RTProj = m_pca.project( frame.reshape(0,1) );
                }
            } catch (cv::Exception& e) {
                // ignore
            }
        }
        // move face toward projected position
        m_RTFacePos = m_RTFacePos.lerp(m_flightInertia, studentized(m_RTProj, 0));

        // move the camera to a certain offset from rt face
        Vec3f flyTarget = m_RTFacePos;
        if (flyTarget.length() == 0) {
            // fly target depends on where the camera is
            flyTarget = Rand::randVec3f();
        }

        flyTarget += flyTarget.normalized() * m_targetDistance;
        Vec3f eye = m_cam.getEyePoint();
        eye = eye.lerp(m_followInertia, flyTarget);
        m_cam.setEyePoint(eye);

        Vec3f dir0 = m_cam.getViewDirection();
        Vec3f dir1 = m_RTFacePos - eye;
        m_cam.setViewDirection( dir0.lerp(m_rotateInertia, dir1) );

        sortSprites();
    }

    void draw() {
        Vec2f sz = Vec2f(1,1) * m_portraitSize;

        gl::clear( Color(0,0,0), true );

        gl::setMatrices( m_cam );
        Vec3f right, up;
        m_cam.getBillboardVectors(&right, &up);


        gl::color( 0.9, 1, 0.9, .6);
        m_RTFace.enableAndBind();
        gl::drawBillboard( m_RTFacePos, sz, right, up);
        m_RTFace.unbind();

        m_texture.enableAndBind();
        int fpl = 2048/64;
        float texScale = 1.0f/fpl;
        BOOST_FOREACH(int i, m_order) {
            Vec2f uv00 = Vec2f(i%fpl, i/fpl)*texScale,
                  uv11 = uv00+Vec2f(texScale,texScale);
            gl::drawBillboard( p(i), sz, right, up, uv00, uv11);
        }
        m_texture.unbind();

        params::InterfaceGl::draw();
    }

    void resize(ResizeEvent e) {
        m_cam.setAspectRatio(e.getAspectRatio());
    }

    void sortSprites() {
        // sort by distance from camera
        std::sort(m_order.begin(), m_order.end(),
                boost::bind(&Sean::further, this, _1, _2));
    }

    //! i-th chosen shuffled axis
    int d(int i) {
        return m_eigenAxes[m_chosenAxes + i];
    }

    //! interpolated position of i-th sprite
    Vec3f p(int i) {
        return m_positions[0][i].lerp(m_interp, m_positions[1][i]);
    }

    void shuffleAxes() {
        push_back( m_eigenAxes, irange(0,m_pca.eigenvectors.rows) );
        random_shuffle( m_eigenAxes );
    }

    void otherAxes() {
        if (!m_animator.isFinished("interp"))
            return;

        int idx;
        if (m_interp < 0.5f) {
            idx = 1;
            m_animator.animate("interp", &m_interp, 1./m_interpTime);
        } else {
            idx = 0;
            m_animator.animate("interp", &m_interp, -1./m_interpTime);
        }

        m_chosenAxes += 3;
        if (m_chosenAxes > m_eigenAxes.size()-3)
            m_chosenAxes = 0;

        for(int i=0; i<m_positions[idx].size(); i++)
            m_positions[idx][i] = studentized(m_projection, i);
    }

    Vec3f studentized(const cv::Mat& mtx, int i) {
        Vec3f v;
        for(int j=0; j<3; j++)
            v[j] = (mtx.at<float>(i,d(j)) - m_mean[d(j)]) / m_stddev[d(j)];
        return v;
    }

    void toggleFullscreen() {
        setFullScreen(! isFullScreen() );
    }

    // fields
    gl::Texture m_texture;
    std::vector<Vec3f> m_positions[2];
    float m_interp, m_interpTime;
    std::vector<int> m_order;

    CameraPersp m_cam;
    PersistentParams m_gui;

    Capture m_capture;
    gl::Texture m_RTFace;
    cv::Mat m_RTProj;
    Vec3f m_RTFacePos;

    //! a shuffled vector of eigenspace axes
    std::vector<int> m_eigenAxes;
    //! offset into m_eigenAxes
    int m_chosenAxes;

    cv::CascadeClassifier m_classifier;
    cv::PCA m_pca;
    cv::Mat m_projection;
    std::vector<float> m_mean, m_stddev;

    float m_targetDistance, m_portraitSize;

    float m_flightInertia, m_followInertia, m_rotateInertia;
    Animator m_animator;
};

CINDER_APP_BASIC( Sean, RendererGl )
