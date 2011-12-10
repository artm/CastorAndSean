#include <cinder/app/AppBasic.h>
#include <cinder/gl/gl.h>
#include <cinder/gl/Texture.h>
#include <cinder/gl/Vbo.h>
#include <cinder/Vector.h>
#include <cinder/Camera.h>
#include <cinder/Capture.h>
#include <cinder/Arcball.h>
#include <cinder/params/Params.h>
#include <cinder/Rand.h>

#include <CinderOpenCV.h>

#include <boost/foreach.hpp>
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

class Sean : public AppBasic {
    gl::Texture m_texture;
    std::vector<Vec3f> m_positions;
    std::vector<int> m_order;

    CameraPersp m_cam;
    params::InterfaceGl m_gui;

    Capture m_capture;
    gl::Texture m_RTFace;
    Vec3f m_RTFaceProj, m_RTFacePos;

    //! a shuffled vector of eigenspace axes
    std::vector<int> m_eigenAxes;
    //! offset into m_eigenAxes
    int m_chosenAxes;

    cv::CascadeClassifier m_classifier;
    cv::PCA m_pca;
    float m_targetDistance, m_portraitSize;

    float m_flightInertia, m_followInertia, m_rotateInertia;

    void setup()
    {
        // load the projected
        std::vector<std::string> seedNames;
        cv::Mat projection;
        { // load projection
            cv::FileStorage yml(
                    loadResource("projection.yml")->getFilePath().native(),
                    cv::FileStorage::READ );
            yml["seednames"] >> seedNames;
            yml["projection"] >> projection;
        }
        { // load pca
            cv::FileStorage yml(
                    loadResource("pca.yml")->getFilePath().native(),
                    cv::FileStorage::READ );
            yml["eigenvectors"] >> m_pca.eigenvectors;
            yml["eigenvalues"] >> m_pca.eigenvalues;
            yml["mean"] >> m_pca.mean;

            // well, actually we only care for the first three...
            //m_pca.eigenvectors = m_pca.eigenvectors.rowRange(0,2);
            //m_pca.eigenvalues = m_pca.eigenvalues.rowRange(0,2);
        }

        Rand::randomize();
        // decide which three axes to use
        m_chosenAxes = 0;
        shuffleAxes();

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
                    projection.at<float>(i,d(0)),
                    projection.at<float>(i,d(1)),
                    projection.at<float>(i,d(2))));
            m_order.push_back(i);
            i++;
        }
        gl::Texture::Format fmt;
        fmt.enableMipmapping();
        fmt.setMinFilter(GL_LINEAR_MIPMAP_NEAREST);
        m_texture = gl::Texture( texture, fmt );

        m_cam.setPerspective( 90.0f, getWindowAspectRatio(), 10.0f, 5000.0f );
        m_gui = params::InterfaceGl( "Sean Archer", Vec2i( 250, 150 ) );


        m_flightInertia = 0.05;
        m_gui.addParam("Flight inertia", &m_flightInertia,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_followInertia = 0.1;
        m_gui.addParam("Follow inertia", &m_followInertia,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_rotateInertia = 0.001;
        m_gui.addParam("Rotate inertia", &m_rotateInertia,
                "min=1e-5 max=1 step=1e-3 precision=5");

        m_targetDistance = 100.0f;
        m_gui.addParam("Target distance", &m_targetDistance,
                "min=1 max=1000 step=10 precision=1");

        m_portraitSize = 50.0f;
        m_gui.addParam("Portrait size", &m_portraitSize,
                "min=1 max=600 step=1 precision=0");


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
        m_RTFaceProj = m_RTFacePos = Rand::randVec3f()*1000.0;
        m_classifier.load(
                loadResource("haarcascade_frontalface_alt.xml")
                ->getFilePath().native());
        m_RTFace = gl::Texture(64,64);
    }

    bool further(int a, int b) {
        Vec3f eye = m_cam.getEyePoint();
        return m_positions[a].distanceSquared(eye)
            > m_positions[b].distanceSquared(eye);
    }

    void update() {
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
                    cv::Mat proj = m_pca.project( frame.reshape(0,1) );
                    m_RTFaceProj = Vec3f(
                            proj.at<float>(0,d(0)),
                            proj.at<float>(0,d(1)),
                            proj.at<float>(0,d(2)));

                }
            } catch (cv::Exception& e) {
                // ignore
            }
        }
        // move face toward projected position
        m_RTFacePos = m_RTFacePos.lerp(m_flightInertia, m_RTFaceProj);

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

    void draw()
    {
        Vec2f sz = Vec2f(1,1) * m_portraitSize;

        gl::clear( Color(0,0,0), true );

        gl::setMatrices( m_cam );
        Vec3f right, up;
        m_cam.getBillboardVectors(&right, &up);


        gl::color( 0.9, 1, 0.9, .6);
        if (m_RTFacePos.length() > m_cam.getNearClip()) {
            m_RTFace.enableAndBind();
            drawBillboard( m_RTFacePos, sz, right, up);
            m_RTFace.unbind();
        }

        m_texture.enableAndBind();
        int fpl = 2048/64;
        float texScale = 1.0f/fpl;
        BOOST_FOREACH(int i, m_order) {
            Vec2f uv00 = Vec2f(i%fpl, i/fpl)*texScale,
                  uv11 = uv00+Vec2f(texScale,texScale);
            drawBillboard( m_positions[i], sz, right, up, uv00, uv11);
        }
        m_texture.unbind();

        params::InterfaceGl::draw();
    }

    void resize(ResizeEvent e)
    {
        m_cam.setAspectRatio(e.getAspectRatio());
    }

    inline static void drawBillboard(const Vec3f& pos, const Vec2f& scale,
            const Vec3f &bbRight, const Vec3f &bbUp, GLfloat texCoords[] )
    {
        glEnableClientState( GL_VERTEX_ARRAY );
        Vec3f verts[4];
        glVertexPointer( 3, GL_FLOAT, 0, &verts[0].x );
        glEnableClientState( GL_TEXTURE_COORD_ARRAY );
        glTexCoordPointer( 2, GL_FLOAT, 0, texCoords );

        Vec2f halfscale = 0.5f * scale;

        verts[0] = pos
            + bbRight * halfscale.x
            + bbUp    * halfscale.y;
        verts[1] = pos
            + bbRight * halfscale.x
            - bbUp    * halfscale.y;
        verts[2] = pos
            - bbRight * halfscale.x
            + bbUp    * halfscale.y;
        verts[3] = pos
            - bbRight * halfscale.x
            - bbUp    * halfscale.y;

        glDrawArrays( GL_TRIANGLE_STRIP, 0, 4 );

        glDisableClientState( GL_VERTEX_ARRAY );
        glDisableClientState( GL_TEXTURE_COORD_ARRAY );
    }

    inline static void drawBillboard(const Vec3f& pos, const Vec2f& scale,
            const Vec3f &bbRight, const Vec3f &bbUp,
            const Vec2f& uv00, const Vec2f& uv11 )
    {
        GLfloat texCoords[8] = {
            uv00.x, uv00.y,
            uv00.x, uv11.y,
            uv11.x, uv00.y,
            uv11.x, uv11.y
        };
        drawBillboard(pos, scale, bbRight, bbUp, texCoords);
    }

    inline static void drawBillboard(const Vec3f& pos, const Vec2f& scale,
            const Vec3f &bbRight, const Vec3f &bbUp )
    {
        drawBillboard(pos, scale,
                bbRight, bbUp,
                Vec2f(0,0), Vec2f(1,1));
    }

    void sortSprites()
    {
        // sort by distance from camera
        std::sort(m_order.begin(), m_order.end(),
                boost::bind(&Sean::further, this, _1, _2));
    }

    //! i-th chosen shuffled axis
    int d(int i)
    {
        return m_eigenAxes[m_chosenAxes + i];
    }

    void shuffleAxes()
    {
        push_back( m_eigenAxes, irange(0,m_pca.eigenvectors.rows) );
        random_shuffle( m_eigenAxes );
    }
};
CINDER_APP_BASIC( Sean, RendererGl )
