#pragma once
#include <cinder/Timer.h>
#include <boost/range/adaptor/map.hpp>
#include <boost/function.hpp>
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <map>

namespace {

using namespace ci;
using namespace boost::adaptors;

class Animator {
    Timer m_timer;
    float m_deltaTime;

    struct Track {
        typedef boost::function<void (float t)> FloatSetter;
        FloatSetter set;
        float t;
        float speed;
        bool finished;

        Track() {}

        Track(FloatSetter setter, float _t, float _speed)
            : set(setter), t(_t), speed(_speed), finished(false)
        {}

        void advance(float deltaTime) {
            t += deltaTime * speed;
            if (t<=0) {
                t = 0;
                finished = true;
            } else if (t>=1) {
                t = 1;
                finished = true;
            }
            set(t);
        }

        bool isFinished() const { return finished; }
        static void setFloat(float* var, float val) { *var = val; }
    };

    std::map<std::string, Track> m_tracks;
    typedef std::pair<std::string, Track> NamedTrack;
public:
    Animator()
        : m_deltaTime(0)
    {}

    void update() {
        m_timer.stop();
        m_deltaTime = m_timer.getSeconds();
        m_timer.start();
        BOOST_FOREACH(Track& t, m_tracks | map_values) {
            t.advance(m_deltaTime);
        }
        BOOST_FOREACH(const std::string& name, m_tracks | map_keys) {
            if (m_tracks[name].isFinished())
                m_tracks.erase(name);
        }
    }

    float deltaTime() const { return m_deltaTime; }
    bool isFinished(const std::string& name)
    {
        return m_tracks.count(name) == 0;
    }

    void animate(const std::string& name, float * target, float speed)
    {
        m_tracks[name] = Track(
                boost::bind(&Track::setFloat,target,_1),
                *target,
                speed);
    }

    typedef boost::function<float (float t)> FloatAdaptor;
    void animate(const std::string& name, float * target, float speed, FloatAdaptor adaptor)
    {
        m_tracks[name] = Track(
                    boost::bind(&Track::setFloat,target,boost::bind(adaptor,_1)),
                    *target,
                    speed);
    }
};

}
