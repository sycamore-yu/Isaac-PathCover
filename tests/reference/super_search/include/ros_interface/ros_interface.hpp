#pragma once
#include <memory>
// Original A* visualization arguments; no message or ROS behavior is used.
struct Color {
    Color() = default;
    Color(Color,double) {}
    static Color Orange(){return {};}
    static Color Blue(){return {};}
    static Color Green(){return {};}
    static Color Red(){return {};}
    static Color Pink(){return {};}
};
namespace ros_interface {
class RosInterface {
public:
    using Ptr = std::shared_ptr<RosInterface>;
    double time=0.0, increment=0.0;
    double getSimTime(){time+=increment;return time;}
    template<class... T> void warn(const T&...) {}
    template<class... T> void error(const T&...) {}
    template<class... T> void vizAstarPoints(const T&...) {}
    template<class... T> void vizAstarBoundingBox(const T&...) {}
};
}
