#pragma once
#include <rog_map/rog_map.h>
namespace rog_map {
class ROGMapROS : public ROGMap {
public:
    using Ptr = std::shared_ptr<ROGMapROS>;
    explicit ROGMapROS(const std::string& path) {
        cfg_ = Config(path);
        cfg_.raycasting_en = false;
        init();
    }
    const double getSystemWalltimeNow() override { return 0.0; }
};
}
