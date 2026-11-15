// Offline oracle: compile unchanged SUPER 2ad3419 ROG-Map sources.
#include <rog_map/rog_map.h>
#include <iomanip>

class OracleMap : public rog_map::ROGMap {
public:
    OracleMap(const std::string& path, bool raycast, bool features) {
        cfg_ = rog_map::Config(path);
        cfg_.raycasting_en = raycast;
        cfg_.esdf_en = features;
        cfg_.frontier_extraction_en = features;
        cfg_.esdf_resolution = .2;
        cfg_.esdf_local_update_box = rog_map::Vec3f(3.2,3.2,2.4);
        init();
    }
    const double getSystemWalltimeNow() override { return 0.0; }
    rog_map::ESDFMap& esdf() { return *esdf_map_; }
    void freePoint(const rog_map::Vec3f& p) { missPointUpdate(p,getHashIndexFromPos(p),999); }
};

int main(int argc, char** argv) {
    const bool features = std::string(argv[2]) == "features";
    OracleMap map(argv[1], std::string(argv[2]) != "hits", features);
    std::cout << std::setprecision(17);
    const rog_map::Vec3f center(0,0,1);
    if (features) map.esdf().updateESDF3D(center);
    std::vector<rog_map::Vec3f> queries;
    for (double x : {-1.95, -1.15, -0.95, -0.05, 0.05, 0.55, 1.05, 1.95})
        for (double y : {-0.95, 0.05, 0.25, 0.95})
            for (double z : {0.55, 1.05, 1.55}) queries.emplace_back(x,y,z);
    queries.emplace_back(0,0,-1.7); queries.emplace_back(0,0,-1.3);
    queries.emplace_back(0,0,3.3); queries.emplace_back(0,0,3.7);
    queries.emplace_back(10,0,1); queries.emplace_back(0,0,10);
    auto snapshot = [&](int phase) {
        for (const auto& p : queries)
            std::cout << "QUERY " << phase << " " << p.transpose() << " " << map.getMapValue(p)
                      << " " << int(map.getGridType(p)) << " " << int(map.getInfGridType(p))
                      << " " << map.isOccupied(p) << " " << map.isOccupiedInflate(p) << "\n";
        rog_map::vec_Vec3f points;
        map.boxSearch(rog_map::Vec3f(-2,-1,0), rog_map::Vec3f(2,1,2), super_utils::OCCUPIED, points);
        for (const auto& p : points) std::cout << "BOX " << phase << " " << p.transpose() << "\n";
        if (features) {
            for (const auto& p : queries)
                std::cout << "FRONTIER " << phase << " " << p.transpose() << " " << map.isFrontier(p) << "\n";
            map.boxSearch(rog_map::Vec3f(-3,-1,0),rog_map::Vec3f(3,1,2),super_utils::FRONTIER,points);
            for (const auto& p : points) std::cout << "FRONTIER_BOX " << phase << " " << p.transpose() << "\n";
            rog_map::Vec3f lo,hi; map.esdf().getUpdatedBbox(lo,hi);
            std::cout << "ESDF_BOUNDS " << phase << " " << lo.transpose() << " " << hi.transpose() << "\n";
            for (double x : {-1.15,-.31,.05,.27,.55,.95})
                for (double y : {-.31,.07,.43})
                    for (double z : {.79,1.09,1.37}) {
                        rog_map::Vec3f p(x,y,z),first,second;
                        double distance;
                        map.esdf().evaluateEDT(p,distance);
                        map.esdf().evaluateFirstGrad(p,first);
                        map.esdf().evaluateSecondGrad(p,second);
                        std::cout << "ESDF " << phase << " " << p.transpose() << " "
                                  << map.esdf().getDistance(p) << " " << distance << " "
                                  << first.transpose() << " " << second.transpose() << "\n";
                    }
        }
    };
    snapshot(0);
    rog_map::PointCloud cloud;
    for (const auto& p : std::vector<rog_map::Vec3f>{{-1.15,.25,1.05},{.55,.05,1.05},{.55,.05,1.05},{1.05,.95,1.55}}) {
        pcl::PointXYZI q; q.x=p.x(); q.y=p.y(); q.z=p.z(); q.intensity=1; cloud.push_back(q);
    }
    map.updateOccPointCloud(cloud);
    rog_map::vec_Vec3f block;
    if (features) {
        rog_map::PointCloud occupied;
        for(int x=-2;x<=2;++x) for(int y=-2;y<=2;++y) for(int z=3;z<=7;++z) {
            rog_map::Vec3f p(x*.2+.05,y*.2+.05,z*.2+.05); block.push_back(p);
            pcl::PointXYZI q; q.x=p.x();q.y=p.y();q.z=p.z();q.intensity=1;occupied.push_back(q);
        }
        map.updateOccPointCloud(occupied);
        map.esdf().updateESDF3D(center);
    }
    snapshot(1);
    rog_map::PointCloud scan;
    for (const auto& p : std::vector<rog_map::Vec3f>{{1.95,.05,1.05},{1.95,.95,1.55},{-.95,-.95,.55}}) {
        pcl::PointXYZI q; q.x=p.x(); q.y=p.y(); q.z=p.z(); q.intensity=1; scan.push_back(q);
    }
    for(int i=0;i<4;++i) map.updateMap(scan,{rog_map::Vec3f(-1.95,.05,1.05),Eigen::Quaterniond::Identity()});
    snapshot(2);
    if (features) {
        for(const auto& p:block) map.freePoint(p.cast<float>().cast<double>());
        rog_map::PointCloud hit;
        pcl::PointXYZI q; q.x=-1.95;q.y=.05;q.z=1.05;q.intensity=1;hit.push_back(q);
        map.updateOccPointCloud(hit);
        map.esdf().updateESDF3D(center);
        snapshot(3);
    }
    std::vector<std::pair<rog_map::Vec3f,rog_map::Vec3f>> rays{
        {{-.05,.05,1.05},{1.05,1.05,2.05}},{{-.3,.3,1.0},{-.3,.3,1.0}},
        {{-.51,-.21,.71},{1.72,.51,1.31}},{{1.72,.51,1.31},{-.51,-.21,.71}}};
    for (size_t i=0;i<rays.size();++i) {
        rog_map::raycaster::RayCaster ray(.1); ray.setInput(rays[i].first,rays[i].second);
        rog_map::Vec3f p;
        while(ray.step(p)) std::cout << "RAY " << i << " " << p.transpose() << "\n";
        std::cout << "LINE " << i << " " << map.isLineFree(rays[i].first,rays[i].second,999999.0)
                  << " " << map.isLineFree(rays[i].first,rays[i].second,true,false) << "\n";
    }
}
