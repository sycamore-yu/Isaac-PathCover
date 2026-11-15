// Uses original A* and ROG-Map; only ROS time/publication are replaced.
#include <path_search/astar.h>
#include <iomanip>

int main(int argc,char** argv) {
    const auto map=std::make_shared<rog_map::ROGMapROS>(argv[1]);
    const auto ros=std::make_shared<ros_interface::RosInterface>();
    path_search::Astar astar(argv[1],ros,map);
    astar.setFineInfNeighbors(2);
    rog_map::PointCloud cloud;
    for(int y=-10;y<=10;++y)for(int z=0;z<20;++z){
        pcl::PointXYZI p;p.x=.05;p.y=(y+.5)*.1;p.z=(z+.5)*.1;p.intensity=1;cloud.push_back(p);
    }
    map->updateOccPointCloud(cloud);
    std::cout<<std::setprecision(17);
    for(int i=0;i<44;++i){
        rog_map::Vec3f start((-37+(i*17)%74)*.1+.013,(-32+(i*13)%64)*.1+.021,(-16+(i*7)%50)*.1+.019);
        rog_map::Vec3f end((-37+(i*23+9)%74)*.1+.013,(-32+(i*29+5)%64)*.1+.021,(-16+(i*11+3)%50)*.1+.019);
        if(i>=36){
            const std::vector<std::pair<rog_map::Vec3f,rog_map::Vec3f>> special{
                {{-.05,.05,1.05},{1.05,1.05,2.05}},{{-.3,.3,1.},{-.3,.3,1.}},
                {{-.51,-.21,.71},{1.72,.51,1.31}},{{1.72,.51,1.31},{-.51,-.21,.71}},
                {{-1.75,-1.55,1.05},{1.65,-1.55,1.05}},{{-1.75,.05,1.05},{1.65,.05,1.05}},
                {{-5.,-2.,1.},{5.,2.,1.}},{{-1.05,-1.05,-1.55},{1.05,1.05,-1.55}}};
            start=special[i-36].first;end=special[i-36].second;
        }
        const int step=i%3;
        const double max_distance=(i%4==0?.55:i%4==1?6.:i%4==2?-1.:0.);
        rog_map::vec_Vec3i neighbors;
        if(step)for(int x=-step;x<=step;++x)for(int y=-step;y<=step;++y)for(int z=-step;z<=step;++z)
            if(x*x+y*y+z*z<=step*step)neighbors.emplace_back(x,y,z);
        rog_map::Vec3f last;
        const bool free=map->isLineFree(start,end,last,max_distance,neighbors);
        std::cout<<"LINE "<<i<<" "<<step<<" "<<max_distance<<" "<<free<<" "<<start.transpose()<<" "<<end.transpose()<<" "<<last.transpose()<<"\n";
    }
    for(const auto&item:std::vector<std::pair<std::string,rog_map::Vec3f>>{
        {"free_raw_center",{-1.73,-1.52,1.04}},
        {"inside_inflation",{-.15,.05,1.05}},
        {"inside_obstacle",{.05,.05,1.05}},
        {"virtual_ground",{-1.73,-1.52,-1.49}}}){
        rog_map::Vec3f nearest;
        bool found=map->getNearestInfCellNot(super_utils::OCCUPIED,item.second,nearest,3.0);
        std::cout<<"NEAREST "<<item.first<<" "<<found<<" "<<item.second.transpose()<<" "<<nearest.transpose()<<"\n";
    }
    auto print=[&](const std::string&name,super_utils::RET_CODE code,const rog_map::vec_Vec3f&path){
        std::cout<<"CASE "<<name<<" "<<super_utils::RET_CODE_STR[code]<<" "<<path.size()<<"\n";
        for(const auto&p:path)std::cout<<"POINT "<<name<<" "<<p.transpose()<<"\n";
    };
    auto search=[&](const std::string&name,rog_map::Vec3f start,rog_map::Vec3f goal,double horizon,bool inflated,bool unknown,bool fine){
        int flags=(inflated?path_search::ON_INF_MAP:path_search::ON_PROB_MAP)
                 |(unknown?path_search::UNKNOWN_AS_OCCUPIED:path_search::UNKNOWN_AS_FREE)
                 |(fine?path_search::USE_INF_NEIGHBOR:path_search::DONT_USE_INF_NEIGHBOR);
        rog_map::vec_Vec3f path;
        auto code=astar.pointToPointPathSearch(start,goal,flags,horizon,path);
        print(name,code,path);
    };
    search("clear",{-1.75,-1.55,1.05},{1.65,-1.55,1.05},9.,true,false,false);
    search("wall",{-1.75,.05,1.05},{1.65,.05,1.05},9.,true,false,false);
    search("horizon",{-1.75,.05,1.05},{3.15,2.45,1.05},1.2,true,false,false);
    search("prob_fine",{-1.75,.05,1.05},{1.65,.05,1.05},9.,false,false,true);
    search("unknown_blocked",{-1.75,-1.55,1.05},{1.65,-1.55,1.05},9.,false,true,false);
    for(const auto&item:std::vector<std::pair<std::string,rog_map::Vec3f>>{
        {"escape_clear",{-1.75,-1.55,1.05}},{"escape_inflated",{-.15,.05,1.05}},
        {"escape_outside",{99.,0.,1.}}}){
        rog_map::vec_Vec3f path;
        auto code=astar.escapePathSearch(item.second,path_search::ON_PROB_MAP|path_search::UNKNOWN_AS_FREE,path);
        print(item.first,code,path);
    }
    ros->increment=.051;
    search("timeout",{-1.75,-1.55,1.05},{1.65,-1.55,1.05},9.,true,false,false);
}
