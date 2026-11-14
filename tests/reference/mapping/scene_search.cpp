#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include <Eigen/Dense>
#include <jps_planner/jps_planner.h>
#include <distance_map_planner/distance_map_planner.h>
using V=Eigen::Vector3d;
class Probe: public DMPlanner<3> { public: double cost() const { return path_cost_; } };
template<class C> void points(const C& ps) { std::cout << '['; bool comma=false; for(const auto& p:ps) { if(comma)std::cout<<',';comma=true;std::cout<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']'; }std::cout<<']'; }
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  std::ifstream in(argv[1]); Eigen::Vector3i dims; V origin,start,goal; double res;
  in>>dims.x()>>dims.y()>>dims.z(); in>>origin.x()>>origin.y()>>origin.z()>>res;
  in>>start.x()>>start.y()>>start.z()>>goal.x()>>goal.y()>>goal.z();
  std::vector<signed char> grid(dims.prod()); int v;
  for(auto& x:grid){in>>v;x=static_cast<signed char>(v);} if(!in)return 3;
  auto util=std::make_shared<JPS::VoxelMapUtil>(); util->setMap(origin,dims,grid,res);
  JPSPlanner3D jps; jps.setMapUtil(util);jps.updateMap();
  auto t=std::chrono::steady_clock::now(); bool ok=jps.plan(start,goal,res,true);
  double jt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  auto jp=jps.getRawPath();double jc=0.;for(size_t i=1;i<jp.size();++i)jc+=(jp[i]-jp[i-1]).norm()/res;
  Probe dm;dm.setPotentialRadius(V(1,1,1));dm.setSearchRadius(V(2,2,2));
  t=std::chrono::steady_clock::now();dm.setMap(util,start);
  double pt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  auto potential=dm.getMapUtil()->getMap(); uint64_t fingerprint=14695981039346656037ULL;
  for(auto x:potential){fingerprint^=static_cast<uint8_t>(x);fingerprint*=1099511628211ULL;}
  t=std::chrono::steady_clock::now();bool dok=dm.computePath(start,goal,jp);
  double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  std::cout<<std::setprecision(17)<<"{\"jps_ok\":"<<ok<<",\"dmp_ok\":"<<dok
    <<",\"jps_cost\":"<<jc<<",\"dmp_cost\":"<<dm.cost()<<",\"jps_s\":"<<jt
    <<",\"dmp_s\":"<<dt<<",\"potential_s\":"<<pt<<",\"potential_fnv64\":"<<fingerprint
    <<",\"jps_path\":";points(jp);std::cout<<",\"dmp_path\":";points(dm.getRawPath());std::cout<<"}\n";
  return ok&&dok?0:4;
}
