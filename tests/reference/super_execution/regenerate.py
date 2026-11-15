"""Build unmodified SUPER trajectory containers and extract original FSM methods.

Only ROS publication/logging and planner results are stubbed. Method bodies are
read from the pinned checkout on every regeneration; their hashes are saved.
"""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/home/tong/tongworkspace/reference_repos/SUPER')
BUILD = ROOT / '.reference/super_execution'
BUILD.mkdir(parents=True, exist_ok=True)
include = SOURCE / 'super_planner/include'
common = ['g++', '-std=c++17', '-O2', '-DCLASS_COLOR_MSG_UTILS', '-ffunction-sections', '-fdata-sections',
          '-I/usr/include/eigen3', f'-I{include}', f'-I{SOURCE / "rog_map/include"}',
          '-Wl,--gc-sections']
executable = BUILD / 'trajectory_oracle'
subprocess.run(common + [str(Path(__file__).with_name('oracle.cpp')),
    str(SOURCE / 'super_planner/src/utils/trajectory.cpp'),
    str(SOURCE / 'super_planner/src/utils/piece.cpp'),
    str(SOURCE / 'super_planner/src/utils/ellipsoid.cpp'),
    str(SOURCE / 'super_planner/src/utils/polytope.cpp'), '-lfmt', '-o', str(executable)], check=True)
rows = {}
for line in subprocess.check_output([str(executable)], text=True).splitlines():
    items = line.split()
    if items and items[0] in ('command', 'prefix', 'allocator', 'exp_only', 'sampling'):
        rows.setdefault(items[0], []).append([float(v) for v in items[1:]])

fsm_source = (SOURCE / 'super_planner/src/super_core/fsm.cpp').read_text()
ros_source = (include / 'ros_interface/ros2/fsm_ros2.hpp').read_text()


def extract(text, marker):
    start = text.index(marker)
    begin = text.index('{', start)
    depth, end = 1, begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


methods = [extract(fsm_source, 'void Fsm::callMainFsmOnce()'),
           extract(fsm_source, 'void Fsm::callReplanOnce()'),
           extract(fsm_source, 'bool Fsm::closeToGoal(')]
command = extract(ros_source, 'void pubCmdTimerCallback()')
stub = r'''
#include <Eigen/Eigen>
#include <iomanip>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
using std::cout; using std::endl; using std::string;
const char *YELLOW="", *GREEN="", *RESET="";
enum RET_CODE {FAILED,NO_NEED,SUCCESS,FINISH,NEW_TRAJ,EMER};
enum GridType {OCCUPIED};
double now=0.;
struct TimeConsuming {TimeConsuming(const char*,bool){} double stop(){return 0.;}};
struct Robot {bool rcv=true; double rcv_time=0.; Eigen::Vector3d p{0,0,0};};
struct Ros {double getSimTime(){return now;}};
struct Map {void getNearestInfCellNot(GridType,Eigen::Vector3d&,Eigen::Vector3d&,double){}};
struct Planner {
    Robot robot; Map map; RET_CODE plan=SUCCESS,replan=SUCCESS; bool valid=true;
    int plans=0, replans=0;
    std::function<void()> during_replan;
    RET_CODE PlanFromRest(Eigen::Vector3d,double,bool){++plans;return plan;}
    RET_CODE ReplanOnce(Eigen::Vector3d,double,bool){++replans;if(during_replan)during_replan();return replan;}
    void getRobotState(Robot& r){r=robot;}
    bool goalValid(){return valid;}
    Map* getMap(){return &map;}
    void getModuleTimeConsuming(std::vector<double>&){}
    int getLatestReplanLog(){return 0;}
};
namespace mars_quadrotor_msgs { namespace msg { struct PolynomialTrajectory {}; } }
struct Publisher {template<class T> void publish(T){}};
class Fsm {
public:
    enum MACHINE_STATE {INIT,WAIT_GOAL,YAWING,GENERATE_TRAJ,FOLLOW_TRAJ,EMER_STOP};
    std::vector<string> MACHINE_STATE_STR{"INIT","WAIT_GOAL","YAWING","GENERATE_TRAJ","FOLLOW_TRAJ","EMER_STOP"};
    bool stop=false, finish_plan=false, plan_from_rest_=false, started_=true, traj_finish_=false;
    MACHINE_STATE machine_state_=INIT;
    Robot robot_state_;
    struct Goal {Eigen::Vector3d goal_p{10,0,0}; double goal_yaw=0; bool new_goal=true;} gi_;
    std::shared_ptr<Planner> planner_ptr_=std::make_shared<Planner>();
    std::shared_ptr<Ros> ros_ptr_=std::make_shared<Ros>();
    std::shared_ptr<Publisher> mpc_cmd_pub_=std::make_shared<Publisher>(), cmd_pub_=std::make_shared<Publisher>();
    std::vector<double> log_module_time=std::vector<double>(9);
    std::vector<int> replan_logs_;
    int pid_cmd_=0;
    bool command_finished=false;
    void ChangeState(const string&, MACHINE_STATE state){machine_state_=state;}
    void publishPolyTraj(){} void resetVisualizedPath(){} void publishCurPoseToPath(){} void WriteTimeToLog(){}
    void getOneHeartBeatMsg(mars_quadrotor_msgs::msg::PolynomialTrajectory&, bool& finish){finish=command_finished;}
    void getOnePositionCommand(int&,bool& finish){finish=command_finished;}
    void callMainFsmOnce(); void callReplanOnce(); bool closeToGoal(const double&);
    COMMAND_METHOD
};
METHODS
int main(){
 Fsm f;
 auto emit=[&](const char* name){
  std::cout<<"fsm "<<name<<' '<<int(f.machine_state_)<<' '<<f.gi_.new_goal<<' '<<f.finish_plan<<' '<<f.plan_from_rest_<<' '<<f.planner_ptr_->plans<<' '<<f.planner_ptr_->replans<<'\n';
  now+=0.01; f.planner_ptr_->robot.rcv_time=now;
 };
 f.callMainFsmOnce();emit("init");
 f.callMainFsmOnce();emit("goal");
 f.planner_ptr_->plan=FAILED;f.callMainFsmOnce();emit("rest_failed");
 f.planner_ptr_->plan=SUCCESS;f.callMainFsmOnce();emit("rest_success");
 f.callReplanOnce();emit("skip_first_replan");
 f.planner_ptr_->replan=FAILED;f.callReplanOnce();emit("replan_failed");
 f.planner_ptr_->replan=SUCCESS;f.callReplanOnce();emit("replan_success");
 f.planner_ptr_->replan=NEW_TRAJ;f.callReplanOnce();emit("replan_new");
 f.planner_ptr_->plan=FINISH;f.callMainFsmOnce();emit("rest_finish");
 f.callReplanOnce();emit("finished_replan_skipped");
 f.command_finished=true;f.pubCmdTimerCallback();emit("local_end");
 f.planner_ptr_->plan=SUCCESS;f.callMainFsmOnce();emit("rest_after_local_end");
 f.callReplanOnce();emit("second_skip");
 f.planner_ptr_->replan=EMER;f.callReplanOnce();emit("emergency");
 f.callMainFsmOnce();emit("emergency_wait");
 f.gi_.new_goal=true;f.callMainFsmOnce();emit("next_goal");
 f.callMainFsmOnce();emit("next_rest");
 f.robot_state_.p=Eigen::Vector3d(9.95,0,0);f.pubCmdTimerCallback();emit("goal_end");
 f.gi_.new_goal=true;f.callMainFsmOnce();emit("race_goal");
 f.callMainFsmOnce();emit("race_rest");
 f.callReplanOnce();emit("race_skip");
 f.planner_ptr_->replan=SUCCESS;
 f.planner_ptr_->during_replan=[&](){f.pubCmdTimerCallback();};
 f.callReplanOnce();emit("replan_crosses_command_end");
 for(int i=0;i<2;++i){
  Fsm r;r.machine_state_=Fsm::FOLLOW_TRAJ;r.gi_.new_goal=false;
  const double cached=i==0?9.8:9.95, observed=i==0?9.95:9.8;
  r.planner_ptr_->robot.p=Eigen::Vector3d(cached,0,0);
  r.callMainFsmOnce();
  r.command_finished=true;
  r.planner_ptr_->during_replan=[&](){
   r.planner_ptr_->robot.p=Eigen::Vector3d(observed,0,0);
   r.pubCmdTimerCallback();
  };
  r.callReplanOnce();
  std::cout<<std::setprecision(17)<<"fsm_snapshot "<<(i==0?"outside_to_inside":"inside_to_outside")
   <<' '<<cached<<' '<<observed<<' '<<int(r.machine_state_)<<' '<<r.robot_state_.p.x();
  r.callMainFsmOnce();
  std::cout<<' '<<int(r.machine_state_)<<' '<<r.robot_state_.p.x()<<'\n';
 }
}
'''
fsm_cpp = BUILD / 'fsm_oracle.cpp'
fsm_cpp.write_text(stub.replace('COMMAND_METHOD', command).replace('METHODS', '\n'.join(methods)))
fsm_exe = BUILD / 'fsm_oracle'
subprocess.run(common + [str(fsm_cpp), '-o', str(fsm_exe)], check=True)
rows['fsm'] = []
rows['fsm_snapshots'] = []
for line in subprocess.check_output([str(fsm_exe)], text=True).splitlines():
    items = line.split()
    if items and items[0] == 'fsm':
        rows['fsm'].append({'event': items[1], 'state': int(items[2]), 'new_goal': bool(int(items[3])),
            'finish_plan': bool(int(items[4])), 'plan_from_rest': bool(int(items[5])),
            'plan_calls': int(items[6]), 'replan_calls': int(items[7])})
    elif items and items[0] == 'fsm_snapshot':
        rows['fsm_snapshots'].append({'case': items[1], 'main_position': [float(items[2]),0.,0.],
            'pub_position': [float(items[3]),0.,0.], 'state_after_pub': int(items[4]),
            'cached_position_after_pub': [float(items[5]),0.,0.], 'state_after_main': int(items[6]),
            'cached_position_after_main': [float(items[7]),0.,0.]})
rows['source'] = {'commit': subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
                  'fsm_methods_sha256': hashlib.sha256('\n'.join(methods).encode()).hexdigest(),
                  'command_method_sha256': hashlib.sha256(command.encode()).hexdigest()}
target = ROOT / 'tests/fixtures/super_execution.json'
target.write_text(json.dumps(rows, indent=2)+'\n')
print(target)
