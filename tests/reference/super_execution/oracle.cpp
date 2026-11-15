// Offline source oracle. Runtime never links or executes this program.
#include <mutex>
#include <iomanip>
#include <iostream>
#include <fmt/color.h>
#include <data_structure/base/polytope.h>
namespace super_planner { using namespace super_utils; using namespace geometry_utils; }
#include <data_structure/cmd_traj.h>
#include <utils/geometry/geometry_utils.h>

using geometry_utils::Trajectory;
using super_planner::ExpTraj;
using super_planner::BackupTraj;
using super_planner::CmdTraj;

Trajectory polynomial(const std::vector<double>& durations, double shift) {
    std::vector<Eigen::MatrixXd> coefficients;
    for (int i = 0; i < static_cast<int>(durations.size()); ++i) {
        Eigen::MatrixXd c = Eigen::MatrixXd::Zero(3, 8);
        for (int d = 0; d < 3; ++d) {
            for (int k = 0; k < 8; ++k) {
                c(d, 7-k) = (i+1)*(d+1)*0.01*(k+1) + (k == 0 ? shift : 0);
            }
        }
        coefficients.push_back(c);
    }
    return Trajectory(durations, coefficients);
}

int main() {
    std::cout << std::setprecision(17);
    ExpTraj exp;
    exp.setTrajectory(7.0, polynomial({2.0, 1.5}, 1.0), polynomial({2.0, 1.5}, 0.1), 0.2, 0.4);
    BackupTraj backup;
    backup.setTrajectory(8.25, 1.25, polynomial({0.8}, 5.0), polynomial({0.8}, 0.7));
    CmdTraj cmd;
    cmd.setTrajectory(exp, backup);
    for (double t : {-0.1, 0.0, 0.2, 0.3, 0.4, 1.25, 1.2500001, 2.05, 2.051}) {
        const double elapsed = (7.0 + t) - cmd.getStartWallTime();
        const bool finished = elapsed > cmd.getTotalDuration();
        const double eval = finished ? cmd.getTotalDuration() : elapsed;
        auto state = cmd.posTraj().getState(eval);
        std::cout << "command " << t << ' ' << cmd.isTTOnBackupTraj(eval) << ' ' << finished;
        for (int d = 0; d < 3; ++d)
            for (int k = 0; k < 4; ++k) std::cout << ' ' << state(d,k);
        std::cout << ' ' << cmd.getYaw(eval)[0] << ' ' << cmd.getYawRate(eval)[0] << '\n';
    }
    Trajectory prefix;
    exp.posTraj().getPartialTrajectoryByTime(0.35, 2.7, prefix);
    for (double t : {0.0, 0.1, 1.65, 1.7, 2.35}) {
        auto state = prefix.getState(t);
        std::cout << "prefix " << t;
        for (int d = 0; d < 3; ++d)
            for (int k = 0; k < 4; ++k) std::cout << ' ' << state(d,k);
        std::cout << '\n';
    }
    for (double v0 : {0.0, 2.0, 9.0})
        for (double length : {0.1, 2.0, 9.0})
            for (double fraction : {0.0, 0.3, 0.9, 1.0}) {
                double time, velocity;
                geometry_utils::simplePMTimeAllocator(15.0, 8.0, v0, length, length*fraction, time, velocity);
                std::cout << "allocator " << v0 << ' ' << length << ' ' << length*fraction
                          << ' ' << time << ' ' << velocity << '\n';
            }
    cmd.setTrajectory(exp);
    std::cout << "exp_only " << cmd.getTotalDuration() << ' ' << cmd.getStartWallTime()
              << ' ' << cmd.backupTrajAvilibale() << ' ' << cmd.isTTOnBackupTraj(0.3) << '\n';
    // super_planner.cpp lines 513 and 854 advance each sample by repeated addition.
    for (double start : {0., .1, .3}) {
        const double stop=1.1, step=.05/8.;
        std::cout << "sampling " << start << ' ' << stop << ' ' << step;
        for (double t=start;t<stop;t+=step) std::cout << ' ' << t;
        std::cout << '\n';
    }
}
