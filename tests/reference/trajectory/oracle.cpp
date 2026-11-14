// Offline executable: call the pinned upstream headers, with no Python binding.
// No PathCover algorithm implementation is copied into this driver.
#include <Eigen/Eigen>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include "gcopter/geo_utils.hpp"
#include "gcopter/minco.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/lbfgs.hpp"
#include "gcopter/trajectory.hpp"
// Test-only access to objective evaluation and coordinate matrices.
// The included algorithm bodies and the upstream checkout remain unchanged.
#define private public
#include "gcopter_solver.hpp"
#undef private

void number(double value)
{
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}

template<class Vector> void vector_json(const Vector &x)
{
    std::cout << "[";
    for (int i = 0; i < x.size(); ++i) {
        if (i) std::cout << ",";
        number(x(i));
    }
    std::cout << "]";
}

template<class Matrix> void matrix_json(const Matrix &x)
{
    std::cout << "[";
    for (int row = 0; row < x.rows(); ++row) {
        if (row) std::cout << ",";
        std::cout << "[";
        for (int col = 0; col < x.cols(); ++col) {
            if (col) std::cout << ",";
            number(x(row, col));
        }
        std::cout << "]";
    }
    std::cout << "]";
}

template<int Order> void minco_reference()
{
    typename trajectory_server::MincoTraits<Order>::Minco minco;
    Eigen::Matrix<double, 3, Order> start = Eigen::Matrix<double, 3, Order>::Zero();
    Eigen::Matrix<double, 3, Order> end = start;
    end.col(0) << 3, 2, 1;
    Eigen::Matrix<double, 3, 2> points;
    points << 1, 2, .5, 1.5, .3, .7;
    Eigen::Vector3d times(1.1, 1.4, 1.2);
    minco.setConditions(start, end, 3);
    minco.setParameters(points, times);
    double energy;
    Eigen::MatrixX3d grad_c;
    Eigen::VectorXd partial_t, grad_t;
    Eigen::Matrix3Xd grad_points;
    minco.getEnergy(energy);
    minco.getEnergyPartialGradByCoeffs(grad_c);
    minco.getEnergyPartialGradByTimes(partial_t);
    minco.propogateGrad(grad_c, partial_t, grad_points, grad_t);
    std::cout << "{\"coefficients\":";
    matrix_json(minco.getCoeffs());
    std::cout << ",\"energy\":"; number(energy);
    std::cout << ",\"grad_c\":"; matrix_json(grad_c);
    std::cout << ",\"partial_grad_t\":"; vector_json(partial_t);
    std::cout << ",\"grad_points\":"; matrix_json(grad_points);
    std::cout << ",\"grad_times\":"; vector_json(grad_t);
    std::cout << "}";
}

void primitives()
{
    std::cout << "{\"minco3\":"; minco_reference<3>();
    std::cout << ",\"minco4\":"; minco_reference<4>();
    flatness::FlatnessMap model;
    model.reset(.752, 9.8066, .1, .15, .08, .001);
    Eigen::Vector3d velocity(.8, -.3, .2), acceleration(.5, .2, .8), jerk(.2, .7, -.4);
    Eigen::Vector3d omega, gp(.2, -.1, .4), gv(.1, .3, -.2), go(.4, -.2, .5);
    Eigen::Vector3d out_p, out_v, out_a, out_j;
    Eigen::Vector4d quat, gq(.3, -.6, .2, .1);
    double thrust, yaw_grad, yaw_rate_grad;
    model.forward(velocity, acceleration, jerk, .3, -.2, thrust, quat, omega);
    model.backward(gp, gv, .7, gq, go, out_p, out_v, out_a, out_j, yaw_grad, yaw_rate_grad);
    std::cout << ",\"flatness\":{\"thrust\":"; number(thrust);
    std::cout << ",\"quat\":"; vector_json(quat);
    std::cout << ",\"omega\":"; vector_json(omega);
    std::cout << ",\"grad_pos\":"; vector_json(out_p);
    std::cout << ",\"grad_vel\":"; vector_json(out_v);
    std::cout << ",\"grad_acc\":"; vector_json(out_a);
    std::cout << ",\"grad_jer\":"; vector_json(out_j);
    std::cout << ",\"grad_yaw\":"; number(yaw_grad);
    std::cout << ",\"grad_yaw_rate\":"; number(yaw_rate_grad);
    std::cout << "}}\n";
}

int iterations = 0;
int progress(void *, const Eigen::VectorXd &, const Eigen::VectorXd &,
             double, double, int k, int)
{
    iterations = k;
    return 0;
}

void gcopter_reference(bool optimize)
{
    using Solver = trajectory_server::GcopterSolver<3>;
    Solver solver;
    Eigen::VectorXd physical(6), bounds(5), weights(5);
    for (int i = 0; i < 6; ++i) std::cin >> physical(i);
    Eigen::Matrix3d start, end;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) std::cin >> start(i, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) std::cin >> end(i, j);
    int count;
    std::cin >> count;
    Solver::PolyhedraH corridor(count);
    for (auto &poly : corridor) {
        int rows;
        std::cin >> rows;
        poly.resize(rows, 4);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < 3; ++col) std::cin >> poly(row, col);
            double bound;
            std::cin >> bound;
            poly(row, 3) = -bound;
        }
    }
    bounds << 3, 3, .7854, 2, 10;
    weights << 10000, 100, 10000, 10000, 10000;
    if (!solver.setup(100, start, end, corridor, .15, 2, 1, .01, 16,
                      bounds, weights, physical)) {
        std::cout << "{\"setup_success\":false}\n";
        return;
    }
    Eigen::VectorXi native_counts(solver.vPolytopes.size());
    for (int i = 0; i < native_counts.size(); ++i)
        native_counts(i) = solver.vPolytopes[i].cols();
    std::cin >> count;
    solver.vPolytopes.resize(count);
    for (auto &poly : solver.vPolytopes) {
        int vertices;
        std::cin >> vertices;
        poly.resize(3, vertices);
        for (int vertex = 0; vertex < vertices; ++vertex)
            for (int axis = 0; axis < 3; ++axis) std::cin >> poly(axis, vertex);
    }
    solver.spatialDim = 0;
    for (int i = 0; i < solver.vPolyIdx.size(); ++i)
        solver.spatialDim += solver.vPolytopes[solver.vPolyIdx(i)].cols();
    std::cin >> count;
    if (count != solver.temporalDim + solver.spatialDim)
        throw std::runtime_error("GCOPTER input variable dimension mismatch");
    Eigen::VectorXd x(count), gradient(count);
    for (int i = 0; i < count; ++i) std::cin >> x(i);
    if (!std::cin) throw std::runtime_error("incomplete GCOPTER input");
    const double cost = Solver::costFunctional(&solver, x, gradient);
    std::cout << "{\"setup_success\":true,\"native_vertex_counts\":";
    vector_json(native_counts);
    std::cout << ",\"initial_cost\":"; number(cost);
    std::cout << ",\"initial_gradient\":"; vector_json(gradient);
    if (optimize) {
        lbfgs::lbfgs_parameter_t parameters;
        parameters.mem_size = 256;
        parameters.past = 3;
        parameters.min_step = 1e-32;
        parameters.g_epsilon = 0;
        parameters.delta = 1e-4;
        parameters.max_iterations = 1000;
        double final_cost;
        const auto started = std::chrono::steady_clock::now();
        const int status = lbfgs::lbfgs_optimize(
            x, final_cost, &Solver::costFunctional, nullptr, progress, &solver, parameters);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << ",\"status\":" << status << ",\"cost\":"; number(final_cost);
        std::cout << ",\"x\":"; vector_json(x);
        std::cout << ",\"iterations\":" << iterations << ",\"elapsed_s\":" << elapsed;
    }
    std::cout << "}\n";
}

int main(int argc, char **argv)
{
    std::cout << std::setprecision(17);
    try {
        if (argc < 2) throw std::runtime_error("usage: oracle primitives | gcopter [optimize]");
        const std::string mode(argv[1]);
        if (mode == "primitives") primitives();
        else if (mode == "gcopter")
            gcopter_reference(argc == 3 && std::string(argv[2]) == "optimize");
        else throw std::runtime_error("unknown oracle mode");
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}

