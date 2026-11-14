// Offline I/O driver; all planning mathematics comes from the upstream header.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <Eigen/Dense>
#include <libqhullcpp/QhullVertex.h>
#include "PathCover.hpp"

using Point = Eigen::Vector3d;
using Planes = Eigen::Matrix<double, -1, 3>;

template<class Matrix>
void matrix(const Matrix& a) {
    std::cout << "[";
    for (int i = 0; i < a.rows(); ++i) {
        if (i) std::cout << ",";
        std::cout << "[";
        for (int j = 0; j < a.cols(); ++j) {
            if (j) std::cout << ",";
            std::cout << a(i, j);
        }
        std::cout << "]";
    }
    std::cout << "]";
}

template<class Vector>
void vector(const Vector& a) {
    std::cout << "[";
    for (int i = 0; i < a.size(); ++i) {
        if (i) std::cout << ",";
        std::cout << a[i];
    }
    std::cout << "]";
}

void polytope(const Planes& a, const Eigen::VectorXd& b) {
    std::cout << "{\"A\":";
    matrix(a);
    std::cout << ",\"b\":";
    vector(b);
    std::cout << "}";
}

int main() {
    int n, m, horizon, random_seed;
    if (!(std::cin >> n >> m >> horizon >> random_seed) || n < 0 || m < 1 || horizon < 1)
        return 2;
    std::vector<Point> obstacles(n), path(m);
    for (auto& p : obstacles) std::cin >> p[0] >> p[1] >> p[2];
    for (auto& p : path) std::cin >> p[0] >> p[1] >> p[2];
    Planes bounds(6, 3);
    Eigen::VectorXd offsets(6);
    for (int i = 0; i < 6; ++i)
        std::cin >> bounds(i, 0) >> bounds(i, 1) >> bounds(i, 2) >> offsets[i];
    if (!std::cin) return 2;

    Planes a;
    Eigen::VectorXd b;
    std::vector<int> count;
    std::srand(random_seed);
    PathCover::RISP<double, 3>(obstacles, path.front(), bounds, offsets, a, b, count);
    std::vector<Planes> aa;
    std::vector<Eigen::VectorXd> bb;
    std::vector<Point> seeds;
    std::vector<std::vector<int>> counts;
    std::srand(random_seed);
    const auto begin = std::chrono::steady_clock::now();
    PathCover::pathCover<double, 3>(obstacles, path, bounds, offsets, aa, bb, seeds, counts, horizon);
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

    std::cout << std::setprecision(17) << "{\"risp\":";
    polytope(a, b);
    std::cout << ",\"risp_counts\":";
    vector(count);
    std::cout << ",\"polytopes\":[";
    for (int i = 0; i < aa.size(); ++i) {
        if (i) std::cout << ",";
        polytope(aa[i], bb[i]);
    }
    std::cout << "],\"seeds\":[";
    for (int i = 0; i < seeds.size(); ++i) {
        if (i) std::cout << ",";
        vector(seeds[i]);
    }
    std::cout << "],\"counts\":[";
    int random_count = 0;
    for (int i = 0; i < counts.size(); ++i) {
        if (i) std::cout << ",";
        vector(counts[i]);
        random_count += counts[i].size() - 1;
    }
    std::cout << "],\"rand_values\":[";
    std::srand(random_seed);
    for (int i = 0; i < random_count; ++i) {
        if (i) std::cout << ",";
        std::cout << std::rand();
    }
    std::cout << "],\"wall_s\":" << wall << "}\n";
}
