// Offline driver; the partition mathematics is the unchanged PathCover RISP.
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
int main() {
    int n, random_seed;
    if (!(std::cin >> n >> random_seed) || n < 0) return 2;
    std::vector<Eigen::Vector3d> obstacles(n);
    for (auto& p : obstacles) std::cin >> p[0] >> p[1] >> p[2];
    Eigen::Vector3d seed;
    std::cin >> seed[0] >> seed[1] >> seed[2];
    Eigen::Matrix<double,-1,3> bounds(6,3), a;
    Eigen::VectorXd offsets(6), b;
    for(int i=0;i<6;++i) std::cin >> bounds(i,0) >> bounds(i,1) >> bounds(i,2) >> offsets[i];
    if (!std::cin) return 2;
    std::vector<int> counts;
    std::srand(random_seed);
    const auto begin=std::chrono::steady_clock::now();
    PathCover::RISP<double,3>(obstacles,seed,bounds,offsets,a,b,counts);
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    std::cout << std::setprecision(17) << "{\"risp\":{\"A\":";
    matrix(a);
    std::cout << ",\"b\":";
    vector(b);
    std::cout << "},\"risp_counts\":";
    vector(counts);
    std::cout << ",\"rand_values\":[";
    std::srand(random_seed);
    for(size_t i=0;i+1<counts.size();++i) {
        if(i) std::cout << ",";
        std::cout << std::rand();
    }
    std::cout << "],\"wall_s\":" << wall << "}\n";
}
