// SOR 구현.
// deploy/pi/cluster_preprocess.py::_statistical_outlier_removal 의 C++ 동등.
//
// Python (참조):
//   nn = NearestNeighbors(n_neighbors=min(k+1, N))
//   nn.fit(pts); dist, _ = nn.kneighbors(pts)        # dist[:, 0] == 0 (self)
//   mean_dist = dist[:, 1:].mean(axis=1)              # k 이웃 평균
//   mu = mean_dist.mean(); sigma = mean_dist.std()    # numpy std = ddof=0
//   mask = mean_dist <= mu + std_ratio * sigma
//   if mask.all(): return pf else return pf[mask]

#include "radar/sor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace radar {
namespace sor {

namespace {

inline float sq_dist3(const Point& a, const Point& b) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

}  // namespace

std::vector<Point> filter(const std::vector<Point>& points, int k, float std_ratio) {
    const size_t n = points.size();
    if (k <= 0 || std_ratio <= 0.0f) return points;
    if (n == 0 || n <= static_cast<size_t>(k)) return points;   // Python no-op

    const int neighbors = k + 1;   // self 포함 k+1 — sklearn 과 동일

    // 1. 각 점의 (self 제외) k 이웃 평균 거리
    //    Brute-force O(N²). Voxel 통과 후 N ~ 수십~수백 — 충분히 빠름.
    std::vector<float> mean_dist(n);
    std::vector<float> dists(n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            dists[j] = std::sqrt(sq_dist3(points[i], points[j]));
        }
        // partial sort: 가장 작은 'neighbors' 개를 [0..neighbors) 에 배치
        std::nth_element(dists.begin(),
                         dists.begin() + neighbors,
                         dists.end());
        // self 거리 (가장 작은 값, 보통 0) 1개 제외 후 평균
        float sum = 0.0f;
        float min_d = dists[0];
        for (int j = 0; j < neighbors; ++j) {
            sum += dists[j];
            if (dists[j] < min_d) min_d = dists[j];
        }
        mean_dist[i] = (sum - min_d) / static_cast<float>(k);
    }

    // 2. μ, σ — numpy 와 동일 ddof=0
    double mu = 0.0;
    for (float v : mean_dist) mu += v;
    mu /= static_cast<double>(n);

    double var = 0.0;
    for (float v : mean_dist) {
        const double d = static_cast<double>(v) - mu;
        var += d * d;
    }
    var /= static_cast<double>(n);
    const float threshold = static_cast<float>(
        mu + static_cast<double>(std_ratio) * std::sqrt(var));

    // 3. mask 적용
    std::vector<Point> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (mean_dist[i] <= threshold) {
            out.push_back(points[i]);
        }
    }
    return out;   // out.size() == n 인 경우 = mask.all() (input 그대로)
}

}  // namespace sor
}  // namespace radar
