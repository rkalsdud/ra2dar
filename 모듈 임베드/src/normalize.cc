// normalize_cluster + sample_or_pad 구현.
// deploy/server/model_preprocess.py 와 line-by-line 동일.

#include "radar/normalize.h"

#include <algorithm>
#include <cmath>

#include "radar/config.h"

namespace radar {
namespace normalize {

namespace {
constexpr float EPS = 1e-6f;
}

Transform normalize_cluster(std::vector<Point>& points) {
    Transform t{};
    t.scale = ROI_RADIUS;

    if (points.empty()) return t;

    // 1. centroid
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (const auto& p : points) { sx += p.x; sy += p.y; sz += p.z; }
    const float inv_n = 1.0f / static_cast<float>(points.size());
    t.cx = static_cast<float>(sx) * inv_n;
    t.cy = static_cast<float>(sy) * inv_n;
    t.cz = static_cast<float>(sz) * inv_n;

    const float roi = std::max(ROI_RADIUS, EPS);
    const float z_span = std::max(Z_MAX - Z_MIN, EPS);
    const float dop = std::max(DOPPLER_NORM, EPS);
    const float pwr = std::max(POWER_NORM, EPS);

    // 2. per-point 변환
    for (auto& p : points) {
        p.x = (p.x - t.cx) / roi;
        p.y = (p.y - t.cy) / roi;
        p.z = (p.z - Z_MIN) / z_span * 2.0f - 1.0f;
        p.doppler = p.doppler / dop;
        p.power = std::log1p(std::max(p.power, 0.0f)) / pwr;
    }
    return t;
}

std::vector<Point> sample_or_pad(
    const std::vector<Point>& points, int num_points, std::mt19937& rng) {

    const int n = static_cast<int>(points.size());
    if (num_points <= 0) return {};

    // n == 0 → zero-filled
    if (n == 0) {
        return std::vector<Point>(static_cast<size_t>(num_points));  // default-constructed = zeros
    }
    if (n == num_points) {
        return points;
    }
    if (n > num_points) {
        // random subsample without replacement — Fisher-Yates 부분 셔플
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        for (int i = 0; i < num_points; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            const int j = dist(rng);
            std::swap(indices[i], indices[j]);
        }
        std::vector<Point> out;
        out.reserve(static_cast<size_t>(num_points));
        for (int i = 0; i < num_points; ++i) out.push_back(points[indices[i]]);
        return out;
    }
    // n < num_points → 원본 그대로 + 부족분 random repeat (with replacement)
    std::vector<Point> out = points;
    out.reserve(static_cast<size_t>(num_points));
    std::uniform_int_distribution<int> dist(0, n - 1);
    while (static_cast<int>(out.size()) < num_points) {
        out.push_back(points[static_cast<size_t>(dist(rng))]);
    }
    return out;
}

}  // namespace normalize
}  // namespace radar
