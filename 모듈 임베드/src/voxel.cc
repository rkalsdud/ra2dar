// Voxel downsample 구현.
// deploy/pi/cluster_preprocess.py::_voxel_downsample 의 C++ 동등.
//
// Python (참조):
//   keys = np.floor(pts[:, :3] / voxel).astype(np.int64)
//   order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))  # primary=x, then y, z
//   keys_sorted = keys[order]; pts_sorted = pts[order]
//   diff = np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)
//   first_mask = np.concatenate([[True], diff])
//   return pts_sorted[first_mask]
//
// 즉: voxel key 별로 첫 번째 입력 점만 유지. 출력 순서는 key 정렬 순.

#include "radar/voxel.h"

#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>

namespace radar {
namespace voxel {

namespace {

using Key = std::tuple<int64_t, int64_t, int64_t>;

inline Key make_key(const Point& p, float voxel_size) {
    // double 로 division — numpy 의 float / Python float 와 같은 정밀도.
    const double inv = 1.0 / static_cast<double>(voxel_size);
    return Key{
        static_cast<int64_t>(std::floor(static_cast<double>(p.x) * inv)),
        static_cast<int64_t>(std::floor(static_cast<double>(p.y) * inv)),
        static_cast<int64_t>(std::floor(static_cast<double>(p.z) * inv)),
    };
}

}  // namespace

std::vector<Point> downsample(const std::vector<Point>& points, float voxel_size) {
    if (points.empty() || voxel_size <= 0.0f) return points;

    // map<Key, first_idx>. emplace 는 키 중복 시 무시 → 첫 점 인덱스 유지.
    // std::map 은 키 정렬 — numpy lexsort (x, y, z) 와 일치.
    std::map<Key, size_t> first_idx;
    for (size_t i = 0; i < points.size(); ++i) {
        first_idx.emplace(make_key(points[i], voxel_size), i);
    }

    std::vector<Point> out;
    out.reserve(first_idx.size());
    for (const auto& kv : first_idx) {
        out.push_back(points[kv.second]);
    }
    return out;
}

}  // namespace voxel
}  // namespace radar
