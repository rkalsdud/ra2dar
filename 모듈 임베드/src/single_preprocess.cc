#include "radar/single_preprocess.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>

#include "radar/config.h"
#include "radar/dbscan.h"
#include "radar/sor.h"

namespace radar {
namespace single {

namespace {
using Key = std::tuple<int64_t, int64_t, int64_t>;

inline Key make_key(const Point& p, float vs) {
    const double inv = 1.0 / static_cast<double>(vs);
    return Key{
        static_cast<int64_t>(std::floor(static_cast<double>(p.x) * inv)),
        static_cast<int64_t>(std::floor(static_cast<double>(p.y) * inv)),
        static_cast<int64_t>(std::floor(static_cast<double>(p.z) * inv)),
    };
}
}  // namespace

// ---- voxel 평균 (cluster_process.voxel_downsample) ----
// 같은 voxel 의 모든 점 xyz/V/P 를 평균. 출력 순서 = voxel key 정렬 (np.unique 와 동일).
std::vector<Point> voxel_mean(const std::vector<Point>& pts, float vs) {
    if (pts.empty() || vs <= 0.0f) return pts;

    std::map<Key, std::vector<size_t>> groups;
    for (size_t i = 0; i < pts.size(); ++i) {
        groups[make_key(pts[i], vs)].push_back(i);
    }

    std::vector<Point> out;
    out.reserve(groups.size());
    for (const auto& kv : groups) {
        double sx = 0, sy = 0, sz = 0, sv = 0, sp = 0;
        for (size_t i : kv.second) {
            sx += pts[i].x; sy += pts[i].y; sz += pts[i].z;
            sv += pts[i].doppler; sp += pts[i].power;
        }
        const double n = static_cast<double>(kv.second.size());
        Point avg;
        avg.x = static_cast<float>(sx / n);
        avg.y = static_cast<float>(sy / n);
        avg.z = static_cast<float>(sz / n);
        avg.doppler = static_cast<float>(sv / n);
        avg.power = static_cast<float>(sp / n);
        avg.track_id = -1;
        out.push_back(avg);
    }
    return out;
}

// ---- cluster local 정규화 (cluster_process._normalize_cluster_points) ----
void normalize_cluster(const std::vector<Point>& cp,
                       std::vector<float>& out, float& cx, float& cy, float& zmin) {
    const size_t n = cp.size();
    double sx = 0, sy = 0;
    float zm = FLT_MAX;
    for (const auto& p : cp) {
        sx += p.x; sy += p.y;
        if (p.z < zm) zm = p.z;
    }
    cx = static_cast<float>(sx / static_cast<double>(n));
    cy = static_cast<float>(sy / static_cast<double>(n));
    zmin = zm;

    out.resize(n * 5);
    for (size_t i = 0; i < n; ++i) {
        const Point& p = cp[i];
        out[i * 5 + 0] = (p.x - cx) / radar::SF_NORM_SCALE;
        out[i * 5 + 1] = (p.y - cy) / radar::SF_NORM_SCALE;
        out[i * 5 + 2] = (p.z - zmin) / radar::SF_NORM_SCALE;
        out[i * 5 + 3] = p.doppler / radar::SF_V_SCALE;
        out[i * 5 + 4] = p.power / radar::SF_P_SCALE;
    }
}

namespace {
// ---- FPS (cluster_process._fps_sample_indices) ----
// n <= target: 0..n-1 + random repetition padding.
// n >  target: farthest point sampling (xyz 거리).
// ⚠️ RNG 가 Python RandomState 와 다르므로 정확한 점 선택은 parity 제외 (cascade sample_or_pad 와 동일).
void fps_sample(const std::vector<float>& pts5, int n, int target,
                std::vector<float>& out, std::mt19937& rng) {
    out.resize(static_cast<size_t>(target) * 5);
    std::vector<int> sel(target);

    if (n <= target) {
        for (int i = 0; i < n; ++i) sel[i] = i;
        std::uniform_int_distribution<int> d(0, n - 1);
        for (int i = n; i < target; ++i) sel[i] = d(rng);
    } else {
        std::vector<double> dist(static_cast<size_t>(n), DBL_MAX);
        std::uniform_int_distribution<int> d(0, n - 1);
        sel[0] = d(rng);
        for (int k = 1; k < target; ++k) {
            const int last = sel[k - 1];
            const float lx = pts5[last * 5 + 0];
            const float ly = pts5[last * 5 + 1];
            const float lz = pts5[last * 5 + 2];
            double best = -1.0;
            int best_idx = 0;
            for (int j = 0; j < n; ++j) {
                const double dx = pts5[j * 5 + 0] - lx;
                const double dy = pts5[j * 5 + 1] - ly;
                const double dz = pts5[j * 5 + 2] - lz;
                const double dd = dx * dx + dy * dy + dz * dz;
                if (dd < dist[j]) dist[j] = dd;
                if (dist[j] > best) { best = dist[j]; best_idx = j; }
            }
            sel[k] = best_idx;
        }
    }

    for (int k = 0; k < target; ++k) {
        const int s = sel[k];
        for (int c = 0; c < 5; ++c) out[k * 5 + c] = pts5[s * 5 + c];
    }
}
}  // namespace

std::vector<ClusterInput> process_frame(const std::vector<Point>& raw, std::mt19937& rng) {
    std::vector<ClusterInput> result;

    // 1) V/P/Z 필터 — cluster_process 기본값 OFF, skip.
    // 2) voxel 평균
    auto v = voxel_mean(raw, radar::SF_VOXEL_SIZE);
    if (static_cast<int>(v.size()) < radar::SF_MIN_AFTER_VOXEL) return result;

    // 3) SOR
    auto s = radar::sor::filter(v, radar::SF_SOR_NB, radar::SF_SOR_STD);
    if (static_cast<int>(s.size()) < radar::SF_DBSCAN_MIN_PTS) return result;

    // 4) 3D DBSCAN (v_alpha=0 → V 미사용) + cluster 필터
    auto db = radar::dbscan::cluster(s, radar::SF_DBSCAN_EPS, radar::SF_DBSCAN_MIN_PTS, /*v_alpha=*/0.0f);
    auto clusters = radar::dbscan::extract_clusters(s, db.labels, db.n_clusters, radar::SF_MIN_CLUSTER);
    if (clusters.empty()) return result;

    // 5) per-cluster: 정규화 + FPS + 절대좌표 bbox
    for (const auto& c : clusters) {
        std::vector<float> pts5;
        ClusterInput ci;
        normalize_cluster(c.points, pts5, ci.centroid_x, ci.centroid_y, ci.z_min);
        ci.n_orig = static_cast<int>(c.points.size());
        fps_sample(pts5, ci.n_orig, radar::SF_NUM_POINTS, ci.points, rng);

        // 절대좌표 bbox (정규화 전 점들)
        ci.min_x = ci.max_x = c.points[0].x;
        ci.min_y = ci.max_y = c.points[0].y;
        ci.min_z = ci.max_z = c.points[0].z;
        for (const auto& p : c.points) {
            if (p.x < ci.min_x) ci.min_x = p.x; if (p.x > ci.max_x) ci.max_x = p.x;
            if (p.y < ci.min_y) ci.min_y = p.y; if (p.y > ci.max_y) ci.max_y = p.y;
            if (p.z < ci.min_z) ci.min_z = p.z; if (p.z > ci.max_z) ci.max_z = p.z;
        }
        result.push_back(std::move(ci));
    }
    return result;
}

}  // namespace single
}  // namespace radar
