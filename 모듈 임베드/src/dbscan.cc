// 4D DBSCAN 구현.
// sklearn.cluster.DBSCAN 과 동일 의미: core point는 eps 안에 (자기 포함) min_samples 이상.

#include "radar/dbscan.h"

#include <queue>
#include <vector>

namespace radar {
namespace dbscan {

namespace {

inline float sq_dist_4d(const Point& a, const Point& b, float v_alpha) {
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    const float dv = (a.doppler - b.doppler) * v_alpha;
    return dx*dx + dy*dy + dz*dz + dv*dv;
}

}  // namespace

Result cluster(const std::vector<Point>& points,
               float eps, int min_samples, float v_alpha) {
    Result out;
    const size_t n = points.size();
    out.labels.assign(n, -1);
    out.n_clusters = 0;
    if (n == 0 || eps <= 0.0f || min_samples <= 0) return out;

    const float eps_sq = eps * eps;

    // 1. 이웃 리스트 (자기 제외) 미리 계산
    std::vector<std::vector<int>> neighbors(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (sq_dist_4d(points[i], points[j], v_alpha) <= eps_sq) {
                neighbors[i].push_back(static_cast<int>(j));
            }
        }
    }

    // sklearn: core 정의 = 이웃 카운트 (자기 포함) >= min_samples
    //   → 자기 제외 neighbors.size() >= (min_samples - 1)
    const int core_min = min_samples - 1;

    std::vector<bool> visited(n, false);
    int cid = 0;

    for (size_t i = 0; i < n; ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        if (static_cast<int>(neighbors[i].size()) < core_min) {
            // 시작점이 core 아님 — noise (단, 나중에 다른 cluster 의 border 로 재배정 가능)
            continue;
        }

        // 새 cluster 시작
        out.labels[i] = cid;

        std::queue<int> q;
        for (int nb : neighbors[i]) q.push(nb);

        while (!q.empty()) {
            const int p = q.front(); q.pop();
            // noise 였더라도 cluster border 로 흡수
            if (out.labels[p] == -1) {
                out.labels[p] = cid;
            }
            if (!visited[p]) {
                visited[p] = true;
                if (static_cast<int>(neighbors[p].size()) >= core_min) {
                    // p 도 core → 그 이웃들 expansion 에 추가
                    for (int nb2 : neighbors[p]) q.push(nb2);
                }
            }
        }
        ++cid;
    }

    out.n_clusters = cid;
    return out;
}

std::vector<Cluster> extract_clusters(
    const std::vector<Point>& points,
    const std::vector<int>& labels,
    int n_clusters,
    int min_cluster_points) {

    std::vector<Cluster> out;
    if (n_clusters <= 0) return out;

    for (int cid = 0; cid < n_clusters; ++cid) {
        Cluster c;
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (size_t i = 0; i < points.size(); ++i) {
            if (labels[i] == cid) {
                c.points.push_back(points[i]);
                sx += points[i].x; sy += points[i].y; sz += points[i].z;
            }
        }
        if (static_cast<int>(c.points.size()) < min_cluster_points) continue;

        const float inv_n = 1.0f / static_cast<float>(c.points.size());
        c.centroid_x = static_cast<float>(sx) * inv_n;
        c.centroid_y = static_cast<float>(sy) * inv_n;
        c.centroid_z = static_cast<float>(sz) * inv_n;
        out.push_back(std::move(c));
    }
    return out;
}

}  // namespace dbscan
}  // namespace radar
