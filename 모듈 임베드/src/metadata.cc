#include "radar/metadata.h"

#include <cfloat>
#include <cmath>

namespace radar::metadata {

namespace {

// numpy ddof=0 std. 2-pass 로 누적 오차 최소화 (TN=2560 정도라 1-pass 도 충분하지만 안전 우선).
inline double pop_std(const double* xs_mean_diff_sq_sum, int n) {
    double v = *xs_mean_diff_sq_sum / static_cast<double>(n);
    return v > 0.0 ? std::sqrt(v) : 0.0;
}

}  // namespace

void extract(const float* seq, int T, int N, float* out) {
    constexpr int F = 5;
    const int TN = T * N;

    // ----- 전체 통계 (z, v, p) 1-pass: mean + |v|.mean + z.max/min -----
    double z_sum = 0.0, v_sum = 0.0, p_sum = 0.0, v_abs_sum = 0.0;
    float  z_max = -FLT_MAX, z_min = FLT_MAX;
    for (int i = 0; i < TN; ++i) {
        const float z = seq[i * F + 2];
        const float v = seq[i * F + 3];
        const float p = seq[i * F + 4];
        z_sum += z;
        v_sum += v;
        p_sum += p;
        v_abs_sum += std::fabs(v);
        if (z > z_max) z_max = z;
        if (z < z_min) z_min = z;
    }
    const double z_mean = z_sum / TN;
    const double v_mean = v_sum / TN;
    const double p_mean = p_sum / TN;

    // ----- 전체 통계 2-pass: variance -----
    double z_sq = 0.0, v_sq = 0.0, p_sq = 0.0;
    for (int i = 0; i < TN; ++i) {
        const double dz = seq[i * F + 2] - z_mean;
        const double dv = seq[i * F + 3] - v_mean;
        const double dp = seq[i * F + 4] - p_mean;
        z_sq += dz * dz;
        v_sq += dv * dv;
        p_sq += dp * dp;
    }
    const double z_std = pop_std(&z_sq, TN);
    const double v_std = pop_std(&v_sq, TN);
    const double p_std = pop_std(&p_sq, TN);

    // ----- x/y spread: 각 timestep 의 N 점 std → T 개 → 평균 -----
    double x_std_sum = 0.0, y_std_sum = 0.0;
    for (int t = 0; t < T; ++t) {
        double xs = 0.0, ys = 0.0;
        for (int n = 0; n < N; ++n) {
            xs += seq[(t * N + n) * F + 0];
            ys += seq[(t * N + n) * F + 1];
        }
        const double xm = xs / N;
        const double ym = ys / N;
        double xsq = 0.0, ysq = 0.0;
        for (int n = 0; n < N; ++n) {
            const double dx = seq[(t * N + n) * F + 0] - xm;
            const double dy = seq[(t * N + n) * F + 1] - ym;
            xsq += dx * dx;
            ysq += dy * dy;
        }
        x_std_sum += pop_std(&xsq, N);
        y_std_sum += pop_std(&ysq, N);
    }

    // cascade v2 — METADATA_DIM=8 (v1: 11 → v2: 8)
    // 제거된 항목: z_max, z_range, v_std
    // 이유: 학습에서 metadata shortcut 차단 → 포복 사람 검출 97%.
    // 학습 코드 (build_notebook_cascade_v2.py 의 extract_metadata_np) 와 1:1 일치 유지.
    out[0]  = static_cast<float>(z_mean);
    out[1]  = static_cast<float>(z_std);
    out[2]  = z_min;                                    // v1 idx 3
    out[3]  = static_cast<float>(v_abs_sum / TN);       // v1 idx 5
    out[4]  = static_cast<float>(p_mean);               // v1 idx 7
    out[5]  = static_cast<float>(p_std);                // v1 idx 8
    out[6]  = static_cast<float>(x_std_sum / T);        // v1 idx 9
    out[7]  = static_cast<float>(y_std_sum / T);        // v1 idx 10
    // 사용 안 함: z_max, z_range (=z_max-z_min), v_std
}

}  // namespace radar::metadata
