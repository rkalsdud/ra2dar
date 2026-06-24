// ROITracker 구현.

#include "radar/roi_tracker.h"

#include <algorithm>
#include <cmath>

#include "radar/config.h"   // EARLY_EMIT_MIN_FRAMES — 신규 ROI 버퍼 생성 시 주입

namespace radar {

ROITracker::ROITracker(int seq_len, int num_points, int stride, int features,
                       float match_dist, int max_concurrent,
                       float centroid_alpha, int idle_timeout_frames,
                       int max_seq_gap)
    : seq_len_(seq_len), num_points_(num_points), stride_(stride), features_(features),
      match_dist_(match_dist), max_concurrent_(max_concurrent),
      centroid_alpha_(centroid_alpha), idle_timeout_frames_(idle_timeout_frames),
      max_seq_gap_(max_seq_gap) {}

ROITracker::PushResult ROITracker::push(
    const std::vector<Point>& sampled_normed, float cx, float cy, uint32_t frame_counter) {

    // 1. 가장 가까운 ROI 찾기 (match_dist_ 안)
    int best_idx = -1;
    float best_dist = match_dist_;
    for (size_t i = 0; i < rois_.size(); ++i) {
        const float dx = rois_[i].centroid_x - cx;
        const float dy = rois_[i].centroid_y - cy;
        const float d = std::sqrt(dx*dx + dy*dy);
        if (d < best_dist) {
            best_dist = d;
            best_idx = static_cast<int>(i);
        }
    }

    if (best_idx >= 0) {
        auto& r = rois_[best_idx];
        // 갭 가드 — 인접 push 사이에 레이더 프레임이 max_seq_gap 초과로 비면
        // 시간축 불연속(학습 분포 밖) → 버퍼 리셋 후 새 시퀀스 시작.
        if (max_seq_gap_ >= 0 && frame_counter > r.last_frame_counter) {
            const uint32_t gap = frame_counter - r.last_frame_counter - 1u;  // 빠진 프레임 수
            if (static_cast<int>(gap) > max_seq_gap_) r.buffer->reset();
        }
        r.last_frame_counter = frame_counter;
        const float a = centroid_alpha_;
        r.centroid_x = (1.0f - a) * r.centroid_x + a * cx;
        r.centroid_y = (1.0f - a) * r.centroid_y + a * cy;
        r.last_seen_global_frame = global_frame_;
        auto seq = r.buffer->push(sampled_normed);
        return PushResult{r.id, std::move(seq)};
    }

    // 2. 새 ROI 생성 (필요 시 가장 오래된 ROI 제거)
    if (static_cast<int>(rois_.size()) >= max_concurrent_) {
        auto it = std::min_element(rois_.begin(), rois_.end(),
            [](const ROI& a, const ROI& b) {
                return a.last_seen_global_frame < b.last_seen_global_frame;
            });
        if (it != rois_.end()) rois_.erase(it);
    }

    ROI new_roi;
    new_roi.id = next_id_++;
    new_roi.centroid_x = cx;
    new_roi.centroid_y = cy;
    new_roi.last_seen_global_frame = global_frame_;
    new_roi.last_frame_counter = frame_counter;
    new_roi.buffer = std::make_unique<SequenceBuffer>(
        seq_len_, num_points_, stride_, features_, EARLY_EMIT_MIN_FRAMES);
    auto seq = new_roi.buffer->push(sampled_normed);
    const uint32_t id = new_roi.id;
    rois_.push_back(std::move(new_roi));
    return PushResult{id, std::move(seq)};
}

void ROITracker::end_frame() {
    ++global_frame_;
    rois_.erase(std::remove_if(rois_.begin(), rois_.end(),
        [this](const ROI& r) {
            return global_frame_ - r.last_seen_global_frame > idle_timeout_frames_;
        }), rois_.end());
}

void ROITracker::reset() {
    rois_.clear();
    next_id_ = 0;
    global_frame_ = 0;
}

void ROITracker::update_inference(uint32_t roi_id, float human_logit, int pose_idx) {
    for (auto& r : rois_) {
        if (r.id == roi_id) {
            r.has_inference = true;
            r.cached_human_logit = human_logit;
            r.cached_pose_idx = pose_idx;
            return;
        }
    }
}

bool ROITracker::get_cached_inference(uint32_t roi_id, float& human_logit, int& pose_idx) const {
    for (const auto& r : rois_) {
        if (r.id == roi_id) {
            if (!r.has_inference) return false;
            human_logit = r.cached_human_logit;
            pose_idx = r.cached_pose_idx;
            return true;
        }
    }
    return false;
}

}  // namespace radar
