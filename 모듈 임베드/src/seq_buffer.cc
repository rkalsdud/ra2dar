// SequenceBuffer 구현.
// deploy/server/model_preprocess.py::TrackSeqBuffer 와 동일 emit 조건.
// + 조기 emit (early_emit_min) — 첫 박스 지연 단축용.

#include "radar/seq_buffer.h"

namespace radar {

SequenceBuffer::SequenceBuffer(int seq_len, int num_points, int stride, int features,
                                int early_emit_min)
    : seq_len_(seq_len), num_points_(num_points), stride_(stride), features_(features),
      early_emit_min_(early_emit_min) {}

void SequenceBuffer::reset() {
    history_.clear();
    frame_idx_ = 0;
    emit_count_ = 0;
}

std::vector<float> SequenceBuffer::push(const std::vector<Point>& sampled) {
    // 입력 → flat float (num_points × features)
    const size_t frame_size = static_cast<size_t>(num_points_) * static_cast<size_t>(features_);
    std::vector<float> flat(frame_size, 0.0f);

    const int n = std::min(static_cast<int>(sampled.size()), num_points_);
    for (int i = 0; i < n; ++i) {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(features_);
        flat[base + 0] = sampled[i].x;
        flat[base + 1] = sampled[i].y;
        flat[base + 2] = sampled[i].z;
        flat[base + 3] = sampled[i].doppler;
        flat[base + 4] = sampled[i].power;
    }

    // ring 동작: 가득 차면 가장 오래된 frame pop
    if (static_cast<int>(history_.size()) >= seq_len_) {
        history_.pop_front();
    }
    history_.push_back(std::move(flat));
    ++frame_idx_;

    const int size = static_cast<int>(history_.size());
    const bool full = (size == seq_len_);
    const bool stride_match = (frame_idx_ % stride_ == 0);

    // 표준 emit: 버퍼 가득 + stride 정렬.
    // 조기 emit: 가득 차기 전이라도 EARLY_EMIT_MIN_FRAMES 이상 쌓였고 stride 정렬이면
    //   가장 오래된 프레임을 앞쪽 슬롯에 복제 패딩해서 한 시퀀스(seq_len) 를 만들어 emit.
    //   → 첫 박스가 SEQ_LEN/FPS(=2s) 안 기다리고 EARLY_EMIT_MIN_FRAMES/FPS(=0.2s) 안에 나옴.
    const bool early = !full && stride_match &&
                       (early_emit_min_ > 0) &&
                       (size >= early_emit_min_);

    if (!(full && stride_match) && !early) return {};

    ++emit_count_;
    std::vector<float> seq;
    seq.reserve(static_cast<size_t>(seq_len_) * frame_size);

    if (early) {
        const int pad_count = seq_len_ - size;
        const auto& oldest = history_.front();
        for (int p = 0; p < pad_count; ++p) {
            seq.insert(seq.end(), oldest.begin(), oldest.end());
        }
    }
    for (const auto& f : history_) {
        seq.insert(seq.end(), f.begin(), f.end());
    }
    return seq;
}

}  // namespace radar
