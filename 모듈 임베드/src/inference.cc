// TFLite C API wrapper — cascade 모드 구현.
// HOST 빌드 (단위 테스트) 에선 stub. ARM TARGET 에서 실제 libtensorflowlite_c 링크.
//
// 활성화: Makefile 에서 ARM TARGET 일 때 -DRADAR_HAS_TFLITE 정의.

#include "radar/inference.h"

#include "radar/config.h"

#ifdef RADAR_HAS_TFLITE
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/c/c_api_types.h>
#include <cmath>
#include <cstring>
#endif

namespace radar {
namespace inference {

#ifdef RADAR_HAS_TFLITE

struct TFLiteInterpreter::Impl {
    TfLiteModel* model = nullptr;
    TfLiteInterpreterOptions* options = nullptr;
    TfLiteInterpreter* interpreter = nullptr;
};

namespace {

constexpr int SEQ_LEN_C = 40;   // 시간축 보상에 사용 (config.h SEQ_LEN 과 동일)

// float buffer → 입력 텐서. dtype 에 따라 자동 quantize (int8 PTQ 모델 대응).
bool fill_input(TfLiteTensor* in, const float* src, std::size_t count) {
    const TfLiteType dtype = TfLiteTensorType(in);
    if (dtype == kTfLiteFloat32) {
        return TfLiteTensorCopyFromBuffer(in, src, count * sizeof(float)) == kTfLiteOk;
    }
    if (dtype == kTfLiteInt8) {
        const TfLiteQuantizationParams qp = TfLiteTensorQuantizationParams(in);
        if (qp.scale == 0.0f) return false;
        // 짧은 시퀀스 (~12800)이라 stack 대신 heap 1회 할당.
        std::vector<int8_t> q(count);
        for (std::size_t i = 0; i < count; ++i) {
            float v = std::round(src[i] / qp.scale + static_cast<float>(qp.zero_point));
            if (v < -128.0f) v = -128.0f;
            if (v >  127.0f) v =  127.0f;
            q[i] = static_cast<int8_t>(v);
        }
        return TfLiteTensorCopyFromBuffer(in, q.data(), count) == kTfLiteOk;
    }
    return false;
}

std::size_t elem_count(const TfLiteTensor* t) {
    const TfLiteType dtype = TfLiteTensorType(t);
    const std::size_t elem_size = (dtype == kTfLiteInt8) ? 1u : sizeof(float);
    return TfLiteTensorByteSize(t) / elem_size;
}

// 출력 텐서 → float buffer. 자동 dequantize + 시간축 (T*K → K) 처리.
//   target_count: 원하는 출력 원소 수 (human=1, pose=3)
bool read_output(const TfLiteTensor* t, float* dst, std::size_t target_count) {
    const TfLiteType dtype = TfLiteTensorType(t);
    const std::size_t total = elem_count(t);
    const TfLiteQuantizationParams qp = TfLiteTensorQuantizationParams(t);

    // raw byte buffer 복사
    std::vector<uint8_t> raw(TfLiteTensorByteSize(t));
    if (TfLiteTensorCopyToBuffer(t, raw.data(), raw.size()) != kTfLiteOk) return false;

    // 시간축 offset 계산: total == target 이면 0, total == target*T 면 마지막 timestep
    std::size_t offset = 0;
    if (total == target_count) {
        offset = 0;
    } else if (total == target_count * static_cast<std::size_t>(SEQ_LEN_C)) {
        offset = (SEQ_LEN_C - 1) * target_count;
    } else {
        return false;
    }

    auto dequant = [&](std::size_t i) -> float {
        if (dtype == kTfLiteFloat32) {
            return reinterpret_cast<const float*>(raw.data())[i];
        }
        if (dtype == kTfLiteInt8) {
            const int8_t v = reinterpret_cast<const int8_t*>(raw.data())[i];
            return (static_cast<float>(v) - static_cast<float>(qp.zero_point)) * qp.scale;
        }
        return 0.0f;
    };

    for (std::size_t i = 0; i < target_count; ++i) {
        dst[i] = dequant(offset + i);
    }
    return true;
}

}  // namespace

TFLiteInterpreter::TFLiteInterpreter(const uint8_t* model_data, std::size_t model_size, int num_threads) {
    impl_ = new Impl{};
    impl_->model = TfLiteModelCreate(model_data, model_size);
    if (!impl_->model) return;

    impl_->options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(impl_->options, num_threads);

    impl_->interpreter = TfLiteInterpreterCreate(impl_->model, impl_->options);
    if (!impl_->interpreter) return;

    if (TfLiteInterpreterAllocateTensors(impl_->interpreter) != kTfLiteOk) return;
    ready_ = true;
}

TFLiteInterpreter::~TFLiteInterpreter() {
    if (impl_) {
        if (impl_->interpreter) TfLiteInterpreterDelete(impl_->interpreter);
        if (impl_->options) TfLiteInterpreterOptionsDelete(impl_->options);
        if (impl_->model) TfLiteModelDelete(impl_->model);
        delete impl_;
    }
}

bool TFLiteInterpreter::infer(const std::vector<float>& sequence_flat,
                              const float* metadata, std::size_t metadata_size,
                              Output& out) {
    if (!ready_ || !impl_) return false;

    // 입력 0: points
    TfLiteTensor* in_pts = TfLiteInterpreterGetInputTensor(impl_->interpreter, 0);
    if (!in_pts) return false;
    if (!fill_input(in_pts, sequence_flat.data(), sequence_flat.size())) return false;

    // 입력 1: metadata (cascade)
    const int n_inputs = TfLiteInterpreterGetInputTensorCount(impl_->interpreter);
    if (n_inputs > 1) {
        if (!metadata) return false;
        TfLiteTensor* in_meta = TfLiteInterpreterGetInputTensor(impl_->interpreter, 1);
        if (!in_meta) return false;
        if (!fill_input(in_meta, metadata, metadata_size)) return false;
    }

    if (TfLiteInterpreterInvoke(impl_->interpreter) != kTfLiteOk) return false;

    // 출력 매핑 — byte size 로 어느 게 human(1) / pose(3) 인지 자동 판별
    // (변환기에 따라 출력 순서 또는 시간축 (T*K) 잔존 여부가 달라지므로)
    const TfLiteTensor* o0 = TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0);
    const TfLiteTensor* o1 = TfLiteInterpreterGetOutputTensor(impl_->interpreter, 1);
    if (!o0 || !o1) return false;

    const std::size_t c0 = elem_count(o0);
    const std::size_t c1 = elem_count(o1);

    auto is_human_count = [](std::size_t c) {
        return c == 1u || c == static_cast<std::size_t>(SEQ_LEN_C);
    };
    auto is_pose_count = [](std::size_t c) {
        return c == 3u || c == 3u * static_cast<std::size_t>(SEQ_LEN_C);
    };

    const TfLiteTensor* t_human = nullptr;
    const TfLiteTensor* t_pose  = nullptr;
    if (is_human_count(c0) && is_pose_count(c1)) {
        t_human = o0; t_pose = o1;
    } else if (is_pose_count(c0) && is_human_count(c1)) {
        t_human = o1; t_pose = o0;
    } else {
        return false;
    }

    if (!read_output(t_human, &out.human_logit, 1)) return false;
    if (!read_output(t_pose,  out.pose_logits,  3)) return false;
    return true;
}

bool TFLiteInterpreter::infer_single(const std::vector<float>& points_flat, SingleOutput& out) {
    if (!ready_ || !impl_) return false;

    TfLiteTensor* in = TfLiteInterpreterGetInputTensor(impl_->interpreter, 0);
    if (!in) return false;
    if (!fill_input(in, points_flat.data(), points_flat.size())) return false;

    if (TfLiteInterpreterInvoke(impl_->interpreter) != kTfLiteOk) return false;

    const TfLiteTensor* o0 = TfLiteInterpreterGetOutputTensor(impl_->interpreter, 0);
    const TfLiteTensor* o1 = TfLiteInterpreterGetOutputTensor(impl_->interpreter, 1);
    if (!o0 || !o1) return false;

    const std::size_t c0 = elem_count(o0);
    // person = 1 원소, pose = 9 원소. byte size 로 판별 (출력 순서 무관).
    const TfLiteTensor* t_person = (c0 == 1u) ? o0 : o1;
    const TfLiteTensor* t_pose   = (c0 == 1u) ? o1 : o0;

    if (!read_output(t_person, &out.person_logit, 1)) return false;
    if (!read_output(t_pose,   out.pose_logits,   9)) return false;
    return true;
}

bool TFLiteInterpreter::is_supported() { return true; }

#else  // !RADAR_HAS_TFLITE — HOST stub

struct TFLiteInterpreter::Impl {};

TFLiteInterpreter::TFLiteInterpreter(const uint8_t* /*model_data*/, std::size_t /*model_size*/, int /*num_threads*/) {
    // HOST 에선 TFLite 없음 — stub. ready_ = false.
}

TFLiteInterpreter::~TFLiteInterpreter() = default;

bool TFLiteInterpreter::infer(const std::vector<float>& /*sequence_flat*/,
                              const float* /*metadata*/, std::size_t /*metadata_size*/,
                              Output& out) {
    out.human_logit = 0.0f;
    out.pose_logits[0] = out.pose_logits[1] = out.pose_logits[2] = 0.0f;
    return false;
}

bool TFLiteInterpreter::infer_single(const std::vector<float>& /*points_flat*/, SingleOutput& out) {
    out.person_logit = 0.0f;
    for (int i = 0; i < 9; ++i) out.pose_logits[i] = 0.0f;
    return false;
}

bool TFLiteInterpreter::is_supported() { return false; }

#endif

}  // namespace inference
}  // namespace radar
