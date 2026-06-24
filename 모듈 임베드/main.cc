// radar_embed entrypoint — 전체 파이프라인 통합 (Phase 7 + 9).
//
// 운영 흐름:
//   SDK callback (raw point cloud)
//     → process_frame()
//       → Z 필터 → Voxel → SOR → 4D DBSCAN → cluster 추출
//       → 각 cluster: normalize + sample_or_pad → ROITracker.push
//                     → 시퀀스 emit 시 TFLite inference
//       → Frame { points (track_id 부여), tracks (status + human_prob) } 구성
//       → Publisher 가 TCP 29172 로 직렬화 송신
//     → ROITracker.end_frame()
//
// 빌드:
//   make user_module          (HOST, demo_run 만 동작 — TFLite/Publisher stub)
//   make TARGET=arm           (ARM 크로스 컴파일, 실제 추론 + TCP 서버)
//
// 실행:
//   ./build/host/user_module             (HOST demo)
//   ./build/host/user_module --no-demo   (SDK 콜백 대기 모드, ARM 에서만 의미)

#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "radar/config.h"
#include "radar/dbscan.h"
#include "radar/inference.h"
#include "radar/json_publisher.h"
#include "radar/metadata.h"
#include "radar/normalize.h"
#include "radar/roi_tracker.h"
#include "radar/single_preprocess.h"
#include "radar/sor.h"
#include "radar/types.h"
#include "radar/voxel.h"

#ifndef _WIN32
#include <csignal>
#include <sched.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include "radar/radar_input.h"
#include "radar/radar_socket.h"
#endif

extern "C" {
extern unsigned char model_tflite[];          // cascade (시퀀스)
extern unsigned int  model_tflite_len;
extern unsigned char model2_tflite[];          // single (warmup)
extern unsigned int  model2_tflite_len;
}

namespace {

// ============ Pipeline state (전역 — main 에서 한 번만 init) ============

std::unique_ptr<radar::JsonPublisher> g_json_publisher;
std::unique_ptr<radar::ROITracker> g_tracker;
std::unique_ptr<radar::inference::TFLiteInterpreter> g_inferencer;        // cascade
std::unique_ptr<radar::inference::TFLiteInterpreter> g_single_inferencer; // warmup
std::mt19937 g_rng(0);          // cascade 파이프라인 전용 (메인 쓰레드)
std::mt19937 g_single_rng(1);   // single 파이프라인 전용 (워커 쓰레드) — 별도 RNG 로 race 방지
uint32_t g_frame_count = 0;

#ifndef _WIN32
// 호출 쓰레드를 특정 코어에 핀 (cascade=2, single=3).
void pin_thread_core(int core);
// 본체는 POSIX 블록에서 정의. compute_single_detections 등 앞쪽에서도 호출용으로 forward decl.
void rlog(const char* msg);
#endif

// pose 3-class → 와이어 TargetStatus 매핑.
// 학습된 모델: 0=upright, 1=horizontal, 2=low
radar::TargetStatus pose_to_status(int pose_idx) {
    switch (pose_idx) {
        case 1: return radar::TargetStatus::Lying;     // horizontal
        case 2: return radar::TargetStatus::Sitting;   // low
        default: return radar::TargetStatus::Standing; // upright
    }
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ============ cascade 파이프라인 (메인 쓰레드, 코어 2) ============
// raw → Z필터 → voxel → SOR → 4D DBSCAN → cluster → normalize → tracker → cascade inference.
// out_frame 에 points(noise+cluster) 와 tracks(시퀀스 추론 결과) 를 채운다.
void run_cascade(const std::vector<radar::Point>& raw_points,
                 uint32_t frame_counter, radar::Frame& out_frame) {
    // 노이즈 환경 안전망: raw 가 RAW_INPUT_CAP 초과면 stride 균등 subsample.
    // SOR/DBSCAN O(N²) 의 최악 케이스 처리시간을 capped 유지. clean 환경에선 no-op.
    const std::vector<radar::Point>* src = &raw_points;
    std::vector<radar::Point> capped;
    if (static_cast<int>(raw_points.size()) > radar::RAW_INPUT_CAP) {
        const std::size_t step = raw_points.size() /
                                 static_cast<std::size_t>(radar::RAW_INPUT_CAP) + 1;
        capped.reserve(radar::RAW_INPUT_CAP);
        for (std::size_t i = 0; i < raw_points.size(); i += step) {
            capped.push_back(raw_points[i]);
        }
        src = &capped;
    }

    std::vector<radar::Point> z_filt;
    z_filt.reserve(src->size());
    for (const auto& p : *src) {
        if (p.z >= radar::Z_FILTER_MIN && p.z <= radar::Z_FILTER_MAX) z_filt.push_back(p);
    }

    auto voxeled = radar::voxel::downsample(z_filt, radar::VOXEL_SIZE);
    auto sored = radar::SOR_ENABLED
        ? radar::sor::filter(voxeled, radar::SOR_NB_NEIGHBORS, radar::SOR_STD_RATIO)
        : voxeled;

    auto dbres = radar::dbscan::cluster(sored,
        radar::DBSCAN_EPS, radar::DBSCAN_MIN_SAMPLES, radar::DBSCAN_V_ALPHA);
    auto clusters = radar::dbscan::extract_clusters(
        sored, dbres.labels, dbres.n_clusters, radar::MIN_CLUSTER_POINTS);

    // 살아남은 cluster 라벨 미리 카운트 — extract_clusters 는 MIN_CLUSTER_POINTS 미만 cluster 를
    // 통째 drop 하므로, 그 점들이 "배경(label≥0)" 도 "노이즈(label=-1)" 도 아니라 publish 누락됐던
    // 버그가 있음. 사람 등장 후 cluster 가 임계까지 자라기 전까지(=수 초) 점이 안 보이던 증상의 원인.
    // 살아남은(=size≥MIN_CLUSTER_POINTS) cluster 라벨의 집합을 먼저 만든 뒤, 그 외 점은 모두
    // background(track_id=-1)로 publish — DBSCAN-노이즈 + 작은 cluster 점 둘 다 즉시 가시화.
    std::vector<int> label_count(dbres.n_clusters > 0 ? dbres.n_clusters : 1, 0);
    for (std::size_t i = 0; i < dbres.labels.size(); ++i) {
        const int lbl = dbres.labels[i];
        if (lbl >= 0 && lbl < dbres.n_clusters) ++label_count[lbl];
    }
    for (std::size_t i = 0; i < sored.size(); ++i) {
        if (i >= dbres.labels.size()) continue;
        const int lbl = dbres.labels[i];
        const bool in_kept_cluster =
            (lbl >= 0 && lbl < dbres.n_clusters &&
             label_count[lbl] >= radar::MIN_CLUSTER_POINTS);
        if (in_kept_cluster) continue;   // cluster 루프가 노랑·박스로 처리
        radar::Point pp = sored[i];
        pp.track_id = -1;
        out_frame.points.push_back(pp);
    }

    for (auto& c : clusters) {
        std::vector<radar::Point> orig = c.points;   // 원본 점 (publish 용, 절대 좌표)
        if (orig.empty()) continue;

        // bbox 사전 계산 — size filter 용 (normalize 전 절대좌표 기준).
        float minx = orig[0].x, maxx = minx;
        float miny = orig[0].y, maxy = miny;
        float minz = orig[0].z, maxz = minz;
        for (const auto& p : orig) {
            if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
            if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
            if (p.z < minz) minz = p.z; if (p.z > maxz) maxz = p.z;
        }
        const float dx = maxx - minx;
        const float dy = maxy - miny;
        const float dz = maxz - minz;
        const float max_axis = std::fmax(std::fmax(dx, dy), dz);
        const float min_axis = std::fmin(std::fmin(dx, dy), dz);

        // ghost 필터 — 박스가 너무 작으면 track/추론 모두 skip, 배경 점만 publish.
        // SequenceBuffer 비우지 않고 그냥 push 안 하는 거라 후속 cluster 영향 0.
        if (max_axis < radar::CLUSTER_MIN_BOX_EXTENT_M ||
            min_axis < radar::CLUSTER_MIN_BOX_AXIS_M) {
            for (auto& p : orig) {
                p.track_id = -1;   // 배경 — viewer 가 작은 점으로 표시
                out_frame.points.push_back(p);
            }
            continue;
        }

        radar::normalize::normalize_cluster(c.points);
        auto sampled = radar::normalize::sample_or_pad(c.points, radar::NUM_POINTS, g_rng);

        auto result = g_tracker->push(sampled, c.centroid_x, c.centroid_y, frame_counter);

        // 원본 점에 roi_id 부여 — bbox 는 위에서 이미 계산.
        for (auto& p : orig) {
            p.track_id = static_cast<int32_t>(result.roi_id);
            out_frame.points.push_back(p);
        }

        // 시퀀스 emit 시 추론 → ROI 캐시 갱신 (라벨용).
        if (!result.sequence.empty() && g_inferencer && g_inferencer->is_ready()) {
            radar::inference::Output inf;
            float meta[radar::METADATA_DIM];
            radar::metadata::extract(result.sequence.data(),
                                     radar::SEQ_LEN, radar::NUM_POINTS, meta);
            if (g_inferencer->infer(result.sequence, meta, radar::METADATA_DIM, inf)) {
                int pose_idx = 0;
                for (int i = 1; i < 3; ++i)
                    if (inf.pose_logits[i] > inf.pose_logits[pose_idx]) pose_idx = i;

                // cascade v2 부터 포복 사람을 모델이 직접 검출 (test pobok recall 0.97).
                // v1 시절 후처리 휴리스틱 (pose=low + 점 25+ → 사람 강제) 은 제거.

                g_tracker->update_inference(result.roi_id, inf.human_logit, pose_idx);
            }
        }

        // 추론 완료 여부와 무관하게 트랙(박스) 발행 — 캐시 있으면 라벨 포함, 없으면 sentinel.
        radar::Track t;
        t.track_id = result.roi_id;
        t.centroid_x = c.centroid_x; t.centroid_y = c.centroid_y;
        t.min_x = minx; t.max_x = maxx;
        t.min_y = miny; t.max_y = maxy;
        t.min_z = minz; t.max_z = maxz;
        float h_logit = 0.0f; int p_idx = 0;
        if (g_tracker->get_cached_inference(result.roi_id, h_logit, p_idx)) {
            t.human_prob = sigmoid(h_logit);
            t.pose_idx = p_idx;
            t.status = pose_to_status(p_idx);
        } else {
            t.human_prob = -1.0f;   // sentinel: 추론 대기중 → 뷰어가 박스만 그리고 라벨 보류.
            t.pose_idx = 0;
            t.status = radar::TargetStatus::Standing;
        }
        out_frame.tracks.push_back(t);
    }
}

// ============ single 파이프라인 (워커 쓰레드, 코어 3) ============
// raw → voxel 0.05 → SOR → 3D DBSCAN → cluster → normalize → FPS 256 → single inference.
// 자세/인간확률(person_prob)만 채우고, cascade track 매칭은 join 후 별도로 수행.
void compute_single_detections(const std::vector<radar::Point>& raw_points,
                               std::vector<radar::SingleDetection>& out) {
    auto sf_clusters = radar::single::process_frame(raw_points, g_single_rng);
    static int dbg = 0;
    static int entry = 0;
#ifndef _WIN32
    // 무조건 마커: compute_single_detections 가 호출됐는지(클러스터 개수 포함) 우선 확인.
    if (entry < 10) {
        char ebuf[160];
        std::snprintf(ebuf, sizeof(ebuf),
                      "[single ENTRY %d] sf_clusters=%lu  raw_pts=%lu",
                      entry, static_cast<unsigned long>(sf_clusters.size()),
                      static_cast<unsigned long>(raw_points.size()));
        rlog(ebuf);
        ++entry;
    }
#endif
    for (const auto& sc : sf_clusters) {
        radar::inference::SingleOutput so;
        if (!g_single_inferencer->infer_single(sc.points, so)) continue;

        // 진단: 첫 20번의 single 추론은 raw person_logit + pose_logits 를 /log2 에 덤프.
        // (person_prob = sigmoid(person_logit). logit 이 매우 음수면 0%, 매우 양수면 100%.)
        if (dbg < 20) {
#ifndef _WIN32
            char buf[320];
            std::snprintf(buf, sizeof(buf),
                "[single dbg %2d] n_orig=%-3d cluster_pts0=[%.3f %.3f %.3f %.3f %.3f] "
                "person_logit=%+8.4f sigmoid=%6.3f%% pose=[%+.2f %+.2f %+.2f %+.2f %+.2f %+.2f %+.2f %+.2f %+.2f]",
                dbg, sc.n_orig,
                sc.points.size() >= 5 ? sc.points[0] : 0.0f,
                sc.points.size() >= 5 ? sc.points[1] : 0.0f,
                sc.points.size() >= 5 ? sc.points[2] : 0.0f,
                sc.points.size() >= 5 ? sc.points[3] : 0.0f,
                sc.points.size() >= 5 ? sc.points[4] : 0.0f,
                so.person_logit,
                100.0f / (1.0f + std::exp(-so.person_logit)),
                so.pose_logits[0], so.pose_logits[1], so.pose_logits[2],
                so.pose_logits[3], so.pose_logits[4], so.pose_logits[5],
                so.pose_logits[6], so.pose_logits[7], so.pose_logits[8]);
            rlog(buf);
#endif
            ++dbg;
        }

        radar::SingleDetection d;
        d.min_x = sc.min_x; d.max_x = sc.max_x;
        d.min_y = sc.min_y; d.max_y = sc.max_y;
        d.min_z = sc.min_z; d.max_z = sc.max_z;

        int pi = 0;
        for (int i = 1; i < radar::SF_POSE_CLASSES; ++i)
            if (so.pose_logits[i] > so.pose_logits[pi]) pi = i;
        d.pose_idx = pi;
        for (int i = 0; i < radar::SF_POSE_CLASSES; ++i) d.pose_logits[i] = so.pose_logits[i];

        d.person_logit = so.person_logit;   // raw logit (진단용)
        d.person_prob = sigmoid(so.person_logit);
        d.human_prob = d.person_prob;   // 기본값(매칭 전) = single. 매칭 시 cascade 로 덮어씀.
        d.from_cascade = false;
        out.push_back(d);
    }
}

// single 검출을 cascade track 과 공간 매칭 — 매칭되면 인간확률을 cascade 로 교체.
//   매칭됨(=시퀀스 버퍼 참)  → human = cascade
//   매칭 안됨(=버퍼 차기 전) → human = single (그대로)
void match_single_to_tracks(std::vector<radar::SingleDetection>& dets,
                            const radar::Frame& out_frame) {
    for (auto& d : dets) {
        const float scx = 0.5f * (d.min_x + d.max_x);
        const float scy = 0.5f * (d.min_y + d.max_y);
        float best_dist = radar::ROI_MATCH_DIST;
        const radar::Track* matched = nullptr;
        for (const auto& t : out_frame.tracks) {
            const float dx = t.centroid_x - scx;
            const float dy = t.centroid_y - scy;
            const float dd = std::sqrt(dx * dx + dy * dy);
            if (dd < best_dist) { best_dist = dd; matched = &t; }
        }
        if (matched) {
            d.human_prob = matched->human_prob;
            d.from_cascade = true;
        }
    }
}

// ============ 핵심 — raw points → publish ============
// 임시 순차 실행 — v3 모델로 ARM 폭주 재현 시 쓰레딩 vs 모델 분리 확인용.
// 순차에서 정상 나오면 → 쓰레딩 + cross-thread interpreter call 이 진짜 원인 확정 →
// 그땐 worker 쓰레드에서 single 인터프리터를 *생성* 까지 시키는 구조로 리팩토링.
void process_frame(const std::vector<radar::Point>& raw_points, uint32_t frame_counter) {
    ++g_frame_count;
    const bool do_publish = (g_frame_count % radar::PUBLISH_STRIDE == 0);
    const bool want_single = radar::ENABLE_SINGLE && do_publish &&
        g_single_inferencer && g_single_inferencer->is_ready();

    radar::Frame out_frame;
    out_frame.frame_count = g_frame_count;
    run_cascade(raw_points, frame_counter, out_frame);   // 메인 쓰레드

    if (want_single) {
        std::vector<radar::SingleDetection> single_dets;
        compute_single_detections(raw_points, single_dets);
        match_single_to_tracks(single_dets, out_frame);
        out_frame.single_detections = std::move(single_dets);
    }

    if (g_json_publisher && do_publish) g_json_publisher->publish(out_frame);
    g_tracker->end_frame();
}


// ============ Demo mode — SDK 없이 합성 데이터로 pipeline 동작 검증 ============

void demo_run(int n_frames) {
    std::printf("[demo] running %d synthetic frames\n", n_frames);
    std::mt19937 rng(42);

    for (int f = 0; f < n_frames; ++f) {
        std::vector<radar::Point> pts;
        // human-like cluster: ~30점, 0.5m 반경
        std::normal_distribution<float> jx(0.0f, 0.15f);
        std::normal_distribution<float> jy(0.7f, 0.15f);
        std::uniform_real_distribution<float> jz(-1.5f, 0.5f);
        for (int i = 0; i < 30; ++i) {
            radar::Point p;
            p.x = jx(rng); p.y = jy(rng); p.z = jz(rng);
            p.doppler = 0.2f + std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng);
            p.power = 20000.0f + std::uniform_real_distribution<float>(-5000.0f, 5000.0f)(rng);
            p.track_id = -1;
            pts.push_back(p);
        }
        // 잡음
        std::uniform_real_distribution<float> nx(-2.0f, 2.0f);
        for (int i = 0; i < 10; ++i) {
            radar::Point p;
            p.x = nx(rng); p.y = nx(rng); p.z = jz(rng);
            p.doppler = 0.0f; p.power = 100.0f; p.track_id = -1;
            pts.push_back(p);
        }
        process_frame(pts, static_cast<uint32_t>(f));
    }

    const size_t clients = g_json_publisher ? g_json_publisher->client_count() : 0;
    std::printf("[demo] done. processed %u frames, json clients=%lu\n",
                g_frame_count, static_cast<unsigned long>(clients));
}

void print_banner() {
    std::printf("=== radar_embed user_module ===\n");
    std::printf("config: SEQ_LEN=%d NUM_POINTS=%d STRIDE=%d ROI_RADIUS=%.2fm\n",
                radar::SEQ_LEN, radar::NUM_POINTS, radar::STRIDE, radar::ROI_RADIUS);
    std::printf("dbscan: eps=%.2f min_samples=%d v_alpha=%.2f min_cluster=%d\n",
                radar::DBSCAN_EPS, radar::DBSCAN_MIN_SAMPLES,
                radar::DBSCAN_V_ALPHA, radar::MIN_CLUSTER_POINTS);
    std::printf("io: input(connect) port=%d, output(json) port=%d, alarm threshold=%.2f\n",
                radar::PUBLISHER_PORT, radar::PUBLISHER_JSON_PORT, radar::ALARM_THRESHOLD);
}

#ifndef _WIN32
// ============ 실시간 모드 (POSIX/ARM) ============

volatile std::sig_atomic_t g_running = 1;

void on_signal(int) { g_running = 0; }

// 진단 로그 — /log2/srs_user_module_YYYYMMDD.log (없으면 stdout fallback).
// /log2 는 부팅 후 ~10초 뒤 mount 되므로 못 열면 stdout 으로.
FILE* g_logf = nullptr;

void open_log() {
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    char path[160];
    std::snprintf(path, sizeof(path), "/log2/srs_user_module_%04d%02d%02d.log",
                  tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday);
    g_logf = std::fopen(path, "a");  // null 이면 rlog 가 stdout 사용
}

// receive(코어2) / process(코어3) 두 쓰레드 동시 호출 시 라인 깨짐 방지용 mutex.
// POSIX write 는 file 에 대해 atomic 보장 없음 (PIPE_BUF 까지만 pipe 한정).
std::mutex g_log_mtx;

void rlog(const char* msg) {
    std::lock_guard<std::mutex> lk(g_log_mtx);
    // /log2 마운트 지연(~10s) 대응: 아직 못 열었으면 매 호출 재시도.
    if (!g_logf) open_log();
    std::time_t t = std::time(nullptr);
    std::tm* tm = std::localtime(&t);
    FILE* o = g_logf ? g_logf : stdout;
    std::fprintf(o, "[%02d:%02d:%02d] %s\n", tm->tm_hour, tm->tm_min, tm->tm_sec, msg);
    std::fflush(o);
    // 디버그용 보조 파일에도 동시 기록 — 로그 파일을 못 찾을 경우 백업.
    // /log2 (마운트 후), /tmp (항상 가능) 양쪽에 append.
    FILE* dbg1 = std::fopen("/log2/srs_user_module_dbg.log", "a");
    if (dbg1) { std::fprintf(dbg1, "[%02d:%02d:%02d] %s\n", tm->tm_hour, tm->tm_min, tm->tm_sec, msg); std::fclose(dbg1); }
    FILE* dbg2 = std::fopen("/tmp/srs_user_module_dbg.log", "a");
    if (dbg2) { std::fprintf(dbg2, "[%02d:%02d:%02d] %s\n", tm->tm_hour, tm->tm_min, tm->tm_sec, msg); std::fclose(dbg2); }
}

// CPU affinity — 호출 쓰레드를 특정 코어 하나에 핀.
// 예제 srs_app_collaboration: 코어 0,1 은 신호처리/포인트클라우드, 2,3 이 user module 몫.
// receive 쓰레드 = 코어 2 (네트워크 I/O), process 쓰레드 = 코어 3 (heavy preprocessing+inference).
void pin_thread_core(int core) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core, &mask);
    char buf[96];
    if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
        std::snprintf(buf, sizeof(buf),
                      "[affinity] core %d 핀 실패 (errno=%d) — default scheduler 로 동작",
                      core, errno);
        rlog(buf);
    } else {
        // 실제로 박혔는지 재확인 (요청 ≠ 실제일 수 있음).
        cpu_set_t got;
        CPU_ZERO(&got);
        if (sched_getaffinity(0, sizeof(got), &got) == 0) {
            int n_set = 0, only = -1;
            for (int c = 0; c < CPU_SETSIZE; ++c) {
                if (CPU_ISSET(c, &got)) { ++n_set; if (only < 0) only = c; }
            }
            std::snprintf(buf, sizeof(buf),
                          "[affinity] core %d 핀 OK (mask=%d cores, only=%d)",
                          core, n_set, only);
            rlog(buf);
        } else {
            std::snprintf(buf, sizeof(buf), "[affinity] core %d set OK (verify 실패)", core);
            rlog(buf);
        }
    }
}

// FrameQueue — receive → process 쓰레드 간 SPSC 큐.
// 처리율이 수신율을 못 따라갈 때 가장 오래된 프레임 드롭 (신선도 우선).
class FrameQueue {
public:
    void push(std::vector<radar::Point>&& pts, uint32_t fn) {
        std::lock_guard<std::mutex> lk(m_);
        if (static_cast<int>(q_.size()) >= CAP) {
            q_.pop_front();
            ++dropped_;
        }
        q_.emplace_back(std::move(pts), fn);
        cv_.notify_one();
    }
    // shutdown 시 false 반환. 항목 올 때까지 블록.
    bool pop(std::vector<radar::Point>& pts, uint32_t& fn) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return !q_.empty() || stop_; });
        if (q_.empty()) return false;
        auto& f = q_.front();
        pts = std::move(f.points);
        fn = f.frame_num;
        q_.pop_front();
        return true;
    }
    void shutdown() {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
        cv_.notify_all();
    }
    int dropped() {
        std::lock_guard<std::mutex> lk(m_);
        return dropped_;
    }
private:
    // CAP 8: 20Hz 에서 ~400ms 버퍼. 일시 process 스파이크(TFLite warm-up 등) 흡수.
    //   - 정상 부하 (process < 50ms) 시 큐 거의 비어 latency 추가 ≈ 0.
    //   - 백프레셔 시 최악 latency 400ms (cascade 2s 시퀀스 대비 미미).
    //   - drop 발생 빈도 ↓ → SEQ_MAX_GAP(15) 위반 가능성 격감.
    static constexpr int CAP = 8;
    struct Item {
        std::vector<radar::Point> points;
        uint32_t frame_num;
        Item(std::vector<radar::Point>&& p, uint32_t fn)
            : points(std::move(p)), frame_num(fn) {}
    };
    std::deque<Item> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
    int dropped_ = 0;
};

int run_realtime() {
    pin_thread_core(2);   // receive 쓰레드 = 코어2 (가벼운 네트워크 I/O)
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    open_log();
    char m[256];
    std::snprintf(m, sizeof(m), "user_module start. connecting to point-cloud source 127.0.0.1:%d",
                  radar::PUBLISHER_PORT);
    rlog(m);

    // 점 cloud 소스 connect (무한 재시도 backoff)
    Socket sock;
    const char* ip = "127.0.0.1";
    auto connect_with_backoff = [&](const char* why) {
        int backoff_ms = 100, attempt = 0;
        while (g_running && !sock.connectSocket(ip, radar::PUBLISHER_PORT)) {
            if (++attempt == 1 || attempt % 20 == 0) {
                std::snprintf(m, sizeof(m), "%s: connecting %s:%d (attempt %d) ...",
                              why, ip, radar::PUBLISHER_PORT, attempt);
                rlog(m);
            }
            usleep(backoff_ms * 1000);
            backoff_ms = backoff_ms < 3000 ? backoff_ms * 2 : 3000;
            sock = Socket();
        }
    };
    connect_with_backoff("init");
    if (!g_running) return 0;
    // 소켓 recv 타임아웃 200ms — receive_frame 이 무한 블록되지 않도록.
    // SIGINT 시 g_running=0 → 다음 타임아웃 이내(≤200ms)에 loop 빠져나옴.
    auto set_recv_timeout = [&](int ms) {
        const int fd = sock.getSocketfd();
        if (fd < 0) return;
        struct timeval tv;
        tv.tv_sec  = ms / 1000;
        tv.tv_usec = (ms % 1000) * 1000;
        if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) < 0) {
            char b[80];
            std::snprintf(b, sizeof(b), "[sock] SO_RCVTIMEO 설정 실패 (errno=%d)", errno);
            rlog(b);
        }
    };
    set_recv_timeout(200);
    rlog("CONNECTED to point-cloud source. entering pipelined receive loop.");

    // ====== 파이프라인 ======
    // receive 쓰레드(현재): 패킷 수신 → 큐 push.
    // process 쓰레드(워커, 코어3): 큐 pop → cascade 파이프라인 → publish.
    FrameQueue queue;
    std::thread process_thread([&]() {
        pin_thread_core(3);
        std::vector<radar::Point> pts;
        uint32_t fn = 0;
        while (queue.pop(pts, fn)) {
            process_frame(pts, fn);
        }
    });

    std::vector<radar::Point> raw;
    uint32_t frame_num = 0;
    uint32_t rx_ok = 0, rx_fail = 0, consec_fail = 0;
    while (g_running) {
        if (!radar::input::receive_frame(sock, raw, frame_num)) {
            ++rx_fail;
            if (++consec_fail >= 500) {
                rlog("연속 수신 실패 — 소켓 재연결 (무한 재시도)");
                sock = Socket();
                connect_with_backoff("reconnect");
                consec_fail = 0;
                if (g_running) {
                    set_recv_timeout(200);   // 새 소켓에도 타임아웃 재설정
                    rlog("재연결 성공");
                }
            }
            if (rx_fail % 2000 == 0) {
                std::snprintf(m, sizeof(m), "receive failing: rx_fail=%u, rx_ok=%u", rx_fail, rx_ok);
                rlog(m);
            }
            usleep(1000);
            continue;
        }
        consec_fail = 0;
        ++rx_ok;
        const std::size_t n_raw = raw.size();
        queue.push(std::move(raw), frame_num);   // 이동 (복사 X) → raw 비워짐
        raw.clear();
        if (rx_ok <= 3 || rx_ok % 100 == 0) {
            const std::size_t pub = g_json_publisher ? g_json_publisher->client_count() : 0;
            std::snprintf(m, sizeof(m),
                "rx_ok=%u frame=%u raw_points=%lu json_clients=%lu q_dropped=%d",
                rx_ok, g_frame_count,
                static_cast<unsigned long>(n_raw),
                static_cast<unsigned long>(pub),
                queue.dropped());
            rlog(m);
        }
    }
    queue.shutdown();
    process_thread.join();
    rlog("realtime loop 종료");
    if (g_logf) std::fclose(g_logf);
    return 0;
}
#endif  // !_WIN32

}  // namespace

int main(int argc, char** argv) {
    print_banner();

    // 1. JSON publisher (TCP 29173) — 라즈베리파이 viewer 가 connect 해서 결과 수신.
    g_json_publisher = std::make_unique<radar::JsonPublisher>(radar::PUBLISHER_JSON_PORT);
    if (g_json_publisher->start()) {
        std::printf("[main] json publisher: listening on port %d\n", radar::PUBLISHER_JSON_PORT);
    } else {
        std::printf("[main] json publisher: NOT started (HOST stub 또는 bind 실패)\n");
    }

    // 2. ROITracker
    g_tracker = std::make_unique<radar::ROITracker>(
        radar::SEQ_LEN, radar::NUM_POINTS, radar::STRIDE, /*features=*/5,
        radar::ROI_MATCH_DIST, radar::ROI_MAX_CONCURRENT,
        radar::ROI_CENTROID_ALPHA, radar::ROI_IDLE_TIMEOUT_FRM,
        radar::SEQ_MAX_GAP);

    // 3a. cascade inferencer (메인, 시퀀스)
    g_inferencer = std::make_unique<radar::inference::TFLiteInterpreter>(
        model_tflite, model_tflite_len, radar::TFLITE_NUM_THREADS);
    if (g_inferencer->is_ready()) {
        std::printf("[main] cascade inference: ready (%u bytes)\n", model_tflite_len);
    } else {
        std::printf("[main] cascade inference: NOT ready (HOST stub)\n");
    }

    // 3b. single-frame inferencer (warmup)
    g_single_inferencer = std::make_unique<radar::inference::TFLiteInterpreter>(
        model2_tflite, model2_tflite_len, radar::TFLITE_NUM_THREADS);
    if (g_single_inferencer->is_ready()) {
        std::printf("[main] single inference: ready (%u bytes)\n", model2_tflite_len);
    } else {
        std::printf("[main] single inference: NOT ready (HOST stub)\n");
    }

    // 4. 운영 모드
    //    --demo : 합성 데이터 검증 (HOST 기본). 실기기에서는 실시간 소켓 수신.
    bool demo_mode = false;
    int demo_frames = 60;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--demo") demo_mode = true;
        else if (a == "--frames" && i + 1 < argc) demo_frames = std::atoi(argv[++i]);
    }

    int rc = 0;
#ifdef _WIN32
    // HOST(Windows): 소켓 입력 불가 → 항상 demo.
    (void)demo_mode;
    demo_run(demo_frames);
#else
    if (demo_mode) {
        demo_run(demo_frames);
    } else {
        rc = run_realtime();   // connect(127.0.0.1:29172) → while loop → receive → process
    }
#endif

    if (g_json_publisher) g_json_publisher->stop();
    std::printf("[main] exit (rc=%d)\n", rc);
    return rc;
}
