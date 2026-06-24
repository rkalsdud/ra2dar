// JsonPublisher 구현. line-delimited JSON over TCP.
// POSIX 실제 listen socket + accept thread; Windows stub.

#include "radar/json_publisher.h"

#include "radar/config.h"

#include <cstdio>
#include <string>

namespace radar {

namespace {

const char* pose_name(int pose_idx) {
    switch (pose_idx) {
        case 0: return "Standing";   // upright
        case 1: return "Lying";      // horizontal
        case 2: return "Sitting";    // low
        default: return "Unknown";
    }
}

// 한 frame → 한 줄 JSON. 수동 직렬화 (외부 의존성 0).
//   {"frame":N,"points":[[x,y,z,V,P,tid],...],"tracks":[{...},...]}\n
std::string serialize(const Frame& f) {
    // 점/트랙 평균 크기 추정해서 사전 할당
    std::string s;
    s.reserve(64 + f.points.size() * 48 + f.tracks.size() * 128);

    char buf[512];   // single_detections JSON 에 raw logits 포함되면서 길어짐

    std::snprintf(buf, sizeof(buf), "{\"frame\":%u,\"points\":[", f.frame_count);
    s += buf;

    // 점 수 cap — MAX_JSON_POINTS 초과 시 균등 subsample (대역 절약).
    // 좌표 정밀도도 mm(.3f)/정수 power 로 낮춰 JSON 크기 축소.
    const std::size_t np = f.points.size();
    std::size_t step = 1;
    if (radar::MAX_JSON_POINTS > 0 && np > static_cast<std::size_t>(radar::MAX_JSON_POINTS)) {
        step = np / static_cast<std::size_t>(radar::MAX_JSON_POINTS) + 1;
    }
    bool first = true;
    for (std::size_t i = 0; i < np; i += step) {
        const auto& p = f.points[i];
        std::snprintf(buf, sizeof(buf),
                      "%s[%.3f,%.3f,%.3f,%.2f,%.0f,%d]",
                      first ? "" : ",",
                      p.x, p.y, p.z, p.doppler, p.power, p.track_id);
        first = false;
        s += buf;
    }
    s += "],\"tracks\":[";

    // 트랙별 bbox 는 publisher 측에서 채워준 값을 사용 (wire publisher 와 동일 로직).
    // Track.min_x..max_z 가 비어있을 수 있으므로 점들로부터 직접 derive.
    for (std::size_t i = 0; i < f.tracks.size(); ++i) {
        const auto& t = f.tracks[i];

        float minx = 1e30f, maxx = -1e30f;
        float miny = 1e30f, maxy = -1e30f;
        float minz = 1e30f, maxz = -1e30f;
        bool any = false;
        for (const auto& p : f.points) {
            if (static_cast<uint32_t>(p.track_id) != t.track_id) continue;
            any = true;
            if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
            if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
            if (p.z < minz) minz = p.z; if (p.z > maxz) maxz = p.z;
        }
        if (!any) { minx = maxx = t.centroid_x; miny = maxy = t.centroid_y; minz = maxz = 0.0f; }

        std::snprintf(buf, sizeof(buf),
                      "%s{\"id\":%u,\"bbox\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
                      "\"human_prob\":%.4f,\"pose_idx\":%d,\"pose\":\"%s\"}",
                      i == 0 ? "" : ",",
                      t.track_id, minx, maxx, miny, maxy, minz, maxz,
                      t.human_prob, t.pose_idx, pose_name(t.pose_idx));
        s += buf;
    }
    s += "]";

    // single 모델 검출 (메인 표시용).
    //   pose      : 항상 single (pose_idx 9-class, 매핑은 viewer 측)
    //   human_prob: 결합 — cascade track 매칭 시 cascade, 아니면 single
    //   human_src : "cascade"(버퍼 참) | "single"(버퍼 전)
    s += ",\"single_detections\":[";
    for (std::size_t i = 0; i < f.single_detections.size(); ++i) {
        const auto& d = f.single_detections[i];
        std::snprintf(buf, sizeof(buf),
                      "%s{\"bbox\":[%.4f,%.4f,%.4f,%.4f,%.4f,%.4f],"
                      "\"pose_idx\":%d,\"human_prob\":%.4f,"
                      "\"person_prob\":%.4f,\"person_logit\":%.4f,"
                      "\"pose_logits\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f],"
                      "\"human_src\":\"%s\"}",
                      i == 0 ? "" : ",",
                      d.min_x, d.max_x, d.min_y, d.max_y, d.min_z, d.max_z,
                      d.pose_idx, d.human_prob, d.person_prob, d.person_logit,
                      d.pose_logits[0], d.pose_logits[1], d.pose_logits[2],
                      d.pose_logits[3], d.pose_logits[4], d.pose_logits[5],
                      d.pose_logits[6], d.pose_logits[7], d.pose_logits[8],
                      d.from_cascade ? "cascade" : "single");
        s += buf;
    }
    s += "]}\n";
    return s;
}

}  // namespace

}  // namespace radar

// ============ POSIX/Windows 분기 ============

#ifdef _WIN32

namespace radar {

struct JsonPublisher::Impl {};

JsonPublisher::JsonPublisher(int port) : impl_(new Impl{}), running_(false), port_(port) {}
JsonPublisher::~JsonPublisher() = default;
bool JsonPublisher::start() { return false; }
void JsonPublisher::stop() {}
std::size_t JsonPublisher::client_count() const { return 0; }
void JsonPublisher::publish(const Frame& /*frame*/) {}
bool JsonPublisher::is_supported() { return false; }

}  // namespace radar

#else

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <mutex>
#include <thread>
#include <vector>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace radar {

struct JsonPublisher::Impl {
    int listen_fd = -1;
    std::thread accept_thread;
    mutable std::mutex clients_mtx;
    std::vector<int> client_fds;
    std::atomic<bool>* running_ptr = nullptr;

    void accept_loop() {
        while (running_ptr && running_ptr->load()) {
            sockaddr_in addr{};
            socklen_t addr_len = sizeof(addr);
            int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len);
            if (fd < 0) {
                if (!running_ptr || !running_ptr->load()) break;
                continue;
            }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            // non-blocking — WiFi 느린 client 가 메인 루프(점 cloud 수신)를 막지 않도록.
            const int fl = ::fcntl(fd, F_GETFL, 0);
            ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
            std::lock_guard<std::mutex> lk(clients_mtx);
            client_fds.push_back(fd);
        }
    }
};

JsonPublisher::JsonPublisher(int port) : impl_(new Impl{}), running_(false), port_(port) {
    impl_->running_ptr = &running_;
}

JsonPublisher::~JsonPublisher() {
    stop();
}

bool JsonPublisher::start() {
    if (!impl_) return false;
    impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->listen_fd < 0) return false;

    int yes = 1;
    ::setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (::bind(impl_->listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(impl_->listen_fd); impl_->listen_fd = -1; return false;
    }
    if (::listen(impl_->listen_fd, 4) < 0) {
        ::close(impl_->listen_fd); impl_->listen_fd = -1; return false;
    }
    running_.store(true);
    impl_->accept_thread = std::thread([this] { impl_->accept_loop(); });
    return true;
}

void JsonPublisher::stop() {
    if (!impl_) return;
    if (running_.exchange(false)) {
        if (impl_->listen_fd >= 0) {
            ::shutdown(impl_->listen_fd, SHUT_RDWR);
            ::close(impl_->listen_fd);
            impl_->listen_fd = -1;
        }
        if (impl_->accept_thread.joinable()) impl_->accept_thread.join();
    }
    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    for (int fd : impl_->client_fds) ::close(fd);
    impl_->client_fds.clear();
}

std::size_t JsonPublisher::client_count() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    return impl_->client_fds.size();
}

void JsonPublisher::publish(const Frame& frame) {
    if (!impl_ || !running_.load()) return;
    const std::string line = serialize(frame);
    if (line.empty()) return;

    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    const ssize_t need = static_cast<ssize_t>(line.size());
    auto it = impl_->client_fds.begin();
    while (it != impl_->client_fds.end()) {
        // non-blocking 단일 send. 한 줄(JSON)은 원자적으로 보내야 viewer 파싱이 안 깨짐.
        const ssize_t n = ::send(*it, line.data(), static_cast<size_t>(need),
                                 MSG_NOSIGNAL | MSG_DONTWAIT);
        if (n == need) {
            ++it;                                   // 전체 송신 OK
        } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            // 송신 버퍼 참 (WiFi 느림) — 0 byte 송신이라 안전. 이번 frame skip, client 유지.
            ++it;
        } else {
            // 부분 송신(스트림 깨짐) 또는 연결 끊김 → client 제거 (viewer 가 재연결).
            ::close(*it);
            it = impl_->client_fds.erase(it);
        }
    }
}

bool JsonPublisher::is_supported() { return true; }

}  // namespace radar

#endif
