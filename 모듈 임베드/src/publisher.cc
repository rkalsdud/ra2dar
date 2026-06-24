// Publisher 구현.
// POSIX (Linux/ARM): 실제 TCP 서버.
// Windows/MinGW: stub.

#include "radar/publisher.h"

#include "radar/wire_protocol.h"

#ifdef _WIN32
// ============ Windows/MinGW HOST stub ============

namespace radar {

struct Publisher::Impl {};

Publisher::Publisher(int /*port*/) { impl_ = new Impl{}; }
Publisher::~Publisher() { delete impl_; }
bool Publisher::start() { return false; }
void Publisher::stop() {}
void Publisher::publish(const Frame& /*frame*/) {}
bool Publisher::is_running() const { return false; }
size_t Publisher::client_count() const { return 0; }
bool Publisher::is_supported() { return false; }

}  // namespace radar

#else
// ============ POSIX (Linux/ARM) — 실제 구현 ============

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace radar {

struct Publisher::Impl {
    int port = 0;
    int listen_fd = -1;
    std::atomic<bool> running{false};
    std::thread accept_thread;

    mutable std::mutex clients_mtx;
    std::vector<int> client_fds;

    void accept_loop() {
        while (running.load()) {
            sockaddr_in addr{};
            socklen_t addr_len = sizeof(addr);
            int fd = ::accept(listen_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len);
            if (fd < 0) {
                if (!running.load()) break;
                continue;
            }
            int yes = 1;
            ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
            std::lock_guard<std::mutex> lk(clients_mtx);
            client_fds.push_back(fd);
        }
    }
};

Publisher::Publisher(int port) {
    impl_ = new Impl{};
    impl_->port = port;
}

Publisher::~Publisher() {
    stop();
    delete impl_;
}

bool Publisher::start() {
    if (!impl_) return false;
    impl_->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (impl_->listen_fd < 0) return false;

    int yes = 1;
    ::setsockopt(impl_->listen_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(impl_->port));

    if (::bind(impl_->listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
        return false;
    }
    if (::listen(impl_->listen_fd, 4) < 0) {
        ::close(impl_->listen_fd);
        impl_->listen_fd = -1;
        return false;
    }

    impl_->running.store(true);
    impl_->accept_thread = std::thread([this] { impl_->accept_loop(); });
    return true;
}

void Publisher::stop() {
    if (!impl_) return;
    if (impl_->running.exchange(false)) {
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

void Publisher::publish(const Frame& frame) {
    if (!impl_ || !impl_->running.load()) return;
    auto bytes = wire::serialize_packet(frame);
    if (bytes.empty()) return;

    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    auto it = impl_->client_fds.begin();
    while (it != impl_->client_fds.end()) {
        ssize_t total = 0;
        const ssize_t need = static_cast<ssize_t>(bytes.size());
        bool dead = false;
        while (total < need) {
            ssize_t n = ::send(*it, bytes.data() + total,
                               static_cast<size_t>(need - total), MSG_NOSIGNAL);
            if (n <= 0) { dead = true; break; }
            total += n;
        }
        if (dead) {
            ::close(*it);
            it = impl_->client_fds.erase(it);
        } else {
            ++it;
        }
    }
}

bool Publisher::is_running() const { return impl_ && impl_->running.load(); }

size_t Publisher::client_count() const {
    if (!impl_) return 0;
    std::lock_guard<std::mutex> lk(impl_->clients_mtx);
    return impl_->client_fds.size();
}

bool Publisher::is_supported() { return true; }

}  // namespace radar

#endif  // _WIN32
