// Retina TCP 와이어 (de)serializer.
// retina.cpp (Downloads/radar.zip) 의 parseSinglePacket 와 동일 의미 + 역방향 직렬화.

#include "radar/wire_protocol.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace radar {
namespace wire {

namespace {

// ============ Raw 구조체 (와이어 메모리 그대로) ============
// retina.cpp 의 RetinaPacketHeaderRaw / RetinaFrameHeaderRaw / RetinaTargetRaw 미러.
// pack(1) 사용 — alignment padding 방지 (와이어는 byte-aligned).
#pragma pack(push, 1)

struct PacketHeaderRaw {
    uint32_t reserved0;
    uint32_t magic;
    uint32_t reserved1;
    uint32_t reserved2;
    uint32_t package_size;
    uint32_t reserved3;
    uint32_t reserved4;
    uint32_t reserved5;
    uint32_t reserved6;
};
static_assert(sizeof(PacketHeaderRaw) == PACKET_HEADER_SIZE, "PacketHeaderRaw size");

struct FrameHeaderRaw {
    uint64_t magic;
    uint32_t frame_count;
    // 첫 frame header 에서는 POINT 수, 두 번째에서는 TARGET 수 — 의미가 다름.
    uint32_t target_number;
};
static_assert(sizeof(FrameHeaderRaw) == FRAME_HEADER_SIZE, "FrameHeaderRaw size");

struct PointRaw {
    float x;
    float y;
    float z;
    float doppler;
    float power;
};
static_assert(sizeof(PointRaw) == POINT_WIRE_SIZE, "PointRaw size");

struct TargetRaw {
    float x;
    float y;
    uint32_t status;          // TargetStatus
    uint32_t target_id;
    float reserved0;
    float reserved1;
    float reserved2;
};
static_assert(sizeof(TargetRaw) == TARGET_WIRE_SIZE, "TargetRaw size");

#pragma pack(pop)

// little-endian 가정 (Cortex-A53 + x86 모두 LE).
// 안전을 위해 memcpy 로 unaligned read/write.

template <typename T>
bool read_le(const uint8_t* base, size_t size, size_t offset, T& out) {
    if (offset + sizeof(T) > size) return false;
    std::memcpy(&out, base + offset, sizeof(T));
    return true;
}

template <typename T>
void write_le(uint8_t* base, size_t offset, const T& v) {
    std::memcpy(base + offset, &v, sizeof(T));
}

}  // namespace

// ============ peek_packet_size ============

bool peek_packet_size(const uint8_t* data, size_t size, uint32_t& out_size) {
    PacketHeaderRaw hdr;
    if (!read_le(data, size, 0, hdr)) return false;
    if (hdr.magic != PACKET_MAGIC) return false;
    out_size = hdr.package_size;
    return true;
}

// ============ parse_packet ============

bool parse_packet(const uint8_t* data, size_t size, ParsedFrame& out) {
    out = ParsedFrame{};

    // 1. PacketHeader
    PacketHeaderRaw pkt_hdr;
    if (!read_le(data, size, 0, pkt_hdr)) return false;
    if (pkt_hdr.magic != PACKET_MAGIC) return false;

    const size_t expected_total = PACKET_HEADER_SIZE + pkt_hdr.package_size;
    if (size < expected_total) return false;

    // 2. FrameHeader_A (points)
    FrameHeaderRaw fh_a;
    if (!read_le(data, size, PACKET_HEADER_SIZE, fh_a)) return false;
    if (fh_a.magic != FRAME_MAGIC) return false;

    out.frame_count = fh_a.frame_count;
    const uint32_t point_number = fh_a.target_number;  // 이 위치에선 point count
    out.points.reserve(point_number);

    // 3. Points (xyzVP)
    size_t off = PACKET_HEADER_SIZE + FRAME_HEADER_SIZE;
    for (uint32_t i = 0; i < point_number; ++i) {
        PointRaw pr;
        if (!read_le(data, size, off, pr)) return false;
        off += POINT_WIRE_SIZE;
        Point p;
        p.x = pr.x; p.y = pr.y; p.z = pr.z;
        p.doppler = pr.doppler; p.power = pr.power;
        p.track_id = -1;
        out.points.push_back(p);
    }

    // 4. Per-point targetIds
    for (uint32_t i = 0; i < point_number; ++i) {
        int32_t tid;
        if (!read_le(data, size, off, tid)) return false;
        off += TARGET_ID_SIZE;
        out.points[i].track_id = tid;
    }

    // 5. 옵션: target section. retina.cpp 와 동일 로직.
    //    - 데이터 끝이면 끝 (targets 없음)
    //    - 현재 위치에서 FRAME_MAGIC peek → 있으면 거기서 시작
    //    - 없으면 고정 오프셋 48056 으로 seek
    if (off >= expected_total) return true;

    FrameHeaderRaw fh_b;
    bool have_target_header = false;
    if (read_le(data, size, off, fh_b) && fh_b.magic == FRAME_MAGIC) {
        have_target_header = true;
    } else if (read_le(data, size, TARGET_FRAME_HEADER_OFFSET, fh_b)
               && fh_b.magic == FRAME_MAGIC) {
        off = TARGET_FRAME_HEADER_OFFSET;
        have_target_header = true;
    }

    if (!have_target_header) return true;   // targets 없음

    off += FRAME_HEADER_SIZE;
    const uint32_t target_number = fh_b.target_number;
    out.tracks.reserve(target_number);

    for (uint32_t i = 0; i < target_number; ++i) {
        TargetRaw tr;
        if (!read_le(data, size, off, tr)) return false;
        off += TARGET_WIRE_SIZE;

        Track t;
        t.track_id = tr.target_id;
        t.centroid_x = tr.x;
        t.centroid_y = tr.y;
        t.status = static_cast<TargetStatus>(tr.status);

        // bbox: 점 멤버에서 derive (retina.cpp 와 동일).
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        bool any = false;
        for (const auto& p : out.points) {
            if (p.track_id == static_cast<int32_t>(t.track_id)) {
                min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
                min_y = std::min(min_y, p.y); max_y = std::max(max_y, p.y);
                min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
                any = true;
            }
        }
        if (any) {
            t.min_x = min_x; t.max_x = max_x;
            t.min_y = min_y; t.max_y = max_y;
            t.min_z = min_z; t.max_z = max_z;
        }
        out.tracks.push_back(t);
    }
    return true;
}

// ============ serialize_packet ============

std::vector<uint8_t> serialize_packet(const Frame& frame) {
    const uint32_t N = static_cast<uint32_t>(frame.points.size());
    const uint32_t M = static_cast<uint32_t>(frame.tracks.size());

    // 1. 본문 크기 계산
    const size_t points_block_size = static_cast<size_t>(N) * POINT_WIRE_SIZE
                                     + static_cast<size_t>(N) * TARGET_ID_SIZE;
    const size_t end_of_points = PACKET_HEADER_SIZE + FRAME_HEADER_SIZE + points_block_size;

    size_t target_section_offset = 0;
    size_t total_size = end_of_points;
    if (M > 0) {
        // viewer 호환을 위해 target FH 를 48056 또는 그 이후로 배치.
        // 점이 적으면 48056 으로 padding, 많으면 (~2000) 그대로 이어 붙임.
        target_section_offset = std::max<size_t>(end_of_points, TARGET_FRAME_HEADER_OFFSET);
        total_size = target_section_offset + FRAME_HEADER_SIZE
                     + static_cast<size_t>(M) * TARGET_WIRE_SIZE;
    }

    std::vector<uint8_t> buf(total_size, 0u);

    // 2. PacketHeader
    PacketHeaderRaw pkt_hdr{};
    pkt_hdr.magic = PACKET_MAGIC;
    pkt_hdr.package_size = static_cast<uint32_t>(total_size - PACKET_HEADER_SIZE);
    write_le(buf.data(), 0, pkt_hdr);

    // 3. FrameHeader_A
    FrameHeaderRaw fh_a{};
    fh_a.magic = FRAME_MAGIC;
    fh_a.frame_count = frame.frame_count;
    fh_a.target_number = N;          // 여기서는 point count
    write_le(buf.data(), PACKET_HEADER_SIZE, fh_a);

    // 4. Points
    size_t off = PACKET_HEADER_SIZE + FRAME_HEADER_SIZE;
    for (const auto& p : frame.points) {
        PointRaw pr{p.x, p.y, p.z, p.doppler, p.power};
        write_le(buf.data(), off, pr);
        off += POINT_WIRE_SIZE;
    }

    // 5. Per-point targetIds
    for (const auto& p : frame.points) {
        int32_t tid = p.track_id;
        write_le(buf.data(), off, tid);
        off += TARGET_ID_SIZE;
    }

    // 6. Targets (있을 때만)
    if (M > 0) {
        FrameHeaderRaw fh_b{};
        fh_b.magic = FRAME_MAGIC;
        fh_b.frame_count = frame.frame_count;
        fh_b.target_number = M;      // 여기서는 target count
        write_le(buf.data(), target_section_offset, fh_b);

        size_t toff = target_section_offset + FRAME_HEADER_SIZE;
        for (const auto& t : frame.tracks) {
            TargetRaw tr{};
            tr.x = t.centroid_x;
            tr.y = t.centroid_y;
            tr.status = static_cast<uint32_t>(t.status);
            tr.target_id = t.track_id;
            // reserved 는 0 으로 둠
            write_le(buf.data(), toff, tr);
            toff += TARGET_WIRE_SIZE;
        }
    }

    return buf;
}

}  // namespace wire
}  // namespace radar
