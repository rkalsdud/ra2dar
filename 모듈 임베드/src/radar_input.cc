// 점 cloud 입력 파서 (POSIX 전용). HOST 빌드에서는 빈 translation unit.

#ifndef _WIN32

#include "radar/radar_input.h"

#include <cstring>

namespace radar {
namespace input {

bool receive_frame(Socket& sock, std::vector<radar::Point>& out, uint32_t& frame_num) {
    // FRAME_SIZE ≈ 100 KB — 스택 폭발 방지 위해 static (단일 수신 스레드라 안전).
    static PACKET_BUFFER pb;
    out.clear();

    // 1) CmdHeader 36B — blocking readData 로 점이 올 때까지 대기 (예제 main.cc 방식).
    //    ⚠️ readCmdHeader(non-blocking magic 재탐색)는 점 프레임 간격(~50ms) 동안
    //       "데이터 없음"으로 즉시 실패해 정상 수신을 깨뜨림 → blocking readData 로 복원.
    //       desync 는 호출자의 소켓 재연결(consec_fail)로 복구.
    int readBytes = sock.readData(reinterpret_cast<uint8_t*>(&pb.cmdHeader),
                                  RADAR_CMD_HEADER_LENGTH, true);
    if (readBytes != static_cast<int>(RADAR_CMD_HEADER_LENGTH)) return false;
    if (std::memcmp(&pb.cmdHeader.header, NETWORK_TX_HEADER, NETWORK_TX_HEADER_LENGTH) != 0)
        return false;
    if (pb.cmdHeader.dataSize == 0 || pb.cmdHeader.dataSize > MAX_BUF_SIZE) return false;

    // 2) payload (dataSize) 수신 + packet 매직 확인
    readBytes = sock.readData(reinterpret_cast<uint8_t*>(&pb.buf),
                              pb.cmdHeader.dataSize, true);
    if (readBytes != static_cast<int>(pb.cmdHeader.dataSize)) return false;
    if (std::memcmp(pb.pkHeader.magicWord, radarMagicWord, RADAR_OUTPUT_MAGIC_WORD_LENGTH) != 0)
        return false;

    frame_num = pb.pkHeader.frame_counter;
    uint32_t nPoints = pb.pkHeader.targetNumber;
    if (nPoints > MAX_NUM_POINTS_PER_FRAME) nPoints = MAX_NUM_POINTS_PER_FRAME;

    // 3) POINT_DATA[] → radar::Point
    out.reserve(nPoints);
    for (uint32_t i = 0; i < nPoints; ++i) {
        const POINT_DATA* p = reinterpret_cast<const POINT_DATA*>(
            pb.data + sizeof(POINT_DATA) * i);
        radar::Point rp;
        rp.x = p->x;
        rp.y = p->y;
        rp.z = p->z;
        rp.doppler = p->doppler;
        rp.power = static_cast<float>(p->power);   // uint32 → float
        rp.track_id = -1;
        out.push_back(rp);
    }
    return true;
}

}  // namespace input
}  // namespace radar

#endif  // !_WIN32
