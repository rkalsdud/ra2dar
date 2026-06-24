// POSIX 전용 (Linux/ARM). HOST(Windows) 빌드에서는 빈 translation unit.
#ifndef _WIN32

#include <iostream>
#include <unistd.h>
#include <sys/ioctl.h>
#include "sys/socket.h"
#include "sys/types.h"
#include "netinet/in.h"
#include "arpa/inet.h"
#include "fcntl.h"
#include "radar/radar_socket.h"

Socket::~Socket()
{
    if(socket_fd >= 0) {
        shutdown(socket_fd, SHUT_RDWR);
        close(socket_fd); 
        socket_fd = -1;
    }
}

bool Socket::connectSocket(const char* ip, int port)
{
    struct sockaddr_in addr;
    int ret;
 
    if(socket_fd < 0) {
        socket_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        std::cout << "Socket created(" << socket_fd << ")" << std::endl;
    }

    if(ip == nullptr) {
        std::cout << "Invalide IP address" << std::endl;
        return false;
    }
    
    if(port < 0) {
        std::cout << "Invalid port number" << std::endl;
        return false;
    }

    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ip);
    addr.sin_port = htons(port);

    // ⚠️ blocking connect — non-blocking 으로 connect 하면 localhost 라도 EINPROGRESS(-1)
    //    를 반환할 수 있어, 호출자의 재시도 루프가 연결 진행 중 소켓을 버리고 무한 반복한다.
    //    연결 판정은 blocking 으로 확실히 하고, readData 용 non-blocking 은 연결 성공 후 설정.
    if(connect(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cout << "Connection failed !" << std::endl;
        return false;
    }

    // 연결 성공 후 readData (ioctl FIONREAD 폴링) 용 non-blocking 설정.
    ret = fcntl(socket_fd, F_GETFL, 0);
    fcntl(socket_fd, F_SETFL, ret | O_NONBLOCK);

    return true;
}

int Socket::readData(uint8_t *buf, uint32_t size, bool isBlocked)
{
    int readBytes = 0;
    int packetSize = 0;
    int getSize = 0;

    if(buf == nullptr) {
        std::cout << "buf is nullptr" << std::endl;
        return readBytes;
    }

    if(size == 0) {
        std::cout << "size is not correct" << std::endl;
        return readBytes;
    }
    
    int remainSize = size;
    int readSize = 0;

    do {
        ioctl(socket_fd, FIONREAD, &packetSize);
        if(packetSize > 0) {            
            getSize = read(socket_fd, &buf[readSize], remainSize);
            readSize += getSize;
            remainSize -= getSize;
        }

        if(remainSize == 0) {      
            readBytes = readSize;      
            break;
        }

        /* Non sleep */
        if(isBlocked != true) {
            break;
        }

        usleep(1000); // 1ms
    } while (isBlocked);

    return readBytes;
}

int Socket::readCmdHeader(NetworkRx_CmdHeader* cmdHeader)
{
    uint8_t dummy[sizeof(NetworkRx_CmdHeader)] = {0, }; 
    uint8_t buffer[sizeof(NetworkRx_CmdHeader)];
    uint32_t dataSize = sizeof(NetworkRx_CmdHeader);
    uint32_t headerIdx = 0;
    int readBytes = 0;
    int numBuf = 0;
    uint32_t cycleCount = 0;
    
    if (cmdHeader == nullptr) {
        std::cout << "Network Rx Cmd header is nullptr" << std::endl;
        return -1;
    }

    while((cycleCount < 100) && (headerIdx != dataSize)) {
        usleep(1); // 1us
        uint32_t getHeaderSize = dataSize - headerIdx;
        readBytes = readData(buffer, getHeaderSize, false);

        if (readBytes <= 0) {
            std::cout << "Cycle Count : " << cycleCount << std::endl;
            return readBytes;
        } else if (readBytes > 0) {
            if (headerIdx == 0) {
                if (memcmp(buffer, dummy, 36) == 0) {
                    continue;
                }
            }
            uint8_t bufIdx = 0;
            cycleCount++;

            while(bufIdx < readBytes - 4) {
                if(headerIdx < 8) {
                    /* found header */
                    if(memcmp(buffer + bufIdx, NETWORK_TX_HEADER, NETWORK_TX_HEADER_LENGTH) == 0) {
                        /* load prev data */
                        if(bufIdx == 0) {
                            cmdHeader->numBuf = numBuf;
                            memcpy((uint8_t *)cmdHeader + 4, buffer, readBytes);
                            headerIdx += readBytes;
                        } else {
                            memcpy((uint8_t *)cmdHeader, buffer + bufIdx - 4, readBytes - bufIdx + 4);
                            headerIdx += readBytes - bufIdx + 4;
                        }

                        bufIdx = readBytes;
                    }
                    else {
                        bufIdx += 4;
                    }
                } else {
                    memcpy((uint8_t *)cmdHeader + headerIdx, buffer, readBytes);
                    headerIdx += readBytes;
                    bufIdx = readBytes;
                    break;
                }
            }

            /* not found header, last data save */
            if (headerIdx == 0) {
                numBuf = (int)GetU32(buffer + readBytes - 4);
            }
        }
    }

    if (cmdHeader->dataSize > MAX_BUF_SIZE) {
        std::cout << "CMD header's datasize exceeded MAX_BUF_SIZE" << std::endl;
        return -2;
    }

    if (headerIdx != dataSize) {
        std::cout << "Mismatch the datasize !" << std::endl;
        return 1;
    }

    return 0;
}

#endif  // !_WIN32