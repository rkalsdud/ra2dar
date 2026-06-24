#!/bin/bash
# ============================================================================
# Retina 4SN Viewer launcher — systemd 가 부팅 시 호출.
#
# 핫스팟 환경에서 매번 바뀌는 레이더 IP 를 MAC 기반 ARP 스캔으로 자동 검색,
# 발견 시 viewer_three.py 를 venv python 으로 실행.
#
# 동작 원칙:
#   1. arp-scan 으로 레이더 검색 (짧게 ~15초)
#   2. 발견 시: 그 IP 로 viewer 실행
#   3. 미발견 시: placeholder IP(0.0.0.0) 로 viewer 실행 — Dash 사이트는 즉시 열림.
#      viewer 의 tcp_loop 가 자체 backoff 로 무한 재접속 시도.
#      레이더가 늦게 들어와도 viewer 가 알아서 잡음.
#      ※ IP 가 바뀌었으면 systemctl restart 로 재스캔 트리거.
#
# 사전 설치 (Pi 에서 한 번만):
#   sudo apt install arp-scan
#   chmod +x launch_viewer.sh
#
# 사용처:
#   systemd ExecStart=/path/to/launch_viewer.sh
#   또는 셸에서 직접 ./launch_viewer.sh
# ============================================================================

set -u   # 미정의 변수 사용 시 즉시 종료
# 주의: set -e 는 일부러 안 씀 — arp-scan 비정상 종료 시에도 retry 루프 살아있게.

# ── 환경별로 수정해야 할 값 ────────────────────────────────────────────────
# 레이더 본체 스티커 또는 평소 `arp -a | grep <레이더IP>` 로 확인.
# 대소문자 무관, 콜론 구분.
RADAR_MAC="b8:27:eb:XX:XX:XX"

# venv 의 python 절대경로. 평소 `source activate && which python` 결과.
VENV_PYTHON="/home/pi/myvenv/.venv/bin/python"

# viewer 가 있는 폴더 (cd 대상).
VIEWER_DIR="/home/pi/radar/raspberry_pi_viewer"

# viewer 스크립트 이름.
VIEWER_SCRIPT="viewer_three.py"

# viewer 추가 인자 (없으면 빈 문자열).
VIEWER_ARGS=""

# 레이더 미발견 시 사용할 placeholder IP. viewer 가 접속 실패 → 무한 재시도 →
# Dash 웹서버는 계속 떠 있어 브라우저 접속 가능.
PLACEHOLDER_IP="0.0.0.0"

# ARP 스캔 인터페이스 (자동 감지). 무선이 wlan0, 유선이 eth0 보통.
# 핫스팟 자동 식별 — 기본 라우트가 가리키는 인터페이스 사용.
SCAN_IFACE=$(ip route show default | awk '{print $5; exit}')
SCAN_IFACE=${SCAN_IFACE:-wlan0}

# 재시도 — 충분히 길게 (90초). 레이더가 Pi 보다 부팅 늦은 게 보통이라
# (SDK 초기화 + /log2 mount 대기 등 30~60초 걸림) 검색 창을 넉넉히.
# 그래도 못 찾으면 placeholder 로 viewer 띄움 → systemctl restart 로 재스캔.
MAX_RETRY=45
RETRY_SLEEP=2

# ── 본문 ────────────────────────────────────────────────────────────────────
log() { echo "[launcher $(date +%H:%M:%S)] $*"; }

# 네트워크 깨어날 시간 약간 더 — systemd After=network-online 으로도 보호되지만 안전판.
# 동시에 핫스팟 / 레이더가 안정화될 시간도 벌어줌.
sleep 10

log "iface=$SCAN_IFACE  searching radar MAC=$RADAR_MAC ..."

RADAR_IP=""
for i in $(seq 1 "$MAX_RETRY"); do
    # arp-scan: 같은 서브넷 전체 ping + ARP 응답 수집 → MAC/IP 매핑.
    # 결과 예시 라인: "192.168.43.5  b8:27:eb:1a:2b:3c  (Unknown)"
    FOUND=$(sudo arp-scan --interface="$SCAN_IFACE" --localnet 2>/dev/null \
            | awk -v mac="$(echo "$RADAR_MAC" | tr 'A-Z' 'a-z')" \
                  'tolower($2)==mac { print $1; exit }')
    if [ -n "$FOUND" ]; then
        RADAR_IP="$FOUND"
        log "radar found at $RADAR_IP (after ${i} tries)"
        break
    fi
    log "not found yet, retry $i/$MAX_RETRY"
    sleep "$RETRY_SLEEP"
done

# 못 찾았으면 placeholder 로 fallback — viewer 는 떠야 하니까.
if [ -z "$RADAR_IP" ]; then
    log "radar (MAC=$RADAR_MAC) not found in $((MAX_RETRY*RETRY_SLEEP))s"
    log "  → fallback: launching with placeholder IP=$PLACEHOLDER_IP"
    log "  → 브라우저는 접속 가능, viewer 가 자체 backoff 로 레이더 재시도"
    log "  → 레이더 늦게 ON 됐다가 안 잡히면: sudo systemctl restart retina-viewer"
    RADAR_IP="$PLACEHOLDER_IP"
fi

log "launching viewer: $VENV_PYTHON $VIEWER_SCRIPT --retina-host $RADAR_IP $VIEWER_ARGS"

cd "$VIEWER_DIR" || { log "FATAL: cd $VIEWER_DIR 실패"; exit 1; }
exec "$VENV_PYTHON" "$VIEWER_SCRIPT" --retina-host "$RADAR_IP" $VIEWER_ARGS
