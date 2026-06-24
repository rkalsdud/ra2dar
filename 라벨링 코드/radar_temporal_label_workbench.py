#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar Temporal Label Workbench v19
- 같은 폴더의 dataset zip 선택
- frame별 rule-based label 생성
- high anchor 기반 non-high 위치/크기 temporal correction
- 전체 frame / 검수 frame 모두 브라우저에서 확인
- 검수 결과 CSV/JSONL export
- v18: separate dataset default count from current-frame count; guard against 0/0/0 load reset
- v19: strict schema export. 메인 파일 labels_final_all_frames_schema.jsonl 에는
       class / pose / box(center, dimensions, yaw) 만 들어가고 n_points / track_id 등
       디버그 필드가 절대 남지 않는다. export 직후 자동 strict validation 실행 +
       markdown 리포트 + CSV. source order 버전
       (labels_final_strict_source_order.jsonl) 도 동시에 생성한다.

실행:
  python radar_temporal_label_workbench.py
  python radar_temporal_label_workbench.py --data-dir . --out workbench_outputs
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import socket
import statistics
import bisect
import struct
import sys
import time
import webbrowser
import zipfile
import zlib
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# -----------------------------
# Config
# -----------------------------

@dataclass
class RuleParams:
    box_w: float = 0.8
    box_l: float = 0.8
    box_h: float = 1.8
    min_w: float = 0.55
    min_l: float = 0.55
    max_w: float = 1.40  # v19: walking/horizontal 처럼 팔이 옆으로 흔들리는 자세도 박스 안에 포함되도록 상한 완화
    max_l: float = 1.40
    min_h: float = 1.25
    max_h: float = 3.0  # v19+: min/max z anchor 와 함께 점군 끝까지 박스가 덮도록 상한 확장
    trk_radius: float = 0.8
    z_pool_radius: float = 1.5  # v19: z range는 trk_radius보다 넓게 잡아서 raised-arm 같은 sparse 상부 점도 포함
    box_padding_xy: float = 0.15  # v19: 박스 모서리에서 점이 새지 않도록 xy 여유 padding (walking 팔 흔들림 커버)
    box_padding_z: float = 0.20   # v19: z 는 머리/발에 좀 더 큰 여유 margin (튀어나오는 점 방지)
    temporal_xy_reanchor_threshold: float = 1.0  # v19: temporal anchor xy가 현재 frame 점군 중심에서 이만큼 멀면 snap
    min_points_for_box: int = 30  # v19: 이 미만의 점만 있는 frame 은 박스 생성 skip (empty 로 export)
    min_tid_points_high: int = 15
    min_tid_points_medium: int = 5
    invalid_tid: int = 255
    max_interpolation_gap_frames: int = 30
    max_anchor_gap_frames: int = 60
    # TRK/cluster가 전혀 잡히지 않았지만 point cloud가 존재하는 frame을
    # objects: []로 버리지 않기 위한 fallback 후보 box.
    fallback_point_box: bool = True
    fallback_min_points: int = 6
    fallback_medium_points: int = 12
    fallback_padding_xy: float = 0.18
    fallback_padding_z: float = 0.12
    enable_cluster_candidates: bool = False  # v9: load 속도 때문에 기본은 끄고, 화면에 보이는 frame에서만 lazy 생성
    cluster_eps_xy: float = 0.42
    min_cluster_points_medium: int = 8
    min_cluster_points_high: int = 18
    max_extra_cluster_candidates: int = 2  # v19: load time 단축 위해 후보 줄임 (dog+human 시나리오 충분)
    cluster_box_margin_xy: float = 0.18
    cluster_box_margin_z: float = 0.18

TRK_STATUS_NAME = {0: "standing", 1: "lying_down", 2: "sitting", 3: "falling", 4: "unknown"}
CONF_ORDER = ["high", "medium", "low", "empty", "error"]

METHOD_TEXT = {
    "auto_high": "자동 사용: high frame",
    "auto_high_plus_lazy_cluster": "자동 high + 현재 frame 추가 cluster 후보",
    "auto_medium_smoothed": "자동 보정: medium frame smoothing",
    "interpolate_between_prev_next_high": "앞뒤 안정 frame 사이 보간",
    "carry_forward_prev_high": "이전 안정 frame 기준 유지",
    "carry_backward_next_high": "다음 안정 frame 기준 역보정",
    "no_temporal_anchor": "보정 불가: 주변 안정 frame 없음",
    "empty_or_error": "point 없음/파일 오류",
    "raw_only": "초기 자동 box만 있음",
}
RECOMMEND_TEXT = {
    "auto_use": "사용 권장",
    "quick_check": "빠른 확인 후 사용",
    "careful_check": "신중 검수",
    "exclude": "제외 권장",
}
RECOMMEND_HELP = {
    "auto_use": "high frame이거나 temporal 보정 근거가 충분한 frame입니다. 샘플 확인 후 대량 사용 후보입니다.",
    "quick_check": "큰 문제는 없어 보이나, box 흐름만 빠르게 확인하면 좋습니다.",
    "careful_check": "point/TRK 근거가 약하거나 보간/유지 방식입니다. 눈으로 확인 후 accept/reject를 결정하세요.",
    "exclude": "empty/error 또는 주변 안정 frame이 부족합니다. 기본 제외 후보입니다.",
}


def is_multi_class_mode(object_class: str) -> bool:
    return object_class in ("multi", "human_animal", "person+animal", "multi_human_animal", "person_animal", "person+animal", "multi_person_animal")

def default_object_class(object_class: str) -> str:
    return "candidate" if is_multi_class_mode(object_class) else normalize_object_class(object_class, multi_mode=False)

def pose_options() -> List[str]:
    return ["unknown", "standing", "walking", "horizontal_movement", "low_movement", "transition", "sitting", "lying_down", "falling"]

def box_values_from_obj(obj: Dict[str,Any]) -> Optional[Tuple[float,float,float,float,float,float,float]]:
    try:
        b = obj["box"]
        return (safe_float(b["position"]["x"]), safe_float(b["position"]["y"]), safe_float(b["position"]["z"]), safe_float(b["dimensions"]["w"]), safe_float(b["dimensions"]["l"]), safe_float(b["dimensions"]["h"]), safe_float(b["rotation"].get("yaw",0)))
    except Exception:
        return None

def obj_track_id(obj: Dict[str,Any]) -> Optional[int]:
    q = obj.get("quality") or {}
    if "track_id" not in q:
        return None
    try:
        return int(q.get("track_id"))
    except Exception:
        return None

# -----------------------------
# Utility
# -----------------------------

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def median(vals: List[float], default: float = 0.0) -> float:
    return statistics.median(vals) if vals else default

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(k)); hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi-k) + vals[hi] * (k-lo)

# -----------------------------
# Zip reading, including damaged central directory fallback
# -----------------------------

class ZipDatasetReader:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        self.mode = "zipfile"
        self._zip: Optional[zipfile.ZipFile] = None
        self.entries: List[Dict[str, Any]] = []

    def open(self) -> None:
        if self._zip is not None or self.entries:
            return
        try:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
            names = self._zip.namelist()
            self.entries = [{"name": n, "mode": "zipfile"} for n in names if n.endswith(".json")]
            self.mode = "zipfile"
        except Exception:
            self.mode = "local_header"
            self.entries = self._scan_local_headers()

    def close(self) -> None:
        if self._zip:
            self._zip.close()
        self._zip = None

    def _scan_local_headers(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        off = 0
        size = self.zip_path.stat().st_size
        with self.zip_path.open("rb") as f:
            while off + 30 <= size:
                f.seek(off)
                hdr = f.read(30)
                if len(hdr) < 30:
                    break
                if hdr[:4] != b"PK\x03\x04":
                    f.seek(off)
                    chunk = f.read(1024 * 1024)
                    idx = chunk.find(b"PK\x03\x04")
                    if idx == -1:
                        break
                    off += idx
                    continue
                sig, ver, flag, comp, mt, md, crc, csize, usize, namelen, extralen = struct.unpack("<IHHHHHIIIHH", hdr)
                name = f.read(namelen).decode("utf-8", "replace")
                f.read(extralen)
                data_off = off + 30 + namelen + extralen
                if name.endswith(".json"):
                    out.append({"name": name, "mode": "local_header", "data_off": data_off, "comp": comp, "csize": csize})
                off = data_off + csize
        return out

    def list_jsons(self) -> List[str]:
        self.open()
        return [e["name"] for e in self.entries]

    def read_json_bytes(self, name: str) -> bytes:
        self.open()
        if self.mode == "zipfile" and self._zip is not None:
            return self._zip.read(name)
        e = next((x for x in self.entries if x["name"] == name), None)
        if not e:
            raise FileNotFoundError(name)
        with self.zip_path.open("rb") as f:
            f.seek(e["data_off"])
            raw = f.read(e["csize"])
        if e["comp"] == 0:
            return raw
        if e["comp"] == 8:
            return zlib.decompress(raw, -15)
        raise ValueError(f"Unsupported zip compression method: {e['comp']}")

# -----------------------------
# Frame parsing and labeling
# -----------------------------

def parse_points(frame: Dict[str, Any]) -> List[Tuple[float,float,float,float,float,int]]:
    """Return (x,y,z,V,P,TID).

    v12 fix:
    - raw frame JSON만 사용하도록 load_dataset에서 먼저 필터링한다.
    - V/P/TID가 일부 비어 있어도 C 좌표가 있으면 점은 그린다.
      TID가 없으면 invalid track(255)로 처리한다.
    """
    C = frame.get("C") or []
    if isinstance(C, dict):
        return []
    V = frame.get("V") or []
    P = frame.get("P") or []
    TID = frame.get("TID") or []

    def v_at(i: int) -> float:
        return safe_float(V[i]) if i < len(V) else 0.0
    def p_at(i: int) -> float:
        return safe_float(P[i]) if i < len(P) else 0.0
    def tid_at(i: int) -> int:
        try:
            return int(TID[i]) if i < len(TID) else 255
        except Exception:
            return 255

    if C and isinstance(C[0], list):
        pts=[]
        for i, c in enumerate(C):
            if not isinstance(c, list) or len(c) < 3:
                continue
            pts.append((safe_float(c[0]), safe_float(c[1]), safe_float(c[2]), v_at(i), p_at(i), tid_at(i)))
        return pts

    n = len(C)//3
    pts = []
    for i in range(n):
        pts.append((safe_float(C[3*i]), safe_float(C[3*i+1]), safe_float(C[3*i+2]), v_at(i), p_at(i), tid_at(i)))
    return pts

def is_raw_frame_json(frame: Any) -> bool:
    """Workbench에 넣을 실제 센서 frame인지 검사.

    이전 버전은 zip 안의 모든 .json을 읽어서, 이미 만들어진 label JSON
    또는 report JSON까지 frame으로 처리할 수 있었다. 그러면 C/V/P가 없어서
    화면에는 box만 보이고 point cloud가 안 보인다.
    """
    if not isinstance(frame, dict):
        return False
    if "C" not in frame:
        return False
    C = frame.get("C")
    if C is None or not isinstance(C, list):
        return False
    # raw frame은 보통 V/P/TID/TRK 중 일부를 같이 가진다.
    # C=[]인 빈 센서 frame도 학습용 negative frame이 될 수 있으므로 허용한다.
    return any(k in frame for k in ("V", "P", "TID", "TRK", "T"))

def parse_tracks(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    trk = frame.get("TRK") or []
    tracks = []
    for i in range(0, len(trk)-3, 4):
        tid = int(safe_float(trk[i], -1))
        status = int(safe_float(trk[i+1], 4))
        x, y = safe_float(trk[i+2]), safe_float(trk[i+3])
        if math.isfinite(x) and math.isfinite(y):
            tracks.append({"track_id": tid, "status": status, "x": x, "y": y})
    return tracks

def frame_id_from_name(name: str) -> str:
    return Path(name).stem

def record_from_name(name: str) -> str:
    return str(Path(name).parent)

def timestamp_from_name(name: str) -> float:
    return safe_float(Path(name).stem, 0.0)

def normalize_object_class(name: Any, multi_mode: bool = False) -> str:
    v = str(name or "").strip().lower()
    aliases = {"human":"person","person":"person","사람":"person","animal":"animal","dog":"animal","개":"animal","non-human":"non_human","non_human":"non_human","nonhuman":"non_human","background":"non_human","candidate":"candidate","unknown":"candidate","":"candidate"}
    out = aliases.get(v, v)
    if out == "candidate" and not multi_mode:
        return "person"
    return out

def normalize_pose_label(name: Any) -> str:
    v = str(name or "").strip().lower()
    aliases = {"stand":"standing","standing":"standing","trk_standing":"standing","scenario_stand":"standing","walk":"walking","walking":"walking","upright_walking":"walking","horizontal":"horizontal_movement","horizontal_movement":"horizontal_movement","side_walking":"horizontal_movement","low":"low_movement","low_movement":"low_movement","transition":"transition","sit":"sitting","sitting":"sitting","lie":"lying_down","lying":"lying_down","lying_down":"lying_down","fall":"falling","falling":"falling","unknown":"unknown","auto":"unknown","":"unknown"}
    return aliases.get(v, v)

MODEL_CLASSES = ["person", "animal", "non_human"]
ALLOWED_CLASSES = set(MODEL_CLASSES)
# v19: 최종 라벨 strict 스키마의 허용 pose 값.
ALLOWED_POSES = {"standing", "walking", "horizontal_movement", "low_movement", "transition", "unknown", "sitting", "lying_down", "falling"}
STRICT_OBJECT_KEYS = {"class", "pose", "box"}
STRICT_BOX_KEYS = {"center", "dimensions", "yaw"}
STRICT_FRAME_KEYS = {"frame_id", "objects"}


def object_to_model_schema(obj: Dict[str,Any], default_class: str = "person", default_pose: str = "unknown", multi_mode: bool = False) -> Optional[Dict[str,Any]]:
    """Strict object → {class, pose, box{center,dimensions,yaw}} only.

    v19: n_points / track_id / quality / label_source 같은 디버그 필드는 절대로 넣지 않는다.
    이 함수가 export / preview / current_payload 모두에서 공통으로 쓰이기 때문에 여기서 한 번
    만 strict하게 만들어 두면 어떤 경로로 export 되더라도 디버그 필드가 새지 않는다.
    """
    b = obj.get("box") or {}
    pos = b.get("position") or {}
    dim = b.get("dimensions") or {}
    rot = b.get("rotation") or {}
    try:
        cx = safe_float(pos.get("x")); cy = safe_float(pos.get("y")); cz = safe_float(pos.get("z"))
        dx = safe_float(dim.get("w", dim.get("dx", 0.0)))
        dy = safe_float(dim.get("l", dim.get("dy", 0.0)))
        dz = safe_float(dim.get("h", dim.get("dz", 0.0)))
        yaw = safe_float(rot.get("yaw", b.get("yaw", 0.0)))
    except Exception:
        return None
    cls_src = obj.get("class") or obj.get("object_class") or default_class
    pose_src = obj.get("pose") or obj.get("pose_class") or default_pose
    cls = normalize_object_class(cls_src, multi_mode=multi_mode)
    if cls not in ALLOWED_CLASSES:
        return None
    pose = normalize_pose_label(pose_src)
    if pose not in ALLOWED_POSES:
        pose = "unknown"
    return {
        "class": cls,
        "pose": pose,
        "box": {
            "center": [round(cx, 6), round(cy, 6), round(cz, 6)],
            "dimensions": [round(dx, 6), round(dy, 6), round(dz, 6)],
            "yaw": round(yaw, 6),
        },
    }


def label_to_model_schema(lab: Dict[str,Any], default_class: str = "person", default_pose: str = "unknown", multi_mode: bool = False) -> Dict[str,Any]:
    """Strict frame → {frame_id, objects:[...strict objects...]}.

    빈 프레임은 반드시 objects: [] 로 떨어진다.
    """
    objects: List[Dict[str,Any]] = []
    for obj in lab.get("objects", []) or []:
        item = object_to_model_schema(obj, default_class=default_class, default_pose=default_pose, multi_mode=multi_mode)
        if item is not None:
            objects.append(item)
    return {"frame_id": str(lab.get("frame_id", "")), "objects": objects}


def strict_label_schema(lab: Dict[str,Any], default_class: str = "person", default_pose: str = "unknown", multi_mode: bool = False) -> Dict[str,Any]:
    """Alias for label_to_model_schema for call-sites that want to make 'strict' explicit."""
    return label_to_model_schema(lab, default_class=default_class, default_pose=default_pose, multi_mode=multi_mode)


def normalize_expected_counts(value: Any) -> Dict[str, int]:
    """Normalize UI-supplied object-count gate values."""
    out = {k: 0 for k in MODEL_CLASSES}
    if isinstance(value, dict):
        for k in MODEL_CLASSES:
            try:
                out[k] = max(0, int(value.get(k, 0) or 0))
            except Exception:
                out[k] = 0
    return out


def class_of_object(obj: Dict[str,Any], default_class: str, multi_mode: bool) -> str:
    return normalize_object_class(obj.get("class") or obj.get("object_class") or default_class, multi_mode=multi_mode)


def default_expected_counts_for_export(lab: Dict[str,Any], default_class: str, multi_mode: bool) -> Dict[str,int]:
    """Fallback gate when a frame has not been manually counted.

    - Single-class datasets: non-empty frames export max 1 object for the dataset class.
    - Multi-class datasets: infer only explicitly assigned person/animal/non_human boxes;
      candidate boxes are not exported unless the user assigns them through slots.
    """
    objects = lab.get("objects", []) or []
    counts = {k: 0 for k in MODEL_CLASSES}
    if not objects:
        return counts
    if not multi_mode:
        cls = normalize_object_class(default_class, multi_mode=False)
        if cls not in MODEL_CLASSES:
            cls = "person"
        counts[cls] = 1
        return counts
    for obj in objects:
        cls = class_of_object(obj, default_class, multi_mode=True)
        if cls in MODEL_CLASSES:
            counts[cls] += 1
    return counts


def enforce_object_count_gate(lab: Dict[str,Any], expected_counts: Optional[Dict[str,int]], default_class: str, default_pose: str, multi_mode: bool) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    """Validate the final slot/count rule before JSONL export.

    Final exported objects must equal person_count + animal_count + non_human_count.
    Extra candidates are dropped. Extra same-class boxes are trimmed. Missing boxes are reported
    but never fabricated because a fake box would contaminate the ground-truth labels.
    """
    out = json.loads(json.dumps(lab, ensure_ascii=False))
    objects = out.get("objects", []) or []
    if expected_counts is None:
        counts = default_expected_counts_for_export(out, default_class, multi_mode)
        source = "derived"
    else:
        counts = normalize_expected_counts(expected_counts)
        source = "manual"

    used = set()
    selected: List[Dict[str,Any]] = []
    missing: Dict[str,int] = {k: 0 for k in MODEL_CLASSES}

    def pick_index(target_cls: str) -> Optional[int]:
        # 1) Prefer an already-classed object.
        for i, obj in enumerate(objects):
            if i in used:
                continue
            if class_of_object(obj, default_class, multi_mode=True) == target_cls:
                return i
        # 2) For manual count gates, allow candidate/unassigned boxes to fill a requested slot.
        if source == "manual":
            for i, obj in enumerate(objects):
                if i in used:
                    continue
                if class_of_object(obj, default_class, multi_mode=True) not in MODEL_CLASSES:
                    return i
            # 3) Last resort: use any remaining box, reclassed to the requested slot.
            for i, obj in enumerate(objects):
                if i not in used:
                    return i
        return None

    for cls in MODEL_CLASSES:
        for _ in range(counts.get(cls, 0)):
            idx = pick_index(cls)
            if idx is None:
                missing[cls] += 1
                continue
            used.add(idx)
            obj = json.loads(json.dumps(objects[idx], ensure_ascii=False))
            obj["object_class"] = cls
            obj["pose_class"] = normalize_pose_label(obj.get("pose") or obj.get("pose_class") or default_pose)
            obj["label_source"] = obj.get("label_source") or "count_gate"
            selected.append(obj)

    selected = sanitize_objects(selected, default_class, default_pose)
    out["objects"] = selected
    out["expected_counts"] = counts
    out["count_gate_applied"] = True
    input_valid_count = sum(1 for obj in objects if class_of_object(obj, default_class, multi_mode=True) in MODEL_CLASSES)
    candidate_count = max(0, len(objects) - input_valid_count)
    expected_total = sum(counts.values())
    diag = {
        "frame_id": out.get("frame_id", ""),
        "gate_source": source,
        "expected_counts": counts,
        "expected_total": expected_total,
        "input_object_count": len(objects),
        "output_object_count": len(selected),
        "trimmed_count": max(0, len(objects) - len(selected)),
        "candidate_or_unassigned_count": candidate_count,
        "missing_counts": missing,
        "missing_total": sum(missing.values()),
        "valid": len(selected) == expected_total and sum(missing.values()) == 0,
    }
    return out, diag


# -----------------------------
# v19: strict JSONL validator
# -----------------------------

def _len_or_none(v: Any) -> Any:
    return len(v) if isinstance(v, list) else "n/a"


def validate_strict_jsonl(path: Path, frame_point_counts: Optional[Dict[str, int]] = None, min_points_for_box: int = 0) -> Dict[str, Any]:
    """Read back the exported JSONL and verify strict schema.

    Returns a summary dict with both aggregate counts and per-issue rows. The caller
    persists these as `validation_report.md` and `strict_validation_issues.csv` so
    a downstream training pipeline can reject the file before consuming it.

    v19: min_points_for_box 임계값 이하의 점만 있는 frame은 의도적으로 empty 로 출력하므로
    "point_count > 0 but no object" 검사에서 제외한다 (frame_point_count >= 임계값 인 경우에만 flag).
    """
    frame_point_counts = frame_point_counts or {}
    total_frames = 0
    total_objects = 0
    empty_frames = 0
    one_object_frames = 0
    multi_object_frames = 0
    seen_fids: set = set()
    duplicate_fids: List[str] = []
    invalid_class_examples: List[str] = []
    invalid_pose_examples: List[str] = []
    bad_center_count = 0
    bad_dims_count = 0
    missing_yaw_count = 0
    missing_box_count = 0
    extra_object_field_count = 0
    extra_box_field_count = 0
    extra_top_field_count = 0
    pc_pos_obj_zero = 0
    pc_zero_obj_pos = 0
    issues: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except Exception as e:
                issues.append({"line_no": line_no, "frame_id": "", "issue": "invalid_json", "detail": str(e)})
                continue
            total_frames += 1
            fid = row.get("frame_id", "")
            if not isinstance(fid, str) or not fid:
                issues.append({"line_no": line_no, "frame_id": str(fid), "issue": "missing_frame_id", "detail": ""})
            elif fid in seen_fids:
                issues.append({"line_no": line_no, "frame_id": fid, "issue": "duplicate_frame_id", "detail": ""})
                duplicate_fids.append(fid)
            else:
                seen_fids.add(fid)
            extras_top = sorted(set(row.keys()) - STRICT_FRAME_KEYS)
            for k in extras_top:
                extra_top_field_count += 1
                issues.append({"line_no": line_no, "frame_id": fid, "issue": "extra_top_field", "detail": k})
            objects = row.get("objects")
            if not isinstance(objects, list):
                issues.append({"line_no": line_no, "frame_id": fid, "issue": "objects_not_list", "detail": str(type(objects).__name__)})
                continue
            n_obj = len(objects)
            if n_obj == 0:
                empty_frames += 1
            elif n_obj == 1:
                one_object_frames += 1
            else:
                multi_object_frames += 1
            total_objects += n_obj
            pc = frame_point_counts.get(fid)
            if pc is not None:
                # v19: 점이 너무 적은 frame 은 의도적으로 empty 처리하므로 flag 에서 제외.
                if pc >= max(1, int(min_points_for_box)) and n_obj == 0:
                    pc_pos_obj_zero += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "point_count_positive_but_no_object", "detail": f"point_count={pc}"})
                if pc <= 0 and n_obj > 0:
                    pc_zero_obj_pos += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "point_count_zero_but_has_object", "detail": f"point_count={pc} n_objects={n_obj}"})
            for obj_i, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "object_not_dict", "detail": f"index={obj_i}"})
                    continue
                obj_extras = sorted(set(obj.keys()) - STRICT_OBJECT_KEYS)
                for k in obj_extras:
                    extra_object_field_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "object_extra_field", "detail": k})
                cls = obj.get("class")
                if cls not in ALLOWED_CLASSES:
                    invalid_class_examples.append(str(cls))
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "invalid_class", "detail": str(cls)})
                pose = obj.get("pose")
                if pose not in ALLOWED_POSES:
                    invalid_pose_examples.append(str(pose))
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "invalid_pose", "detail": str(pose)})
                box = obj.get("box")
                if not isinstance(box, dict):
                    missing_box_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "missing_box", "detail": ""})
                    continue
                box_extras = sorted(set(box.keys()) - STRICT_BOX_KEYS)
                for k in box_extras:
                    extra_box_field_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "box_extra_field", "detail": k})
                center = box.get("center")
                if not (isinstance(center, list) and len(center) == 3 and all(isinstance(v, (int, float)) for v in center)):
                    bad_center_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "center_bad_length_or_type", "detail": f"len={_len_or_none(center)}"})
                dims = box.get("dimensions")
                if not (isinstance(dims, list) and len(dims) == 3 and all(isinstance(v, (int, float)) for v in dims)):
                    bad_dims_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "dimensions_bad_length_or_type", "detail": f"len={_len_or_none(dims)}"})
                if "yaw" not in box or not isinstance(box.get("yaw"), (int, float)):
                    missing_yaw_count += 1
                    issues.append({"line_no": line_no, "frame_id": fid, "issue": "missing_or_non_numeric_yaw", "detail": ""})
    summary = {
        "path": str(path),
        "total_frames": total_frames,
        "total_objects": total_objects,
        "empty_frames": empty_frames,
        "one_object_frames": one_object_frames,
        "multi_object_frames": multi_object_frames,
        "unique_frame_ids": len(seen_fids),
        "duplicate_frame_id_count": len(duplicate_fids),
        "duplicate_frame_id_examples": sorted(set(duplicate_fids))[:10],
        "invalid_class_count": len(invalid_class_examples),
        "invalid_class_examples": sorted(set(invalid_class_examples))[:10],
        "invalid_pose_count": len(invalid_pose_examples),
        "invalid_pose_examples": sorted(set(invalid_pose_examples))[:10],
        "missing_box_count": missing_box_count,
        "center_bad_count": bad_center_count,
        "dimensions_bad_count": bad_dims_count,
        "missing_yaw_count": missing_yaw_count,
        "extra_top_field_count": extra_top_field_count,
        "extra_object_field_count": extra_object_field_count,
        "extra_box_field_count": extra_box_field_count,
        "point_count_positive_but_no_object": pc_pos_obj_zero,
        "point_count_zero_but_has_object": pc_zero_obj_pos,
        "issue_count": len(issues),
        "is_strict_valid": (
            len(issues) == 0
            and len(duplicate_fids) == 0
            and len(invalid_class_examples) == 0
            and len(invalid_pose_examples) == 0
            and missing_box_count == 0
            and bad_center_count == 0
            and bad_dims_count == 0
            and missing_yaw_count == 0
            and extra_top_field_count == 0
            and extra_object_field_count == 0
            and extra_box_field_count == 0
        ),
        "issues": issues,
    }
    return summary


def write_label_readme(d: Path, primary_path: Path, summary: Dict[str, Any], dataset_meta: Optional[Dict[str, Any]] = None, minimal: bool = False) -> Path:
    """v19: strict JSONL 옆에 학습 데이터 가이드라인 README.md 를 만든다.

    학습/제출 받아가는 사람이 이 파일만 봐도 schema/허용값/좌표계/검증결과를 알 수 있도록.
    """
    dataset_meta = dataset_meta or {}
    scenario = dataset_meta.get("scenario", "")
    object_class = dataset_meta.get("object_class", "")
    pose_class = dataset_meta.get("pose_class", "")
    lec = dataset_meta.get("load_expected_counts") or {}
    total = summary.get("total_frames", 0)
    nonempty = summary.get("one_object_frames", 0) + summary.get("multi_object_frames", 0)
    empty = summary.get("empty_frames", 0)
    obj_total = summary.get("total_objects", 0)
    valid = summary.get("is_strict_valid", False)
    text = f"""# `{primary_path.name}` 사용 가이드

`{scenario}` 데이터셋의 radar point cloud 자동 라벨 결과 (strict JSONL).
이 파일을 그대로 학습 입력으로 쓰면 됩니다. 후처리 스크립트 필요 없음.

## 1. 파일 포맷

JSON Lines (`.jsonl`). 한 줄 = 한 frame. UTF-8 인코딩.

### 빈 frame (검출 객체 없음)
```json
{{"frame_id": "1778545853.523353", "objects": []}}
```

### 객체가 있는 frame
```json
{{"frame_id": "1778545854.727346", "objects": [
  {{"class": "person", "pose": "standing", "box": {{
      "center": [0.407637, 0.901024, -0.860000],
      "dimensions": [1.0, 1.0, 1.94],
      "yaw": 0.0
  }}}}
]}}
```

## 2. Schema 정의

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `frame_id` | string | frame 고유 식별자. 이 데이터셋에서는 **유닉스 timestamp (초)** 형식. dataset 안에서 유일 (중복 보장 없음 → 검증 통과). |
| `objects` | list | 검출된 객체 배열. 비어 있으면 `[]`. |
| `objects[].class` | string | 객체 클래스. **허용값: `person` / `animal` / `non_human` 만**. |
| `objects[].pose` | string | 자세 라벨. **허용값**: `standing`, `walking`, `horizontal_movement`, `low_movement`, `transition`, `unknown`, `sitting`, `lying_down`, `falling`. |
| `objects[].box.center` | `[x, y, z]` | 박스 중심 좌표 (meters). 순서 고정. |
| `objects[].box.dimensions` | `[dx, dy, dz]` | 박스 크기 (meters). 순서 **`[dx, dy, dz]` 고정**. |
| `objects[].box.yaw` | float | 박스 z축 회전 (radians). 회전 없으면 `0.0`. |

**디버그 필드는 절대 들어가지 않음** (`n_points`, `track_id`, `confidence`, `expected_counts`, `quality`, `source` 등). strict validator 가 export 직후 자동으로 확인함.

## 3. 좌표계 / 단위

- 단위: **meters**, 각도: **radians**
- z 축 부호: **원본 radar 좌표계 그대로 유지** (workbench가 변환하지 않음). 이 데이터셋에서 z 가 작을수록(음수) 위쪽, z 가 클수록 아래쪽인 radar mount 좌표계를 따름.
- 박스 좌표는 raw frame JSON 의 좌표와 동일한 frame 에 있음.

## 4. 박스 의미

- `center` 는 박스 정중앙. 박스의 외곽:
  - x: `[cx - dx/2, cx + dx/2]`
  - y: `[cy - dy/2, cy + dy/2]`
  - z: `[cz - dz/2, cz + dz/2]`
- `yaw` 는 z 축 기준 시계방향(혹은 사용 convention) 회전. 회전 박스가 필요하면 `box.yaw` 만으로 충분.

## 5. 빈 frame 처리

- `objects: []` 가 의미하는 것:
  1. radar 점이 0개 (sensor empty), 또는
  2. 점이 `min_points_for_box=30` 미만 (signal too sparse → 박스 신뢰도 낮음), 또는
  3. 검토 단계에서 reject 처리된 frame
- 빈 frame 도 학습 데이터에 포함해야 negative 샘플 학습 가능.

## 6. 자동 라벨 생성 파이프라인 (참고)

1. raw point cloud + TRK (tracker) 데이터 로드
2. TRK 가 잡힌 frame: TRK xy 중심으로 박스 생성 (radius 0.8m), z 는 trk_radius+1.5m 범위 점들의 percentile p1/p99 + 0.15m padding 으로 결정. 직립 pose 면 박스 바닥을 점군 발 위치에 anchor.
3. TRK 가 없는 frame: 점군 자체에서 클러스터 fallback box 추정.
4. high confidence frame을 anchor로 삼아 인접 frame에 temporal interpolation 적용. 단 현재 frame 의 실제 점군 위치로 z/xy re-anchor 해서 어긋남 방지.
5. load-time count gate: 사용자가 정한 person/animal/non_human 개수에 맞춰 박스 개수 제한.
6. export gate: 같은 개수 제한을 다시 한 번 적용 + strict schema 출력.
7. validation: schema/duplicate/허용값/extra-field 자동 검사.

## 7. 데이터셋 통계 (이 파일)

| 항목 | 값 |
| --- | --- |
| dataset | `{scenario}` |
| mode (object_class) | `{object_class}` |
| 기본 pose | `{pose_class}` |
| dataset default count | person={lec.get('person',0)}, animal={lec.get('animal',0)}, non_human={lec.get('non_human',0)} |
| 전체 frame | {total} |
| 객체 있는 frame | {nonempty} |
| empty frame | {empty} |
| 전체 객체 수 | {obj_total} |
| 중복 frame_id | {summary.get('duplicate_frame_id_count', 0)} |
| strict schema 검증 | {'✅ PASS' if valid else '❌ FAIL'} |

## 8. Python 로딩 예시

```python
import json

labels = {{}}
with open("{primary_path.name}") as f:
    for line in f:
        row = json.loads(line)
        labels[row["frame_id"]] = row["objects"]

# 특정 frame의 객체
frame_id = "1778545854.727346"
for obj in labels.get(frame_id, []):
    cls = obj["class"]
    pose = obj["pose"]
    cx, cy, cz = obj["box"]["center"]
    dx, dy, dz = obj["box"]["dimensions"]
    yaw = obj["box"]["yaw"]
    print(f"{{cls}}/{{pose}} center=({{cx:.2f}}, {{cy:.2f}}, {{cz:.2f}}) size=({{dx:.2f}}, {{dy:.2f}}, {{dz:.2f}}) yaw={{yaw:.3f}}")

# 통계
n_empty = sum(1 for objs in labels.values() if not objs)
n_obj = sum(len(objs) for objs in labels.values())
print(f"frames={{len(labels)}} empty={{n_empty}} objects={{n_obj}}")
```

## 9. 보장사항

이 파일은 다음 검증을 자동으로 통과한 상태로 생성됨:

- 모든 frame_id 가 유일 (중복 없음)
- 모든 object 의 `class` 가 허용 enum 내
- 모든 object 의 `pose` 가 허용 enum 내
- 모든 object 가 `class / pose / box` 3개 키만 가짐 (extra field 없음)
- 모든 box 가 `center / dimensions / yaw` 3개 키만 가짐
- `center` 와 `dimensions` 모두 길이 3 의 숫자 배열
- `yaw` 는 항상 숫자
- 빈 frame 은 `objects: []` 형식

## 10. 파일 출처

- 생성 도구: **Radar Temporal Label Workbench v19** (strict export 모드)
- 생성 시각: `{now_stamp()}`
- export 모드: `{'minimal (이 README + JSONL 만)' if minimal else 'full (디버그/검수 파일 포함)'}`
- 같은 폴더의 다른 파일은 보조용. 학습에는 이 JSONL 하나만 있으면 됨.
"""
    readme_path = d / "README.md"
    readme_path.write_text(text, encoding="utf-8")
    return readme_path


def write_validation_outputs(d: Path, summary: Dict[str, Any], source_order_summary: Optional[Dict[str, Any]] = None, dataset_meta: Optional[Dict[str, Any]] = None) -> Tuple[Path, Path]:
    """Persist strict_validation_issues.csv + validation_report.md."""
    dataset_meta = dataset_meta or {}
    issues_path = d / "strict_validation_issues.csv"
    with issues_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line_no", "frame_id", "issue", "detail"])
        w.writeheader()
        for r in summary.get("issues", []):
            w.writerow({k: r.get(k, "") for k in ["line_no", "frame_id", "issue", "detail"]})
    md_path = d / "validation_report.md"
    lines: List[str] = []
    lines.append(f"# Strict schema validation report")
    lines.append("")
    if dataset_meta:
        lines.append(f"- dataset: `{dataset_meta.get('scenario','')}`")
        lines.append(f"- object_class (mode): `{dataset_meta.get('object_class','')}`")
        lines.append(f"- pose_class (default): `{dataset_meta.get('pose_class','')}`")
        lec = dataset_meta.get("load_expected_counts") or {}
        lines.append(f"- dataset default count (load-time): person={lec.get('person',0)}, animal={lec.get('animal',0)}, non_human={lec.get('non_human',0)}")
        lines.append(f"- generated_at: `{now_stamp()}`")
        lines.append("")
    lines.append(f"## main file: `{summary.get('path','')}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    metric_rows = [
        ("total_frames", summary.get("total_frames", 0)),
        ("unique_frame_ids", summary.get("unique_frame_ids", 0)),
        ("duplicate_frame_id_count", summary.get("duplicate_frame_id_count", 0)),
        ("total_objects", summary.get("total_objects", 0)),
        ("empty_frames (objects: [])", summary.get("empty_frames", 0)),
        ("one_object_frames", summary.get("one_object_frames", 0)),
        ("multi_object_frames (>=2)", summary.get("multi_object_frames", 0)),
        ("point_count_positive_but_no_object", summary.get("point_count_positive_but_no_object", 0)),
        ("point_count_zero_but_has_object", summary.get("point_count_zero_but_has_object", 0)),
        ("invalid_class_count", summary.get("invalid_class_count", 0)),
        ("invalid_pose_count", summary.get("invalid_pose_count", 0)),
        ("missing_box_count", summary.get("missing_box_count", 0)),
        ("center_bad_count", summary.get("center_bad_count", 0)),
        ("dimensions_bad_count", summary.get("dimensions_bad_count", 0)),
        ("missing_yaw_count", summary.get("missing_yaw_count", 0)),
        ("extra_top_field_count", summary.get("extra_top_field_count", 0)),
        ("extra_object_field_count", summary.get("extra_object_field_count", 0)),
        ("extra_box_field_count", summary.get("extra_box_field_count", 0)),
        ("issue_count", summary.get("issue_count", 0)),
        ("is_strict_valid", str(summary.get("is_strict_valid", False))),
    ]
    for name, val in metric_rows:
        lines.append(f"| {name} | {val} |")
    if summary.get("duplicate_frame_id_examples"):
        lines.append("")
        lines.append("### duplicate frame_id examples")
        for fid in summary["duplicate_frame_id_examples"]:
            lines.append(f"- `{fid}`")
    if summary.get("invalid_class_examples"):
        lines.append("")
        lines.append("### invalid class values")
        for v in summary["invalid_class_examples"]:
            lines.append(f"- `{v}`")
    if summary.get("invalid_pose_examples"):
        lines.append("")
        lines.append("### invalid pose values")
        for v in summary["invalid_pose_examples"]:
            lines.append(f"- `{v}`")
    if source_order_summary:
        lines.append("")
        lines.append(f"## source-order companion file: `{source_order_summary.get('path','')}`")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        lines.append(f"| total_frames | {source_order_summary.get('total_frames',0)} |")
        lines.append(f"| unique_frame_ids | {source_order_summary.get('unique_frame_ids',0)} |")
        lines.append(f"| duplicate_frame_id_count | {source_order_summary.get('duplicate_frame_id_count',0)} |")
        lines.append(f"| total_objects | {source_order_summary.get('total_objects',0)} |")
        lines.append(f"| empty_frames | {source_order_summary.get('empty_frames',0)} |")
        lines.append(f"| issue_count | {source_order_summary.get('issue_count',0)} |")
        lines.append(f"| is_strict_valid | {source_order_summary.get('is_strict_valid', False)} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, issues_path


def object_sort_score(obj: Dict[str,Any]) -> Tuple[float, float]:
    """Score auto candidates so load-time count limiting keeps the most plausible boxes first."""
    q = obj.get("quality") or {}
    selected = safe_float(q.get("selected_points"), 0.0)
    point_count = safe_float(q.get("point_count"), 0.0)
    src = str(obj.get("label_source", ""))
    source_bonus = 0.0
    if "trk" in src:
        source_bonus = 1000.0
    elif "fallback" in src:
        source_bonus = 250.0
    elif "cluster" in src:
        source_bonus = 100.0
    return (source_bonus + selected, point_count)


def classify_candidate_by_size(obj: Dict[str,Any]) -> str:
    """v19+: 사람 vs 동물(강아지)을 박스 높이로 자동 분별 (엄격한 임계).

    rule:
      - h >= 1.4 m → person (성인/청소년 직립)
      - h < 1.4 m → animal (강아지, 또는 부분 cluster — 사람이면 검토에서 manual fix)

    radar 가 강아지 몸 전체 + 사람 일부만 잡거나 partial occlusion 으로 사람 박스가
    1.4m 미만으로 나오는 경우가 있어서 검토용. 학습 후 1-2 frame manual fix 권장.
    """
    box = obj.get("box") or {}
    dims = box.get("dimensions") or {}
    if isinstance(dims, dict):
        h = safe_float(dims.get("h"), 1.6)
    elif isinstance(dims, list) and len(dims) >= 3:
        h = safe_float(dims[2], 1.6)
    else:
        h = 1.6
    return "person" if h >= 1.4 else "animal"


def apply_loadtime_count_limit(lab: Dict[str,Any], expected_counts: Optional[Dict[str,int]], default_class: str, default_pose: str, multi_mode: bool) -> Dict[str,Any]:
    """Limit generated/temporal objects at dataset-load time.

    v17 behavior: once the user supplies person/animal/non_human counts before Load,
    the label shown in the UI is already constrained to that count. Extra TRK/cluster
    boxes are debug candidates only and are not kept as final objects.
    """
    if not isinstance(expected_counts, dict):
        return lab
    counts = normalize_expected_counts(expected_counts)
    expected_total = sum(counts.values())
    out = json.loads(json.dumps(lab, ensure_ascii=False))
    objects = out.get("objects", []) or []
    before_count = len(objects)
    q = out.setdefault("quality", {})
    q["loadtime_expected_counts"] = counts
    q["loadtime_object_count_before_limit"] = before_count

    if expected_total == 0 or out.get("confidence") == "empty" or safe_float((out.get("quality") or {}).get("point_count"), 0.0) <= 0:
        # True sensor-empty frames stay negative even when the dataset default count is person=1.
        # Frame-level empty overrides are still saved as objects: [].
        if expected_total > 0 and (out.get("confidence") == "empty" or safe_float((out.get("quality") or {}).get("point_count"), 0.0) <= 0):
            counts = {k: 0 for k in MODEL_CLASSES}
            expected_total = 0
        out["objects"] = []
        out["expected_counts"] = counts
        out["loadtime_count_limited"] = True
        out["loadtime_count_missing"] = {k: 0 for k in MODEL_CLASSES}
        q["object_count"] = 0
        q["expected_object_count"] = 0
        out["recommendation"] = "exclude"
        if out.get("confidence") not in ("empty", "error"):
            out["method"] = "loadtime_count_empty"
        return out

    pool = sorted([json.loads(json.dumps(o, ensure_ascii=False)) for o in objects], key=object_sort_score, reverse=True)
    selected: List[Dict[str,Any]] = []
    missing: Dict[str,int] = {k: 0 for k in MODEL_CLASSES}

    if multi_mode:
        # v19 size-based class assignment: 사람=키큰 박스, 동물=짧은 박스. 후보들을 분류한 뒤
        # 클래스별로 sort_score 순으로 N개씩 채택. 사람/강아지 동시 촬영 시나리오에서 박스의
        # class 가 점군 크기에 따라 자동 배정됨.
        classified: Dict[str, List[Dict[str,Any]]] = {k: [] for k in MODEL_CLASSES}
        unassigned: List[Dict[str,Any]] = []
        for obj in pool:
            cls_pred = classify_candidate_by_size(obj)
            if cls_pred in MODEL_CLASSES:
                classified[cls_pred].append(obj)
            else:
                unassigned.append(obj)
        for cls in MODEL_CLASSES:
            want = counts.get(cls, 0)
            bucket = classified.get(cls, [])
            for i in range(want):
                if i < len(bucket):
                    obj = bucket[i]
                elif unassigned:
                    # 해당 클래스 후보가 부족하면 미분류 candidate 를 reclass 해서 채움
                    obj = unassigned.pop(0)
                else:
                    missing[cls] += 1
                    continue
                obj["object_class"] = cls
                obj["pose_class"] = normalize_pose_label(obj.get("pose") or obj.get("pose_class") or default_pose)
                obj["label_source"] = str(obj.get("label_source") or "auto") + "+size_classify+loadtime_count"
                selected.append(obj)
    else:
        cursor = 0
        for cls in MODEL_CLASSES:
            for _ in range(counts.get(cls, 0)):
                if cursor >= len(pool):
                    missing[cls] += 1
                    continue
                obj = pool[cursor]
                cursor += 1
                obj["object_class"] = cls
                obj["pose_class"] = normalize_pose_label(obj.get("pose") or obj.get("pose_class") or default_pose)
                obj["label_source"] = str(obj.get("label_source") or "auto") + "+loadtime_count"
                selected.append(obj)

    selected = sanitize_objects(selected, default_class, default_pose)
    out["objects"] = selected
    out["expected_counts"] = counts
    out["loadtime_count_limited"] = True
    out["loadtime_count_missing"] = missing
    q["object_count"] = len(selected)
    q["expected_object_count"] = expected_total
    q["loadtime_object_count_after_limit"] = len(selected)
    q["loadtime_trimmed_object_count"] = max(0, before_count - len(selected))
    q["loadtime_missing_object_count"] = sum(missing.values())
    if sum(missing.values()) > 0:
        out["recommendation"] = "careful_check"
        if out.get("confidence") == "high":
            out["confidence"] = "medium"
        out["method"] = "loadtime_count_missing"
    elif before_count > len(selected):
        out["method"] = str(out.get("method") or "auto") + "+loadtime_count_limit"
    return out


UPRIGHT_POSES = {"standing", "walking", "horizontal_movement", "transition", "unknown"}  # v19+: 514 horizontal 데이터셋 포함


def make_box_for_track(points: List[Tuple[float,float,float,float,float,int]], tr: Dict[str,Any], params: RuleParams, pose_class: str = "unknown") -> Tuple[Dict[str,Any], Dict[str,Any]]:
    """Build a candidate box around a TRK report.

    v19 z-fix:
    Same-tid radar points often only cover one body part (e.g. feet or torso), which
    pulled the v18 box center far below the actual body. The v19 box uses same_tid
    points for xy positioning but **all near points within trk_radius** for the z
    range, and anchors the box bottom at the lowest visible z when the pose is
    upright (standing/walking). This makes the rendered box span feet→head instead
    of dipping into the floor.
    """
    tx, ty, tid = tr["x"], tr["y"], tr["track_id"]
    r2 = params.trk_radius ** 2
    z_r2 = (getattr(params, "z_pool_radius", params.trk_radius) or params.trk_radius) ** 2
    same_tid_near = [p for p in points if p[5] == tid and p[5] != params.invalid_tid and (p[0]-tx)**2 + (p[1]-ty)**2 <= r2]
    valid_near = [p for p in points if p[5] != params.invalid_tid and (p[0]-tx)**2 + (p[1]-ty)**2 <= r2]
    all_near = [p for p in points if (p[0]-tx)**2 + (p[1]-ty)**2 <= r2]
    # v19: z 범위는 trk_radius보다 넓은 z_pool_radius(기본 1.5m)로 수집해서 raised-arm 같은
    # 옆으로 뻗은 점도 포함. xy 정렬에는 영향 없음.
    z_pool_wide = [p for p in points if (p[0]-tx)**2 + (p[1]-ty)**2 <= z_r2]
    selected = same_tid_near if len(same_tid_near) >= params.min_tid_points_medium else valid_near
    source = "same_tid_near_trk" if selected is same_tid_near else "valid_tid_near_trk"
    upright = (pose_class or "").lower() in UPRIGHT_POSES
    z_pool = z_pool_wide if z_pool_wide else (all_near if all_near else (valid_near or selected))
    if selected:
        xs = [p[0] for p in selected]; ys = [p[1] for p in selected]
        xlo, xhi = percentile(xs, 10), percentile(xs, 90)
        ylo, yhi = percentile(ys, 10), percentile(ys, 90)
        raw_w, raw_l = abs(xhi-xlo), abs(yhi-ylo)
        # v19: xy 도 padding 추가해서 body 외곽 점이 박스 모서리 밖으로 나가지 않게.
        pad_xy = params.box_padding_xy if upright else 0.0
        w = clamp(max(raw_w + 2 * pad_xy, params.box_w), params.min_w, params.max_w)
        l = clamp(max(raw_l + 2 * pad_xy, params.box_l), params.min_l, params.max_l)
        zs_pool = [p[2] for p in z_pool] if z_pool else [p[2] for p in selected]
        # v19+: 직립 pose 는 점군 min/max 까지 박스가 wrap 하도록 변경. max_h=3.0 으로 상한이
        # 있어 절대 무한 확장은 안 되고 (한 점 sensor glitch 가 +5m 같은 곳에 있어도 cap),
        # 정상 frame 은 모든 visible 점이 박스 안에 들어옴. 그 외 pose 는 robust p5/p95 유지.
        if upright:
            zlo, zhi = min(zs_pool), max(zs_pool)
        else:
            zlo, zhi = percentile(zs_pool, 5), percentile(zs_pool, 95)
        raw_h = abs(zhi - zlo)
        pad_z = params.box_padding_z if upright else 0.0
        h = clamp(max(raw_h + 2 * pad_z, params.box_h), params.min_h, params.max_h)
        if upright:
            cz = (zlo - pad_z) + h / 2
        else:
            cz = (zlo + zhi) / 2 if raw_h > 0 else median(zs_pool, 0.0)
    else:
        w, l, h, cz = params.box_w, params.box_l, params.box_h, 0.0
    box = {"position": {"x": tx, "y": ty, "z": cz}, "dimensions": {"w": w, "l": l, "h": h}, "rotation": {"yaw": 0.0}}
    q = {
        "track_id": tid,
        "selected_points": len(selected),
        "same_tid_near_points": len(same_tid_near),
        "valid_near_points": len(valid_near),
        "all_near_points": len(all_near),
        "selection_source": source,
        "z_anchor": "feet" if upright else "center",
        "trk_status": tr.get("status", 4),
        "trk_status_name": TRK_STATUS_NAME.get(tr.get("status", 4), "unknown"),
    }
    return box, q


def point_inside_box_xy(p: Tuple[float,float,float,float,float,int], box: Dict[str,Any], margin: float = 0.0) -> bool:
    try:
        pos, dim = box["position"], box["dimensions"]
        cx, cy = safe_float(pos.get("x")), safe_float(pos.get("y"))
        w, l = safe_float(dim.get("w")), safe_float(dim.get("l"))
        yaw = safe_float((box.get("rotation") or {}).get("yaw"), 0.0)
        dx, dy = p[0]-cx, p[1]-cy
        ca, sa = math.cos(-yaw), math.sin(-yaw)
        lx = dx*ca - dy*sa
        ly = dx*sa + dy*ca
        return abs(lx) <= w/2 + margin and abs(ly) <= l/2 + margin
    except Exception:
        return False

def make_box_for_cluster(cluster: List[Tuple[float,float,float,float,float,int]], params: RuleParams) -> Tuple[Dict[str,Any], Dict[str,Any]]:
    xs = [p[0] for p in cluster]; ys = [p[1] for p in cluster]; zs = [p[2] for p in cluster]
    xlo, xhi = percentile(xs, 5), percentile(xs, 95)
    ylo, yhi = percentile(ys, 5), percentile(ys, 95)
    zlo, zhi = percentile(zs, 5), percentile(zs, 95)
    raw_w, raw_l, raw_h = abs(xhi-xlo), abs(yhi-ylo), abs(zhi-zlo)
    # cluster 후보는 사람/동물 공용이므로 사람 키 template을 강제하지 않음.
    # 실제 높이/폭은 point extent + 여유 margin 기준으로 둔다.
    w = clamp(raw_w + params.cluster_box_margin_xy * 2, 0.25, 1.8)
    l = clamp(raw_l + params.cluster_box_margin_xy * 2, 0.25, 1.8)
    h = clamp(raw_h + params.cluster_box_margin_z * 2, 0.25, 1.9)
    cx, cy, cz = median(xs), median(ys), (zlo + zhi) / 2 if raw_h > 1e-6 else median(zs, 0.0)
    # xy 분산의 주축을 yaw로 사용. 너무 불안정하면 0에 가까움.
    yaw = 0.0
    if len(cluster) >= 3:
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        sxx = sum((x-mx)*(x-mx) for x in xs) / len(xs)
        syy = sum((y-my)*(y-my) for y in ys) / len(ys)
        sxy = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / len(xs)
        if abs(sxy) + abs(sxx-syy) > 1e-9:
            yaw = 0.5 * math.atan2(2*sxy, sxx-syy)
    box = {"position": {"x": cx, "y": cy, "z": cz}, "dimensions": {"w": w, "l": l, "h": h}, "rotation": {"yaw": yaw}}
    q = {
        "track_id": params.invalid_tid,
        "selected_points": len(cluster),
        "selection_source": "point_cluster_candidate",
        "cluster_points": len(cluster),
        "trk_status_name": "none",
    }
    return box, q

def _connected_components_from_pairs(num_nodes: int, pairs):
    """Union-Find로 빠르게 connected components 계산."""
    parent = list(range(num_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for a, b in pairs:
        union(int(a), int(b))
    groups: Dict[int, List[int]] = {}
    for i in range(num_nodes):
        r = find(i)
        groups.setdefault(r, []).append(i)
    return list(groups.values())


try:
    import numpy as _np  # type: ignore
    from scipy.spatial import cKDTree as _cKDTree  # type: ignore
    from scipy.sparse import csr_matrix as _csr_matrix  # type: ignore
    from scipy.sparse.csgraph import connected_components as _connected_components  # type: ignore
    _HAS_FAST_CLUSTER = True
except Exception:
    _HAS_FAST_CLUSTER = False


def _fast_point_cluster_candidates(base, params):
    """v19+: 코어스 그리드 다운샘플 → scipy KDTree + sparse-csgraph connected components.

    Dense point cloud (예: 515 dog+human frame 당 2000+ points) 에서는 query_ball_tree 가
    O(n^2) pairs 를 반환해서 Python loop 가 폭발. 먼저 점들을 5cm xy grid 로 다운샘플해서
    cell 당 1개 대표점만 남기면 n 이 100배 줄어듦. 그 다음 다운샘플 점으로 cluster 잡고
    어느 cluster 에 어느 cell 이 속하는지 → 원본 점들을 cell 매핑으로 재배정.
    """
    if len(base) < params.min_cluster_points_medium:
        return []
    cell = max(0.05, params.cluster_eps_xy / 3.0)  # 5cm 또는 eps/3 중 큰 값
    # 원본 점들을 cell key 로 그룹화
    cell_to_pts: Dict[Tuple[int, int], List[int]] = {}
    for i, p in enumerate(base):
        key = (int(p[0] // cell), int(p[1] // cell))
        cell_to_pts.setdefault(key, []).append(i)
    cell_keys = list(cell_to_pts.keys())
    if len(cell_keys) < 2:
        # 모든 점이 하나의 cell → 하나의 cluster
        if len(base) >= params.min_cluster_points_medium:
            box, q = make_box_for_cluster(base, params)
            return [(box, q)]
        return []
    # cell 중심으로 cluster (xy 좌표는 cell key * cell + cell/2)
    cell_xy = _np.array([(k[0] * cell + cell / 2, k[1] * cell + cell / 2) for k in cell_keys], dtype=_np.float64)
    tree = _cKDTree(cell_xy)
    nbr_lists = tree.query_ball_tree(tree, r=params.cluster_eps_xy)
    rows: List[int] = []; cols: List[int] = []
    for i, nbrs in enumerate(nbr_lists):
        for j in nbrs:
            if i < j:
                rows.append(i); cols.append(j)
    if rows:
        data = _np.ones(len(rows), dtype=bool)
        adj = _csr_matrix((data, (rows, cols)), shape=(len(cell_keys), len(cell_keys)))
        _, labels = _connected_components(adj + adj.T, directed=False)
    else:
        labels = list(range(len(cell_keys)))
    # 각 cluster 에 원본 점들 재배정
    cluster_to_pts: Dict[int, List[int]] = {}
    for ci, lab in enumerate(labels):
        cluster_to_pts.setdefault(int(lab), []).extend(cell_to_pts[cell_keys[ci]])
    components = [v for v in cluster_to_pts.values() if len(v) >= params.min_cluster_points_medium]
    components.sort(key=len, reverse=True)
    out = []
    for comp_idx in components[:params.max_extra_cluster_candidates]:
        comp_pts = [base[i] for i in comp_idx]
        box, q = make_box_for_cluster(comp_pts, params)
        out.append((box, q))
    return out


def _slow_point_cluster_candidates(base, params):
    """기존 pure-Python grid + BFS 구현. scipy 가 없을 때 fallback."""
    eps = params.cluster_eps_xy
    eps2 = eps * eps
    grid: Dict[Tuple[int,int], List[int]] = {}
    for i, p in enumerate(base):
        key = (math.floor(p[0]/eps), math.floor(p[1]/eps))
        grid.setdefault(key, []).append(i)
    visited = [False] * len(base)
    clusters: List[List[Tuple[float,float,float,float,float,int]]] = []
    def neigh(i: int) -> List[int]:
        p = base[i]
        kx, ky = math.floor(p[0]/eps), math.floor(p[1]/eps)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in grid.get((kx+dx, ky+dy), []):
                    if j == i:
                        continue
                    q = base[j]
                    if (p[0]-q[0])**2 + (p[1]-q[1])**2 <= eps2:
                        out.append(j)
        return out
    for i in range(len(base)):
        if visited[i]:
            continue
        stack = [i]; visited[i] = True; comp = []
        while stack:
            a = stack.pop(); comp.append(base[a])
            for b in neigh(a):
                if not visited[b]:
                    visited[b] = True; stack.append(b)
        if len(comp) >= params.min_cluster_points_medium:
            clusters.append(comp)
    clusters.sort(key=len, reverse=True)
    out = []
    for comp in clusters[:params.max_extra_cluster_candidates]:
        box, q = make_box_for_cluster(comp, params)
        out.append((box, q))
    return out


def point_cluster_candidates(points: List[Tuple[float,float,float,float,float,int]], existing_objects: List[Dict[str,Any]], params: RuleParams) -> List[Tuple[Dict[str,Any], Dict[str,Any]]]:
    if not params.enable_cluster_candidates:
        return []
    # 이미 TRK box 안에 들어간 point는 제거하고, 나머지 point에서 추가 후보를 찾음.
    # 즉 TRK가 사람 하나만 잡고 개/다른 객체를 놓친 경우를 보완하기 위한 단계.
    base = []
    for p in points:
        if any(point_inside_box_xy(p, obj.get("box") or {}, margin=0.12) for obj in existing_objects):
            continue
        base.append(p)
    if len(base) < params.min_cluster_points_medium:
        return []
    if _HAS_FAST_CLUSTER:
        return _fast_point_cluster_candidates(base, params)
    return _slow_point_cluster_candidates(base, params)

def make_point_extent_fallback_object(pts: List[Tuple[float,float,float,float,float,int]], object_class: str, pose_class: str, params: RuleParams) -> Optional[Dict[str,Any]]:
    """Create one conservative candidate box when TRK/cluster failed.

    This is not a trusted label. It exists so sparse medium/low frames can be
    reviewed in the workbench instead of showing objects: []. For human standing
    data, dimensions are kept close to the scenario template because a person's
    physical size should not change frame-to-frame. Center is estimated from
    robust percentiles of the visible points.
    """
    if not params.fallback_point_box or len(pts) < params.fallback_min_points:
        return None
    xs = sorted([p[0] for p in pts]); ys = sorted([p[1] for p in pts]); zs = sorted([p[2] for p in pts])
    def pct(arr, q):
        if not arr: return 0.0
        k = max(0, min(len(arr)-1, int(round((len(arr)-1)*q))))
        return float(arr[k])
    # Robust visible extent. Outliers at 5/95% are ignored.
    x0,x1 = pct(xs,0.05), pct(xs,0.95)
    y0,y1 = pct(ys,0.05), pct(ys,0.95)
    z0,z1 = pct(zs,0.05), pct(zs,0.95)
    # v19: 박스 xy 중심은 p5/p95 midpoint 가 아니라 median (= 50th percentile) 사용.
    # 외곽 stray point가 있어도 박스가 cluster 중심에 그대로 박힘.
    cx = pct(xs, 0.5)
    cy = pct(ys, 0.5)
    # Keep dimensions stable for human labels; do not let sparse points shrink height.
    w = max(params.box_w, min(1.20, (x1-x0) + params.fallback_padding_xy))
    l = max(params.box_l, min(1.20, (y1-y0) + params.fallback_padding_xy))
    # v19 z-fix: 직립 pose 면 p2/p98 percentile + padding 으로 outlier 노이즈는 배제하면서
    # 박스 안에 body 점이 충분히 들어가도록 한다.
    upright = (pose_class or "").lower() in UPRIGHT_POSES
    if upright:
        # v19+: min/max 로 박스가 점군 전체를 wrap 하도록
        z_lo = float(zs[0]); z_hi = float(zs[-1])
        raw_h = max(0.0, z_hi - z_lo)
        pad_z = params.box_padding_z
        h = clamp(max(raw_h + 2 * pad_z, params.box_h), params.min_h, params.max_h)
        cz = (z_lo - pad_z) + h / 2.0
        # xy 도 padding 추가
        pad_xy = params.box_padding_xy
        w = max(params.box_w, min(1.20, (x1-x0) + 2 * pad_xy))
        l = max(params.box_l, min(1.20, (y1-y0) + 2 * pad_xy))
    else:
        h = max(params.box_h * 0.85, min(params.box_h * 1.40, (z1-z0) + params.fallback_padding_z))
        cz = (z0+z1)/2.0
    box = make_box(cx, cy, cz, w, l, h, 0.0)
    return {
        "object_id": "obj_001",
        "object_class": default_object_class(object_class),
        "pose_class": pose_class,
        "box": box,
        "confidence": "pending",
        "label_source": "point_extent_fallback",
        "quality": {
            "selected_points": len(pts),
            "point_count": len(pts),
            "method": "point_extent_fallback",
            "note": "TRK/cluster failed; candidate box estimated from visible point extent"
        }
    }

def frame_point_z_stats(pts: List[Tuple[float,float,float,float,float,int]]) -> Dict[str, float]:
    """v19: temporal correction이 z 위치를 현재 frame 점군 기준으로 재조정할 수 있도록
    raw_label_frame 단계에서 점군 z/xy 통계를 한 번만 계산해서 lab.quality 에 박아둔다."""
    if not pts:
        return {}
    zs = sorted(p[2] for p in pts)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return {
        "point_z_min": float(zs[0]),
        "point_z_max": float(zs[-1]),
        "point_z_p1": float(percentile(zs, 1)),
        "point_z_p2": float(percentile(zs, 2)),
        "point_z_p5": float(percentile(zs, 5)),
        "point_z_p95": float(percentile(zs, 95)),
        "point_z_p98": float(percentile(zs, 98)),
        "point_z_p99": float(percentile(zs, 99)),
        "point_xy_cx": float(sum(xs) / len(xs)),
        "point_xy_cy": float(sum(ys) / len(ys)),
    }


def raw_label_frame(meta: Dict[str,Any], frame: Dict[str,Any], object_class: str, pose_class: str, params: RuleParams) -> Dict[str,Any]:
    pts = parse_points(frame)
    tracks = parse_tracks(frame)
    pt_stats = frame_point_z_stats(pts)
    if len(pts) == 0:
        return {**meta, "objects": [], "confidence": "empty", "recommendation": "exclude", "method": "empty_or_error", "quality": {"point_count": 0, "track_count": len(tracks)}}
    # v19: 점이 너무 적은 frame은 box 신뢰도가 낮으므로 empty 로 처리.
    # 학습 라벨에 노이즈 박스를 섞지 않으려면 이 단계에서 잘라내는 게 안전함.
    if len(pts) < int(getattr(params, "min_points_for_box", 0) or 0):
        return {**meta, "objects": [], "confidence": "empty", "recommendation": "exclude", "method": "too_few_points", "quality": {"point_count": len(pts), "track_count": len(tracks), "reason": f"point_count<{params.min_points_for_box}", **pt_stats}}
    objects = []
    selected_counts = []
    for i, tr in enumerate(tracks):
        box, q = make_box_for_track(pts, tr, params, pose_class=pose_class)
        selected_counts.append(q["selected_points"])
        objects.append({"object_id": f"obj_{i+1:03d}", "object_class": default_object_class(object_class), "pose_class": pose_class, "box": box, "confidence": "pending", "label_source": "trk_rule_based", "quality": q})

    # TRK가 1개만 생기는 센서/tracker 상황 보완: 남은 point cluster를 추가 후보 box로 생성
    cluster_pairs = point_cluster_candidates(pts, objects, params)
    for box, q in cluster_pairs:
        selected_counts.append(q["selected_points"])
        objects.append({"object_id": f"obj_{len(objects)+1:03d}", "object_class": default_object_class(object_class), "pose_class": pose_class, "box": box, "confidence": "pending", "label_source": "cluster_rule_based", "quality": q})

    if not objects:
        # v13: TRK/cluster가 실패해도 point가 충분히 있으면 검수 가능한 후보 box를 만든다.
        # 이 후보는 자동 사용 대상이 아니라 review 대상이다.
        fb = make_point_extent_fallback_object(pts, object_class, pose_class, params)
        if fb is not None:
            sel = len(pts)
            conf = "medium" if sel >= params.fallback_medium_points else "low"
            fb["confidence"] = conf
            return {**meta, "objects": [fb], "confidence": conf, "recommendation": "careful_check", "method": "point_extent_fallback", "quality": {"point_count": len(pts), "track_count": len(tracks), "reason": "no_trk_or_cluster_point_fallback", "object_count": 1, "max_selected_points": sel, **pt_stats}}
        return {**meta, "objects": [], "confidence": "low", "recommendation": "exclude", "method": "no_candidate", "quality": {"point_count": len(pts), "track_count": len(tracks), "reason": "no_trk_or_cluster_too_sparse", **pt_stats}}

    max_sel = max(selected_counts) if selected_counts else 0
    if max_sel >= params.min_tid_points_high:
        conf, rec = "high", "auto_use"
    elif max_sel >= params.min_tid_points_medium:
        conf, rec = "medium", "quick_check"
    else:
        conf, rec = "low", "careful_check"
    for o in objects:
        o["confidence"] = conf
    return {**meta, "objects": objects, "confidence": conf, "recommendation": rec, "method": "auto_high" if conf == "high" else "raw_only", "quality": {"point_count": len(pts), "track_count": len(tracks), "cluster_candidate_count": len(cluster_pairs), "object_count": len(objects), "max_selected_points": max_sel, **pt_stats}}

def box_values(lab: Dict[str,Any]) -> Optional[Tuple[float,float,float,float,float,float,float]]:
    objs = lab.get("objects") or []
    if not objs:
        return None
    b = objs[0]["box"]
    return (safe_float(b["position"]["x"]), safe_float(b["position"]["y"]), safe_float(b["position"]["z"]), safe_float(b["dimensions"]["w"]), safe_float(b["dimensions"]["l"]), safe_float(b["dimensions"]["h"]), safe_float(b["rotation"].get("yaw",0)))

def make_box(cx: float, cy: float, cz: float, w: float, l: float, h: float, yaw: float) -> Dict[str,Any]:
    return {"position": {"x": cx, "y": cy, "z": cz}, "dimensions": {"w": w, "l": l, "h": h}, "rotation": {"yaw": yaw}}

def apply_temporal_correction(labels: List[Dict[str,Any]], params: RuleParams) -> List[Dict[str,Any]]:
    """Temporal correction that keeps multiple objects/tracks in a frame.

    v6 corrected only the first object. v7 treats each TRK track as a candidate object,
    so person+animal frames can keep more than one box. Class is still only a candidate;
    the user confirms human / animal / non_human in the UI.
    """
    out = [json.loads(json.dumps(x, ensure_ascii=False)) for x in labels]
    by_record: Dict[str, List[int]] = {}
    for i, lab in enumerate(out):
        by_record.setdefault(lab["record"], []).append(i)

    for rec, idxs in by_record.items():
        idxs.sort(key=lambda i: out[i].get("timestamp", 0))

        # Anchors: high-confidence object boxes, separated by track_id when available.
        anchors_by_tid: Dict[int, List[Tuple[int, Tuple[float,float,float,float,float,float,float], Dict[str,Any]]]] = {}
        global_dims: List[Tuple[float,float,float]] = []
        global_z: List[float] = []
        for i in idxs:
            lab = out[i]
            if lab.get("confidence") != "high":
                continue
            for obj in lab.get("objects", []) or []:
                vals = box_values_from_obj(obj)
                if not vals:
                    continue
                global_dims.append(vals[3:6])
                global_z.append(vals[2])
                tid = obj_track_id(obj)
                if tid is not None and tid != params.invalid_tid:
                    anchors_by_tid.setdefault(tid, []).append((i, vals, obj))

        def med_dims_for_tid(tid: Optional[int]) -> Tuple[float,float,float,float]:
            source = []
            zsource = []
            if tid is not None and tid in anchors_by_tid:
                for _, vals, _ in anchors_by_tid[tid]:
                    source.append(vals[3:6])
                    zsource.append(vals[2])
            if not source:
                source = global_dims
                zsource = global_z
            med_w = median([d[0] for d in source], params.box_w)
            med_l = median([d[1] for d in source], params.box_l)
            med_h = median([d[2] for d in source], params.box_h)
            med_z = median(zsource, 0.0)
            return med_w, med_l, med_h, med_z

        # v12: nearest_anchor를 매번 list comprehension으로 찾으면 큰 데이터셋에서 매우 느림.
        # track별 anchor timestamp를 미리 만들어 이진 탐색으로 찾는다.
        anchor_ts_by_tid: Dict[int, List[float]] = {}
        for tid, arr in anchors_by_tid.items():
            arr.sort(key=lambda a: out[a[0]].get("timestamp", 0))
            anchor_ts_by_tid[tid] = [out[a[0]].get("timestamp", 0) for a in arr]

        def nearest_anchor(tid: int, pos: int, direction: int):
            arr = anchors_by_tid.get(tid, [])
            ts_arr = anchor_ts_by_tid.get(tid, [])
            if not arr or not ts_arr:
                return None
            cur_i = idxs[pos]
            tcur = out[cur_i].get("timestamp", 0)
            k = bisect.bisect_left(ts_arr, tcur)
            if direction < 0:
                return arr[k-1] if k-1 >= 0 else None
            return arr[k] if k < len(arr) and ts_arr[k] > tcur else (arr[k+1] if k+1 < len(arr) else None)

        all_anchor_tids = sorted(anchors_by_tid.keys())

        for pos, i in enumerate(idxs):
            lab = out[i]
            current_objs = lab.get("objects", []) or []

            # v19: 점이 너무 적은 frame은 temporal 보간으로도 박스를 만들지 않는다.
            # 그렇지 않으면 temporal carry forward 가 sparse frame에 빈 위치의 박스를 박는다.
            cur_pc = (lab.get("quality") or {}).get("point_count", 0)
            try:
                cur_pc = int(cur_pc)
            except Exception:
                cur_pc = 0
            if cur_pc < int(getattr(params, "min_points_for_box", 0) or 0):
                lab["objects"] = []
                lab["confidence"] = "empty"
                lab["recommendation"] = "exclude"
                lab["method"] = "too_few_points"
                continue

            if lab.get("confidence") == "high":
                # Stabilize dimensions while preserving every object candidate.
                for obj in current_objs:
                    vals = box_values_from_obj(obj)
                    if not vals:
                        continue
                    tid = obj_track_id(obj)
                    # TRK 기반 사람 box만 median template으로 안정화.
                    # cluster 기반 후보는 개/낮은 객체일 수 있으므로 높이를 사람 template으로 강제하지 않음.
                    if tid is not None and tid != params.invalid_tid:
                        med_w, med_l, med_h, med_z = med_dims_for_tid(tid)
                        cx, cy, cz, w, l, h, yaw = vals
                        # v19: median stabilization 이 raised-arm 같은 큰 박스를 median으로 줄여서
                        # 점군 머리/팔이 박스 위로 새는 문제 발생. h 는 per-frame 값이 더 크면
                        # 그대로 유지(=커진 박스를 보존). w,l 은 median 안정화 유지.
                        # cz 는 박스 바닥을 유지하면서 새 h 에 맞춰 평행이동.
                        new_h = max(h, med_h)
                        new_cz = cz + (new_h - h) / 2.0
                        obj["box"] = make_box(cx, cy, new_cz, med_w, med_l, new_h, yaw)
                    obj["label_source"] = obj.get("label_source", "rule_based")
                lab["method"] = "auto_high"
                lab["recommendation"] = "auto_use"
                continue

            # For non-high frames, keep the current tracks if they exist; otherwise infer
            # candidate tracks from neighboring high anchors.
            current_by_tid: Dict[int, Dict[str,Any]] = {}
            no_tid_objs: List[Dict[str,Any]] = []
            for obj in current_objs:
                tid = obj_track_id(obj)
                if tid is None or tid == params.invalid_tid:
                    no_tid_objs.append(obj)
                else:
                    current_by_tid[tid] = obj

            candidate_tids = sorted(current_by_tid.keys())
            if not candidate_tids:
                # Only create temporal candidates for tracks that have at least one nearby anchor.
                for tid in all_anchor_tids:
                    if nearest_anchor(tid, pos, -1) or nearest_anchor(tid, pos, 1):
                        candidate_tids.append(tid)

            new_objects: List[Dict[str,Any]] = []
            methods: List[str] = []
            recs: List[str] = []
            for tid in candidate_tids:
                cur_obj = current_by_tid.get(tid)
                prev_a = nearest_anchor(tid, pos, -1)
                next_a = nearest_anchor(tid, pos, 1)
                med_w, med_l, med_h, med_z = med_dims_for_tid(tid)
                new_box = None
                method = "raw_only"
                recm = "careful_check"
                if prev_a is not None and next_a is not None:
                    p_i, pvals, pobj = prev_a
                    n_i, nvals, nobj = next_a
                    t0, t1, t = out[p_i]["timestamp"], out[n_i]["timestamp"], lab["timestamp"]
                    ratio = 0.5 if abs(t1-t0) < 1e-9 else clamp((t-t0)/(t1-t0), 0, 1)
                    cx = pvals[0] + (nvals[0]-pvals[0]) * ratio
                    cy = pvals[1] + (nvals[1]-pvals[1]) * ratio
                    yaw = math.atan2(nvals[1]-pvals[1], nvals[0]-pvals[0]) if abs(nvals[0]-pvals[0]) + abs(nvals[1]-pvals[1]) > 1e-6 else 0.0
                    new_box = make_box(cx, cy, med_z, med_w, med_l, med_h, yaw)
                    gap = out[n_i].get("order",0) - out[p_i].get("order",0)
                    method = "interpolate_between_prev_next_high"
                    recm = "quick_check" if gap <= params.max_interpolation_gap_frames else "careful_check"
                elif prev_a is not None:
                    p_i, pvals, pobj = prev_a
                    cx, cy, _, _, _, _, yaw = pvals
                    new_box = make_box(cx, cy, med_z, med_w, med_l, med_h, yaw)
                    dist = lab.get("order",0) - out[p_i].get("order",0)
                    method = "carry_forward_prev_high"
                    recm = "careful_check" if dist <= params.max_anchor_gap_frames else "exclude"
                elif next_a is not None:
                    n_i, nvals, nobj = next_a
                    cx, cy, _, _, _, _, yaw = nvals
                    new_box = make_box(cx, cy, med_z, med_w, med_l, med_h, yaw)
                    dist = out[n_i].get("order",0) - lab.get("order",0)
                    method = "carry_backward_next_high"
                    recm = "careful_check" if dist <= params.max_anchor_gap_frames else "exclude"
                elif cur_obj is not None:
                    vals = box_values_from_obj(cur_obj)
                    if vals:
                        cx, cy, cz, w, l, h, yaw = vals
                        new_box = make_box(cx, cy, cz, w, l, h, yaw)

                if new_box is None:
                    continue
                obj_class = (cur_obj or {}).get("object_class") or default_object_class(lab.get("object_class_hint", "person"))
                pose_class = (cur_obj or {}).get("pose_class") or lab.get("pose_class_hint", "unknown")
                # v19 z/xy-reanchor: temporal anchor의 박스를 그대로 가져오면 현재 frame
                # 의 점군과 어긋날 수 있다. 직립 pose 면:
                #   - z 는 p2/p98 + padding 으로 재계산해서 stray 노이즈는 배제 + 여유
                #   - xy 는 anchor가 점군 중심에서 너무 멀면 (>임계) 중심으로 snap
                if (pose_class or "").lower() in UPRIGHT_POSES:
                    cur_q = lab.get("quality") or {}
                    # v19+: min/max 로 점군 전체를 박스가 wrap
                    cur_zlo = cur_q.get("point_z_min")
                    cur_zhi = cur_q.get("point_z_max")
                    if isinstance(cur_zlo, (int, float)) and isinstance(cur_zhi, (int, float)) and cur_zhi > cur_zlo:
                        raw_extent = float(cur_zhi) - float(cur_zlo)
                        pad_z = params.box_padding_z
                        new_h = clamp(max(raw_extent + 2 * pad_z, params.box_h), params.min_h, params.max_h)
                        new_cz = (float(cur_zlo) - pad_z) + new_h / 2
                        new_box["dimensions"]["h"] = new_h
                        new_box["position"]["z"] = new_cz
                        method = (method or "raw_only") + "+cz_reanchor"
                    cur_xc = cur_q.get("point_xy_cx")
                    cur_yc = cur_q.get("point_xy_cy")
                    if isinstance(cur_xc, (int, float)) and isinstance(cur_yc, (int, float)):
                        anchor_xy_off = math.hypot(new_box["position"]["x"] - float(cur_xc), new_box["position"]["y"] - float(cur_yc))
                        if anchor_xy_off > params.temporal_xy_reanchor_threshold:
                            new_box["position"]["x"] = float(cur_xc)
                            new_box["position"]["y"] = float(cur_yc)
                            method = (method or "raw_only") + "+xy_reanchor"
                new_objects.append({
                    "object_id": f"obj_{len(new_objects)+1:03d}",
                    "object_class": obj_class,
                    "pose_class": pose_class,
                    "box": new_box,
                    "confidence": lab.get("confidence", "low"),
                    "label_source": "temporal_corrected" if method != "raw_only" else "rule_based",
                    "quality": {"raw_confidence": lab.get("confidence"), "method": method, "track_id": tid},
                })
                methods.append(method)
                recs.append(recm)

            # TRK가 없는 cluster 후보도 유지. 개/사람 동시 촬영에서는 한쪽 객체가 TRK에 안 잡히는 경우가 많음.
            if no_tid_objs:
                for obj in no_tid_objs:
                    obj2 = json.loads(json.dumps(obj, ensure_ascii=False))
                    obj2["object_id"] = f"obj_{len(new_objects)+1:03d}"
                    obj2["label_source"] = obj2.get("label_source") or "cluster_rule_based"
                    new_objects.append(obj2)
                methods.append("raw_only")
                recs.append("careful_check")

            if new_objects:
                lab["objects"] = new_objects
                # Overall method/recommendation for filter display.
                if "interpolate_between_prev_next_high" in methods:
                    lab["method"] = "interpolate_between_prev_next_high"
                elif methods:
                    lab["method"] = methods[0]
                else:
                    lab["method"] = "raw_only"
                if "exclude" in recs:
                    lab["recommendation"] = "exclude"
                elif "careful_check" in recs:
                    lab["recommendation"] = "careful_check"
                elif "quick_check" in recs:
                    lab["recommendation"] = "quick_check"
                else:
                    lab["recommendation"] = "auto_use"
            else:
                lab["objects"] = []
                lab["method"] = "empty_or_error" if lab.get("confidence") in ("empty","error") else "no_temporal_anchor"
                lab["recommendation"] = "exclude"
    return out

# -----------------------------
# SVG rendering
# -----------------------------

def project_box(box: Dict[str, Any], plane: str) -> Tuple[float,float,float,float]:
    pos, dim = box["position"], box["dimensions"]
    cx, cy, cz = safe_float(pos["x"]), safe_float(pos["y"]), safe_float(pos["z"])
    w, l, h = safe_float(dim["w"]), safe_float(dim["l"]), safe_float(dim["h"])
    if plane == "top": return cx-w/2, cy-l/2, w, l
    if plane == "side": return cy-l/2, cz-h/2, l, h
    return cx-w/2, cz-h/2, w, h

def iso_project(x: float, y: float, z: float) -> Tuple[float,float]:
    return (x - y*0.45, z*1.05 - y*0.25)

def box_edges_3d(box: Dict[str,Any]) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    pos, dim = box["position"], box["dimensions"]
    cx, cy, cz = safe_float(pos["x"]), safe_float(pos["y"]), safe_float(pos["z"])
    w, l, h = safe_float(dim["w"]), safe_float(dim["l"]), safe_float(dim["h"])
    yaw = safe_float(box.get("rotation",{}).get("yaw",0))
    ca, sa = math.cos(yaw), math.sin(yaw)
    corners=[]
    for dx in (-w/2,w/2):
        for dy in (-l/2,l/2):
            rx = dx*ca - dy*sa; ry = dx*sa + dy*ca
            for dz in (-h/2,h/2):
                corners.append((cx+rx, cy+ry, cz+dz))
    proj=[iso_project(*c) for c in corners]
    edges=[]
    for i in range(8):
        for j in range(i+1,8):
            if bin(i^j).count("1") == 1:
                edges.append((proj[i], proj[j]))
    return edges

def render_svg(points: List[Tuple[float,float,float,float,float,int]], label: Dict[str,Any], plane: str, width: int = 500, height: int = 340) -> str:
    """Render point cloud + boxes.

    v11 change:
    - viewport is fitted from point cloud first, not from box size.
      In earlier versions, one oversized candidate box could expand the SVG scale so much
      that sparse radar points became nearly invisible.
    - boxes are still drawn, but they do not dominate the zoom unless there are no points.
    - points are darker/larger and drawn after boxes so they stay visible.
    """
    if plane == "perspective":
        point_coords = [(iso_project(p[0], p[1], p[2])[0], iso_project(p[0], p[1], p[2])[1], p[4]) for p in points]
        title = "Perspective"
        lines = []
        for obj in label.get("objects", []):
            try:
                lines.extend(box_edges_3d(obj["box"]))
            except Exception:
                pass
        box_range_coords = [(a[0], a[1], 1.0) for a, b in lines] + [(b[0], b[1], 1.0) for a, b in lines]
        rects = []
    else:
        if plane == "top":
            point_coords = [(p[0], p[1], p[4]) for p in points]
            title = "Top x-y"
        elif plane == "side":
            point_coords = [(p[1], p[2], p[4]) for p in points]
            title = "Side y-z"
        else:
            point_coords = [(p[0], p[2], p[4]) for p in points]
            title = "Front x-z"
        lines = []
        rects = []
        box_range_coords = []
        for obj in label.get("objects", []):
            try:
                rx, ry, rw, rh = project_box(obj["box"], plane)
                rects.append((rx, ry, rw, rh))
                box_range_coords += [(rx, ry, 1.0), (rx + rw, ry + rh, 1.0)]
            except Exception:
                pass

    # Fit view mainly by points. If no points exist, fall back to boxes.
    fit_coords = point_coords if point_coords else box_range_coords
    if not fit_coords:
        xmin, xmax, ymin, ymax = -1, 1, -1, 1
    else:
        xs = [c[0] for c in fit_coords]
        ys = [c[1] for c in fit_coords]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

        # Minimum visible window prevents 1~2 sparse points from becoming a giant dot cloud.
        min_span_x = 1.2 if plane != "side" else 1.0
        min_span_y = 1.2 if plane == "top" else 1.0
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if xmax - xmin < min_span_x:
            xmin, xmax = cx - min_span_x / 2, cx + min_span_x / 2
        if ymax - ymin < min_span_y:
            ymin, ymax = cy - min_span_y / 2, cy + min_span_y / 2

        px = max(0.20, (xmax - xmin) * 0.20)
        py = max(0.20, (ymax - ymin) * 0.20)
        xmin -= px; xmax += px; ymin -= py; ymax += py

        # If the box is only slightly outside the point range, include it.
        # If it is far too large, ignore it for zoom so points remain readable.
        if point_coords and box_range_coords:
            cur_w = max(xmax - xmin, 1e-6)
            cur_h = max(ymax - ymin, 1e-6)
            bx = [c[0] for c in box_range_coords]
            by = [c[1] for c in box_range_coords]
            bxmin, bxmax, bymin, bymax = min(bx), max(bx), min(by), max(by)
            merged_w = max(xmax, bxmax) - min(xmin, bxmin)
            merged_h = max(ymax, bymax) - min(ymin, bymin)
            if merged_w <= cur_w * 2.2 and merged_h <= cur_h * 2.2:
                xmin, xmax = min(xmin, bxmin), max(xmax, bxmax)
                ymin, ymax = min(ymin, bymin), max(ymax, bymax)

        if abs(xmax - xmin) < 1e-6:
            xmax += 1; xmin -= 1
        if abs(ymax - ymin) < 1e-6:
            ymax += 1; ymin -= 1

    def sx(x):
        return 35 + (x - xmin) / (xmax - xmin) * (width - 60)
    def sy(y):
        return height - 30 - (y - ymin) / (ymax - ymin) * (height - 62)

    ps = [c[2] for c in point_coords]
    pmin, pmax = (min(ps), max(ps)) if ps else (0, 1)
    if abs(pmax - pmin) < 1e-9:
        pmax = pmin + 1

    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="10" y="20" font-size="14" font-family="Arial" fill="#333">{title}</text>',
    ]

    # Draw boxes first.
    for a, b in lines:
        parts.append(f'<line x1="{sx(a[0]):.1f}" y1="{sy(a[1]):.1f}" x2="{sx(b[0]):.1f}" y2="{sy(b[1]):.1f}" stroke="#e53935" stroke-width="2.2" stroke-opacity="0.95"/>')
    for rx, ry, rw, rh in rects:
        x1, y1 = sx(rx), sy(ry)
        x2, y2 = sx(rx + rw), sy(ry + rh)
        parts.append(f'<rect x="{min(x1,x2):.1f}" y="{min(y1,y2):.1f}" width="{abs(x2-x1):.1f}" height="{abs(y2-y1):.1f}" fill="none" stroke="#e53935" stroke-width="2.4" stroke-opacity="0.95"/>')

    # Then draw points on top so they remain visible.
    step = max(1, len(point_coords) // 2400)
    for x, y, pwr in point_coords[::step]:
        t = (pwr - pmin) / (pmax - pmin)
        sh = int(190 - 145 * t)
        # Stronger points become darker. Sparse points are shown larger.
        r = 2.2 if len(point_coords) < 60 else 1.8
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" fill="rgb({sh},{sh},{sh})" fill-opacity="0.78"/>')

    # Small notice for very large boxes.
    if point_coords and box_range_coords:
        try:
            bx = [c[0] for c in box_range_coords]; by = [c[1] for c in box_range_coords]
            p_x = [c[0] for c in point_coords]; p_y = [c[1] for c in point_coords]
            p_span = max(max(p_x)-min(p_x), max(p_y)-min(p_y), 1e-6)
            b_span = max(max(bx)-min(bx), max(by)-min(by), 1e-6)
            if b_span > p_span * 3.0:
                parts.append('<text x="10" y="38" font-size="11" font-family="Arial" fill="#b71c1c">box가 point 분포보다 커서 point 기준으로 확대 표시 중</text>')
        except Exception:
            pass

    parts.append('</svg>')
    return ''.join(parts)


def sanitize_objects(objects: List[Dict[str,Any]], default_class: str, default_pose: str) -> List[Dict[str,Any]]:
    out: List[Dict[str,Any]] = []
    for i, obj in enumerate(objects):
        b = obj.get("box") or {}
        pos = b.get("position") or {}
        dim = b.get("dimensions") or {}
        rot = b.get("rotation") or {}
        box = make_box(
            safe_float(pos.get("x")), safe_float(pos.get("y")), safe_float(pos.get("z"), 0.8),
            safe_float(dim.get("w"), 0.8), safe_float(dim.get("l"), 0.8), safe_float(dim.get("h"), 1.6),
            safe_float(rot.get("yaw"), 0.0),
        )
        q = obj.get("quality") if isinstance(obj.get("quality"), dict) else {}
        out.append({
            "object_id": obj.get("object_id") or f"obj_{i+1:03d}",
            "object_class": obj.get("object_class") or default_object_class(default_class),
            "pose_class": obj.get("pose_class") or default_pose,
            "box": box,
            "confidence": obj.get("confidence") or "manual",
            "label_source": obj.get("label_source") or "manual_adjusted",
            "quality": q,
        })
    for i, obj in enumerate(out):
        obj["object_id"] = f"obj_{i+1:03d}"
    return out

# -----------------------------
# Workbench state
# -----------------------------

class Workbench:
    def __init__(self, data_dir: Path, out_dir: Path, params: RuleParams):
        self.data_dir = data_dir
        self.out_dir = out_dir
        self.params = params
        self.dataset_zip: Optional[Path] = None
        self.reader: Optional[ZipDatasetReader] = None
        self.labels: List[Dict[str,Any]] = []
        self.frames: Dict[str, Dict[str,Any]] = {}
        self.corrections: Dict[str, Dict[str,Any]] = {}
        self.filter_view = "review"
        self.filter_conf = "all"
        self.filter_recommend = "all"
        self.pos = 0
        self.object_class = "person"
        self.pose_class = "unknown"
        self.load_expected_counts: Optional[Dict[str,int]] = {"person": 1, "animal": 0, "non_human": 0}
        self.scenario = ""
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[str]:
        return sorted([p.name for p in self.data_dir.glob("*.zip") if not p.name.startswith("radar_temporal_label_workbench")])

    def infer_pose(self, name: str) -> str:
        low=name.lower()
        if "walk" in low: return "walking"
        if "stand" in low: return "standing"
        if "horizontal" in low: return "horizontal_movement"
        if "low" in low: return "low_movement"
        return "unknown"

    def load_dataset(self, zip_name: str, object_class: str = "person", pose_class: Optional[str] = None, expected_counts: Optional[Dict[str,int]] = None) -> Dict[str,Any]:
        self.dataset_zip = (self.data_dir / zip_name).resolve()
        self.reader = ZipDatasetReader(self.dataset_zip)
        self.reader.open()
        self.object_class = object_class
        self.pose_class = pose_class or self.infer_pose(zip_name)
        if isinstance(expected_counts, dict):
            self.load_expected_counts = normalize_expected_counts(expected_counts)
        else:
            self.load_expected_counts = default_expected_counts_for_export({"objects": [{"object_class": default_object_class(object_class)}]}, object_class, is_multi_class_mode(object_class))
        # v18 safety guard: dataset-level 0/0/0 would delete all auto boxes.
        # Frame-level empty should be handled by the UI correction button, not by loading the whole dataset with zero expected objects.
        if sum(self.load_expected_counts.values()) == 0:
            if is_multi_class_mode(object_class):
                self.load_expected_counts = {"person": 1, "animal": 1, "non_human": 0}
            else:
                cls = default_object_class(object_class)
                if cls not in MODEL_CLASSES:
                    cls = "person"
                self.load_expected_counts = {k: 0 for k in MODEL_CLASSES}
                self.load_expected_counts[cls] = 1
        self.scenario = Path(zip_name).stem
        self.corrections = {}
        self.pos = 0
        self.skipped_json_count = 0
        self.invalid_json_count = 0

        raw=[]; self.frames={}
        order = 0
        self.duplicate_frame_count = 0
        self.dedup_source_policy = "prefer _frames/raw json over _preprocessed; keep one record per frame_id"

        def source_priority(path_name: str) -> tuple:
            # 같은 frame_id가 _frames와 _preprocessed에 동시에 있으면 하나만 써야 export가 중복되지 않음.
            # 라벨 기준은 원본 좌표계가 안정적이므로 _frames/raw를 우선하고, 없을 때만 preprocessed를 사용.
            low = path_name.lower()
            parts = [p.lower() for p in Path(path_name).parts]
            if any(p.endswith("_frames") or p == "_frames" for p in parts):
                return (0, path_name)
            if "preprocessed" in low or "processed" in low:
                return (1, path_name)
            return (2, path_name)

        candidates_by_fid = {}
        json_names = sorted(self.reader.list_jsons(), key=lambda n: (record_from_name(n), timestamp_from_name(n), n))
        for name in json_names:
            try:
                frame=json.loads(self.reader.read_json_bytes(name))
            except Exception:
                self.invalid_json_count += 1
                continue
            if not is_raw_frame_json(frame):
                self.skipped_json_count += 1
                continue
            fid = frame_id_from_name(name)
            candidates_by_fid.setdefault(fid, []).append((name, frame))

        for fid in sorted(candidates_by_fid.keys(), key=lambda x: (timestamp_from_name(x), x)):
            cand = candidates_by_fid[fid]
            if len(cand) > 1:
                self.duplicate_frame_count += len(cand) - 1
            name, frame = sorted(cand, key=lambda nf: source_priority(nf[0]))[0]
            meta={"frame_id": fid, "source_json": name, "record": record_from_name(name), "timestamp": timestamp_from_name(name), "order": order, "dataset": zip_name, "object_class_hint": self.object_class, "pose_class_hint": self.pose_class}
            order += 1
            self.frames[fid]=frame
            try:
                # v19: multi-class (예: dog+human) 인 경우 load 단계에서 cluster 후보를 강제로
                # 생성. 그래야 TRK 가 사람 1명만 잡아도 강아지가 cluster 후보로 추가됨.
                prev_enable = getattr(self.params, "enable_cluster_candidates", False)
                if is_multi_class_mode(self.object_class):
                    self.params.enable_cluster_candidates = True
                try:
                    lab0 = raw_label_frame(meta, frame, self.object_class, self.pose_class, self.params)
                finally:
                    self.params.enable_cluster_candidates = prev_enable
                lab0 = apply_loadtime_count_limit(lab0, self.load_expected_counts, self.object_class, self.pose_class, is_multi_class_mode(self.object_class))
                raw.append(lab0)
            except Exception as e:
                raw.append({**meta, "objects": [], "confidence": "error", "recommendation":"exclude", "method":"empty_or_error", "quality": {"error": str(e), "point_count":0, "track_count":0}})

        self.labels = apply_temporal_correction(raw, self.params)
        self.labels = [apply_loadtime_count_limit(lab, self.load_expected_counts, self.object_class, self.pose_class, is_multi_class_mode(self.object_class)) for lab in self.labels]
        # Always keep review order chronological: first frame -> later frame.
        self.labels.sort(key=lambda lab: (lab.get("record", ""), lab.get("timestamp", 0), lab.get("order", 0), lab.get("frame_id", "")))
        self.save_outputs()
        return self.stats()

    def save_outputs(self) -> None:
        if not self.dataset_zip: return
        d = self.out_dir / self.scenario
        d.mkdir(parents=True, exist_ok=True)
        with (d / "labels_temporal_corrected_all.jsonl").open("w",encoding="utf-8") as f:
            for lab in self.labels: f.write(json.dumps(lab,ensure_ascii=False)+"\n")
        with (d / "frame_report.csv").open("w",newline="",encoding="utf-8") as f:
            fields=["dataset","frame_id","record","confidence","method","method_text","recommendation","recommendation_text","point_count","track_count","object_count","source_json"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            for lab in self.labels:
                q=lab.get("quality",{})
                w.writerow({"dataset": lab.get("dataset"), "frame_id": lab.get("frame_id"), "record": lab.get("record"), "confidence": lab.get("confidence"), "method": lab.get("method"), "method_text": METHOD_TEXT.get(lab.get("method"), lab.get("method")), "recommendation": lab.get("recommendation"), "recommendation_text": RECOMMEND_TEXT.get(lab.get("recommendation"), lab.get("recommendation")), "point_count": q.get("point_count",0), "track_count": q.get("track_count",0), "object_count": len(lab.get("objects",[])), "source_json": lab.get("source_json")})

    def filtered_indices(self) -> List[int]:
        idxs=[]
        for i,lab in enumerate(self.labels):
            if self.filter_view == "review" and lab.get("recommendation") == "auto_use":
                continue
            if self.filter_view == "corrected" and lab.get("frame_id") not in self.corrections:
                continue
            if self.filter_conf != "all" and lab.get("confidence") != self.filter_conf:
                continue
            if self.filter_recommend != "all" and lab.get("recommendation") != self.filter_recommend:
                continue
            idxs.append(i)
        idxs.sort(key=lambda i: (self.labels[i].get("record", ""), self.labels[i].get("timestamp", 0), self.labels[i].get("order", i), self.labels[i].get("frame_id", "")))
        return idxs

    def stats(self) -> Dict[str,Any]:
        cnt: Dict[str,int] = {}; rec: Dict[str,int] = {}; meth: Dict[str,int] = {}
        review_cnt: Dict[str,int] = {}; review_total = 0
        for lab in self.labels:
            c = lab.get("confidence", "unknown")
            r = lab.get("recommendation", "unknown")
            m = lab.get("method", "unknown")
            cnt[c] = cnt.get(c, 0) + 1
            rec[r] = rec.get(r, 0) + 1
            meth[m] = meth.get(m, 0) + 1
            if c != "high":
                review_cnt[c] = review_cnt.get(c, 0) + 1
                review_total += 1
        return {
            "loaded": self.dataset_zip.name if self.dataset_zip else None,
            "mode": self.reader.mode if self.reader else None,
            "frame_count": len(self.labels),
            "confidence_counts": cnt,
            "non_high_confidence_counts": review_cnt,
            "non_high_total": review_total,
            "recommendation_counts": rec,
            "method_counts": meth,
            "method_text": METHOD_TEXT,
            "recommendation_text": RECOMMEND_TEXT,
            "object_class": self.object_class,
            "pose_class": self.pose_class,
            "scenario": self.scenario,
            "datasets": self.list_datasets(),
            "object_class_options": ["person", "animal", "non_human", "candidate"],
            "pose_class_options": pose_options(),
            "out_dir": str(self.out_dir),
            "skipped_json_count": getattr(self, "skipped_json_count", 0),
            "invalid_json_count": getattr(self, "invalid_json_count", 0),
            "duplicate_frame_count": getattr(self, "duplicate_frame_count", 0),
            "dedup_source_policy": getattr(self, "dedup_source_policy", ""),
            "load_expected_counts": getattr(self, "load_expected_counts", {"person": 1, "animal": 0, "non_human": 0}),
        }


    def ensure_lazy_cluster_candidates(self, idx: int) -> None:
        """대용량 mixed dataset용.
        Load 단계에서는 TRK 기반 box만 빠르게 만들고,
        실제로 화면에서 보는 frame에 대해서만 남은 point cluster 후보를 추가한다.
        """
        if idx < 0 or idx >= len(self.labels):
            return
        if not is_multi_class_mode(self.object_class):
            return
        lab = self.labels[idx]
        if lab.get("_lazy_cluster_done"):
            return
        frame = self.frames.get(lab.get("frame_id", ""), {})
        pts = parse_points(frame)
        objects = list(lab.get("objects", []) or [])
        old_flag = getattr(self.params, "enable_cluster_candidates", False)
        self.params.enable_cluster_candidates = True
        try:
            cluster_pairs = point_cluster_candidates(pts, objects, self.params)
        finally:
            self.params.enable_cluster_candidates = old_flag

        start = len(objects) + 1
        added = 0
        for box, q in cluster_pairs:
            objects.append({
                "object_id": f"obj_{start+added:03d}",
                "object_class": "candidate",
                "pose_class": self.pose_class if self.pose_class != "auto" else "unknown",
                "box": box,
                "confidence": lab.get("confidence", "medium"),
                "label_source": "lazy_point_cluster_candidate",
                "review_status": "pending",
                "quality": q,
            })
            added += 1

        lab["objects"] = objects
        if isinstance(getattr(self, "load_expected_counts", None), dict):
            limited = apply_loadtime_count_limit(lab, self.load_expected_counts, self.object_class, self.pose_class, is_multi_class_mode(self.object_class))
            lab.clear(); lab.update(limited)
        lab["_lazy_cluster_done"] = True
        lab["lazy_cluster_added"] = added
        q = lab.setdefault("quality", {})
        q["lazy_cluster_added"] = added
        q["object_count"] = len(lab.get("objects", []))
        if added > 0 and lab.get("method") == "auto_high":
            lab["method"] = "auto_high_plus_lazy_cluster"
        if added > 0 and lab.get("recommendation") == "auto_use":
            lab["recommendation"] = "quick_check"

    def temporal_context_views(self, center_idx: int) -> List[Dict[str,Any]]:
        """Return small perspective SVGs for previous/current/next chronological frames."""
        items: List[Dict[str,Any]] = []
        names = {-1: "이전 frame", 0: "현재 frame", 1: "다음 frame"}
        keys = {-1: "prev", 0: "current", 1: "next"}
        for rel in [-1, 0, 1]:
            ni = center_idx + rel
            if 0 <= ni < len(self.labels):
                self.ensure_lazy_cluster_candidates(ni)
                lab2 = self.apply_correction(self.labels[ni])
                frame2 = self.frames.get(lab2.get("frame_id", ""), {})
                pts2 = parse_points(frame2)
                items.append({
                    "key": keys[rel],
                    "title": names[rel],
                    "frame_id": lab2.get("frame_id", ""),
                    "confidence": lab2.get("confidence", ""),
                    "recommendation": lab2.get("recommendation", ""),
                    "object_count": len(lab2.get("objects", [])),
                    "svg": render_svg(pts2, lab2, "perspective", width=300, height=185),
                })
            else:
                items.append({
                    "key": keys[rel],
                    "title": names[rel],
                    "frame_id": "-",
                    "confidence": "none",
                    "recommendation": "none",
                    "object_count": 0,
                    "svg": '<svg viewBox="0 0 300 185" width="100%" height="185" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="#fafafa"/><text x="16" y="94" font-size="13" font-family="Arial" fill="#999">frame 없음</text></svg>',
                })
        return items

    def current_payload(self) -> Dict[str,Any]:
        idxs=self.filtered_indices()
        if not idxs: return {"empty": True, "stats": self.stats()}
        self.pos=int(clamp(self.pos,0,len(idxs)-1))
        idx=idxs[self.pos]
        self.ensure_lazy_cluster_candidates(idx)
        lab=self.apply_correction(self.labels[idx])
        frame=self.frames.get(lab["frame_id"], {})
        pts=parse_points(frame)
        return {"empty": False, "pos": self.pos+1, "total": len(idxs), "global_index": idx+1, "global_total": len(self.labels), "label": lab, "model_label": label_to_model_schema(lab, self.object_class, self.pose_class, is_multi_class_mode(self.object_class)), "correction": self.corrections.get(lab["frame_id"], {}), "stats": self.stats(), "svgs": {p: render_svg(pts, lab, p) for p in ["perspective","top","side","front"]}, "context_views": self.temporal_context_views(idx), "method_text": METHOD_TEXT.get(lab.get("method"), lab.get("method")), "recommendation_text": RECOMMEND_TEXT.get(lab.get("recommendation"), lab.get("recommendation")), "recommendation_help": RECOMMEND_HELP.get(lab.get("recommendation"), "")}

    def preview_payload(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        """Return SVG/JSON preview for the current frame with unsaved object values."""
        idxs = self.filtered_indices()
        if not idxs:
            return {"empty": True, "stats": self.stats()}
        self.pos = int(clamp(self.pos, 0, len(idxs)-1))
        idx = idxs[self.pos]
        self.ensure_lazy_cluster_candidates(idx)
        base = self.apply_correction(self.labels[idx])
        if isinstance(payload.get("objects"), list):
            base["objects"] = sanitize_objects(payload.get("objects") or [], self.object_class, self.pose_class)
        else:
            box = make_box(
                safe_float(payload.get("x")), safe_float(payload.get("y")), safe_float(payload.get("z"), 0.8),
                safe_float(payload.get("w"), 0.8), safe_float(payload.get("l"), 0.8), safe_float(payload.get("h"), 1.6),
                safe_float(payload.get("yaw"), 0.0),
            )
            base["objects"] = [{
                "object_id": "obj_001",
                "object_class": default_object_class(self.object_class),
                "pose_class": self.pose_class,
                "box": box,
                "confidence": base.get("confidence", "medium"),
                "label_source": "preview_unsaved",
            }]
        base["preview_unsaved"] = True
        frame = self.frames.get(base["frame_id"], {})
        pts = parse_points(frame)
        return {
            "empty": False,
            "label": base,
            "model_label": label_to_model_schema(base, self.object_class, self.pose_class, is_multi_class_mode(self.object_class)),
            "svgs": {p: render_svg(pts, base, p) for p in ["perspective", "top", "side", "front"]},
            "context_views": self.temporal_context_views(idx),
        }

    def apply_correction(self, lab: Dict[str,Any]) -> Dict[str,Any]:
        out=json.loads(json.dumps(lab,ensure_ascii=False))
        corr=self.corrections.get(out["frame_id"])
        if not corr: return out
        action=corr.get("action")
        out["review_action"]=action
        if isinstance(corr.get("expected_counts"), dict):
            out["expected_counts"] = normalize_expected_counts(corr.get("expected_counts"))
        if action == "reject":
            out["objects"]=[]
            out["expected_counts"] = {k: 0 for k in MODEL_CLASSES}
            out["recommendation"]="exclude"
        elif action == "uncertain":
            out["recommendation"]="careful_check"
        elif action in ("accept","adjust","add","set_objects"):
            out["recommendation"]="auto_use"
            if isinstance(corr.get("objects"), list):
                out["objects"] = sanitize_objects(corr.get("objects") or [], self.object_class, self.pose_class)
                for obj in out["objects"]:
                    obj["label_source"] = "manual_adjusted"
            elif action in ("adjust","add"):
                box=make_box(safe_float(corr.get("x")), safe_float(corr.get("y")), safe_float(corr.get("z")), safe_float(corr.get("w"),.8), safe_float(corr.get("l"),.8), safe_float(corr.get("h"),1.6), safe_float(corr.get("yaw"),0))
                out["objects"]=[{"object_id":"obj_001", "object_class":default_object_class(self.object_class), "pose_class":self.pose_class, "box":box, "confidence":out.get("confidence","medium"), "label_source":"manual_adjusted"}]
        return out

    def save_correction(self, action: str, payload: Dict[str,Any]) -> Dict[str,Any]:
        idxs=self.filtered_indices()
        if not idxs: return {"ok": False, "error": "no frame"}
        lab=self.labels[idxs[self.pos]]; fid=lab["frame_id"]
        corr={"frame_id":fid,"action":action,"note":payload.get("note",""),"updated_at":now_stamp()}
        if isinstance(payload.get("expected_counts"), dict):
            corr["expected_counts"] = normalize_expected_counts(payload.get("expected_counts"))
        elif action == "reject":
            corr["expected_counts"] = {k: 0 for k in MODEL_CLASSES}
        if isinstance(payload.get("objects"), list):
            corr["objects"] = sanitize_objects(payload.get("objects") or [], self.object_class, self.pose_class)
        elif action in ("adjust","add"):
            for k in ["x","y","z","w","l","h","yaw"]: corr[k]=safe_float(payload.get(k),0)
        self.corrections[fid]=corr
        return {"ok": True, "correction": corr}

    def export_review(self, mode: str = "all_frames_schema", minimal: bool = False) -> Dict[str,Any]:
        """Export final model-label schema (v19 strict).

        v19 changes:
        - 메인 파일 `labels_final_all_frames_schema.jsonl` 은 항상 모든 frame을 strict
          schema로 출력한다 (rejected/empty → objects: []). mode 인자와 무관.
        - source-order 동반 파일 `labels_final_strict_source_order.jsonl` 도 항상 같이 출력
          (chronological / load order 그대로).
        - mode != "all_frames_schema" 인 경우, 추가로 `labels_final_{mode}.jsonl`도 만든다
          (예: accepted_only). 단 이 파일도 strict schema 그대로.
        - export 직후 strict validation을 돌리고 `validation_report.md` / `strict_validation_issues.csv` 를 작성한다.
        - `review_targets_non_auto_frames.csv` 도 같이 작성한다.
        디버그 필드 (n_points, track_id, confidence, source, expected_counts ...) 는 strict
        파일에 절대 들어가지 않는다.
        """
        if not self.dataset_zip:
            raise RuntimeError("No dataset loaded")
        d = self.out_dir / self.scenario
        d.mkdir(parents=True, exist_ok=True)
        default_cls = self.object_class
        default_pose = self.pose_class
        multi_mode = is_multi_class_mode(self.object_class)
        all_rows_strict: List[Dict[str, Any]] = []           # 모든 frame strict, source order
        mode_rows_strict: List[Dict[str, Any]] = []          # mode 필터 결과 strict
        gate_diags: List[Dict[str, Any]] = []
        review_targets: List[Dict[str, Any]] = []
        frame_point_counts: Dict[str, int] = {}

        for base in self.labels:
            fid = base.get("frame_id", "")
            corr = self.corrections.get(fid, {})
            action = corr.get("action")
            lab_corr = self.apply_correction(base)

            # 메인 파일은 mode와 상관없이 항상 전체 frame을 담는다.
            lab_main = lab_corr
            if action == "reject" or ((lab_main.get("confidence") in ("empty", "error")) and not lab_main.get("objects")):
                lab_main = {**lab_main, "objects": [], "expected_counts": {k: 0 for k in MODEL_CLASSES}}
            expected_main = corr.get("expected_counts") if isinstance(corr.get("expected_counts"), dict) else lab_main.get("expected_counts")
            lab_main_gated, diag = enforce_object_count_gate(
                lab_main,
                expected_main if isinstance(expected_main, dict) else None,
                default_cls, default_pose, multi_mode,
            )
            gate_diags.append(diag)
            strict_main = strict_label_schema(lab_main_gated, default_cls, default_pose, multi_mode)
            all_rows_strict.append(strict_main)

            point_count = int(safe_float((base.get("quality") or {}).get("point_count"), 0.0))
            frame_point_counts[strict_main["frame_id"]] = point_count

            # mode 필터: 사용자가 누른 export 버튼에 따라 보조 파일에 들어갈 frame을 정한다.
            include_in_mode = True
            lab_mode = lab_corr
            if mode == "accepted_only":
                include_in_mode = action in ("accept", "adjust", "add", "set_objects")
            elif mode == "accepted_plus_high":
                if action == "reject":
                    include_in_mode = False
                elif action not in ("accept", "adjust", "add", "set_objects") and lab_mode.get("recommendation") != "auto_use":
                    include_in_mode = False
            elif mode == "all_except_rejected":
                if action == "reject":
                    lab_mode = {**lab_mode, "objects": [], "expected_counts": {k: 0 for k in MODEL_CLASSES}}
            else:  # all_frames_schema (default) — 동일 콘텐츠
                if action == "reject" or ((lab_mode.get("confidence") in ("empty", "error")) and not lab_mode.get("objects")):
                    lab_mode = {**lab_mode, "objects": [], "expected_counts": {k: 0 for k in MODEL_CLASSES}}
            if include_in_mode:
                expected_mode = corr.get("expected_counts") if isinstance(corr.get("expected_counts"), dict) else lab_mode.get("expected_counts")
                lab_mode_gated, _ = enforce_object_count_gate(
                    lab_mode,
                    expected_mode if isinstance(expected_mode, dict) else None,
                    default_cls, default_pose, multi_mode,
                )
                mode_rows_strict.append(strict_label_schema(lab_mode_gated, default_cls, default_pose, multi_mode))

            recommendation = base.get("recommendation") or ""
            if recommendation != "auto_use" or action in ("uncertain", "reject"):
                review_targets.append({
                    "frame_id": strict_main["frame_id"],
                    "record": base.get("record", ""),
                    "confidence": base.get("confidence", ""),
                    "recommendation": recommendation,
                    "method": base.get("method", ""),
                    "review_action": action or "",
                    "point_count": point_count,
                    "object_count": len(strict_main.get("objects", [])),
                })

        # 정렬: 메인 파일은 frame_id (숫자 변환 가능하면 숫자) 오름차순.
        def sort_key(row: Dict[str, Any]):
            fid = row.get("frame_id", "")
            try:
                return (0, float(fid), fid)
            except Exception:
                return (1, 0.0, fid)
        sorted_main = sorted(all_rows_strict, key=sort_key)

        primary_path = d / "labels_final_all_frames_schema.jsonl"
        with primary_path.open("w", encoding="utf-8") as f:
            for r in sorted_main:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # v19: minimal 모드면 strict 메인 파일 한 개만 만들고 나머지 보조 파일은 다 skip.
        # 학습용으로 배포할 때 디버그/검수 파일을 끼우지 않으려는 사용자 편의용.
        if minimal:
            # 기존 보조 파일이 남아있으면 정리해서 폴더가 깔끔해지도록 한다.
            for aux_name in (
                "labels_final_strict_source_order.jsonl",
                "labels_temporal_corrected_all.jsonl",
                "auto_box_quality.csv",
                "frame_report.csv",
                "review_corrections.csv",
                "review_targets_non_auto_frames.csv",
                "strict_validation_issues.csv",
                "validation_report.md",
            ):
                p = d / aux_name
                if p.exists():
                    try: p.unlink()
                    except Exception: pass
            for old in d.glob("labels_final_*.jsonl"):
                if old.name != "labels_final_all_frames_schema.jsonl":
                    try: old.unlink()
                    except Exception: pass
            min_pts_thresh = int(getattr(self.params, "min_points_for_box", 0) or 0)
            validation = validate_strict_jsonl(primary_path, frame_point_counts, min_points_for_box=min_pts_thresh)
            dataset_meta_min = {
                "scenario": self.scenario,
                "object_class": self.object_class,
                "pose_class": self.pose_class,
                "load_expected_counts": getattr(self, "load_expected_counts", None),
            }
            readme_path_min = write_label_readme(d, primary_path, validation, dataset_meta_min, minimal=True)
            return {
                "ok": True,
                "path": str(primary_path),
                "readme": str(readme_path_min),
                "minimal": True,
                "count": len(all_rows_strict),
                "object_count": sum(len(r.get("objects", [])) for r in all_rows_strict),
                "strict_validation": {
                    "total_frames": validation.get("total_frames", 0),
                    "total_objects": validation.get("total_objects", 0),
                    "empty_frames": validation.get("empty_frames", 0),
                    "duplicate_frame_id_count": validation.get("duplicate_frame_id_count", 0),
                    "issue_count": validation.get("issue_count", 0),
                    "is_strict_valid": validation.get("is_strict_valid", False),
                },
                "format": "frame_id + objects[].class/pose/box.center/box.dimensions/yaw, 디버그 필드 없음, 빈 frame은 objects: []",
                "sample": sorted_main[:2],
            }

        source_order_path = d / "labels_final_strict_source_order.jsonl"
        with source_order_path.open("w", encoding="utf-8") as f:
            for r in all_rows_strict:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        extra_mode_path: Optional[Path] = None
        if mode != "all_frames_schema":
            extra_mode_path = d / f"labels_final_{mode}.jsonl"
            sorted_mode = sorted(mode_rows_strict, key=sort_key)
            with extra_mode_path.open("w", encoding="utf-8") as f:
                for r in sorted_mode:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        cpath = d / "review_corrections.csv"
        with cpath.open("w", newline="", encoding="utf-8") as f:
            fields = ["frame_id", "action", "person_count", "animal_count", "non_human_count", "x", "y", "z", "w", "l", "h", "yaw", "note", "updated_at"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for fid, c in self.corrections.items():
                row = {k: c.get(k, "") for k in fields}
                ec = normalize_expected_counts(c.get("expected_counts"))
                row["person_count"] = ec.get("person", 0)
                row["animal_count"] = ec.get("animal", 0)
                row["non_human_count"] = ec.get("non_human", 0)
                w.writerow(row)

        review_targets_path = d / "review_targets_non_auto_frames.csv"
        with review_targets_path.open("w", newline="", encoding="utf-8") as f:
            fields = ["frame_id", "record", "confidence", "recommendation", "method", "review_action", "point_count", "object_count"]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in review_targets:
                w.writerow({k: row.get(k, "") for k in fields})

        # v19: per-frame box-vs-pointcloud quality CSV. Reviewer가 의심 frame부터 열어볼 수 있게.
        quality_path = d / "auto_box_quality.csv"
        with quality_path.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "frame_id", "record", "n_points", "n_objects",
                "point_z_min", "point_z_max", "point_z_p1", "point_z_p5", "point_z_p95", "point_z_p99", "point_z_median",
                "point_xy_centroid_x", "point_xy_centroid_y",
                "box_center_x", "box_center_y", "box_center_z",
                "box_bottom_z", "box_top_z", "box_h",
                "z_bottom_offset", "z_top_offset", "xy_offset",
                "local_stray_above", "local_stray_below",
                "suspect", "suspect_reason",
            ]
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for base, strict_row in zip(self.labels, all_rows_strict):
                fid = strict_row.get("frame_id", "")
                frame_raw = self.frames.get(fid, {})
                pts = parse_points(frame_raw) if frame_raw else []
                objects = strict_row.get("objects") or []
                obj = objects[0] if objects else None
                row = {k: "" for k in fields}
                row["frame_id"] = fid
                row["record"] = base.get("record", "")
                row["n_points"] = len(pts)
                row["n_objects"] = len(objects)
                if pts:
                    zs = sorted(p[2] for p in pts)
                    row["point_z_min"] = round(zs[0], 4)
                    row["point_z_max"] = round(zs[-1], 4)
                    row["point_z_p1"] = round(percentile(zs, 1), 4)
                    row["point_z_p5"] = round(percentile(zs, 5), 4)
                    row["point_z_p95"] = round(percentile(zs, 95), 4)
                    row["point_z_p99"] = round(percentile(zs, 99), 4)
                    row["point_z_median"] = round(zs[len(zs)//2], 4)
                    cx_p = sum(p[0] for p in pts) / len(pts)
                    cy_p = sum(p[1] for p in pts) / len(pts)
                    row["point_xy_centroid_x"] = round(cx_p, 4)
                    row["point_xy_centroid_y"] = round(cy_p, 4)
                if obj is not None:
                    box = obj.get("box", {})
                    center = box.get("center") or [0.0, 0.0, 0.0]
                    dims = box.get("dimensions") or [0.0, 0.0, 0.0]
                    bcx, bcy, bcz = (safe_float(center[i]) for i in range(3))
                    bh = safe_float(dims[2])
                    row["box_center_x"] = round(bcx, 4)
                    row["box_center_y"] = round(bcy, 4)
                    row["box_center_z"] = round(bcz, 4)
                    row["box_bottom_z"] = round(bcz - bh / 2, 4)
                    row["box_top_z"] = round(bcz + bh / 2, 4)
                    row["box_h"] = round(bh, 4)
                    if pts:
                        # v19: 다중 사람 frame에서는 frame-전체 z extent가 다른 사람을 포함할 수
                        # 있으니, 박스 주변 (box xy + 1.5m) 의 점들만으로 floor/head 를 비교한다.
                        local_radius2 = 1.5 ** 2
                        local_pts = [p for p in pts if (p[0]-bcx)**2 + (p[1]-bcy)**2 <= local_radius2]
                        ref_pts = local_pts if len(local_pts) >= 5 else pts
                        local_zs = sorted(p[2] for p in ref_pts)
                        # v19+: offset 은 p1/p99 기준으로 계산 (1~2개 stray 노이즈는 무시).
                        p_lo = percentile(local_zs, 1)
                        p_hi = percentile(local_zs, 99)
                        cx_p = sum(p[0] for p in ref_pts) / len(ref_pts)
                        cy_p = sum(p[1] for p in ref_pts) / len(ref_pts)
                        box_bot_z = bcz - bh / 2
                        box_top_z = bcz + bh / 2
                        row["z_bottom_offset"] = round(box_bot_z - p_lo, 4)
                        row["z_top_offset"] = round(box_top_z - p_hi, 4)
                        row["xy_offset"] = round(math.hypot(bcx - cx_p, bcy - cy_p), 4)
                        # 박스 밖에 떨어진 점 개수 (사용자에게 시각화하기 위해 카운트)
                        row["local_stray_above"] = sum(1 for p in ref_pts if p[2] > box_top_z + 0.01)
                        row["local_stray_below"] = sum(1 for p in ref_pts if p[2] < box_bot_z - 0.01)
                # Suspect heuristic (v19+):
                # 박스가 padding 0.20m 만큼 양쪽으로 늘어나도록 만들어졌기 때문에 padding 양만큼은
                # offset 이 -0.20 ~ +0.20 정도로 떨어지는 게 정상. suspect 는 padding 을 한참 넘는
                # 진짜 어긋남만 잡도록 임계값을 풀고, top 측 (점이 박스 위로 새는 케이스) 만 좀 더 타이트하게.
                reasons: List[str] = []
                stray_above = row.get("local_stray_above", 0) or 0
                stray_below = row.get("local_stray_below", 0) or 0
                significant_stray_threshold = max(3, int((row["n_points"] or 0) * 0.02))  # 2% of pts
                if isinstance(row["z_bottom_offset"], (int, float)):
                    if row["z_bottom_offset"] < -0.60:
                        reasons.append(f"box bottom {row['z_bottom_offset']:+.2f}m below floor (p1)")
                    elif row["z_bottom_offset"] > 0.40:
                        reasons.append(f"box bottom {row['z_bottom_offset']:+.2f}m above floor (p1)")
                if isinstance(row["z_top_offset"], (int, float)):
                    if row["z_top_offset"] < -0.30:
                        reasons.append(f"box top {row['z_top_offset']:+.2f}m below head (p99) — head cut")
                    elif row["z_top_offset"] > 1.00:
                        # v19+: min/max anchor 때문에 sparse 점군에서는 박스가 클 수 있음. 1m 이상 떴을 때만 flag.
                        reasons.append(f"box top {row['z_top_offset']:+.2f}m above head — too tall (likely sparse pts)")
                if isinstance(row["xy_offset"], (int, float)) and row["xy_offset"] > 0.6:
                    reasons.append(f"xy off {row['xy_offset']:.2f}m")
                # 진짜 많은 점이 박스 밖에 있을 때만 stray flag (1-2개 outlier 는 무시)
                if stray_above >= significant_stray_threshold:
                    reasons.append(f"{stray_above} pts above box (signif)")
                if stray_below >= significant_stray_threshold:
                    reasons.append(f"{stray_below} pts below box (signif)")
                if row["n_points"] and not row["n_objects"]:
                    reasons.append("pts>0 but no object")
                if not row["n_points"] and row["n_objects"]:
                    reasons.append("pts=0 but object exists")
                row["suspect"] = "yes" if reasons else ""
                row["suspect_reason"] = "; ".join(reasons)
                w.writerow(row)

        # strict validation을 export 직후 자동 실행. 메인 파일과 source-order 파일 둘 다 검증.
        min_pts_thresh = int(getattr(self.params, "min_points_for_box", 0) or 0)
        validation = validate_strict_jsonl(primary_path, frame_point_counts, min_points_for_box=min_pts_thresh)
        source_validation = validate_strict_jsonl(source_order_path, frame_point_counts, min_points_for_box=min_pts_thresh)
        dataset_meta = {
            "scenario": self.scenario,
            "object_class": self.object_class,
            "pose_class": self.pose_class,
            "load_expected_counts": getattr(self, "load_expected_counts", None),
        }
        md_path, issues_path = write_validation_outputs(d, validation, source_validation, dataset_meta)
        readme_path = write_label_readme(d, primary_path, validation, dataset_meta, minimal=False)

        return {
            "ok": True,
            "path": str(primary_path),
            "readme": str(readme_path),
            "source_order_path": str(source_order_path),
            "extra_mode_path": str(extra_mode_path) if extra_mode_path else None,
            "count": len(all_rows_strict),
            "object_count": sum(len(r.get("objects", [])) for r in all_rows_strict),
            "corrections_csv": str(cpath),
            "review_targets_csv": str(review_targets_path),
            "auto_box_quality_csv": str(quality_path),
            "validation_report_md": str(md_path),
            "validation_issues_csv": str(issues_path),
            "format": "frame_id + objects[].class/pose/box.center/box.dimensions/yaw, 디버그 필드 없음, 빈 frame은 objects: []",
            "count_gate": {
                "applied_frames": len(gate_diags),
                "manual_gated_frames": sum(1 for d in gate_diags if d.get("gate_source") == "manual"),
                "derived_gated_frames": sum(1 for d in gate_diags if d.get("gate_source") == "derived"),
                "invalid_frames": sum(1 for d in gate_diags if not d.get("valid")),
                "trimmed_objects": sum(int(d.get("trimmed_count", 0) or 0) for d in gate_diags),
                "missing_objects": sum(int(d.get("missing_total", 0) or 0) for d in gate_diags),
            },
            "strict_validation": {
                "total_frames": validation.get("total_frames", 0),
                "total_objects": validation.get("total_objects", 0),
                "empty_frames": validation.get("empty_frames", 0),
                "one_object_frames": validation.get("one_object_frames", 0),
                "multi_object_frames": validation.get("multi_object_frames", 0),
                "duplicate_frame_id_count": validation.get("duplicate_frame_id_count", 0),
                "invalid_class_count": validation.get("invalid_class_count", 0),
                "invalid_pose_count": validation.get("invalid_pose_count", 0),
                "missing_box_count": validation.get("missing_box_count", 0),
                "center_bad_count": validation.get("center_bad_count", 0),
                "dimensions_bad_count": validation.get("dimensions_bad_count", 0),
                "missing_yaw_count": validation.get("missing_yaw_count", 0),
                "extra_top_field_count": validation.get("extra_top_field_count", 0),
                "extra_object_field_count": validation.get("extra_object_field_count", 0),
                "extra_box_field_count": validation.get("extra_box_field_count", 0),
                "point_count_positive_but_no_object": validation.get("point_count_positive_but_no_object", 0),
                "point_count_zero_but_has_object": validation.get("point_count_zero_but_has_object", 0),
                "issue_count": validation.get("issue_count", 0),
                "is_strict_valid": validation.get("is_strict_valid", False),
            },
            "sample": sorted_main[:2],
        }


# -----------------------------
# Web UI
# -----------------------------

INDEX_HTML = r"""
<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Radar Temporal Label Workbench v19</title>
<style>
body{margin:0;background:#f4f4f4;color:#222;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:#111;color:#fff;padding:12px 18px;display:flex;justify-content:space-between;gap:12px;align-items:center}main{display:grid;grid-template-columns:375px 1fr;gap:12px;padding:12px;align-items:start}.panel{background:#fff;border-radius:13px;padding:12px;box-shadow:0 1px 5px #0001}.sidePanel{position:sticky;top:10px;max-height:calc(100vh - 76px);overflow:auto}.sidePanel h3{font-size:18px;margin:12px 0 8px}.sidePanel h3:first-child{margin-top:0}.row{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:6px 0}.controlRow{display:grid;grid-template-columns:74px 1fr;gap:7px;align-items:center;margin:6px 0}.controlRow label{font-weight:600;color:#333}select,input,button,textarea{font:inherit;border:1px solid #ccc;border-radius:8px;padding:7px 9px}select{max-width:100%}button{background:#fff;cursor:pointer}.primary{background:#111;color:#fff;border-color:#111}.good{background:#e8f5e9;border-color:#66bb6a}.bad{background:#ffebee;border-color:#ef5350}.warn{background:#fff8e1;border-color:#ffca28}.contextStrip{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}.contextCard{background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:8px;min-height:214px}.contextTitle{display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:13px;font-weight:700;margin-bottom:4px}.contextMeta{font-size:11px;color:#666;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.views{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.svgbox{background:#fff;border:1px solid #e0e0e0;border-radius:12px;padding:8px;min-height:285px}.kv{display:grid;grid-template-columns:86px 1fr;gap:4px;font-size:12px;margin-top:6px}.small{font-size:12px;color:#666;line-height:1.42}.badge{display:inline-block;border-radius:999px;padding:3px 8px;font-size:12px;background:#eee}.high,.auto_use{background:#e8f5e9;color:#1b5e20}.medium,.quick_check{background:#fff8e1;color:#795548}.interpolated{background:#e3f2fd;color:#0d47a1}.low,.careful_check{background:#ffebee;color:#b71c1c}.empty,.exclude{background:#eceff1;color:#37474f}.error{background:#f3e5f5;color:#4a148c}.num{width:62px}input[type=range]{width:165px;padding:0}.sliderRow{display:grid;grid-template-columns:30px 1fr 70px;gap:5px;align-items:center;margin:4px 0}details{border:1px solid #eee;border-radius:10px;padding:7px;margin:7px 0;background:#fafafa}summary{font-weight:700;cursor:pointer}pre{background:#fafafa;border:1px solid #eee;border-radius:8px;padding:8px;max-height:180px;overflow:auto;font-size:12px}.helpBox{background:#eef4ff;border:1px solid #bbdefb;border-radius:10px;padding:8px;margin:8px 0}.note{width:100%;min-height:36px;box-sizing:border-box}.statsTable{width:100%;border-collapse:collapse;font-size:12px;margin-top:6px}.statsTable th,.statsTable td{border:1px solid #e0e0e0;padding:5px;text-align:left}.statsTable td:last-child,.statsTable th:last-child{text-align:right}.statsFloat{position:fixed;right:16px;bottom:16px;z-index:30;width:265px;max-height:42vh;overflow:auto;box-shadow:0 4px 18px #0002;background:#fffffff2;backdrop-filter:blur(3px)}.actionGrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}.actionGrid button{font-size:12px;line-height:1.15;padding:8px 4px;min-height:42px}.filterButtons{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}.filterButtons button{padding:7px 5px}.wideSelect{width:100%}.datasetRow{display:grid;grid-template-columns:1fr 72px;gap:8px;align-items:center}.tight{margin-top:2px;margin-bottom:2px}.objList{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0}.objBtn{font-size:12px;padding:5px 7px;border-radius:999px}.objBtn.active{background:#111;color:#fff;border-color:#111}.objBox{border:1px solid #eee;border-radius:10px;background:#fafafa;padding:8px;margin:8px 0}.objControl{display:grid;grid-template-columns:56px 1fr;gap:6px;align-items:center;margin:5px 0}.dangerText{color:#b71c1c}.countBox{border:1px solid #e0e0e0;border-radius:12px;background:#fbfbfb;padding:9px;margin:8px 0}.countGrid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0}.countCard{border:1px solid #e5e5e5;border-radius:10px;background:#fff;padding:7px}.countCard label{display:block;font-size:12px;font-weight:700;margin-bottom:4px}.countInput{width:100%;box-sizing:border-box;text-align:center}.statusOk{color:#1b5e20}.statusWarn{color:#b71c1c}.statusNeutral{color:#666}@media(max-width:1100px){main{grid-template-columns:340px 1fr}.statsFloat{position:static;width:auto;margin:0 12px 12px}.actionGrid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><div><b>Radar Temporal Label Workbench v19</b> <span class="small">load-time count 기반 auto box 생성 / person·animal·non_human 라벨링</span></div><div id="summary" class="small"></div></header>
<main><aside class="panel sidePanel"><h3>0) Dataset 선택</h3><div class="datasetRow"><select id="dataset" class="wideSelect"></select><button class="primary" onclick="loadDataset()">Load</button></div><div class="controlRow"><label>mode</label><select id="objectClass" onchange="modePresetCounts()"><option value="person">person 단일</option><option value="animal">animal 단일</option><option value="multi_person_animal">person + animal 동시</option><option value="non_human">non_human 단일</option></select></div><div class="controlRow"><label>pose</label><select id="poseClass"><option value="auto">auto</option><option value="standing">standing</option><option value="walking">walking</option><option value="horizontal_movement">horizontal_movement</option><option value="low_movement">low_movement</option><option value="transition">transition</option><option value="unknown">unknown</option></select></div><div class="small">단일 사람 데이터는 <b>person 단일</b>, 사람+동물 데이터는 <b>person + animal 동시</b>로 Load. v17에서는 Load 전에 person/animal/non_human 개수를 정하고, 자동 박스 생성 단계부터 그 개수만큼만 object를 만듦.</div>
<h3>0-1) Load-time Object Count</h3><div class="countBox"><div class="small"><b>Load 전에 실제 개체 수를 입력</b>. 이제 자동 박스 생성 단계부터 이 개수만큼만 object를 만듦. 이미 Load한 뒤에는 Apply slots로 현재 frame만 다시 맞출 수 있음.</div><div class="countGrid"><div class="countCard"><label>person</label><input id="count_person" class="countInput" type="number" min="0" max="9" step="1" value="1" oninput="updateCountStatus()"></div><div class="countCard"><label>animal</label><input id="count_animal" class="countInput" type="number" min="0" max="9" step="1" value="0" oninput="updateCountStatus()"></div><div class="countCard"><label>non_human</label><input id="count_non_human" class="countInput" type="number" min="0" max="9" step="1" value="0" oninput="updateCountStatus()"></div></div><div class="row"><button class="primary" onclick="applyCountSlots()">Apply object slots</button><button onclick="presetPersonOne()">1 person</button><button onclick="setEmptyFrame()">empty frame</button></div><div id="countStatus" class="small statusNeutral"></div></div>
<h3>1) 보기 기준</h3><div class="controlRow"><label>view</label><select id="view"><option value="review">검수 필요만</option><option value="all">전체 프레임</option><option value="corrected">내가 처리한 것</option></select></div><div class="controlRow"><label>confidence</label><select id="conf"><option value="all">전체</option><option>high</option><option>medium</option><option>interpolated</option><option>low</option><option>empty</option><option>error</option></select></div><div class="controlRow"><label>사용 판단</label><select id="recommend"><option value="all">전체</option><option value="auto_use">사용 권장</option><option value="quick_check">빠른 확인 후 사용</option><option value="careful_check">신중 검수</option><option value="exclude">제외 권장</option></select></div><div class="filterButtons"><button onclick="applyFilters()">필터 적용</button><button onclick="move(-1)">Prev</button><button onclick="move(1)">Next</button></div>
<details><summary>보기 기준 / confidence / 사용 판단 설명</summary><div class="small"><b>view</b>: 어떤 frame 묶음을 볼지 선택<br>· 검수 필요만 = high 사용권장 제외<br>· 전체 프레임 = high 포함 전체 frame<br>· 내가 처리한 것 = Accept/Reject/Uncertain/수정 저장한 frame만<br><br><b>confidence</b>: 자동 라벨 신뢰도. high일수록 box 근거가 강함.<br><br><b>사용 판단</b>: 지금 frame을 학습 라벨로 써도 되는지에 대한 권장 처리.</div><div class="helpBox small" id="methodHelp"></div></details>
<div class="kv" id="meta"></div>
<h3>3) Object / Box 편집</h3><div class="objBox"><div class="small"><b>현재 frame의 object slot</b></div><div id="objectList" class="objList"></div><div class="objControl"><label>class</label><select id="editClass" onchange="updateActiveObjectMeta()"><option>candidate</option><option>person</option><option>animal</option><option>non_human</option></select></div><div class="objControl"><label>pose</label><select id="editPose" onchange="updateActiveObjectMeta()"><option>unknown</option><option>standing</option><option>walking</option><option>horizontal_movement</option><option>low_movement</option><option>transition</option><option>sitting</option><option>lying_down</option><option>falling</option></select></div><div class="row"><button onclick="addObject()">+ Add Box</button><button onclick="deleteObject()" class="bad">Delete Box</button></div><div class="small">여러 개체가 있으면 object 버튼을 눌러 각각 class/pose/box를 수정.</div></div>
<div id="boxSliders"></div><textarea class="note" id="note" placeholder="note"></textarea>
<h3>4) 검수 처리</h3><div class="actionGrid"><button class="good" onclick="saveAction('accept',true)">Accept<br>+ Next</button><button class="bad" onclick="saveAction('reject',true)">Reject<br>+ Next</button><button class="warn" onclick="saveAction('uncertain',true)">Uncertain<br>+ Next</button><button class="primary" onclick="saveAction('adjust',true)">수정값 저장<br>(Accept)+Next</button></div><div class="small tight">슬라이더 변경 = 오른쪽 view 즉시 미리보기. v11은 point cloud 기준으로 화면을 먼저 맞춰서 큰 박스가 있어도 점이 작게 사라지지 않게 표시함. 반영하려면 수정값 저장 버튼을 누름.</div>
<h3>5) Export</h3><div class="row"><button onclick="exportFinal('accepted_only')">Export accepted only</button></div><div class="row"><button class="primary" onclick="exportFinal('all_frames_schema')">Export 전체 프레임 schema(objects: [] 포함)</button></div><div class="row"><button onclick="exportFinal('accepted_plus_high')">Export accepted + 사용권장</button></div><div class="row"><button onclick="exportFinal('all_except_rejected')">Export all except rejected</button></div><div id="exportResult" class="small"></div></aside>
<section><div class="panel"><div class="contextStrip" id="contextStrip"></div><div class="views"><div class="svgbox" id="perspective"></div><div class="svgbox" id="top"></div><div class="svgbox" id="side"></div><div class="svgbox" id="front"></div></div></div><div class="panel"><h3>Export Label JSON preview</h3><pre id="json"></pre></div></section></main><div id="statsTable" class="statsFloat small">Load 후 confidence 개수표가 여기에 표시됨.</div>
<script>
let current=null;let previewTimer=null;let editObjects=[];let activeObj=0;const keys={x:[-8,8,.01],y:[-2,16,.01],z:[-2,4,.01],w:[.1,4,.01],l:[.1,4,.01],h:[.2,3,.01],yaw:[-3.14,3.14,.01]};
const defaultBox=()=>({position:{x:0,y:5,z:.8},dimensions:{w:.8,l:.8,h:1.6},rotation:{yaw:0}});
async function api(path,body=null){const opt=body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{};const r=await fetch(path,opt);return await r.json()}
function badge(t,cls){return `<span class="badge ${cls||t}">${t}</span>`}
function clone(o){return JSON.parse(JSON.stringify(o))}
function normalizeObj(o,i){return {object_id:`obj_${String(i+1).padStart(3,'0')}`,object_class:o.object_class||'candidate',pose_class:o.pose_class||'unknown',box:o.box||defaultBox(),confidence:o.confidence||'manual',label_source:o.label_source||'manual_adjusted',quality:o.quality||{}}}
const MODEL_CLASSES=['person','animal','non_human'];
function intVal(id){return Math.max(0,parseInt(document.getElementById(id).value||'0',10)||0)}
function getExpectedCounts(){return {person:intVal('count_person'),animal:intVal('count_animal'),non_human:intVal('count_non_human')}}
function setExpectedCounts(c){c=c||{};document.getElementById('count_person').value=c.person||0;document.getElementById('count_animal').value=c.animal||0;document.getElementById('count_non_human').value=c.non_human||0;updateCountStatus()}
function modePresetCounts(){const m=document.getElementById('objectClass').value;if(m==='person')setExpectedCounts({person:1,animal:0,non_human:0});else if(m==='animal')setExpectedCounts({person:0,animal:1,non_human:0});else if(m==='non_human')setExpectedCounts({person:0,animal:0,non_human:1});else if(m.includes('multi'))setExpectedCounts({person:1,animal:1,non_human:0});}
function validClass(c){return MODEL_CLASSES.includes(c)}
function countObjects(objs){let c={person:0,animal:0,non_human:0,candidate:0};(objs||[]).forEach(o=>{let cls=o.object_class||o.class||'candidate';if(validClass(cls))c[cls]++;else c.candidate++});return c}
function countsEqual(a,b){return MODEL_CLASSES.every(k=>(a[k]||0)===(b[k]||0))}
function deriveCountsForCurrent(l){let corr=(current&&current.correction)||{};if(corr.expected_counts)return corr.expected_counts;if(l.expected_counts)return l.expected_counts;let objs=(l.objects||[]);let mode=document.getElementById('objectClass').value;let c={person:0,animal:0,non_human:0};if(!objs.length)return c;if(!mode.includes('multi')){let cls=validClass(mode)?mode:'person';c[cls]=1;return c}let actual=countObjects(objs);MODEL_CLASSES.forEach(k=>c[k]=actual[k]||0);return c}
function updateCountStatus(){let expected=getExpectedCounts();let actual=countObjects(editObjects);let total=MODEL_CLASSES.reduce((a,k)=>a+expected[k],0);let ok=countsEqual(expected,actual)&&actual.candidate===0;let el=document.getElementById('countStatus');if(!el)return;el.className='small '+(ok?'statusOk':'statusWarn');el.innerHTML=ok?`OK · export objects = ${total}`:`count mismatch · expected person ${expected.person}, animal ${expected.animal}, non_human ${expected.non_human} / current person ${actual.person}, animal ${actual.animal}, non_human ${actual.non_human}${actual.candidate?`, candidate ${actual.candidate}`:''}`}
function applyCountSlots(){let counts=getExpectedCounts();let src=editObjects.map(clone);let used=new Set();let slots=[];function take(cls){for(let i=0;i<src.length;i++){if(!used.has(i)&&(src[i].object_class||'candidate')===cls)return i}for(let i=0;i<src.length;i++){if(!used.has(i)&&!validClass(src[i].object_class||'candidate'))return i}for(let i=0;i<src.length;i++){if(!used.has(i))return i}return -1}MODEL_CLASSES.forEach(cls=>{for(let n=0;n<(counts[cls]||0);n++){let idx=take(cls);let base=idx>=0?clone(src[idx]):{box:(src[0]&&src[0].box)||defaultBox(),confidence:'manual',label_source:'manual_slot'};if(idx>=0)used.add(idx);base.object_class=cls;base.pose_class=base.pose_class||((document.getElementById('poseClass').value==='auto')?'unknown':document.getElementById('poseClass').value);slots.push(base)}});editObjects=slots.map(normalizeObj);activeObj=0;renderObjectList();queuePreview()}
function presetPersonOne(){setExpectedCounts({person:1,animal:0,non_human:0});applyCountSlots()}
function setEmptyFrame(){setExpectedCounts({person:0,animal:0,non_human:0});editObjects=[];activeObj=0;renderObjectList();queuePreview()}
function makeSliders(){let h='';for(const k of ['x','y','z','w','l','h','yaw']){const b=keys[k];h+=`<div class="sliderRow"><label>${k}</label><input id="range_${k}" type="range" min="${b[0]}" max="${b[1]}" step="${b[2]}" oninput="syncRange('${k}')"><input class="num" id="${k}" oninput="syncNum('${k}')"></div>`}document.getElementById('boxSliders').innerHTML=h}
function setVal(k,v){document.getElementById(k).value=(+v||0).toFixed(4);document.getElementById('range_'+k).value=+v||0}
function syncRange(k){document.getElementById(k).value=Number(document.getElementById('range_'+k).value).toFixed(4);writeSlidersToActive();queuePreview()}
function syncNum(k){document.getElementById('range_'+k).value=document.getElementById(k).value;writeSlidersToActive();queuePreview()}
function sliderBox(){return {position:{x:Number(x.value||0),y:Number(y.value||0),z:Number(z.value||0)},dimensions:{w:Number(w.value||.8),l:Number(l.value||.8),h:Number(h.value||1.6)},rotation:{yaw:Number(yaw.value||0)}}}
function setSlidersFromObj(){let o=editObjects[activeObj];if(!o){setVal('x',0);setVal('y',5);setVal('z',.8);setVal('w',.8);setVal('l',.8);setVal('h',1.6);setVal('yaw',0);return}let b=o.box||defaultBox();setVal('x',b.position.x);setVal('y',b.position.y);setVal('z',b.position.z);setVal('w',b.dimensions.w);setVal('l',b.dimensions.l);setVal('h',b.dimensions.h);setVal('yaw',(b.rotation||{}).yaw||0);document.getElementById('editClass').value=o.object_class||'candidate';document.getElementById('editPose').value=o.pose_class||'unknown'}
function writeSlidersToActive(){if(!editObjects[activeObj])return;editObjects[activeObj].box=sliderBox();editObjects=editObjects.map(normalizeObj);renderObjectList(false)}
function updateActiveObjectMeta(){if(!editObjects[activeObj])return;editObjects[activeObj].object_class=document.getElementById('editClass').value;editObjects[activeObj].pose_class=document.getElementById('editPose').value;editObjects=editObjects.map(normalizeObj);renderObjectList(false);queuePreview()}
function selectObject(i){activeObj=Math.max(0,Math.min(i,editObjects.length-1));renderObjectList();setSlidersFromObj();queuePreview()}
function renderObjectList(updateSliders=true){const el=document.getElementById('objectList');if(!editObjects.length){el.innerHTML='<span class="small dangerText">object slot 없음. count가 0이면 empty frame으로 export됨.</span>';updateCountStatus();return}el.innerHTML=editObjects.map((o,i)=>`<button class="objBtn ${i===activeObj?'active':''}" onclick="selectObject(${i})">${i+1}. ${o.object_class||'candidate'} / ${o.pose_class||'unknown'}</button>`).join('');if(updateSliders)setSlidersFromObj();updateCountStatus()}
function addObject(){let obj=normalizeObj({object_class:document.getElementById('objectClass').value.includes('multi')?'candidate':document.getElementById('objectClass').value,pose_class:document.getElementById('poseClass').value==='auto'?'unknown':document.getElementById('poseClass').value,box:sliderBox()},editObjects.length);editObjects.push(obj);activeObj=editObjects.length-1;renderObjectList();queuePreview()}
function deleteObject(){if(!editObjects.length)return;editObjects.splice(activeObj,1);activeObj=Math.max(0,activeObj-1);editObjects=editObjects.map(normalizeObj);renderObjectList();queuePreview()}
function correctionBody(action){return {action,note:document.getElementById('note').value,objects:editObjects.map(normalizeObj),expected_counts:getExpectedCounts()}}
async function init(){makeSliders();let s=await api('/api/datasets');document.getElementById('dataset').innerHTML=s.datasets.map(x=>`<option>${x}</option>`).join('');modePresetCounts();updateSummary(s.stats)}
function countRows(counts){const order=['high','medium','interpolated','low','empty','error'];let keys=[...order.filter(k=>counts&&counts[k]!=null),...Object.keys(counts||{}).filter(k=>!order.includes(k)).sort()];let total=keys.reduce((a,k)=>a+(counts[k]||0),0);return keys.map(k=>`<tr><td><code>${k}</code></td><td>${counts[k]}</td></tr>`).join('')+`<tr><th>합계</th><th>${total}</th></tr>`}
function renderStatsTable(s){if(!s||!s.loaded){document.getElementById('statsTable').innerHTML='Load 후 confidence 개수표가 여기에 표시됨.';return}document.getElementById('statsTable').innerHTML=`<b>Confidence 개수</b><table class="statsTable"><tr><th>confidence</th><th>개수</th></tr>${countRows(s.confidence_counts)}</table><br><b>high 제외 검수 후보</b><table class="statsTable"><tr><th>confidence</th><th>개수</th></tr>${countRows(s.non_high_confidence_counts)}</table>`}
function updateSummary(s){document.getElementById('summary').innerHTML=s&&s.loaded?`${s.loaded} / frames ${s.frame_count} / non-high ${s.non_high_total}`:'dataset not loaded';renderStatsTable(s)}
async function loadDataset(){const z=document.getElementById('dataset').value;const obj=document.getElementById('objectClass').value;const pose=document.getElementById('poseClass').value;let s=await api('/api/load',{zip:z,object_class:obj,pose_class:pose==='auto'?null:pose,expected_counts:getExpectedCounts()});updateSummary(s);await reload()}
async function applyFilters(){await api('/api/filter',{view:document.getElementById('view').value,confidence:document.getElementById('conf').value,recommendation:document.getElementById('recommend').value});await reload()}
async function move(d){await api('/api/move',{delta:d});await reload()}
function renderViews(svgs){for(const p of ['perspective','top','side','front'])document.getElementById(p).innerHTML=svgs[p]||''}
function renderContext(items){const el=document.getElementById('contextStrip');if(!el)return;el.innerHTML=(items||[]).map(it=>`<div class="contextCard"><div class="contextTitle"><span>${it.title}</span>${badge(it.confidence,it.confidence)}</div><div class="contextMeta">${it.frame_id} · objects ${it.object_count}</div><div>${it.svg||''}</div></div>`).join('')}
function queuePreview(){if(!current||current.empty)return;clearTimeout(previewTimer);previewTimer=setTimeout(previewBox,120)}
async function previewBox(){let r=await api('/api/preview',{objects:editObjects.map(normalizeObj),expected_counts:getExpectedCounts()});if(r.empty)return;renderViews(r.svgs);document.getElementById('json').textContent=JSON.stringify(r.model_label||r.label,null,2);updateCountStatus()}
async function reload(){current=await api('/api/current');if(current.empty){document.getElementById('meta').innerHTML='표시할 frame 없음';return}renderViews(current.svgs);renderContext(current.context_views);const l=current.label;const ml=current.model_label||l;updateSummary(current.stats);editObjects=(l.objects||[]).map(normalizeObj);activeObj=0;setExpectedCounts(deriveCountsForCurrent(l));renderObjectList();document.getElementById('meta').innerHTML=`<div>queue</div><div>${current.pos}/${current.total} · all ${current.global_index}/${current.global_total}</div><div>frame</div><div>${l.frame_id}</div><div>confidence</div><div>${badge(l.confidence)}</div><div>사용 판단</div><div>${badge(current.recommendation_text,l.recommendation)}</div><div>보정 방식</div><div>${current.method_text}</div><div>record</div><div>${l.record}</div><div>objects</div><div>${(l.objects||[]).length}</div>`;document.getElementById('methodHelp').innerHTML=`<b>${current.method_text}</b><br>${current.recommendation_help}`;document.getElementById('json').textContent=JSON.stringify(ml,null,2);document.getElementById('note').value=(current.correction&&current.correction.note)||''}
async function saveAction(action,next=false){let body;if(action==='reject'){body={action,note:document.getElementById('note').value,objects:[],expected_counts:{person:0,animal:0,non_human:0}}}else if(action==='uncertain'){body={action,note:document.getElementById('note').value,expected_counts:getExpectedCounts()}}else{body=correctionBody(action)}await api('/api/correction',body);if(next)await api('/api/move',{delta:1});await reload()}
async function exportFinal(mode){let r=await api('/api/export',{mode});let g=r.count_gate||{};document.getElementById('exportResult').innerHTML=`saved: ${r.path}<br>frames: ${r.count} / objects: ${r.object_count}<br>validation: manual ${g.manual_gated_frames||0}, derived ${g.derived_gated_frames||0}, trimmed objects ${g.trimmed_objects||0}, missing objects ${g.missing_objects||0}, invalid frames ${g.invalid_frames||0}<br>corrections: ${r.corrections_csv}`}
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1);if(e.key==='a')saveAction('accept',true);if(e.key==='r')saveAction('reject',true);if(e.key==='u')saveAction('uncertain',true);if(e.key==='s')saveAction('adjust',true)});init();
</script>
<script>
(function(){
  const oldFetch = window.fetch;
  window.__wb_set_status = function(msg){
    let el=document.getElementById('loadStatusBox');
    if(!el){
      el=document.createElement('div');
      el.id='loadStatusBox';
      el.style.cssText='position:fixed;right:18px;top:72px;z-index:99999;background:#111;color:#fff;padding:10px 12px;border-radius:10px;font:14px system-ui;box-shadow:0 6px 20px #0003;display:none';
      document.body.appendChild(el);
    }
    el.textContent=msg;
    el.style.display=msg?'block':'none';
  };
  window.fetch = async function(input, init){
    try{
      const url = (typeof input === 'string') ? input : (input && input.url) || '';
      if(url.includes('/api/load')) window.__wb_set_status('Loading dataset... 대용량이면 30초~2분 정도 걸릴 수 있음');
      const res = await oldFetch(input, init);
      if(url.includes('/api/load')) window.__wb_set_status('');
      return res;
    }catch(e){
      window.__wb_set_status('Load 실패: 터미널 오류 메시지를 확인해줘');
      throw e;
    }
  };
})();
</script>
</body></html>

"""

def json_resp(h: BaseHTTPRequestHandler, data: Dict[str,Any], status: int=200) -> None:
    raw=json.dumps(data,ensure_ascii=False).encode('utf-8')
    h.send_response(status); h.send_header('Content-Type','application/json; charset=utf-8'); h.send_header('Content-Length',str(len(raw))); h.end_headers(); h.wfile.write(raw)

def body_json(h: BaseHTTPRequestHandler) -> Dict[str,Any]:
    n=int(h.headers.get('Content-Length','0') or '0')
    if n<=0: return {}
    try: return json.loads(h.rfile.read(n).decode('utf-8'))
    except Exception: return {}

def make_handler(wb: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None: return
        def do_GET(self):
            path=urlparse(self.path).path
            if path in ('/','/index.html'):
                
                ui_path = Path(__file__).with_name('index.html')
                if ui_path.exists():
                    raw = ui_path.read_bytes()
                else:
                    raw = INDEX_HTML.encode('utf-8')
                self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            elif path=='/api/datasets': json_resp(self,{"datasets":wb.list_datasets(),"stats":wb.stats()})
            elif path=='/api/current': json_resp(self,wb.current_payload())
            elif path=='/api/stats': json_resp(self,wb.stats())
            else: json_resp(self,{"error":"not found"},404)
        def do_POST(self):
            path=urlparse(self.path).path; b=body_json(self)
            if path=='/api/load': json_resp(self, wb.load_dataset(b.get('zip',''), b.get('object_class','person'), b.get('pose_class'), b.get('expected_counts')))
            elif path=='/api/filter': wb.filter_view=b.get('view','review'); wb.filter_conf=b.get('confidence','all'); wb.filter_recommend=b.get('recommendation','all'); wb.pos=0; json_resp(self,{"ok":True})
            elif path=='/api/move': wb.pos += int(b.get('delta',0)); idxs=wb.filtered_indices(); wb.pos=int(clamp(wb.pos,0,max(0,len(idxs)-1))); json_resp(self,{"ok":True})
            elif path=='/api/correction': json_resp(self, wb.save_correction(b.get('action','accept'), b))
            elif path=='/api/preview': json_resp(self, wb.preview_payload(b))
            elif path=='/api/export': json_resp(self, wb.export_review(b.get('mode','all_frames_schema'), minimal=bool(b.get('minimal', False))))
            else: json_resp(self,{"error":"not found"},404)
    return Handler

def free_port(start: int) -> int:
    for p in range(start,start+80):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            try: s.bind(('127.0.0.1',p)); return p
            except OSError: pass
    return start

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='.', help='dataset zip들이 있는 폴더. 기본값: 현재 폴더')
    ap.add_argument('--out', default='workbench_outputs')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8787)
    ap.add_argument('--no-browser', action='store_true')
    args=ap.parse_args()
    wb=Workbench(Path(args.data_dir).resolve(), Path(args.out).resolve(), RuleParams())
    port=free_port(args.port); url=f'http://{args.host}:{port}'
    print(f'Data dir: {wb.data_dir}')
    print(f'Output dir: {wb.out_dir}')
    print(f'Open: {url}')
    if not args.no_browser: webbrowser.open(url)
    server=ThreadingHTTPServer((args.host,port), make_handler(wb))
    try: server.serve_forever()
    except KeyboardInterrupt: print('\nStopping.')
    finally: server.server_close()

if __name__=='__main__':
    main()
