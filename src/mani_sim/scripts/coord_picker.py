"""범용 MuJoCo 좌표 피커 — 어떤 MJCF든 로드해서 커서(노란 구)를 키보드로 움직이고,
**여러 개의 마커(초록 구)를 자유롭게 떨어뜨려가며** 여러 지점의 world 좌표를 한 번에
모을 수 있다.

오늘 밤 반복된 문제(스크린샷 보고 좌표 짐작 → 틀림 → 재시도)를 없애기 위한 도구 -
robosuite/mani_sim 전용이 아니라 **아무 MJCF 파일에나** 씀(내일 G1 씬에도 그대로 적용 가능).
로봇 컨트롤러·액션공간을 전혀 안 타서(mocap body를 직접 이동) 어떤 씬이든 동작한다.

"로봇/물건을 자유롭게 생성"은 지원 안 함 — 로봇은 로드하는 MJCF 자체에 이미 있어야 하고
(예: G1 씬 파일), 이 도구는 그 위에 "여기에 뭔가 놓을 거다"라는 위치 마커만 여러 개
찍는 용도. 정확한 좌표만 나오면 실제 물체(에셋)는 이후 코드에서 채워 넣는다.

사용:
    python -m mani_sim.scripts.coord_picker --mjcf /path/to/scene.xml --out positions.json

조작: 방향키=커서 XY 이동 · PageUp/PageDown=커서 Z 이동 · **S=현재 위치에 마커
      떨어뜨리고 좌표 저장**(초록 구가 씬에 남고, 커서는 계속 노란 채로 다음 지점 이동
      가능 — 여러 번 눌러서 여러 지점 모으기) · R=커서만 시작 위치로 리셋(찍은 마커는
      안 지워짐) · 창 닫으면 종료. 저장된 이름(point_N)은 나중에 positions.json에서
      직접 바꿔도 된다.
"""

import argparse
import json
import os

import mujoco
import mujoco.viewer

STEP = 0.01  # 한 번 누를 때 이동량(m)
MAX_MARKERS = 40  # 씬에 미리 심어두는 마커 슬롯 개수(MuJoCo는 런타임에 body 추가가 안 돼서
                   # 넉넉히 미리 만들어두고, 안 쓰는 건 화면 밖(z=-100)에 숨겨둔다)

# GLFW 키코드(표준, 안정적) — https://www.glfw.org/docs/latest/group__keys.html
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267
KEY_S, KEY_R = 83, 82
# PageUp/PageDown이 노트북 등에서 Fn 조합 필요 등으로 안 먹는 경우를 대비한 대체 키
# ( '[' = 91, ']' = 93 — 브래킷 키, 대부분 키보드에 있고 Fn 불필요)
KEY_BRACKET_LEFT, KEY_BRACKET_RIGHT = 91, 93


def _inject_bodies(mjcf_path, n_markers=MAX_MARKERS):
    """<worldbody> 바로 뒤에 커서(노랑) + 마커 풀(초록, 처음엔 화면 밖에 숨김)을 삽입한
    임시 파일을 같은 디렉터리에 만든다(상대경로 mesh/texture 참조가 안 깨지도록)."""
    with open(mjcf_path) as f:
        xml = f.read()

    extra = (
        '<body name="__cursor__" mocap="true" pos="0 0 1">'
        '<geom type="sphere" size="0.02" rgba="1 1 0 0.8" contype="0" conaffinity="0"/>'
        "</body>"
    )
    for i in range(n_markers):
        extra += (
            f'<body name="__marker_{i}__" mocap="true" pos="0 0 -100">'
            f'<geom type="sphere" size="0.018" rgba="0 1 0 0.85" contype="0" conaffinity="0"/>'
            "</body>"
        )

    if "<worldbody>" not in xml:
        raise ValueError(f"{mjcf_path}에 <worldbody> 태그가 없음 — 유효한 MJCF가 맞는지 확인")
    xml = xml.replace("<worldbody>", "<worldbody>" + extra, 1)

    src_dir = os.path.dirname(os.path.abspath(mjcf_path))
    tmp_path = os.path.join(src_dir, f".__coord_picker_tmp_{os.getpid()}.xml")
    with open(tmp_path, "w") as f:
        f.write(xml)
    return tmp_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", required=True, help="로드할 MJCF(.xml) 파일 경로")
    p.add_argument("--out", default="positions.json", help="저장할 좌표 JSON 경로")
    p.add_argument("--start", default="0,0,1", help="커서 시작 위치 'x,y,z'")
    args = p.parse_args()
    start_pos = [float(v) for v in args.start.split(",")]

    tmp_path = _inject_bodies(args.mjcf)
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.remove(tmp_path)
    data = mujoco.MjData(model)

    cursor_id = model.body("__cursor__").mocapid[0]
    marker_ids = [model.body(f"__marker_{i}__").mocapid[0] for i in range(MAX_MARKERS)]
    data.mocap_pos[cursor_id] = start_pos

    saved = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            saved = json.load(f)
        print(f"[coord_picker] 기존 {args.out}에서 {len(saved)}개 좌표 불러옴 "
              "(마커 풀엔 다시 안 띄움 — 새로 찍는 것만 이번 세션에 보임)")

    def key_callback(keycode):
        pos = data.mocap_pos[cursor_id]
        if keycode == KEY_LEFT:
            pos[1] += STEP
        elif keycode == KEY_RIGHT:
            pos[1] -= STEP
        elif keycode == KEY_UP:
            pos[0] += STEP
        elif keycode == KEY_DOWN:
            pos[0] -= STEP
        elif keycode in (KEY_PAGE_UP, KEY_BRACKET_RIGHT):
            pos[2] += STEP
        elif keycode in (KEY_PAGE_DOWN, KEY_BRACKET_LEFT):
            pos[2] -= STEP
        elif keycode == KEY_R:
            pos[:] = start_pos
        elif keycode == KEY_S:
            if len(saved) >= MAX_MARKERS:
                print(f"[coord_picker] 마커 슬롯({MAX_MARKERS}개) 다 씀 — 더 못 찍음")
                return
            name = f"point_{len(saved) + 1}"
            saved[name] = [round(float(v), 4) for v in pos]
            data.mocap_pos[marker_ids[len(saved) - 1]] = pos  # 초록 마커로 그 자리에 남김
            with open(args.out, "w") as f:
                json.dump(saved, f, indent=2, ensure_ascii=False)
            print(f"[저장] {name} = {saved[name]}  (총 {len(saved)}개, {args.out})")
            return
        else:
            # 매핑 안 된 키 — Z 이동이 안 될 때 실제 keycode 확인용 디버그 출력
            print(f"[디버그] 인식 못한 키 keycode={keycode}")
            return
        print(f"[커서] x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}")

    print(
        "조작: 방향키=커서 XY 이동 · PageUp/PageDown 또는 [ ]=커서 Z 이동 · S=마커 떨어뜨리고 저장"
        "(여러 번 가능) · R=커서 리셋 · 창 닫으면 종료."
    )
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
