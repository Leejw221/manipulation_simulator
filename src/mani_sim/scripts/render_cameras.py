"""MJCF 씬에 정의된 카메라(전체 또는 지정)로 렌더링한 이미지를 PNG로 저장 — GUI 없이
카메라 시점을 빠르게 확인하기 위한 도구. coord_picker.py와 마찬가지로 아무 MJCF에나 씀.

사용:
    python -m mani_sim.scripts.render_cameras --mjcf scene.xml --out-dir /path/to/out
    python -m mani_sim.scripts.render_cameras --mjcf scene.xml --out-dir /path/to/out \
        --cameras agentview head_camera
"""

import argparse
import os

import mujoco
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mjcf", required=True, help="로드할 MJCF(.xml) 파일 경로")
    p.add_argument("--out-dir", required=True, help="PNG 저장할 디렉터리")
    p.add_argument("--cameras", nargs="*", default=None,
                    help="렌더링할 카메라 이름(생략 시 씬에 정의된 전체 카메라)")
    p.add_argument("--size", type=int, default=480, help="렌더링 해상도(정사각형)")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    cam_names = args.cameras or [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)
    ]
    if not cam_names:
        print("[render_cameras] 씬에 정의된 카메라가 없음")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.size, width=args.size)
    for cn in cam_names:
        renderer.update_scene(data, camera=cn)
        img = renderer.render()
        out_path = os.path.join(args.out_dir, f"cam_{cn}.png")
        Image.fromarray(img).save(out_path)
        print(f"[render_cameras] {cn} -> {out_path}")


if __name__ == "__main__":
    main()
