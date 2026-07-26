"""piper_sort_return task용 eval/rollout 어댑터.

mani_sim_external/piper_capstone에 vendored된 PiperMujocoEnv(raw MuJoCo, robosuite
아님 - FLARE/Cashier_policy 원본)를 감싸서 make_eval_env/rollout_policy/collect.py가
기대하는 계약(reset()->obs dict, step(action)->(obs,reward,done,info),
is_success()->{"task": bool}, env.env.close())을 만족시킨다.

FLARE 원본은 python>=3.12 + lerobot/placo(텔레옵 전용) 의존인데, mani_sim의 학습·평가
환경(robomimic, python 3.10)엔 그 의존성이 없다 - 물리 시뮬레이션 자체(PiperMujocoEnv)는
numpy+mujoco만 있으면 되므로(torch도 optional), 파일을 그대로 복사해 두고 그것만 쓴다
(2026-07-26). 데이터 수집(텔레옵)은 별도 conda env(piper_collect)에서 flare를 그대로
import해서 진행 - 이 어댑터와는 무관.
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

_PIPER_CAPSTONE_DIR = Path(__file__).resolve().parents[4] / "mani_sim_external" / "piper_capstone"
if str(_PIPER_CAPSTONE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPER_CAPSTONE_DIR))
from piper_mujoco_env import PiperMujocoEnv  # noqa: E402

# spot_i<->zone_i 1:1 대응(2026-07-26 확정 설계) - piper_sort_return.xml의 실제 좌표와 일치시킬 것.
SPOT_ZONE_XY = {
    1: {"spot": np.array([0.24, 0.18]), "zone": np.array([0.42, 0.18])},
    2: {"spot": np.array([0.24, 0.0]), "zone": np.array([0.42, 0.0])},
    3: {"spot": np.array([0.24, -0.18]), "zone": np.array([0.42, -0.18])},
}
_VISIT_THRESH = 0.03  # m - 패드 half-size(0.028)와 정합


class PiperSortReturnEnv:
    """block1/2/3을 각자 spot_i -> zone_i -> spot_i로 왕복시키는 task의 eval 환경.

    성공 조건: 3개 블록 전부 "자기 zone을 한 번이라도 방문(visited)"한 뒤 "에피소드
    종료 시점에 자기 spot으로 복귀"해 있어야 한다. 방문 이력 없이 애초에 안 움직인
    경우(정책이 아무것도 안 함)를 성공으로 오판하지 않기 위해 방문 여부를 별도로 추적한다.
    """

    def __init__(self, xml_path, camera_names=None, image_size=(84, 84)):
        self._env = PiperMujocoEnv(
            robot_mode="single",
            xml_path=str(xml_path),
            camera_names=camera_names,
            image_size=image_size,
            enable_neck=False,
        )
        self.env = self  # eval.py/collect.py가 쓰는 `env.env.close()`/`env.env.has_renderer` 관례 호환용
        self._visited_zone = {i: False for i in SPOT_ZONE_XY}
        self._render_hw = None
        self._renderer = None

    def _block_xy(self, i):
        bid = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_BODY, f"block{i}")
        return self._env.data.xpos[bid][:2].copy()

    def _update_visit_tracking(self):
        for i, xy in SPOT_ZONE_XY.items():
            if np.linalg.norm(self._block_xy(i) - xy["zone"]) < _VISIT_THRESH:
                self._visited_zone[i] = True

    def _to_obs(self, dp_obs, raw_images):
        obs = {"state": np.asarray(dp_obs["observation.state"], dtype=np.float32)}
        for key, img in raw_images.items():
            obs[f"{key}_image"] = img  # (H,W,3) uint8 - rollout.py._to_chw01이 알아서 변환
        return obs

    def reset(self):
        self._visited_zone = {i: False for i in SPOT_ZONE_XY}
        dp_obs, raw_images = self._env.reset()
        self._update_visit_tracking()
        return self._to_obs(dp_obs, raw_images)

    def step(self, action):
        dp_obs, raw_images = self._env.step(np.asarray(action, dtype=np.float64))
        self._update_visit_tracking()
        return self._to_obs(dp_obs, raw_images), 0.0, False, {}

    def is_success(self):
        all_returned = all(
            np.linalg.norm(self._block_xy(i) - xy["spot"]) < _VISIT_THRESH
            for i, xy in SPOT_ZONE_XY.items()
        )
        task_success = all_returned and all(self._visited_zone.values())
        return {"task": task_success}

    def render(self, mode="rgb_array", height=240, width=320, camera_name="front_cam"):
        # eval.py의 save_gif 경로가 쓰는 시그니처(robosuite EnvRobosuite.render와 동일 관례)에 맞춤.
        if self._renderer is None or self._render_hw != (height, width):
            self._renderer = mujoco.Renderer(self._env.model, height=height, width=width)
            self._render_hw = (height, width)
        self._renderer.update_scene(self._env.data, camera=camera_name)
        return self._renderer.render()

    def close(self):
        self._env.disconnect()

    def attach_viewer(self):
        """자유 시점 3D 씬 뷰어(mujoco 온스크린, GLFW) - eval 때 보던 것과 같은 종류.
        PiperMujocoEnv.apply_action()이 매 스텝 알아서 self._sync_viewer()를 호출하므로
        (원본 코드, 손 안 댐) self._env.viewer에 핸들만 꽂아두면 그 뒤론 자동 갱신된다.
        조작 기기(키보드/PICO)와 무관하게 이 env 자체가 아는 시각화라 여기 둔다
        (2026-07-26 - 기기 구현 파일에서 env._env 내부를 직접 건드리던 걸 분리).

        ⚠ 실측(2026-07-26, "moai-pobi"): NVIDIA 드라이버/커널모듈 버전 불일치로 실패한 적
        있음(재부팅으로 해결, HANDOFF.md 참고) - 재발 대비로 실패해도 죽지 않게 감싼다."""
        import logging

        import mujoco.viewer

        try:
            self._env.viewer = mujoco.viewer.launch_passive(self._env.model, self._env.data)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"mujoco 온스크린 뷰어 생성 실패({type(e).__name__}: {e}) - rerun 시각화만으로 "
                "계속 진행합니다."
            )

    def close_viewer(self):
        """attach_viewer()로 띄운 창을 닫는다 - collect.py/teleoperate.py가 종료 시 공유 호출."""
        if self._env.viewer is not None:
            self._env.viewer.close()

    def cycle_camera(self):
        """자유시점 <-> front_cam 토글(2026-07-26, PICO A버튼용) - attach_viewer()로 만든
        온스크린 뷰어의 카메라를 코드에서 직접 바꾼다(뷰어 자체 키보드 단축키 대신 컨트롤러
        버튼으로 조작하려는 용도). 뷰어가 없으면(attach_viewer 실패/미호출) 아무 것도 안 함."""
        import mujoco

        viewer = self._env.viewer
        if viewer is None:
            return
        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            cam_id = mujoco.mj_name2id(self._env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
