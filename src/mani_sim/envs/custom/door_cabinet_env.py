"""DoorCabinet: 문 열기 -> 문 닫기 -> 초기 자세로 복귀, 한 에피소드 안에.

robosuite 내장 Door 태스크(environments/manipulation/door.py)를 템플릿으로 확장했다. Door와의
차이: 성공 판정에 "닫힌 채로 시작 -> 한 번은 열림 -> 다시 닫힘 -> 초기 자세 근처로 복귀"를
요구해 open-only로 완주되는 것을 막았다 (state-aliasing 실험의 핵심 조건).

같은 손잡이 접근 관측이 에피소드 안에서 "열 때"와 "닫을 때" 두 번 나오는 게 이 태스크의
목적이다 - stage 입력이 실제로 필요한지(교수님이 원래 검증하고 싶어했던 것) 확인하는 최소 구조.

2026-07-18: 원래 물체(큐브)를 옮기는 단계를 추가했었으나(문 뒤로 옮기기), 물체가 실제로
"안쪽"에 들어갈 공간이 없고(DoorObject는 안쪽 공간이 없는 경첩 판) 목표 위치의 도달 가능성
검증도 신뢰할 수 없어서(단순 스크립트 추종이 실제 PICO 조작보다 훨씬 못함) 제거했다. 물체
없이 문 자체의 왕복만으로 aliasing 조건은 이미 충분하다(교수님 미팅 원문 분석 및 세션 논의,
research_design.md 참고).
"""

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import DoorObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler

# Door._check_success()와 동일한 임계값 관례(열림 판정)
OPEN_THRESHOLD = 0.3
# 닫힘 판정은 열림보다 더 타이트하게 (문이 살짝만 밀려도 "닫힘"으로 오판하지 않도록)
CLOSED_THRESHOLD = 0.05
# 에피소드 종료(복귀 판정): eef가 초기 위치에서 이 거리 이내로 돌아오면 "복귀"로 판정
RETURN_DIST_THRESHOLD = 0.08


class DoorCabinet(ManipulationEnv):
    """문 열기 -> 문 닫기 -> 초기 자세로 복귀. 단일팔 전용(Door와 동일 제약)."""

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mjviewer",
        renderer_config=None,
    ):
        self.table_full_size = (0.8, 0.3, 0.05)
        self.table_offset = (-0.2, -0.35, 0.8)

        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        self.use_object_obs = use_object_obs
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action=None):
        if self._check_success():
            return self.reward_scale
        if not self.reward_shaping:
            return 0.0
        return 0.25 * (1 - np.tanh(10.0 * np.linalg.norm(self._gripper_to_handle)))

    def _load_model(self):
        super()._load_model()

        # 커스텀 base 위치(-0.48,-0.28)는 문 관절과 충돌했다(2026-07-18 실기 확인) - 반대로
        # 오늘 밤 첫 실제 PICO 테스트(손잡이·큐브 둘 다 0.017m 정밀도로 성공, 충돌 없음)는
        # 이 오버라이드 없이 로보스위트 기본값(base_xpos_offset["table"])을 그대로 썼었다.
        # 짐작으로 더 손대지 않고 실측으로 검증된 기본값으로 되돌린다.
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5986131746834771, -4.392035683362857e-09, 1.5903500240372423],
            quat=[0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349],
        )

        self.door = DoorObject(name="Door", friction=0.0, damping=0.1, lock=False)

        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.door)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.door,
                x_range=[0.07, 0.09],
                y_range=[-0.01, 0.01],
                rotation=(-np.pi / 2.0 - 0.25, -np.pi / 2.0),
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
            )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.door],
        )

    def _setup_references(self):
        super()._setup_references()

        self.object_body_ids = dict()
        self.object_body_ids["door"] = self.sim.model.body_name2id(self.door.door_body)
        self.object_body_ids["frame"] = self.sim.model.body_name2id(self.door.frame_body)
        self.door_handle_site_id = self.sim.model.site_name2id(self.door.important_sites["handle"])
        self.hinge_qpos_addr = self.sim.model.get_joint_qpos_addr(self.door.joints[0])

    def _setup_observables(self):
        observables = super()._setup_observables()
        if not self.use_object_obs:
            return observables
        modality = "object"

        @sensor(modality=modality)
        def handle_pos(obs_cache):
            return self._handle_xpos

        @sensor(modality=modality)
        def hinge_qpos(obs_cache):
            return np.array([self.sim.data.qpos[self.hinge_qpos_addr]])

        arm_prefixes = self._get_arm_prefixes(self.robots[0], include_robot_name=False)
        full_prefixes = self._get_arm_prefixes(self.robots[0])
        sensors = [handle_pos, hinge_qpos]
        sensors += [
            self._get_obj_eef_sensor(full_pf, "handle_pos", f"handle_to_{arm_pf}eef_pos", modality)
            for arm_pf, full_pf in zip(arm_prefixes, full_prefixes)
        ]
        names = [s.__name__ for s in sensors]
        for name, s in zip(names, sensors):
            observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)

        return observables

    def _reset_internal(self):
        super()._reset_internal()
        self._door_was_opened = False
        # 초기 eef 위치 기록 - 에피소드가 "복귀"했는지 판정하는 기준점.
        arm = self.robots[0].arms[0]
        self._initial_eef_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])

        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            door_pos, door_quat, _ = object_placements[self.door.name]
            door_body_id = self.sim.model.body_name2id(self.door.root_body)
            self.sim.model.body_pos[door_body_id] = door_pos
            self.sim.model.body_quat[door_body_id] = door_quat

    def _check_success(self):
        """열림->닫힘 왕복 + 초기 자세 복귀가 실제로 일어났는지 추적.

        물체 없이 문 자체만으로 판정한다(2026-07-18, 물체 관련 문제로 제거 - 모듈
        docstring 참고). "복귀"까지 요구하는 이유: 그냥 열고 닫기만 하면 에피소드가 문
        바로 앞에서 끝나버려 "닫으려는 접근"과 "일상적으로 지나가는 접근"을 구분할
        근거가 약해짐 - 초기 자세로 돌아오는 것까지 포함하면 완결된 왕복이 된다.
        """
        hinge_qpos = self.sim.data.qpos[self.hinge_qpos_addr]
        if hinge_qpos > OPEN_THRESHOLD:
            self._door_was_opened = True
        door_closed_now = hinge_qpos < CLOSED_THRESHOLD
        arm = self.robots[0].arms[0]
        eef_pos = np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
        returned = np.linalg.norm(eef_pos - self._initial_eef_pos) < RETURN_DIST_THRESHOLD
        return bool(self._door_was_opened and door_closed_now and returned)

    @property
    def _handle_xpos(self):
        return self.sim.data.site_xpos[self.door_handle_site_id]

    @property
    def _gripper_to_handle(self):
        dists = []
        for arm in self.robots[0].arms:
            diff = self._handle_xpos - np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id[arm]])
            dists.append(np.linalg.norm(diff))
        return min(dists)
