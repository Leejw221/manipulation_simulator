"""robosuite mjviewer 위에 시점 프리셋 순환을 얹는 헬퍼 (텔레오퍼레이션 깊이 판단 보조).

시뮬은 실물처럼 고개를 돌려 시점을 바꿀 수 없어, 2D 단일 시점만으론 그리퍼가 블록 위에
있는지(앞뒤·높이)를 판단하기 어렵다. 그래서 버튼 하나로 각진뷰↔top뷰↔측면뷰를 순환한다:
top으로 XY 정렬을 맞추고, front/angled/side로 높이를 보며 내린다.

robosuite의 자체 뷰어(env.render → MjviewerRenderer)를 쓴다. robosuite는 env.reset()마다
sim/data를 새로 만들고 뷰어를 부순 뒤 다음 render에서 새 sim에 재바인딩하므로, 직접
launch_passive로 잡은 뷰어는 reset 후 죽은 데이터를 보여준다(화면 멈춤). 그래서 여기선
robosuite 뷰어를 그대로 쓰고 카메라만 만진다. reset로 뷰어가 새로 생기면 현재 프리셋을
1회 재적용하고, 그 외엔 건드리지 않아 마우스 시점 조작을 방해하지 않는다.
"""

# lookat = 작업공간 중심(테이블 위). 마우스로도 시점 조절 가능.
CAM_PRESETS = [
    ("front", dict(azimuth=180, elevation=-10, distance=1.6, lookat=[0.0, 0.0, 0.9])),
    ("angled", dict(azimuth=150, elevation=-25, distance=1.6, lookat=[0.0, 0.0, 0.9])),
    ("top", dict(azimuth=180, elevation=-89, distance=1.4, lookat=[0.0, 0.0, 0.85])),
    ("side", dict(azimuth=90, elevation=-10, distance=1.6, lookat=[0.0, 0.0, 0.9])),
]


class CameraCycler:
    """robosuite env.render()를 호출하며 시점 프리셋을 순환 적용한다.

    render()를 매 스텝 부르고, cycle()로 다음 시점으로 넘긴다. 창이 닫히면 render()가
    False를 반환한다.
    """

    def __init__(self, env, presets=CAM_PRESETS):
        self.env = env  # robomimic EnvRobosuite 래퍼 (env.env = raw robosuite)
        self.presets = presets
        self.idx = 0
        self._last_handle = None

    def _handle(self):
        """현재 robosuite passive 뷰어 핸들 (없으면 None)."""
        renderer = getattr(self.env.env, "viewer", None)
        return getattr(renderer, "viewer", None)

    def _apply(self, handle):
        # passive 뷰어는 렌더 스레드가 따로 돌아 cam을 읽으므로 lock 안에서 수정한다.
        # lookat은 robosuite MjviewerRenderer와 동일하게 직접 대입(슬라이스 대입은
        # getter가 반환하는 임시 복사본에만 쓰여 반영 안 될 수 있음).
        #
        # cam.type을 FREE(0)로 강제하는 게 핵심: robomimic의 EnvRobosuite.render(mode="human")는
        # 매번 self.env.viewer.set_camera(agentview_id)를 부르고, 뷰어 최초 생성 시점에 그
        # camera_id 때문에 cam.type이 FIXED(2)로 박힌다. FIXED면 azimuth/elevation/distance/
        # lookat을 전부 무시하고 XML의 고정 카메라만 그리므로, 이걸 안 풀면 이 값들을 아무리
        # 바꿔도 화면이 안 바뀐다.
        cfg = self.presets[self.idx][1]
        with handle.lock():
            cam = handle.cam
            cam.type = 0  # mjCAMERA_FREE
            cam.azimuth = cfg["azimuth"]
            cam.elevation = cfg["elevation"]
            cam.distance = cfg["distance"]
            cam.lookat = cfg["lookat"]

    def cycle(self):
        """다음 시점으로 전환하고 그 이름을 반환."""
        self.idx = (self.idx + 1) % len(self.presets)
        handle = self._handle()
        if handle is not None:
            self._apply(handle)
        return self.presets[self.idx][0]

    def render(self):
        """robosuite 뷰어 동기화(+reset 후 새 뷰어면 프리셋 재적용). 창 살아있으면 True."""
        self.env.render(mode="human")
        handle = self._handle()
        if handle is None:
            return True
        if handle is not self._last_handle:  # reset로 새로 생긴 뷰어 → 프리셋 재적용
            self._apply(handle)
            self._last_handle = handle
        return handle.is_running()

    def is_running(self):
        """창이 사용자에 의해 닫혔으면 False (뷰어 미생성/에피소드 간 파괴 시 True)."""
        handle = self._handle()
        return handle is None or handle.is_running()
