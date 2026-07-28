"""사람 개입이 가능한 닫힌 루프 rollout — 개입 데이터 수집용.

정책이 receding horizon으로 돌다가, 사람이 토글키를 누르면 그 순간부터 키보드가
로봇을 직접 몬다. 개입 중 프레임은 INTV, 정책 프레임은 ROLLOUT으로 라벨링하고,
에피소드가 끝나면 각 개입 직전 구간을 PREINTV로 재라벨한다(SIRIUS 스킴).

개입 판단은 `intervention_fn(step, obs_raw)`로 주입한다:
  - None 반환      → 그 스텝은 정책이 실행 (ROLLOUT)
  - action 반환    → 그 스텝은 사람이 실행 (INTV), 반환된 7-dim action을 그대로 step

이 주입 구조 덕에 키보드(사람) 없이도 스크립트 개입(테스트/오라클)으로 같은 경로를
돌릴 수 있다. 키보드용 구현은 아래 KeyboardIntervention.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import robomimic.utils.obs_utils as ObsUtils

from mani_sim.datasets.labels import LABEL_INTV, LABEL_ROLLOUT, relabel_preintv
from mani_sim.runners.rollout import _build_obs_batch


def _to_storage_obs(key, value):
    """저장용 raw 포맷으로 변환. EnvRobosuite.get_observation()이 rgb 키는 이미
    ObsUtils.process_obs로 정책 입력용(CHW, float32, [0,1])까지 처리해서 돌려주는데
    (postprocess_visual_obs=True가 기본), robomimic hdf5 데이터셋은 원본 raw(HWC,
    uint8, [0,255])로 저장하고 학습 시점에 이 처리를 다시 한다 - 그래서 라이브 rollout
    obs를 그대로 저장하면 원본 데모 hdf5와 이미지 shape/dtype이 어긋난다(2026-07-27
    발견 - round.py가 만드는 병합 데이터셋이 collate 단계에서 shape mismatch로 죽던
    원인). ObsUtils.unprocess_obs가 정확히 이 역변환(robomimic 자체 제공 API)이다."""
    if ObsUtils.key_is_obs_modality(key=key, obs_modality="rgb"):
        return ObsUtils.unprocess_obs(value, obs_key=key)
    return np.asarray(value, dtype=np.float32)


def _predict_chunk(policy, normalizer, obs_history, obs_keys, device, rgb_keys=()):
    """rgb_keys가 주어지면 그 키들은 정규화하지 않고 CHW float[0,1]로만 변환한다(image task 지원
    — runners/rollout.py의 _build_obs_batch와 동일 규약 재사용, 여기서 중복 구현하지 않음)."""
    obs_batch = _build_obs_batch(obs_history, obs_keys, rgb_keys, normalizer, device, extra_obs_fn=None)
    chunk = policy.predict_action_chunk(obs_batch)  # (1, Tp, Da), normalized, on device
    chunk = normalizer.unnormalize_action(chunk[0].cpu())  # (Tp, Da), CPU
    return chunk.detach().numpy()


def _start_async_inference(predict_fn, obs_horizon, merger_name, te_coeff):
    """moai_policy(flare/inference) 이식 - PolicyServer(추론 스레드)+ClientManager(중계
    스레드)를 백그라운드로 띄우고, 메인 루프가 매 스텝 merger에서 action을 꺼내 쓰게 한다.
    action_horizon 스텝마다 동기 블로킹하던 것과 달리 추론이 끊김 없이 계속 돈다.

    chunk_latencies_ms/stall_count/stall_time_total는 진단용 계측(2026-07-27 추가) -
    "action_horizon마다 크게 멈추는 문제"는 없앴지만 "조금씩 끊긴다"는 관찰이 나와서,
    그게 (a) merger가 가끔 비어 _wait_async_action이 대기하는 것 때문인지 (b) 백그라운드
    추론이 메인 스레드(mujoco step+render)와 GIL/CPU를 다퉈서 그런지 추측이 아니라
    실측으로 구분하기 위함 - collect_episode 종료 시 요약 출력."""
    from mani_sim.inference.client_manager import ClientManager
    from mani_sim.inference.merger import make_merger
    from mani_sim.inference.obs_provider import ObsProvider
    from mani_sim.inference.policy_server import PolicyServer

    obs_provider = ObsProvider()
    policy_obs_queue = queue.Queue(maxsize=1)
    policy_chunk_queue = queue.Queue(maxsize=4)
    merger = make_merger(merger_name, te_coeff=te_coeff)
    merger_lock = threading.Lock()
    stop_event = threading.Event()

    async_ctx = {
        "obs_provider": obs_provider,
        "merger": merger,
        "merger_lock": merger_lock,
        "stop_event": stop_event,
        "chunk_latencies_ms": [],
        "stall_count": 0,
        "stall_time_total": 0.0,
        "anchor_gaps": [],  # 연속 제출된 청크끼리 t_obs(anchor) 차이 - pred_horizon과 비교용
        "stall_remaining_at_start": [],  # stall 시작 순간 merger.remaining_after(step) - 0이면 진짜 커버리지 소진
        "_last_anchor": None,
    }

    def _on_chunk_submitted(t_obs, latency_ms):
        async_ctx["chunk_latencies_ms"].append(latency_ms)
        prev = async_ctx["_last_anchor"]
        if prev is not None:
            async_ctx["anchor_gaps"].append(t_obs - prev)
        async_ctx["_last_anchor"] = t_obs

    policy_server = PolicyServer(predict_fn, policy_obs_queue, policy_chunk_queue, stop_event)
    client_manager = ClientManager(
        obs_provider, policy_obs_queue, policy_chunk_queue, merger, merger_lock, obs_horizon, stop_event,
        on_chunk_submitted=_on_chunk_submitted,
    )
    policy_server.start()
    client_manager.start()
    return async_ctx


def _get_async_action(async_ctx, step):
    with async_ctx["merger_lock"]:
        action = async_ctx["merger"].get_action(step)
    return None if action is None else np.asarray(action, dtype=np.float32)


def _wait_async_action(async_ctx, step, timeout=30.0):
    """merger에 해당 step의 예측이 아직 없으면(에피소드 시작 직후 콜드 스타트, 또는 background
    추론이 정말 밀린 드문 경우) 짧게 폴링하며 기다린다. moai_policy eval_runner의 "콜드 스타트 -
    첫 chunk 대기" 개념과 동일 - 다만 거긴 유휴 자세를 유지하고 여긴 그냥 기다린다(시뮬은
    실시간 하드웨어 안전 이슈가 없어 한 스텝 정도 밀려도 무해함). 대기가 실제로 발생한
    횟수·누적 시간을 async_ctx에 기록한다(진단용, _start_async_inference 참고)."""
    deadline = time.time() + timeout
    wait_start = time.time()
    action = _get_async_action(async_ctx, step)
    if action is None:
        async_ctx["stall_count"] += 1
        with async_ctx["merger_lock"]:
            async_ctx["stall_remaining_at_start"].append(async_ctx["merger"].remaining_after(step))
    while action is None:
        if time.time() > deadline:
            raise RuntimeError(
                f"async_infer: step {step} action을 {timeout}초 안에 못 받음 "
                "(PolicyServer/ClientManager 스레드가 멈췄는지 확인)"
            )
        time.sleep(0.005)
        action = _get_async_action(async_ctx, step)
    async_ctx["stall_time_total"] += time.time() - wait_start
    return action


def _print_async_stats(async_ctx, num_steps):
    latencies = async_ctx["chunk_latencies_ms"]
    lat_desc = (
        f"평균 {np.mean(latencies):.0f}ms 최대 {np.max(latencies):.0f}ms (n={len(latencies)})"
        if latencies else "청크 0개(에피소드가 첫 청크 전에 끝남)"
    )
    gaps = async_ctx["anchor_gaps"]
    gap_desc = (
        f"평균 {np.mean(gaps):.1f}스텝 최대 {np.max(gaps)}스텝 (n={len(gaps)})"
        if gaps else "N/A(청크 1개 이하)"
    )
    remainings = async_ctx["stall_remaining_at_start"]
    remaining_desc = (
        f"{sum(1 for r in remainings if r == 0)}/{len(remainings)}회가 remaining_after=0"
        if remainings else "해당 없음(stall 없음)"
    )
    print(
        f"[async_infer 진단] chunk 지연: {lat_desc} | "
        f"anchor 간격(연속 청크 t_obs 차이, pred_horizon과 비교): {gap_desc} | "
        f"stall(merger 대기): {async_ctx['stall_count']}회, 누적 {async_ctx['stall_time_total'] * 1000:.0f}ms "
        f"/ 총 {num_steps}스텝 | stall 시점 remaining_after: {remaining_desc}"
    )


def _print_timing_stats(step_times_ms, render_times_ms, control_fps):
    """env.step()/render() 자체의 소요 시간(진단용, 2026-07-27) - "끊김"이 merger/추론 문제가
    아니라 렌더링·물리 스텝 자체의 비용/변동성 때문일 가능성을 실측으로 확인한다. control_fps
    예산(1000/control_fps ms)과 나란히 비교."""
    budget_ms = 1000.0 / control_fps if control_fps > 0 else None
    budget_desc = f"{budget_ms:.0f}ms" if budget_ms is not None else "N/A(페이싱 없음)"

    def _desc(times_ms):
        if not times_ms:
            return "N/A"
        arr = np.asarray(times_ms)
        return f"평균 {arr.mean():.0f}ms 최대 {arr.max():.0f}ms p95 {np.percentile(arr, 95):.0f}ms (n={len(arr)})"

    print(
        f"[타이밍 진단] control_fps 예산: {budget_desc}/스텝 | "
        f"env.step(): {_desc(step_times_ms)} | render(): {_desc(render_times_ms)}"
    )


def collect_episode(
    env,
    policy,
    normalizer,
    obs_keys,
    obs_horizon,
    action_horizon,
    device,
    intervention_fn,
    max_steps=None,
    should_end_fn=None,
    preintv_length=15,
    render=False,
    render_fn=None,
    control_fps=0.0,
    predict_fn=None,
    async_infer=False,
    merger_name="overwrite",
    te_coeff=0.01,
):
    """한 에피소드를 개입 가능 상태로 돌려 프레임별 (obs, action, action_mode)를 수집.

    에피소드는 성공·env done·사람 종료(should_end_fn)·max_steps 중 먼저 오는 것에서 끝난다.
    배포 데이터 수집에선 max_steps=None(무제한)으로 두고 사람이 종료키로 끝내는 게 기본.
    테스트·오프라인 수집에선 max_steps로 상한을 준다.

    개입에서 정책으로 돌아올 때는(동기 모드) 남은 action chunk를 버리고 현재 관측에서
    재계획한다(사람이 상태를 바꿔놨을 수 있으므로). 비동기 모드는 애초에 개입 중에도
    background 추론이 최신 관측으로 계속 재계획하고 있어 별도 처리가 필요 없다.

    control_fps > 0이면 각 스텝을 그 주기에 맞춰 실시간으로 늦춘다(사람이 보고 개입할
    시간 확보). 0이면 페이싱 없이 최대 속도(테스트·오프라인 수집용).

    render_fn(obs_raw)이 주어지면 매 스텝 obs_raw와 함께 호출하고 False 반환 시 에피소드를
    끝낸다(커스텀 뷰어용, render보다 우선). 없고 render=True면 기존 env.render(mode="human")를 쓴다.

    predict_fn(obs_history) -> (T, Da) ndarray로 청크 예측을 대체할 수 있다(예: robomimic
    체크포인트처럼 자체 정규화·RNN 은닉상태를 갖는 정책 — 이 경우 T=1로 매 스텝 재계획해도
    무방). None이면 기존 (policy, normalizer, obs_keys, obs_horizon, device) 기반
    _predict_chunk를 그대로 쓴다(우리 자체 DP/BC-RNN 정책, 기존 동작 그대로).

    async_infer=True면 predict_fn을 PolicyServer 스레드에서 끊임없이 돌리고(moai_policy
    flare/inference 이식, 2026-07-27), 메인 루프는 매 스텝 merger에서 그 시점 action을
    꺼내 쓴다 — action_horizon 스텝마다 동기 블로킹(예: diffusion DDPM 100-step 추론
    ~0.4-0.5초)하던 것이 없어져 로봇이 끊기지 않고 움직인다. action_horizon>1인 diffusion류
    배포에만 의미가 있다 - robomimic은 이미 매 스텝(action_horizon=1) 재계획이라 켤 필요 없다.
    """
    if predict_fn is None:
        predict_fn = lambda history: _predict_chunk(policy, normalizer, history, obs_keys, device)
    if hasattr(policy, "eval"):
        policy.eval()
    obs_raw = env.reset()
    obs_history = deque([obs_raw] * obs_horizon, maxlen=obs_horizon)

    obs_seq, action_seq, mode_seq = [], [], []
    chunk, chunk_ptr = None, 0
    success = False
    step_times_ms, render_times_ms = [], []  # 진단용(2026-07-27) - env.step()/render() 자체가
    # 얼마나 걸리는지 측정. async_infer 여부와 무관 - "끊김"이 merger/추론 문제가 아니라
    # 렌더링·물리 스텝 자체의 비용/변동성 때문일 가능성을 실측으로 확인하기 위함.

    async_ctx = None
    if async_infer:
        async_ctx = _start_async_inference(predict_fn, obs_horizon, merger_name, te_coeff)
        async_ctx["obs_provider"].put(0, obs_raw)

    try:
        step = 0
        while max_steps is None or step < max_steps:
            if should_end_fn is not None and should_end_fn():
                break
            loop_start = time.time()

            intv_action = intervention_fn(step, obs_raw)

            if intv_action is not None:
                action = np.asarray(intv_action, dtype=np.float32)
                mode = LABEL_INTV
                chunk = None  # 개입 후 정책 복귀 시 강제 재계획(동기 모드 전용)
            elif async_ctx is not None:
                action = _wait_async_action(async_ctx, step)
                mode = LABEL_ROLLOUT
            else:
                if chunk is None or chunk_ptr >= action_horizon:
                    chunk = predict_fn(obs_history)
                    chunk_ptr = 0
                action = np.asarray(chunk[chunk_ptr], dtype=np.float32)
                chunk_ptr += 1
                mode = LABEL_ROLLOUT

            obs_seq.append({key: _to_storage_obs(key, obs_raw[key]) for key in obs_keys})
            action_seq.append(action)
            mode_seq.append(mode)

            t_step0 = time.time()
            obs_raw, _reward, done, _info = env.step(action)
            step_times_ms.append((time.time() - t_step0) * 1000.0)
            obs_history.append(obs_raw)
            step += 1
            if async_ctx is not None:
                async_ctx["obs_provider"].put(step, obs_raw)
            t_render0 = time.time()
            if render_fn is not None:
                keep_going = render_fn(obs_raw)
                render_times_ms.append((time.time() - t_render0) * 1000.0)
                if not keep_going:  # 창이 닫히면 에피소드 종료
                    break
            elif render:
                env.render(mode="human")
                render_times_ms.append((time.time() - t_render0) * 1000.0)

            if env.is_success()["task"]:
                success = True
            if success or done:
                break

            if control_fps > 0:
                remaining = 1.0 / control_fps - (time.time() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        if async_ctx is not None:
            async_ctx["stop_event"].set()
            async_ctx["obs_provider"].stop()
            _print_async_stats(async_ctx, len(mode_seq))
        _print_timing_stats(step_times_ms, render_times_ms, control_fps)

    modes = relabel_preintv(np.asarray(mode_seq, dtype=np.int64), preintv_length)
    return {
        "obs": obs_seq,
        "actions": np.asarray(action_seq, dtype=np.float32),
        "action_modes": modes,
        "success": success,
    }


class KeyboardIntervention:
    """robosuite 키보드 device를 개입 신호원으로 감싼 intervention_fn.

    토글키(기본 좌·우 Ctrl)로 개입 on/off를 전환한다. 개입 on일 때만 키보드 입력을
    7-dim action으로 변환해 반환하고, off면 None(정책이 실행)을 반환한다. 종료키(기본
    Enter)로 현재 에피소드를 끝낸다(정책이 실패해 사람이 포기할 때) — should_end()로 노출.

    주의: pynput 리스너는 X 디스플레이가 있어야 동작하므로 실제 화면(로컬/포워딩)에서
    실행해야 한다. 헤드리스 환경에선 스크립트 intervention_fn을 대신 쓴다.
    """

    def __init__(
        self, raw_env, toggle_key="ctrl", end_key="enter", pos_sensitivity=1.0, rot_sensitivity=1.0
    ):
        from pynput import keyboard as pynput_keyboard

        from robosuite.devices import Keyboard

        self.raw_env = raw_env
        self.device = Keyboard(
            env=raw_env, pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity
        )
        self.device.start_control()
        self.intervening = False
        self.end_requested = False

        self._toggle_keys = self._resolve_keys(toggle_key, pynput_keyboard)
        self._end_keys = self._resolve_keys(end_key, pynput_keyboard)
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    @staticmethod
    def _resolve_keys(name, pynput_keyboard):
        """키 이름 → 허용 키 집합. "ctrl"/"alt"/"shift"는 좌·우 양쪽 모두 인식한다
        (pynput은 좌/우를 구분하므로 한쪽만 지정하면 반대쪽을 눌렀을 때 안 걸린다).
        단일 문자면 그 문자, 그 외엔 특수키 이름(enter, tab 등)."""
        Key = pynput_keyboard.Key
        aliases = {
            "ctrl": {Key.ctrl, Key.ctrl_l, Key.ctrl_r},
            "alt": {Key.alt, Key.alt_l, Key.alt_r},
            "shift": {Key.shift, Key.shift_l, Key.shift_r},
        }
        if name in aliases:
            return aliases[name]
        if len(name) == 1:
            return {pynput_keyboard.KeyCode.from_char(name)}
        return {getattr(Key, name)}

    def _on_press(self, key):
        if key in self._toggle_keys:
            self.intervening = not self.intervening
            if self.intervening:
                self.device.start_control()  # 델타 누적 초기화 (점프 방지)
        elif key in self._end_keys:
            self.end_requested = True

    def should_end(self):
        return self.end_requested

    def reset(self):
        """에피소드 시작 시 개입/종료 상태 초기화."""
        self.intervening = False
        self.end_requested = False

    def __call__(self, step, obs_raw):
        if not self.intervening:
            return None

        ac_dict = self.device.input2action()
        if ac_dict is None:  # device reset('q') → 개입 종료
            self.intervening = False
            return None

        active_robot = self.raw_env.robots[self.device.active_robot]
        action_dict = dict(ac_dict)
        for arm in active_robot.arms:
            action_dict[arm] = ac_dict[f"{arm}_delta"]  # OSC_POSE: delta 입력
        return active_robot.create_action_vector(action_dict)

    def close(self):
        self._listener.stop()
