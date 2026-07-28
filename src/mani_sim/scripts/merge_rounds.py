"""여러 라운드 hdf5(각 collect.py 산출물)를 하나의 누적 데이터셋으로 병합.

SIRIUS/APO는 라운드가 쌓일수록 "그 시점까지의 전체 데이터"로 재학습한다(sirius.json의
"data": "round01.hdf5"처럼 라운드 번호가 누적을 뜻함) — round0(데모)·round1(배포+개입)·
round2(round1 정책 배포+개입)... 순으로 모은 각 hdf5를 이 스크립트로 이어붙여 다음 라운드
학습 입력을 만든다. demo_i 키만 재번호(0..N-1)하고 나머지 구조(obs/actions/action_mode)는
intervention_writer.py 포맷을 그대로 유지한다.

사용: python -m mani_sim.scripts.merge_rounds out.hdf5 round0.hdf5 round1.hdf5 [round2.hdf5 ...]
"""

import sys

import h5py


def merge_rounds(output_path, input_paths):
    total = 0
    demo_idx = 0
    env_args = None
    with h5py.File(output_path, "w") as fout:
        data_out = fout.create_group("data")
        for path in input_paths:
            with h5py.File(path, "r") as fin:
                if env_args is None:
                    # derive_task_meta_from_hdf5가 학습 시 필수로 읽음 - 입력 전부 같은
                    # task에서 나왔으므로(같은 env_name/env_kwargs) 첫 입력 값을 그대로 승계
                    # (2026-07-27 발견: 이게 빠져서 병합 파일로 학습하면 KeyError).
                    env_args = fin["data"].attrs["env_args"]
                demo_keys = sorted(
                    fin["data"].keys(), key=lambda k: int(k.split("_")[1])
                )
                for key in demo_keys:
                    fin.copy(f"data/{key}", data_out, name=f"demo_{demo_idx}")
                    total += int(data_out[f"demo_{demo_idx}"].attrs["num_samples"])
                    demo_idx += 1
            print(f"{path}: {len(demo_keys)} demo 병합 (누적 {demo_idx})")
        data_out.attrs["total"] = total
        data_out.attrs["env_args"] = env_args
    print(f"저장: {output_path} | 총 {demo_idx} demo, {total} frame")
    return output_path


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    merge_rounds(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
