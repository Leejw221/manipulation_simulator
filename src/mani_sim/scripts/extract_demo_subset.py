"""robomimic HDF5의 mask 서브셋(예: demo20)을 독립 HDF5로 추출.

`merge_rounds.py`가 라운드 배포+개입 데이터와 합칠 수 있는 형태(demo_0.. 재번호, action_mode
등 기존 구조 그대로)로 만든다. 원본 HDF5는 읽기만 하고 건드리지 않는다.

`round.py`의 누적 재학습("그 시점까지 전체")이 실제로는 라운드에서 수집한 배포+개입 데이터만
합치고 원본 데모를 빠뜨리고 있던 gap(2026-07-27 발견, `merge_rounds.py`의 "round0(데모)"
컨벤션과 불일치)을 메우기 위해 필요 - 이 스크립트의 출력을 round0_demo_hdf5로 줘서
round_hdf5s를 데모로 미리 채우고 시작하게 한다.

사용: python -m mani_sim.scripts.extract_demo_subset <입력.hdf5> <mask 이름> <출력.hdf5>
"""

import sys

import h5py


def extract_demo_subset(input_path, mask_name, output_path):
    with h5py.File(input_path, "r") as fin:
        demo_names = [d.decode() if isinstance(d, bytes) else d for d in fin["mask"][mask_name][:]]
        demo_names = sorted(demo_names, key=lambda k: int(k.split("_")[1]))
        with h5py.File(output_path, "w") as fout:
            data_out = fout.create_group("data")
            total = 0
            for demo_idx, name in enumerate(demo_names):
                fin.copy(f"data/{name}", data_out, name=f"demo_{demo_idx}")
                total += int(data_out[f"demo_{demo_idx}"].attrs["num_samples"])
            data_out.attrs["total"] = total
            # derive_task_meta_from_hdf5가 학습 시 필수로 읽음 - 원본 값 그대로 승계
            # (2026-07-27 발견: 이게 빠져서 이 스크립트 산출물로 학습하면 KeyError).
            data_out.attrs["env_args"] = fin["data"].attrs["env_args"]
    print(f"저장: {output_path} | {len(demo_names)}개 demo, {total} 프레임 (원본 mask={mask_name})")
    return output_path


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    extract_demo_subset(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()
