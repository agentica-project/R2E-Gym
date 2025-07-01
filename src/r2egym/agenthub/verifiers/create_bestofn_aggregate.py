import glob
import json
from collections import defaultdict

import fire

from r2egym.agenthub.trajectory.trajectory import Trajectory


def run_ef_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    pass


def run_eb_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    pass


def run_hybrid_verifier(sub_trajs: list[Trajectory]) -> Trajectory:
    pass


def run(
    traj_file_glob: str,
    verifier_mode: str,
    output_json_path: str,
):
    assert verifier_mode in ["ef", "eb", "hybrid"]
    verifier_fn_dict = {
        "ef": run_ef_verifier,
        "eb": run_eb_verifier,
        "hybrid": run_hybrid_verifier,
    }
    verifier_fn = verifier_fn_dict[verifier_mode]

    traj_files = glob.glob(traj_file_glob)
    all_trajs_by_docker: list[Trajectory] = defaultdict(list)

    for traj_file in traj_files:
        with open(traj_file, "r") as f:
            for line in f:
                traj = Trajectory.model_validate_json(line)
                all_trajs_by_docker[traj.docker_image].append(traj)

    submission = []
    for docker_image, sub_trajs in all_trajs_by_docker.items():
        selected_traj = verifier_fn(sub_trajs)
        selected_traj.docker_image = docker_image
        submission.append(selected_traj.create_swebench_submission())

    with open(output_json_path, "w") as f:
        json.dump(submission, f)


if __name__ == "__main__":
    fire.Fire(run)
