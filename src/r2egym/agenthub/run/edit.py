# editagent_script.py

import openai
import re
import yaml
from tqdm import tqdm
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading

from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    condense_history: bool = True,
    swesmith_wrapper: bool = False,
):
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    for idx in range(num_restarts):
        logger.warning(f"running agent at idx: {idx+1}")
        trajectory = agent.run(
            env,
            max_steps=steps_per_agent,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            condense_history=condense_history,
            swesmith_wrapper=swesmith_wrapper,
        )
        # remove reproduce.py
        # env.runtime.run('rm reproduce_issue.py')
    return trajectory


def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    condense_history: bool = True,
    swesmith_wrapper: bool = False,
    max_reward_calc_time: int = 300,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """
    logger = setup_logging(
        name=ds["docker_image"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)

    # Initialize the RepoEnv
    env = RepoEnv(env_args, logger=logger, backend=backend)
    # set agent args
    if use_fn_calling:
        agent_args = AgentArgs.from_yaml(
            Path("./src/r2egym/agenthub/config/edit_fn_calling.yaml")
        )
    else:
        # agent_args = AgentArgs.from_yaml(
        #     Path("./src/r2egym/agenthub/config/edit_non_fn_calling.yaml")
        # )
        if swesmith_wrapper:
            agent_args = AgentArgs.from_yaml(
                Path("./src/r2egym/agenthub/config/edit_swesmith.yaml")
            )
        else:
            agent_args = AgentArgs.from_yaml(
                Path("./src/r2egym/agenthub/config/edit_non_fn_calling.yaml")
            )
    agent_args.llm_name = llm_name

    # Initialize the agent
    agent = Agent(name="EditAgent", args=agent_args, logger=logger)

    # run agent editagent
    try:
        trajectory = run_agent_with_restarts(
            agent,
            env,
            max_steps=max_steps,
            num_restarts=num_restarts,
            temperature=temperature,
            max_steps_absolute=max_steps_absolute,
            use_fn_calling=use_fn_calling,
            condense_history=condense_history,
            swesmith_wrapper=swesmith_wrapper,
        )
    except Exception as e:
        logger.error(
            f"Error during agent run for Docker image {ds['docker_image']}: {e}"
        )
        return None

    # also get the gt outputs
    reward_calc_time = time.time()
    reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
    reward_calc_time = time.time() - reward_calc_time
    # Close the environment and runtime
    env.close()

    # update the trajectory object
    trajectory.reward = reward
    trajectory.test_output = test_output
    trajectory.ds = ds
    trajectory.exp_name = exp_name
    trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
    logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")

    logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
    # close env and docker runtime
    logger.info(f"Closing environment for Docker image: {ds['docker_image']}")
    return trajectory.model_dump_json()


def runagent_multiple(
    dataset: str,
    subset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    condense_history: bool = True,
    swesmith_wrapper: bool = False,
    max_reward_calc_time: int = 300,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
    """
    # Load the dataset
    ds = load_dataset(dataset, subset, split=split)
    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # get selected idxs
    if start_idx >= len(ds):
        raise ValueError(f"start_idx ({start_idx}) is larger than or equal to dataset size ({len(ds)})")
    
    end_idx = min(start_idx + k, len(ds))
    selected_idx = range(start_idx, end_idx)
    ds_selected = [ds[i] for i in selected_idx]

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                for line in f.readlines():
                    try:
                        existing_dockers.append(
                            Trajectory.load_from_model_dump_json(line).ds[
                                "docker_image"
                            ]
                        )
                    except:
                        print("error in jsonl file")

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    if skip_existing:
        old_jsonl_files_glob = f"{exp_name[:-1]}*"
        for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
            with open(old_jsonl_file) as f:
                existing_dockers = [
                    loadline["ds"]["docker_image"]
                    for line in f
                    for loadline in [json.loads(line)]
                    if loadline["reward"] == 1
                ]

            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["docker_image"] not in existing_dockers
            ]

    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                condense_history=condense_history,
                swesmith_wrapper=swesmith_wrapper,
                max_reward_calc_time=max_reward_calc_time,
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for ds_entry in ds_selected
        }

        with open(jsonl_file, "a") as f:
            for future in tqdm(concurrent.futures.as_completed(future_to_image), total=len(future_to_image)):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                try:
                    result = future.result()
                    if result is not None:
                        with file_lock:
                            f.write(result + "\n")
                    else:
                        raise Exception("result is None")
                    
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"Exception for Docker image {docker_image}: {e}")
                    raise e

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )
