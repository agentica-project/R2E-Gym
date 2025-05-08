from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from pathlib import Path
from datasets import load_dataset

# get logger
from r2egym.agenthub.utils.log import get_logger
logger = get_logger(__name__)

# load gym dataset [R2E-Gym/R2E-Gym-Subset, R2E-Gym/R2E-Gym-Full, R2E-Gym/SWE-Bench-Verified, R2E-Gym/SWE-Bench-Lite]
ds = load_dataset("r2e-gym/swe-bench-verified")
split = 'test' # split of the dataset [train, test]

# 
ds = ds[split]

# index
env_index = [idx for idx, x in enumerate(ds) if x['docker_image'] == 'slimshetty/swebench-verified:sweb.eval.x86_64.pydata__xarray-4629'][0]

# loop through first 10 indices
for idx in range(10):
    logger.info(f"Index: {idx}; docker_image: {ds[idx]['docker_image']}")

    # load gym environment
    env_args = EnvArgs(ds = ds[idx])
    env = RepoEnv(env_args, backend='kubernetes')
    gt_patch = env.runtime.commit.get_patch(test_file=False, non_test_file=True)

    # check the reward before applying the patch (no need to get test output)
    reward_before = env.runtime._calculate_reward(get_test_output=False)
    logger.info(f"Reward before: {reward_before}")

    # apply gt patch
    env.runtime.apply_patch(gt_patch)

    # run tests after applying the patch (no need to get test output)
    reward_after = env.runtime._calculate_reward(get_test_output=False)
    logger.info(f"Reward after: {reward_after}")
    logger.info ("--------------------------------")
    # reward_after = env.runtime._calculate_reward(get_test_output=True)
    # success, out = env.runtime._calculate_reward(get_test_output=True)
    # logger.info(f"Success: {success}")
    # logger.info(f"Output: {out}")