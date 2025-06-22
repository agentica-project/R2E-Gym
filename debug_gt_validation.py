import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset
from src.r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from src.r2egym.agenthub.agent.agent import AgentArgs, Agent
from src.r2egym.agenthub.utils.log import get_logger

logger = get_logger('validation')

# Configuration
DATASET_NAME = 'r2e-edits/SweSmith-RL-Dataset'
CONFIG_PATH = Path('./src/r2egym/agenthub/config/v1/edit_non_fn_calling.yaml')
LLM_NAME = 'openai/mistralai/Devstral-Small-2505'
MAX_WORKERS = 32

# Load the dataset
smith_ds = load_dataset(DATASET_NAME, split='train')

def process_example(idx):
    """Validate a single dataset example and return metrics."""
    record = {'idx': idx}
    try:
        ds_item = smith_ds[idx]
        env = RepoEnv(EnvArgs(ds=ds_item))
        agent_args = AgentArgs.from_yaml(CONFIG_PATH)
        agent_args.llm_name = LLM_NAME
        _ = Agent(name='EditingAgent', args=agent_args)

        gt_patch = env.runtime.ds['patch']
        reward_before = env.runtime._calculate_reward()
        env.runtime.reverse_patch(gt_patch)
        reward_after = env.runtime._calculate_reward()

        record.update({
            'reward_before': reward_before,
            'reward_after': reward_after,
            'success': reward_before == 0.0 and reward_after == 1.0
        })
        # remove env and container
        env.close()
    except Exception as e:
        record.update({
            'success': False,
            'error': str(e)
        })
    return record

def main():
    total = len(smith_ds)
    results = []
    output_file = Path('validation_results.jsonl')
    output_file.unlink(missing_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_example, idx): idx for idx in range(total)}
        for future in tqdm(as_completed(futures), total=total, desc="Validating"):
            res = future.result()
            results.append(res)
            # Append to JSONL file
            with output_file.open('a') as f:
                f.write(json.dumps(res) + '\n')

            # Print live stats
            succ = sum(1 for r in results if r['success'])
            fail = len(results) - succ
            logger.info(f'Processed {len(results)}/{total}: success={succ}, fail={fail}')

    # Write summary
    summary = {
        'total': total,
        'success': succ,
        'fail': fail,
        'error_types': {}
    }
    for r in results:
        if not r['success']:
            err = r.get('error', 'Unknown')
            summary['error_types'][err] = summary['error_types'].get(err, 0) + 1

    with open('validation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info('Validation complete. Summary: %s', summary)


if __name__ == '__main__':
    main()