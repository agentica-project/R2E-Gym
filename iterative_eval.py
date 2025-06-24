#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import fire
from r2egym.agenthub.trajectory.trajectory import Trajectory

FILE_NAMES = [
    "traj/r2egym-qwen3-32b-rl-100-steps.jsonl",
    "traj/r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter1-swesmithdockerbranch-fixededitor-v1.jsonl",
    "traj/r2egym-qwen3-32b-maxtoken64k-nworkers54-maxiter1-swesmithdockerbranch-fixededitor-temp1-v1.jsonl",
]

FILE_NAMES = [
    "traj/r2egym-qwen3-32b-rl-100-steps.jsonl",
    "traj/r2egym-deepswe-64k-100-steps-attempt-0.jsonl",
    "traj/r2egym-deepswe-64k-100-steps-attempt-1.jsonl",
    "traj/r2egym-deepswe-64k-100-steps-attempt-2.jsonl",
    "traj/r2egym-deepswe-64k-100-steps-attempt-3.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run1.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run2.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run3.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run4.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run5.jsonl",
]


FILE_NAMES = [
    "traj/r2egym-qwen3-32b-rl-100-steps.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run1.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run2.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run3.jsonl",
    "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run4.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run5.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run6.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run7.jsonl",
    "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run8.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run9.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run10.jsonl",
]

FILE_NAMES = [
    # "traj/r2egym-qwen3-32b-rl-100-steps.jsonl",
    "../deepswe/r2egym-deepswe-64k-100-steps-attempt-3.jsonl",
    "../deepswe/r2egym-deepswe-64k-100-steps-attempt-0.jsonl",
    # "../deepswe/r2egym-deepswe-64k-100-steps-attempt-1.jsonl",
    "../deepswe/r2egym-deepswe-64k-100-steps-attempt-2.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run1.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run2.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run3.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run4.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run5.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run6.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run7.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run8.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run9.jsonl",
    # "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run10.jsonl",
]

# FILE_NAMES = [
#     # "traj/r2egym-qwen3-32b-rl-100-steps.jsonl",
#     "../deepswe/r2egym-deepswe-64k-100-steps-attempt-0.jsonl",
#     "../deepswe/r2egym-deepswe-64k-100-steps-attempt-1.jsonl",
#     "../deepswe/r2egym-deepswe-64k-100-steps-attempt-2.jsonl",
#     "../deepswe/r2egym-deepswe-64k-100-steps-attempt-3.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run1.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run2.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run3.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run4.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run5.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run6.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run7.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run8.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run9.jsonl",
#     "../r2e-gym-swe-bench-verified-qwen3-trajectory/r2egym-qwen3-32b-rl-step200-run10.jsonl",
# ]



def load_trajectories(filename: str) -> List[Trajectory]:
    """Load trajectories from a JSONL file."""
    trajectories = []
    with open(filename, "r") as f:
        for idx, line in enumerate(f):
            try:
                trajectory = Trajectory.load_from_model_dump_json(line)
                trajectories.append(trajectory)
            except Exception as e:
                print(f"Error loading trajectory at line {idx} in {filename}: {e}")
                continue
    return trajectories

def compute_individual_scores(filenames: List[str]) -> Dict[str, Tuple[float, int, int]]:
    """Compute individual success rates for each file."""
    results = {}
    
    for filename in filenames:
        print(f"\n=== Analyzing {filename} ===")
        trajectories = load_trajectories(filename)
        
        if not trajectories:
            print(f"No valid trajectories found in {filename}")
            results[filename] = (0.0, 0, 0)
            continue
            
        num_trajectories = len(trajectories)
        num_success = sum(1 for t in trajectories if t.reward == 1)
        success_rate = num_success / num_trajectories if num_trajectories > 0 else 0.0
        
        print(f"Total trajectories: {num_trajectories}")
        print(f"Successful trajectories: {num_success}")
        print(f"Success rate: {success_rate*100:.2f}%")
        
        # Exit reason distribution
        exit_reasons = {}
        for t in trajectories:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        print(f"Exit reasons: {exit_reasons}")
        
        results[filename] = (success_rate, num_success, num_trajectories)
    
    return results

def iterative_evaluation(filenames: List[str]) -> Tuple[float, int, int]:
    """
    Implement iterative evaluation logic:
    1. For each docker_image, collect trajectories from all files
    2. If any trajectory has exit_reason == 'agent', use that trajectory's reward
    3. Otherwise, use the trajectory with minimum steps
    4. Compute overall success rate across all docker_images
    """
    
    # Group trajectories by docker_image across all files
    docker_trajectories = defaultdict(list)
    
    for i, filename in enumerate(filenames):
        print(f"\nLoading trajectories from {filename}...")
        trajectories = load_trajectories(filename)
        
        for trajectory in trajectories:
            docker_image = trajectory.ds.get("docker_image", "unknown")
            docker_trajectories[docker_image].append((trajectory, i, filename))
    
    print(f"\nFound {len(docker_trajectories)} unique docker images across all files")
    
    # Apply iterative evaluation logic
    final_results = []
    agent_exit_count = 0
    min_steps_count = 0
    first_count = 0
    
    for docker_image, traj_list in docker_trajectories.items():
        # Sort by file order (temperature: 0.0, 0.1, 0.2 as per the logic)
        traj_list.sort(key=lambda x: x[1])  # Sort by file index
        
        selected_trajectory = None
        selection_reason = ""
        
        # First, check if any trajectory has exit_reason == 'agent'
        for trajectory, file_idx, filename in traj_list:
            if trajectory.exit_reason == "agent":
                selected_trajectory = trajectory
                selection_reason = f"agent_exit_file_{file_idx}"
                agent_exit_count += 1
                break
        
        # # If no agent exit, select first trajectory
        # if selected_trajectory is None:
        #     selected_trajectory = traj_list[0][0]
        #     selection_reason = f"first_file_0"
        #     first_count += 1
        
        if selected_trajectory is None:
            min_steps_traj = min(traj_list, key=lambda x: x[0].num_steps)
            selected_trajectory = min_steps_traj[0]
            selection_reason = f"min_steps_file_{min_steps_traj[1]}"
            min_steps_count += 1
        
        final_results.append({
            'docker_image': docker_image,
            'reward': selected_trajectory.reward,
            'num_steps': selected_trajectory.num_steps,
            'exit_reason': selected_trajectory.exit_reason,
            'selection_reason': selection_reason
        })
    
    # Compute overall statistics
    total_problems = len(final_results)
    successful_problems = sum(1 for r in final_results if r['reward'] == 1)
    overall_success_rate = successful_problems / total_problems if total_problems > 0 else 0.0
    
    print(f"\n=== ITERATIVE EVALUATION RESULTS ===")
    print(f"Total problems (docker images): {total_problems}")
    print(f"Successful problems: {successful_problems}")
    print(f"Overall success rate: {overall_success_rate*100:.2f}%")
    print(f"Problems selected by agent exit: {agent_exit_count}")
    print(f"Problems selected by minimum steps: {min_steps_count}")
    print(f"Problems selected by first: {first_count}")

    # Show some examples
    print(f"\nFirst 10 results:")
    for i, result in enumerate(final_results[:10]):
        print(f"  {i+1}. {result['docker_image'].split('/')[-1]}: "
              f"reward={result['reward']}, steps={result['num_steps']}, "
              f"exit={result['exit_reason']}, selected_by={result['selection_reason']}")
    
    # Distribution by selection reason
    selection_reasons = {}
    for result in final_results:
        reason = result['selection_reason']
        selection_reasons[reason] = selection_reasons.get(reason, 0) + 1
    print(f"\nSelection reason distribution: {selection_reasons}")
    
    return overall_success_rate, successful_problems, total_problems

def main(
    filenames: List[str] = FILE_NAMES
):
    """
    Perform iterative evaluation on three JSONL files.
    
    Args:
        file1: First JSONL file (temperature 0.0)
        file2: Second JSONL file (temperature 0.1) 
        file3: Third JSONL file (temperature 0.2)
    """
    
    # filenames = [file1, file2, file3]
    
    # Check if files exist
    for filename in filenames:
        if not Path(filename).exists():
            print(f"Error: File {filename} does not exist!")
            return
    
    print("=== INDIVIDUAL FILE ANALYSIS ===")
    individual_results = compute_individual_scores(filenames)
    
    print("\n=== SUMMARY OF INDIVIDUAL RESULTS ===")
    for filename, (success_rate, num_success, num_total) in individual_results.items():
        print(f"{Path(filename).name}: {success_rate*100:.2f}% ({num_success}/{num_total})")
    
    print("\n" + "="*50)
    print("=== ITERATIVE EVALUATION ===")
    iterative_success_rate, iterative_success, iterative_total = iterative_evaluation(filenames)
    
    print(f"\n=== FINAL COMPARISON ===")
    print("Individual file results:")
    for i, (filename, (success_rate, num_success, num_total)) in enumerate(individual_results.items()):
        temp = [0.0, 0.1, 0.2][i] if i < 3 else "unknown"
        print(f"  File {i+1}: {success_rate*100:.2f}% ({num_success}/{num_total})")
    
    print(f"\nIterative evaluation result: {iterative_success_rate*100:.2f}% ({iterative_success}/{iterative_total})")

if __name__ == "__main__":
    fire.Fire(main) 