#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import fire
from r2egym.agenthub.trajectory.trajectory import Trajectory




# Glob files from ./traj/qwen3-context
FILE_NAMES = [f for f in Path("./traj/deepswe").glob("*.jsonl")]

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

def load_all_trajectories(filenames: List[str]) -> Dict[str, List[Trajectory]]:
    """Load all trajectories from all files once."""
    all_trajectories = {}
    
    for filename in filenames:
        print(f"Loading trajectories from {filename}...")
        trajectories = load_trajectories(filename)
        all_trajectories[filename] = trajectories
    
    return all_trajectories

def compute_individual_scores(all_trajectories: Dict[str, List[Trajectory]]) -> Dict[str, Tuple[float, int, int]]:
    """Compute individual success rates for each file."""
    results = {}
    
    for filename, trajectories in all_trajectories.items():
        print(f"\n=== Analyzing {filename} ===")
        
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

def iterative_evaluation_with_n_files(all_trajectories: Dict[str, List[Trajectory]], filenames: List[str], n_files: int) -> Tuple[float, int, int]:
    """
    Implement iterative evaluation logic using only the first n_files:
    1. For each docker_image, collect trajectories from the first n files
    2. If any trajectory has exit_reason == 'agent', use that trajectory's reward
    3. Otherwise, use the trajectory with minimum steps
    4. Compute overall success rate across all docker_images
    """
    
    # Use only the first n_files
    selected_filenames = filenames[:n_files]
    
    # Group trajectories by docker_image across selected files
    docker_trajectories = defaultdict(list)
    
    for i, filename in enumerate(selected_filenames):
        trajectories = all_trajectories[filename]
        
        for trajectory in trajectories:
            docker_image = trajectory.ds.get("docker_image", "unknown")
            docker_trajectories[docker_image].append((trajectory, i, filename))
    
    print(f"\nUsing {n_files} files for iterative evaluation")
    print(f"Found {len(docker_trajectories)} unique docker images across selected files")
    
    # Apply iterative evaluation logic
    final_results = []
    agent_exit_count = 0
    min_steps_count = 0
    
    for docker_image, traj_list in docker_trajectories.items():
        # Sort by file order
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
    
    print(f"\n=== ITERATIVE EVALUATION RESULTS (using {n_files} files) ===")
    print(f"Total problems (docker images): {total_problems}")
    print(f"Successful problems: {successful_problems}")
    print(f"Overall success rate: {overall_success_rate*100:.2f}%")
    print(f"Problems selected by agent exit: {agent_exit_count}")
    print(f"Problems selected by minimum steps: {min_steps_count}")
    
    # Distribution by selection reason
    selection_reasons = {}
    for result in final_results:
        reason = result['selection_reason']
        selection_reasons[reason] = selection_reasons.get(reason, 0) + 1
    print(f"Selection reason distribution: {selection_reasons}")
    
    return overall_success_rate, successful_problems, total_problems

def iterative_evaluation(all_trajectories: Dict[str, List[Trajectory]], filenames: List[str]) -> Tuple[float, int, int]:
    """
    Run iterative evaluation for 1 file, 2 files, ... n files
    """
    
    print(f"\n=== ITERATIVE EVALUATION WITH PROGRESSIVE FILES ===")
    
    results = []
    
    for n_files in range(1, len(filenames) + 1):
        print(f"\n" + "="*30)
        print(f"=== Using first {n_files} file(s) ===")
        
        success_rate, successful, total = iterative_evaluation_with_n_files(all_trajectories, filenames, n_files)
        results.append((n_files, success_rate, successful, total))
    
    # Summary
    print(f"\n=== ITERATIVE EVALUATION SUMMARY ===")
    print("Files Used | Success Rate | Successful/Total")
    print("-" * 45)
    for n_files, success_rate, successful, total in results:
        print(f"{n_files:9d} | {success_rate*100:11.2f}% | {successful:10d}/{total}")
    
    # Return the result using all files
    return results[-1][1], results[-1][2], results[-1][3]

def pass_at_n_evaluation(all_trajectories: Dict[str, List[Trajectory]], filenames: List[str], n: int = None) -> Tuple[float, int, int]:
    """
    Implement pass@N evaluation logic:
    For each docker_image, check if ANY of the first N trajectories is successful.
    If n is None, use all available trajectories.
    """
    
    # Group trajectories by docker_image across all files
    docker_trajectories = defaultdict(list)
    
    for i, filename in enumerate(filenames):
        trajectories = all_trajectories[filename]
        
        for trajectory in trajectories:
            docker_image = trajectory.ds.get("docker_image", "unknown")
            docker_trajectories[docker_image].append((trajectory, i, filename))
    
    print(f"\nFound {len(docker_trajectories)} unique docker images across all files")
    
    # Apply pass@N evaluation logic
    final_results = []
    
    for docker_image, traj_list in docker_trajectories.items():
        # Sort by file order
        traj_list.sort(key=lambda x: x[1])  # Sort by file index
        
        # Consider first N trajectories (or all if n is None)
        trajectories_to_consider = traj_list[:n] if n is not None else traj_list
        
        # Check if any trajectory is successful
        is_successful = any(traj[0].reward == 1 for traj in trajectories_to_consider)
        
        # For reporting, get the first successful trajectory or the first one
        selected_trajectory = None
        for trajectory, file_idx, filename in trajectories_to_consider:
            if trajectory.reward == 1:
                selected_trajectory = trajectory
                break
        
        if selected_trajectory is None and trajectories_to_consider:
            selected_trajectory = trajectories_to_consider[0][0]
        
        final_results.append({
            'docker_image': docker_image,
            'is_successful': is_successful,
            'num_attempts': len(trajectories_to_consider),
            'total_available': len(traj_list),
            'selected_trajectory': selected_trajectory
        })
    
    # Compute overall statistics
    total_problems = len(final_results)
    successful_problems = sum(1 for r in final_results if r['is_successful'])
    overall_success_rate = successful_problems / total_problems if total_problems > 0 else 0.0
    
    n_str = str(n) if n is not None else "ALL"
    print(f"\n=== PASS@{n_str} EVALUATION RESULTS ===")
    print(f"Total problems (docker images): {total_problems}")
    print(f"Successful problems: {successful_problems}")
    print(f"Pass@{n_str} success rate: {overall_success_rate*100:.2f}%")
    
    # Show distribution of attempts used
    attempts_used = [r['num_attempts'] for r in final_results]
    attempts_dist = {}
    for attempts in attempts_used:
        attempts_dist[attempts] = attempts_dist.get(attempts, 0) + 1
    print(f"Distribution of attempts used: {attempts_dist}")
    
    # Show some examples
    # print(f"\nFirst 10 results:")
    # for i, result in enumerate(final_results[:10]):
    #     docker_name = result['docker_image'].split('/')[-1]
    #     success_str = "SUCCESS" if result['is_successful'] else "FAIL"
    #     print(f"  {i+1}. {docker_name}: {success_str} "
    #           f"({result['num_attempts']}/{result['total_available']} attempts)")
    
    return overall_success_rate, successful_problems, total_problems

def main(
    filenames: List[str] = FILE_NAMES,
    eval_type: str = "all"
):
    """
    Perform evaluation on JSONL files.
    
    Args:
        filenames: List of JSONL files to evaluate
        eval_type: Type of evaluation ("individual", "iterative", "pass_at_n", "all")
    """
    
    # Check if files exist
    for filename in filenames:
        if not Path(filename).exists():
            print(f"Error: File {filename} does not exist!")
            return
    
    # Load all trajectories once
    print("=== LOADING ALL TRAJECTORIES ===")
    all_trajectories = load_all_trajectories(filenames)
    
    if eval_type in ["individual", "all"]:
        print("\n=== INDIVIDUAL FILE ANALYSIS ===")
        individual_results = compute_individual_scores(all_trajectories)
        
        print("\n=== SUMMARY OF INDIVIDUAL RESULTS ===")
        for filename, (success_rate, num_success, num_total) in individual_results.items():
            print(f"{Path(filename).name}: {success_rate*100:.2f}% ({num_success}/{num_total})")
    
    if eval_type in ["iterative", "all"]:
        print("\n" + "="*50)
        print("=== ITERATIVE EVALUATION ===")
        iterative_success_rate, iterative_success, iterative_total = iterative_evaluation(all_trajectories, filenames)
    
    if eval_type in ["pass_at_n", "all"]:
        print("\n" + "="*50)
        print("=== PASS@N EVALUATION ===")
        
        # Evaluate pass@1, pass@3, pass@5, and pass@all
        for n in range(1, len(filenames) + 1):
            pass_success_rate, pass_success, pass_total = pass_at_n_evaluation(all_trajectories, filenames, n)

if __name__ == "__main__":
    fire.Fire(main) 