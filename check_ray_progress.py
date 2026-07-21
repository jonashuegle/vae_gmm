#!/usr/bin/env python3

import os
import json
import argparse
from datetime import datetime

def check_ray_progress(experiment_dir):
    """Reports the progress of a Ray Tune experiment."""

    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        return

    trials_dir = os.path.join(experiment_dir, "vae_gmm")
    if not os.path.exists(trials_dir):
        print(f"No trials found in: {trials_dir}")
        return

    trial_dirs = [d for d in os.listdir(trials_dir) if d.startswith("train_vae_")]

    if not trial_dirs:
        print(f"No trial directories found in: {trials_dir}")
        return

    print(f"Experiment: {os.path.basename(experiment_dir)}")
    print(f"Path: {experiment_dir}")
    print(f"Trials found: {len(trial_dirs)}")

    completed = 0
    running = 0
    error = 0

    best_silhouette = -float('inf')
    best_trial_id = None

    for trial_dir in trial_dirs:
        full_path = os.path.join(trials_dir, trial_dir)

        result_file = os.path.join(full_path, "result.json")
        params_file = os.path.join(full_path, "params.json")

        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)

                completed += 1

                if "silhouette" in result:
                    silhouette = result["silhouette"]
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_trial_id = trial_dir
            except Exception:
                # A half-written result.json means the trial is still running.
                running += 1
        elif os.path.exists(params_file):
            # Started, but no result yet.
            running += 1
        else:
            error += 1

    print("SUMMARY:")
    print(f"Completed trials: {completed}")
    print(f"Running trials:   {running}")
    print(f"Failed trials:    {error}")

    if best_trial_id:
        print(f"Best trial:       {best_trial_id}")
        print(f"Silhouette score: {best_silhouette:.4f}")

    print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check the progress of a Ray Tune run.')
    parser.add_argument(
        '--dir', type=str,
        default=os.getenv("RAY_LOG_DIR", "./logs_ray"),
        help='Experiment directory',
    )

    args = parser.parse_args()
    check_ray_progress(args.dir)
