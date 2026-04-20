"""Single-command benchmark runner.

Usage:
    python benchmarks/run_all.py --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Autodidact benchmarks")
    parser.add_argument("--output", default="results", help="Output directory for results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    all_results = {}

    print("=" * 60)
    print("Autodidact Benchmark Suite")
    print("=" * 60)

    # 1. Learning Curve
    print("\n[1/3] Learning Curve Benchmark")
    print("-" * 40)
    t0 = time.time()
    from benchmarks.learning_curve import run_learning_curve
    all_results["learning_curve"] = run_learning_curve(output_dir=args.output)
    print(f"  Time: {time.time() - t0:.1f}s")

    # 2. Thompson Calibration
    print("\n[2/3] Thompson Sampling Calibration Benchmark")
    print("-" * 40)
    t0 = time.time()
    from benchmarks.thompson_calibration import run_thompson_calibration
    all_results["thompson_calibration"] = run_thompson_calibration(output_dir=args.output)
    print(f"  Time: {time.time() - t0:.1f}s")

    # 3. Ebbinghaus Decay
    print("\n[3/3] Ebbinghaus Decay Benchmark")
    print("-" * 40)
    t0 = time.time()
    from benchmarks.ebbinghaus_decay import run_ebbinghaus_decay
    all_results["ebbinghaus_decay"] = run_ebbinghaus_decay(output_dir=args.output)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Summary
    summary_path = os.path.join(args.output, "summary.json")
    summary = {
        "learning_curve": {
            "final_local_rate": all_results["learning_curve"]["final_local_rate"],
            "total_escalations": all_results["learning_curve"]["total_escalations"],
        },
        "thompson_calibration": all_results["thompson_calibration"]["final_accuracy"],
        "ebbinghaus_decay": {
            "with_decay_final_precision": all_results["ebbinghaus_decay"]["results_with_decay"][-1]["precision"],
            "without_decay_final_precision": all_results["ebbinghaus_decay"]["results_without_decay"][-1]["precision"],
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("All benchmarks complete!")
    print(f"Summary: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
