#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tabulate import tabulate
from test_whitebox import main as wb_main
from src.misc_utils import load_config

def aggregate_seeds(config_path, num_seeds):
    base_cfg = load_config(config_path)
    all_results = {}

    for seed in range(num_seeds):
        cfg = base_cfg.copy()
        cfg['seed'] = seed
        cfg['experiment_name'] = f"{base_cfg['experiment_name']}_{seed}"
        cfg['uninformed_students_path'] = f"models/cifar_us_{seed}_best"

        print(f"\n--- Running seed {seed} ---")
        headers, rows = wb_main(cfg)

        for row in rows:
            k = row[0]
            metrics = []
            for x in row[1:]:
                try:
                    metrics.append(float(x))
                except ValueError:
                    metrics.append(x)
            all_results.setdefault(k, []).append(metrics)

        aggregated = []
        for k, seed_lists in all_results.items():
            n_metrics = len(seed_lists[0])
            formatted_row = [k]
            for idx in range(n_metrics):
                col_vals = [seed[idx] for seed in seed_lists]
                # numeric column?
                if all(isinstance(v, (int, float)) for v in col_vals):
                    arr = np.array(col_vals, dtype=float)
                    m, s = arr.mean(), arr.std()
                    hdr = headers[1 + idx].lower()
                    if hdr.startswith("pauc"):
                        formatted_row.append(f"{m:.5f}±{s:.6f}")
                    else:
                        formatted_row.append(f"{m:.2f}±{s:.3f}")
                else:
                    # non‐numeric: leave as is (take first)
                    formatted_row.append(col_vals[0])
            aggregated.append(formatted_row)

    return headers, aggregated

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate PGD‐whitebox results over multiple seeds."
    )
    parser.add_argument(
        '--config', type=str, default='cfg/cifar_benchmark.json',
        help='Path to the configuration file.'
    )
    parser.add_argument(
        '--num_seeds', type=int, default=10,
        help='Number of random seeds to run.'
    )
    args = parser.parse_args()

    headers, aggregated_rows = aggregate_seeds(args.config, args.num_seeds)
    

    print("\nAggregated Results (mean ± std over seeds):\n")
    table_str = tabulate(aggregated_rows, headers=headers, tablefmt='latex_raw', stralign='center')
    print(table_str)

    cfg_base = os.path.splitext(args.config)[0]
    out_fname = f"results{cfg_base[3:]}.txt"
    with open(out_fname, 'w') as f:
        f.write(table_str + "\n")
    print(f"\nSaved aggregated results to {out_fname}")

if __name__ == "__main__":
    main()
