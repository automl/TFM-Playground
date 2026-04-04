"""Generate synthetic regression data using configurable priors.
This script uses the main tfmplayground.priors library with configuration from config.yaml.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List

from tfmplayground.priors.experiments.utils.general import load_config


def generate_prior_data(
    prior_name: str,
    lib: str,
    prior_type: str,
    config: Dict,
    save_dir: str,
    max_classes: int,
    prior_info: Dict = None,
) -> str:
    """Generate synthetic data from a prior using the main CLI.
    
    Args:
        prior_name: Name identifier for the prior (e.g., 'ticl_gp')
        lib: Library name (e.g., 'ticl')
        prior_type: Prior type (e.g., 'gp', 'mlp')
        config: Configuration dictionary
        save_dir: Directory to save the HDF5 file
        max_classes: Maximum number of classes (0 for regression, >0 for classification)
        
    Returns:
        Path to the generated HDF5 file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # extract data generation settings
    gen_config = config['data_generation']
    
    # real data prior is always batch_size=1 (enforced by the CLI)
    # to keep total sample count equal (num_batches × batch_size),
    # multiply num_batches by the configured batch_size for real priors
    effective_batch_size = 1 if lib == "real" else gen_config['batch_size']
    effective_num_batches = gen_config['num_batches'] * gen_config['batch_size'] if lib == "real" else gen_config['num_batches']
    total_samples = effective_num_batches * effective_batch_size
    
    # create output filename (absolute path)
    save_path = os.path.abspath(os.path.join(
        save_dir,
        f"prior_{prior_name}_{effective_num_batches}x{effective_batch_size}_"
        f"{gen_config['max_seq_len']}x{gen_config['max_features']}.h5"
    ))
    
    print(f"\nGenerating data for {prior_name.upper()}...")
    print(f"  Library: {lib}")
    print(f"  Prior type: {prior_type}")
    print(f"  Batches: {effective_num_batches}")
    print(f"  Batch size: {effective_batch_size}")
    print(f"  Total samples: {total_samples}")
    print(f"  Max sequence length: {gen_config['max_seq_len']}")
    print(f"  Max features: {gen_config['max_features']}")
    print(f"  Seed: {gen_config['seed']}")
    print(f"  Output: {save_path}")
    
    # get project root
    project_root = Path(__file__).parent.parent.parent.parent
    
    # build command to call the main CLI
    cmd = [
        sys.executable, "-m", "tfmplayground.priors",
        "--lib", lib,
        "--prior_type", prior_type,
        "--max_classes", str(max_classes),
        "--num_batches", str(effective_num_batches),
        "--batch_size", str(effective_batch_size),
        "--max_seq_len", str(gen_config['max_seq_len']),
        "--max_features", str(gen_config['max_features']),
        "--min_eval_pos", str(gen_config['min_eval_pos']),
        "--device", gen_config['device'],
        "--save_path", save_path,
        "--np_seed", str(gen_config['seed']),
        "--torch_seed", str(gen_config['seed']),
    ]

    # append real-data-specific args
    if lib == "real":
        real_cfg = config['real_data']
        sampling_mode = (prior_info or {}).get('sampling_mode', 'only')
        task_type = "classification" if max_classes > 0 else "regression"

        # pick the right pool for the sampling_mode
        if sampling_mode == "mixed":
            train_pool = real_cfg['pools']['all']
            fallback_pool = real_cfg['pools'][task_type]
        else:
            train_pool = real_cfg['pools'][task_type]
            fallback_pool = None

        cmd += ["--cache_dir", real_cfg['cache_dir'],
                "--train_pool", train_pool,
                "--mode", sampling_mode]
        if fallback_pool:
            cmd += ["--fallback_pool", fallback_pool]
    
    # run the subprocess command from project root
    print(cmd)
    print("="*50)
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(project_root))
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate {prior_name} data")
    
    print(f"[OK] Data saved to {save_path}")
    return save_path


def select_priors_interactive(available_priors: Dict) -> List[str]:
    """Interactive CLI to select which priors to generate.
    
    Args:
        available_priors: Dictionary of available priors from config
        
    Returns:
        List of selected prior names
    """
    print("\n" + "=" * 50)
    print("AVAILABLE PRIORS")
    print("=" * 50)
    
    prior_list = list(available_priors.keys())
    for i, (name, info) in enumerate(available_priors.items(), 1):
        print(f"{i}. {name:<15} - {info['description']}")
    
    print("\n" + "=" * 50)
    print("SELECT PRIORS TO GENERATE")
    print("=" * 50)
    print("Options:")
    print("  - Enter numbers separated by commas (e.g., '1,2')")
    print("  - Enter 'all' to select all priors")
    print("  - Enter 'quit' to exit")
    
    while True:
        selection = input("\nYour selection: ").strip().lower()
        
        if selection == 'quit':
            print("Exiting...")
            sys.exit(0)
        
        if selection == 'all':
            return prior_list
        
        try:
            # parse comma-separated numbers
            indices = [int(x.strip()) for x in selection.split(',')]
            selected = [prior_list[i - 1] for i in indices if 1 <= i <= len(prior_list)]
            
            if not selected:
                print(f"Invalid selection. Please enter numbers between 1 and {len(prior_list)}.")
                continue
            
            # confirm selection
            print("\nYou selected:")
            for name in selected:
                print(f"  - {name}: {available_priors[name]['description']}")
            
            confirm = input("\nProceed with these priors? (y/n): ").strip().lower()
            if confirm == 'y':
                return selected
            
        except (ValueError, IndexError):
            print(f"Invalid input. Please enter numbers between 1 and {len(prior_list)} or 'all'.")


def resolve_priors(priors_arg: List[str], available_priors: Dict) -> List[str]:
    """Resolve --priors CLI argument to a list of prior names."""
    if priors_arg == ["all"]:
        return list(available_priors.keys())

    unknown = [p for p in priors_arg if p not in available_priors]
    if unknown:
        print(f"ERROR: Unknown prior(s): {unknown}")
        print(f"Available: {list(available_priors.keys())}")
        sys.exit(1)
    return priors_arg


def main():
    """Generate data for selected priors using configuration."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data from configurable priors"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Task type: classification or regression",
    )
    parser.add_argument(
        "--priors",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Prior(s) to generate (non-interactive). "
            "Use 'all' for every prior, or list names e.g. 'ticl_gp tabpfn_mlp'"
            "Leave it empty for interactive selection."
        ),
    )
    args = parser.parse_args()

    mode = args.mode
    config = load_config(str(Path(__file__).parent / "config.yaml"))
    available_priors = config['available_priors'][mode]
    save_dir = Path(mode) / config["output"]["data_dir"]
    
    # set max_classes: 0 for regression, from config for classification
    max_classes = 0 if mode == 'regression' else config['data_generation']['max_classes_classification']
    
    print("=" * 50)
    print("DATA GENERATION")
    print("=" * 50)
    print(f"Mode: {mode.upper()}")
    print(f"Output: {save_dir}")
    
    # non-interactive when --priors is given, interactive otherwise
    if args.priors is not None:
        selected_priors = resolve_priors(args.priors, available_priors)
        print(f"\nSelected priors: {selected_priors}")
    else:
        selected_priors = select_priors_interactive(available_priors)
    
    print("\n" + "=" * 50)
    print("GENERATING DATA")
    print("=" * 50)
    
    generated_files = {}
    for prior_name in selected_priors:
        prior_info = available_priors[prior_name]
        file_path = generate_prior_data(
            prior_name=prior_name,
            lib=prior_info['lib'],
            prior_type=prior_info['prior_type'],
            config=config,
            save_dir=save_dir,
            max_classes=max_classes,
            prior_info=prior_info,
        )
        generated_files[prior_name] = file_path
    
    print("\n" + "=" * 50)
    print("DATA GENERATION COMPLETE")
    print("=" * 50)
    for prior_name, file_path in generated_files.items():
        print(f"[OK] {prior_name:<15}: {file_path}")


if __name__ == "__main__":
    main()
