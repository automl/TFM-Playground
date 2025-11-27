"""Generate synthetic regression data using configurable priors.
This script uses the main tfmplayground.priors library with configuration from config.yaml.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List

from utils import load_config


def generate_prior_data(
    prior_name: str,
    lib: str,
    prior_type: str,
    config: Dict,
    save_dir: str,
) -> str:
    """Generate synthetic data from a prior using the main CLI.
    
    Args:
        prior_name: Name identifier for the prior (e.g., 'ticl_gp')
        lib: Library name (e.g., 'ticl')
        prior_type: Prior type (e.g., 'gp', 'mlp')
        config: Configuration dictionary
        save_dir: Directory to save the HDF5 file
        
    Returns:
        Path to the generated HDF5 file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # extract data generation settings
    gen_config = config['data_generation']
    
    # create output filename (absolute path)
    save_path = os.path.abspath(os.path.join(
        save_dir,
        f"prior_{prior_name}_{gen_config['num_batches']}x{gen_config['batch_size']}_"
        f"{gen_config['max_seq_len']}x{gen_config['max_features']}.h5"
    ))
    
    print(f"\nGenerating data for {prior_name.upper()}...")
    print(f"  Library: {lib}")
    print(f"  Prior type: {prior_type}")
    print(f"  Batches: {gen_config['num_batches']}")
    print(f"  Batch size: {gen_config['batch_size']}")
    print(f"  Max sequence length: {gen_config['max_seq_len']}")
    print(f"  Max features: {gen_config['max_features']}")
    print(f"  Seed: {gen_config['seed']}")
    print(f"  Output: {save_path}")
    
    # get project root
    project_root = Path(__file__).parent.parent.parent
    
    # build command to call the main CLI
    cmd = [
        sys.executable, "-m", "tfmplayground.priors",
        "--lib", lib,
        "--prior_type", prior_type,
        "--num_batches", str(gen_config['num_batches']),
        "--batch_size", str(gen_config['batch_size']),
        "--max_seq_len", str(gen_config['max_seq_len']),
        "--max_features", str(gen_config['max_features']),
        "--min_eval_pos", str(gen_config['min_eval_pos']),
        "--device", gen_config['device'],
        "--save_path", save_path,
        "--np_seed", str(gen_config['seed']),
        "--torch_seed", str(gen_config['seed']),
    ]
    
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


def main():
    """Generate data for selected priors using configuration."""
    mode = sys.argv[1]  # Mode: classification, regression
    
    config = load_config(str(Path(__file__).parent / "config.yaml"))
    available_priors = config['available_priors'][mode]
    save_dir = config['output']['data_dir']
    
    print("=" * 50)
    print("DATA GENERATION")
    print("=" * 50)
    print(f"Output: {save_dir}")
    
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
        )
        generated_files[prior_name] = file_path
    
    print("\n" + "=" * 50)
    print("DATA GENERATION COMPLETE")
    print("=" * 50)
    for prior_name, file_path in generated_files.items():
        print(f"[OK] {prior_name:<15}: {file_path}")


if __name__ == "__main__":
    main()
