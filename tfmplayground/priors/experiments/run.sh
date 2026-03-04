#!/bin/bash
# Run script for comparison experiments
#
# Usage:
#   sh run.sh classification              # interactive
#   sh run.sh regression --priors all      # non-interactive data generation (SLURM)
#   sh run.sh regression --priors ticl_gp tabpfn_mlp

mode=$1  # Mode: classification, regression
shift    # remaining args are forwarded as --priors to generate_data.py

echo "=================================================="
echo "Comparison Experiments - Mode: $mode"
echo "=================================================="
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "config.yaml not found!"
    echo "Please create a config.yaml file first."
    exit 1
fi

# Forward remaining args (e.g. --priors ticl_gp tabpfn_mlp) to generate_data.py
extra_args="$@"

# Step 1: Data Generation
echo "Step 1: Data Generation"
echo "--------------------------------------------------"
if [ -n "$extra_args" ]; then
    echo "Running non-interactively with: $extra_args"
    python generate_data.py --mode $mode $extra_args
    if [ $? -ne 0 ]; then
        echo "Data generation failed!"
        exit 1
    fi
else
    echo "This will generate synthetic data from selected priors."
    read -p "Run data generation? (y/n): " run_gen
    if [ "$run_gen" = "y" ] || [ "$run_gen" = "Y" ]; then
        python generate_data.py --mode $mode
        if [ $? -ne 0 ]; then
            echo "Data generation failed!"
            exit 1
        fi
    fi
fi

echo ""
echo "Step 2: Data Analysis"
echo "--------------------------------------------------"
echo "This will analyze the generated data and create reports."
read -p "Run data analysis? (y/n): " run_analysis

if [ "$run_analysis" = "y" ] || [ "$run_analysis" = "Y" ]; then
    python analyze_priors.py --mode $mode
    if [ $? -ne 0 ]; then
        echo "Data analysis failed!"
        exit 1
    fi
fi

echo ""
echo "=================================================="
echo "Experiment Complete"
echo "=================================================="
echo "Check the results directory for outputs."
