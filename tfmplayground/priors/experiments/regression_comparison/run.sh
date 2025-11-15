#!/bin/bash
# Run script for regression comparison experiments

echo "=================================================="
echo "Regression Comparison Experiments"
echo "=================================================="
echo ""

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "config.yaml not found!"
    echo "Please create a config.yaml file first."
    exit 1
fi

# Step 1: Data Generation
echo "Step 1: Data Generation"
echo "--------------------------------------------------"
echo "This will generate synthetic data from selected priors."
read -p "Run data generation? (y/n): " run_gen

if [ "$run_gen" = "y" ] || [ "$run_gen" = "Y" ]; then
    python 1_data_generation.py
    if [ $? -ne 0 ]; then
        echo "Data generation failed!"
        exit 1
    fi
fi

echo ""
echo "Step 2: Data Analysis"
echo "--------------------------------------------------"
echo "This will analyze the generated data and create reports."
read -p "Run data analysis? (y/n): " run_analysis

if [ "$run_analysis" = "y" ] || [ "$run_analysis" = "Y" ]; then
    python 2_run_analysis.py
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
