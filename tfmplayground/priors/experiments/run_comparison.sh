#!/bin/bash
# Quick test script to compare two classification models

echo "==================================================================="
echo "Model Comparison Test - Training on 2 priors for 10 epochs each"
echo "==================================================================="
echo ""
echo "This will train two nanoTabPFN models and compare their accuracy."
echo "Expected runtime: ~5-10 minutes per model (depends on hardware)"
echo ""

# Run the comparison
python compare_models.py \
    --prior1 ./results/data/prior_ticl_classification_adapter_10x8_50x3.h5 \
    --prior2 ./results/data/prior_tabicl_10x8_50x3.h5 \
    --epochs 3 \
    --batch_size 8 \
    --steps 1 \
    --lr 1e-4 \
    --seed 2402

echo ""
echo "==================================================================="
echo "Comparison complete! Check results above."
echo "==================================================================="
