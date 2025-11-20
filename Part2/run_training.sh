#!/bin/bash

# Script to train T5 model for Text-to-SQL task
# This script provides recommended hyperparameters to achieve good performance

echo "======================================"
echo "Training T5 for Text-to-SQL"
echo "======================================"

# Create necessary directories
mkdir -p data
mkdir -p results
mkdir -p records
mkdir -p checkpoints

# Training configuration
EXPERIMENT_NAME="ft_experiment"
FINETUNE="--finetune"  # Use --finetune for fine-tuning, remove for training from scratch
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
BATCH_SIZE=8
TEST_BATCH_SIZE=8
MAX_EPOCHS=35
PATIENCE=5
WARMUP_EPOCHS=1
SCHEDULER="cosine"

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Patience: $PATIENCE"
echo ""

# Run training
python train_t5.py \
    $FINETUNE \
    --experiment_name "$EXPERIMENT_NAME" \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --batch_size $BATCH_SIZE \
    --test_batch_size $TEST_BATCH_SIZE \
    --max_n_epochs $MAX_EPOCHS \
    --patience_epochs $PATIENCE \
    --num_warmup_epochs $WARMUP_EPOCHS \
    --scheduler_type "$SCHEDULER" \
    --optimizer_type "AdamW"

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
echo ""
echo "Results saved in:"
echo "  - results/t5_ft_${EXPERIMENT_NAME}_dev.sql"
echo "  - results/t5_ft_${EXPERIMENT_NAME}_test.sql"
echo "  - records/t5_ft_${EXPERIMENT_NAME}_dev.pkl"
echo "  - records/t5_ft_${EXPERIMENT_NAME}_test.pkl"
echo ""
echo "To evaluate on dev set, run:"
echo "python evaluate.py \\"
echo "  --predicted_sql results/t5_ft_${EXPERIMENT_NAME}_dev.sql \\"
echo "  --predicted_records records/t5_ft_${EXPERIMENT_NAME}_dev.pkl \\"
echo "  --development_sql data/dev.sql \\"
echo "  --development_records records/ground_truth_dev.pkl"