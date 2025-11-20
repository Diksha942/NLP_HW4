#!/bin/bash

# Script to train T5 from scratch for Extra Credit
# Target: ≥50 F1 on test set for 1.5% course grade

echo "========================================"
echo "Training T5 FROM SCRATCH (Extra Credit)"
echo "========================================"

# Create necessary directories
mkdir -p data
mkdir -p results
mkdir -p records
mkdir -p checkpoints

# Training configuration for FROM SCRATCH
EXPERIMENT_NAME="scr_experiment"
FINETUNE=""  # NO --finetune flag = train from scratch
LEARNING_RATE=1e-4  # Higher LR for from scratch
WEIGHT_DECAY=0.01
BATCH_SIZE=8
TEST_BATCH_SIZE=16
MAX_EPOCHS=65  # More epochs needed for from scratch
PATIENCE=10  # More patience
WARMUP_EPOCHS=3  # More warmup
SCHEDULER="cosine"
EVAL_EVERY=10

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Training: FROM SCRATCH (random initialization)"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Max Epochs: $MAX_EPOCHS"
echo "  Evaluate Every: $EVAL_EVERY epochs"
echo "  Patience: $PATIENCE epochs"
echo "  Max Reloads: $MAX_RELOADS"
echo "  Warmup: $WARMUP_EPOCHS epochs"
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
    --optimizer_type "AdamW" \
    --eval_every_n_epochs $EVAL_EVERY \

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - results/t5_scr_${EXPERIMENT_NAME}_dev.sql"
echo "  - results/t5_scr_${EXPERIMENT_NAME}_test.sql"
echo "  - records/t5_scr_${EXPERIMENT_NAME}_dev.pkl"
echo "  - records/t5_scr_${EXPERIMENT_NAME}_test.pkl"
echo ""
echo "To evaluate on dev set, run:"
echo "python evaluate.py \\"
echo "  --predicted_sql results/t5_scr_${EXPERIMENT_NAME}_dev.sql \\"
echo "  --predicted_records records/t5_scr_${EXPERIMENT_NAME}_dev.pkl \\"
echo "  --development_sql data/dev.sql \\"
echo "  --development_records records/ground_truth_dev.pkl"
echo ""
echo "For extra credit submission, use:"
echo "  - results/t5_scr_${EXPERIMENT_NAME}_test.sql"
echo "  - records/t5_scr_${EXPERIMENT_NAME}_test.pkl"
echo "Target: ≥50% F1 on test set for 1.5% course credit"