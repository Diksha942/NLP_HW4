"""
Comprehensive evaluation script for trained T5 models
Supports both dev and test set evaluation

Usage Examples:
  # Evaluate on dev set
  python evaluate_model.py --checkpoint_path checkpoints/ft_experiments/ft_experiment/best_model.pt --split dev
  
  # Evaluate on test set
  python evaluate_model.py --checkpoint_path checkpoints/ft_experiments/ft_experiment/best_model.pt --split test
  
  # Evaluate on both
  python evaluate_model.py --checkpoint_path checkpoints/ft_experiments/ft_experiment/best_model.pt --split both
"""

import os
import argparse
import torch
from tqdm import tqdm

from t5_utils import initialize_model
from transformers import T5TokenizerFast
from load_data import get_dataloader
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate trained T5 model')
    
    # Model loading
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--finetune', action='store_true', default=True,
                        help='Whether model was finetuned')
    
    # Evaluation settings
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test', 'both'],
                        help='Which split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    
    # Output
    parser.add_argument('--experiment_name', type=str, default='eval',
                        help='Name for output files')
    
    return parser.parse_args()

def generate_predictions(model, dataloader, tokenizer, split):
    """Generate predictions for a dataset split"""
    model.eval()
    generated_queries = []
    
    print(f"\nGenerating predictions for {split} set...")
    
    with torch.no_grad():
        if split == 'test':
            # Test set uses different collate function
            for encoder_input, encoder_mask, _ in tqdm(dataloader, desc=f"Generating ({split})"):
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)
                
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
                
                for gen_ids in generated_ids:
                    query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    generated_queries.append(query)
        else:
            # Dev/train sets
            for encoder_input, encoder_mask, _, _, _ in tqdm(dataloader, desc=f"Generating ({split})"):
                encoder_input = encoder_input.to(DEVICE)
                encoder_mask = encoder_mask.to(DEVICE)
                
                generated_ids = model.generate(
                    input_ids=encoder_input,
                    attention_mask=encoder_mask,
                    max_length=512,
                    num_beams=5,
                    early_stopping=True
                )
                
                for gen_ids in generated_ids:
                    query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    generated_queries.append(query)
    
    print(f"✓ Generated {len(generated_queries)} predictions")
    return generated_queries

def evaluate_split(args, model, tokenizer, split):
    """Evaluate model on a specific split"""
    print("\n" + "="*80)
    print(f"EVALUATING ON {split.upper()} SET")
    print("="*80)
    
    # Load data
    dataloader = get_dataloader(args.batch_size, split)
    print(f"Loaded {len(dataloader.dataset)} examples")
    
    # Setup paths
    model_type = 'ft' if args.finetune else 'scr'
    model_sql_path = f'results/t5_{model_type}_{args.experiment_name}_{split}.sql'
    model_record_path = f'records/t5_{model_type}_{args.experiment_name}_{split}.pkl'
    
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    # Generate predictions
    predictions = generate_predictions(model, dataloader, tokenizer, split)
    
    # Save predictions
    print(f"\nSaving predictions...")
    save_queries_and_records(predictions, model_sql_path, model_record_path)
    print(f"✓ Saved to {model_sql_path}")
    print(f"✓ Saved to {model_record_path}")
    
    # Compute metrics (only for dev set)
    if split == 'dev':
        gt_sql_path = f'data/{split}.sql'
        gt_record_path = f'records/ground_truth_{split}.pkl'
        
        print(f"\nComputing metrics...")
        sql_em, record_em, record_f1, error_msgs = compute_metrics(
            gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        
        error_rate = sum(1 for msg in error_msgs if msg != "") / len(error_msgs) if error_msgs else 0
        
        # Print results
        print("\n" + "-"*80)
        print("RESULTS")
        print("-"*80)
        print(f"Record F1:        {record_f1:.4f} ({record_f1*100:.2f}%)")
        print(f"Record EM:        {record_em:.4f} ({record_em*100:.2f}%)")
        print(f"SQL EM:           {sql_em:.4f} ({sql_em*100:.2f}%)")
        print(f"Error Rate:       {error_rate:.4f} ({error_rate*100:.2f}%)")
        print("-"*80)
        
        if record_f1 >= 0.65:
            print("✅ Achieves ≥65% F1 (required for full credit)")
        else:
            print(f"⚠️  Achieves {record_f1*100:.2f}% F1 (need ≥65% for full credit)")
        
        return record_f1
    else:
        print(f"\n✓ Test predictions generated (no ground truth for metrics)")
        return None

def main():
    args = get_args()
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Checkpoint:  {args.checkpoint_path}")
    print(f"Split:       {args.split}")
    print(f"Experiment:  {args.experiment_name}")
    print(f"Device:      {DEVICE}")
    print("="*80)
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"\n❌ ERROR: Checkpoint not found at {args.checkpoint_path}")
        print("\nLooking for available checkpoints...")
        
        if os.path.exists('checkpoints'):
            found = False
            for root, dirs, files in os.walk('checkpoints'):
                for file in files:
                    if file.endswith('.pt'):
                        print(f"  ✓ {os.path.join(root, file)}")
                        found = True
            if not found:
                print("  No checkpoints found in checkpoints/ directory")
        else:
            print("  checkpoints/ directory not found")
        return
    
    # Initialize model
    print("\nInitializing model...")
    model = initialize_model(args)
    
    # Load checkpoint
    print(f"Loading weights from checkpoint...")
    checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    print("✓ Model loaded successfully")
    
    # Load tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Evaluate
    if args.split == 'both':
        # Evaluate on both dev and test
        evaluate_split(args, model, tokenizer, 'dev')
        evaluate_split(args, model, tokenizer, 'test')
    else:
        # Evaluate on single split
        evaluate_split(args, model, tokenizer, args.split)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()