import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig, T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=5,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")
    parser.add_argument('--eval_every_n_epochs', type=int, default=5,
                        help="Evaluate on dev set every N epochs")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='ft_experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0
    epochs_since_last_eval = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = args.experiment_name
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss:.4f}\n")

        epochs_since_last_eval += 1

        # Only evaluate every N epochs or on the last epoch
        should_evaluate = (epochs_since_last_eval >= args.eval_every_n_epochs) or (epoch == args.max_n_epochs - 1)
        
        if should_evaluate:
            epochs_since_last_eval = 0  # Reset counter
            
            eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                             gt_sql_path, model_sql_path,
                                                                             gt_record_path, model_record_path)
            print(f"Epoch {epoch}: Dev loss: {eval_loss:.4f}, Record F1: {record_f1:.4f}, Record EM: {record_em:.4f}, SQL EM: {sql_em:.4f}")
            print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors\n")

            if args.use_wandb:
                result_dict = {
                    'train/loss' : tr_loss,
                    'dev/loss' : eval_loss,
                    'dev/record_f1' : record_f1,
                    'dev/record_em' : record_em,
                    'dev/sql_em' : sql_em,
                    'dev/error_rate' : error_rate,
                    'epoch': epoch
                }
                wandb.log(result_dict, step=epoch)

            # Update best model tracking
            if record_f1 > best_f1:
                best_f1 = record_f1
                epochs_since_improvement = 0
                save_model(checkpoint_dir, model, best=True)
                print(f"  → New best F1: {best_f1:.4f}")
            else:
                epochs_since_improvement += args.eval_every_n_epochs
                
            # Check early stopping (only on eval epochs)
            # if epochs_since_imp/rovement >= args.patience_epochs:
                # print(f"Early stopping after {epoch + 1} epochs ({epochs_since_improvement} epochs without improvement)")
                # break
        else:
            # If not evaluating, just log training loss
            if args.use_wandb:
                wandb.log({'train/loss': tr_loss, 'epoch': epoch}, step=epoch)

        # Always save last checkpoint
        save_model(checkpoint_dir, model, best=False)

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    '''
    Evaluation loop for training. Computes model loss on SQL queries and various metrics
    including exact match and F1 scores.
    '''
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()
    
    # Initialize tokenizer for generation
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # Lists to store generated and ground truth queries
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader, desc="Evaluating"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            
            # Compute loss
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']
            
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode generated queries
            for gen_ids in generated_ids:
                generated_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(generated_query)
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    
    # Save generated queries and compute records
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)
    
    # Compute metrics
    sql_em, record_em, record_f1, model_error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    
    # Compute error rate
    error_rate = sum(1 for msg in model_error_msgs if msg != "") / len(model_error_msgs) if model_error_msgs else 0
    
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    Inference function to compute model's generated SQL queries and associated database records
    for the test set.
    '''
    model.eval()
    
    # Initialize tokenizer for generation
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    # List to store generated queries
    generated_queries = []
    
    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader, desc="Test Inference"):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            
            # Generate SQL queries
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_length=512,
                num_beams=5,
                early_stopping=True
            )
            
            # Decode generated queries
            for gen_ids in generated_ids:
                generated_query = tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_queries.append(generated_query)
    
    # Save generated queries and compute records
    print(f"Saving {len(generated_queries)} test predictions to {model_sql_path}")
    save_queries_and_records(generated_queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    print("\n" + "="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model: {'Fine-tuning' if args.finetune else 'From scratch'}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_n_epochs}")
    print(f"Evaluate every: {args.eval_every_n_epochs} epochs")
    print(f"Patience: {args.patience_epochs} epochs")
    print(f"Device: {DEVICE}")
    print("="*80 + "\n")

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    print("\n" + "="*80)
    print("FINAL EVALUATION ON DEV SET")
    print("="*80)
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss:.4f}, Record F1: {dev_record_f1:.4f}, Record EM: {dev_record_em:.4f}, SQL EM: {dev_sql_em:.4f}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    print("\n" + "="*80)
    print("GENERATING TEST SET PREDICTIONS")
    print("="*80)
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)
    print(f"\nTest predictions saved to:")
    print(f"  - {model_sql_path}")
    print(f"  - {model_record_path}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()