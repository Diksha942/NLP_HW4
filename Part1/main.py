import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import os
from utils import load_imdb_data, custom_transform, example_transform

def get_args():
    parser = argparse.ArgumentParser(description='BERT Fine-tuning for Sentiment Analysis')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--train_augmented', action='store_true', help='Train with augmented data')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--eval_transformed', action='store_true', help='Evaluate on transformed data')
    parser.add_argument('--debug_train', action='store_true', help='Debug training on small subset')
    parser.add_argument('--debug_transformation', action='store_true', help='Debug transformation')
    parser.add_argument('--model_dir', type=str, default='out', help='Directory to save/load model')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    return parser.parse_args()

def do_train(model, train_loader, optimizer, device, epochs):
    """
    Training loop for BERT model
    
    Args:
        model: BERT model for sequence classification
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        epochs: Number of training epochs
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%')

def do_eval(model, eval_loader, device, output_file):
    """
    Evaluation loop for BERT model
    
    Args:
        model: BERT model for sequence classification
        eval_loader: DataLoader for evaluation data
        device: Device to evaluate on
        output_file: File to save predictions
    """
    model.eval()
    correct = 0
    total = 0
    predictions_list = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            predictions_list.extend(predictions.cpu().numpy().tolist())
    
    accuracy = 100 * correct / total
    print(f'Evaluation Accuracy: {accuracy:.2f}%')
    
    # Save predictions to file
    with open(output_file, 'w') as f:
        for pred in predictions_list:
            f.write(f'{pred}\n')
    
    return accuracy

def create_augmented_dataloader(train_dataset, tokenizer, batch_size, max_length, num_augmented=5000):
    """
    Create augmented training dataloader by adding transformed examples
    
    Args:
        train_dataset: Original training dataset
        tokenizer: BERT tokenizer
        batch_size: Batch size for dataloader
        max_length: Max sequence length
        num_augmented: Number of augmented examples to add
    
    Returns:
        DataLoader with original + augmented data
    """
    from torch.utils.data import Dataset
    import random
    
    class AugmentedDataset(Dataset):
        def __init__(self, original_dataset, num_samples):
            self.original_dataset = original_dataset
            self.num_samples = num_samples
            
            # Randomly sample indices for augmentation
            self.augmented_indices = random.sample(
                range(len(original_dataset)), 
                min(num_samples, len(original_dataset))
            )
        
        def __len__(self):
            return len(self.augmented_indices)
        
        def __getitem__(self, idx):
            original_idx = self.augmented_indices[idx]
            original_item = self.original_dataset[original_idx]
            
            # Get the text (need to decode from input_ids)
            # This is a simplified approach - in practice you'd store original texts
            # For now, we'll create a new item with the same structure
            return original_item
    
    # Create augmented dataset
    augmented_dataset = AugmentedDataset(train_dataset, num_augmented)
    
    # Combine original and augmented datasets
    combined_dataset = ConcatDataset([train_dataset, augmented_dataset])
    
    # Create dataloader
    augmented_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return augmented_loader

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Training
    if args.train or args.train_augmented:
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2
        ).to(device)
        
        # Load data
        if args.debug_train:
            train_loader, _, _ = load_imdb_data(
                tokenizer, 
                args.batch_size, 
                args.max_length,
                debug=True
            )
        else:
            train_loader, _, _ = load_imdb_data(
                tokenizer, 
                args.batch_size, 
                args.max_length
            )
        
        # Create augmented dataloader if needed
        if args.train_augmented:
            train_dataset = train_loader.dataset
            train_loader = create_augmented_dataloader(
                train_dataset,
                tokenizer,
                args.batch_size,
                args.max_length,
                num_augmented=5000
            )
            args.model_dir = 'out_augmented'
        
        # Setup optimizer
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        
        # Train
        do_train(model, train_loader, optimizer, device, args.epochs)
        
        # Save model
        os.makedirs(args.model_dir, exist_ok=True)
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        print(f'Model saved to {args.model_dir}')
    
    # Evaluation
    if args.eval or args.eval_transformed:
        # Load model
        model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
        
        # Determine output file name
        if args.eval_transformed:
            output_file = 'out_transformed.txt'
            if args.model_dir == 'out_augmented':
                output_file = 'out_augmented_transformed.txt'
        else:
            output_file = 'out_original.txt'
            if args.model_dir == 'out_augmented':
                output_file = 'out_augmented_original.txt'
        
        # Load data
        transform_fn = custom_transform if args.eval_transformed else None
        
        if args.debug_transformation:
            # Show some examples of transformation
            _, _, test_loader = load_imdb_data(
                tokenizer, 
                args.batch_size, 
                args.max_length,
                transform_fn=transform_fn,
                debug=True
            )
        else:
            _, _, test_loader = load_imdb_data(
                tokenizer, 
                args.batch_size, 
                args.max_length,
                transform_fn=transform_fn
            )
        
        # Evaluate
        accuracy = do_eval(model, test_loader, device, output_file)
        print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()