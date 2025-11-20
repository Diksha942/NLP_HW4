import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    '''Initialize Weights & Biases for experiment tracking'''
    wandb.init(
        project="text-to-sql-t5",
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "max_epochs": args.max_n_epochs,
            "optimizer": args.optimizer_type,
            "scheduler": args.scheduler_type,
            "weight_decay": args.weight_decay,
            "num_warmup_epochs": args.num_warmup_epochs,
            "finetune": args.finetune,
            "max_gen_length": args.max_gen_length,
        }
    )

def initialize_model(args):
    '''
    Initialize T5 model - either load pretrained or initialize from scratch.
    
    Args:
        args: Must have args.finetune (bool)
    
    Returns:
        model: T5ForConditionalGeneration on DEVICE
    '''
    model_name = 'google-t5/t5-small'
    
    if args.finetune:
        print(f"Loading pretrained T5 model from {model_name}...")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        print(f"✓ Loaded pretrained T5-small")
    else:
        print(f"Initializing T5 from scratch with {model_name} config...")
        config = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(config)
        model.apply(model._init_weights)
        print(f"✓ Initialized T5-small from random weights")
    
    model = model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {DEVICE}")
    
    return model

def mkdir(dirpath):
    '''Create directory if it doesn't exist'''
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best=False):
    '''
    Save model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model to save
        best: If True, save as best_model.pt, else latest_model.pt
    '''
    mkdir(checkpoint_dir)
    
    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving best model to {save_path}")
    else:
        save_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

def load_model_from_checkpoint(args, best=True):
    '''
    Load model from checkpoint.
    
    Args:
        args: Must have args.checkpoint_dir and args.finetune
        best: If True, load best_model.pt, else latest_model.pt
    
    Returns:
        model: Loaded model
    '''
    # Initialize model architecture
    model = initialize_model(args)
    
    # Load checkpoint
    if best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest_model.pt')
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return model
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded checkpoint successfully")
    
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    '''Initialize optimizer and learning rate scheduler'''
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    '''
    Initialize AdamW optimizer with weight decay.
    Weight decay is NOT applied to bias and layer norm parameters.
    '''
    # Get parameter names that should have weight decay
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    # Split parameters into two groups
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer_type} not implemented")

    print(f"Initialized AdamW: lr={args.learning_rate}, weight_decay={args.weight_decay}")
    return optimizer

def initialize_scheduler(args, optimizer, epoch_length):
    '''
    Initialize learning rate scheduler with warmup.
    
    Args:
        args: Must have scheduler_type, max_n_epochs, num_warmup_epochs
        optimizer: Optimizer to schedule
        epoch_length: Number of batches per epoch
    
    Returns:
        scheduler or None
    '''
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        print("No LR scheduler")
        return None
    elif args.scheduler_type == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        print(f"Cosine scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
        return scheduler
    elif args.scheduler_type == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        print(f"Linear scheduler: {num_warmup_steps} warmup steps, {num_training_steps} total steps")
        return scheduler
    else:
        raise NotImplementedError(f"Scheduler {args.scheduler_type} not implemented")

def get_parameter_names(model, forbidden_layer_types):
    '''
    Get all parameter names, excluding forbidden layer types (LayerNorm).
    Used to determine which parameters should have weight decay.
    '''
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result