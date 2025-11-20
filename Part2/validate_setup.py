"""
Script to validate that your setup is correct before running training.
Run this first to catch any issues early!
"""

import os
import sys

def check_file_exists(path, description):
    """Check if a file exists"""
    if os.path.exists(path):
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ MISSING {description}: {path}")
        return False

def check_imports():
    """Check if required packages can be imported"""
    print("\n" + "="*80)
    print("Checking Python package imports...")
    print("="*80)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Hugging Face Transformers'),
        ('nltk', 'NLTK'),
        ('tqdm', 'tqdm'),
        ('numpy', 'NumPy'),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name} ({package})")
        except ImportError:
            print(f"✗ MISSING: {name} ({package})")
            all_ok = False
    
    return all_ok

def check_data_files():
    """Check if data files exist"""
    print("\n" + "="*80)
    print("Checking data files...")
    print("="*80)
    
    required_files = [
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/train.nl', 'Training NL queries'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/train.sql', 'Training SQL queries'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/dev.nl', 'Dev NL queries'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/dev.sql', 'Dev SQL queries'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/test.nl', 'Test NL queries'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/flight_database.db', 'Flight database'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/flight_database.schema', 'Database schema'),
        ('/content/drive/MyDrive/uni_things/NLP/part-2-code/records/ground_truth_dev.pkl', 'Ground truth dev records'),
    ]
    
    all_ok = True
    for path, description in required_files:
        if not check_file_exists(path, description):
            all_ok = False
    
    return all_ok

def check_code_files():
    """Check if implementation files exist"""
    print("\n" + "="*80)
    print("Checking implementation files...")
    print("="*80)
    
    required_files = [
        ('load_data.py', 'Data loading'),
        ('t5_utils.py', 'T5 utilities'),
        ('train_t5.py', 'Training script'),
        ('utils.py', 'Utility functions'),
        ('evaluate.py', 'Evaluation script'),
    ]
    
    all_ok = True
    for path, description in required_files:
        if not check_file_exists(path, description):
            all_ok = False
    
    return all_ok

def test_data_loading():
    """Test if data can be loaded"""
    print("\n" + "="*80)
    print("Testing data loading...")
    print("="*80)
    
    try:
        from load_data import load_t5_data
        print("✓ Imported load_t5_data")
        
        # Try loading with small batch
        train_loader, dev_loader, test_loader = load_t5_data(2, 2)
        print(f"✓ Created data loaders")
        
        # Get one batch
        batch = next(iter(train_loader))
        print(f"✓ Retrieved training batch")
        print(f"  - Batch has {len(batch)} elements")
        
        batch = next(iter(dev_loader))
        print(f"✓ Retrieved dev batch")
        
        batch = next(iter(test_loader))
        print(f"✓ Retrieved test batch")
        print(f"  - Test batch has {len(batch)} elements")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test if model can be initialized"""
    print("\n" + "="*80)
    print("Testing model initialization...")
    print("="*80)
    
    try:
        from t5_utils import initialize_model
        import argparse
        print("✓ Imported initialize_model")
        
        # Test fine-tuning
        args = argparse.Namespace(finetune=True)
        model = initialize_model(args)
        print(f"✓ Initialized fine-tuning model")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  - Total parameters: {num_params:,}")
        
        # Test from scratch
        args = argparse.Namespace(finetune=False)
        model = initialize_model(args)
        print(f"✓ Initialized from-scratch model")
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenization():
    """Test tokenization"""
    print("\n" + "="*80)
    print("Testing tokenization...")
    print("="*80)
    
    try:
        from transformers import T5TokenizerFast
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        print("✓ Loaded T5 tokenizer")
        
        # Test encoding
        text = "show me flights from Boston to Seattle"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        print(f"✓ Encoded example text")
        print(f"  Input: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Length: {len(tokens)}")
        
        # Test decoding
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Decoded: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error with tokenization: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n" + "="*80)
    print("Checking GPU availability...")
    print("="*80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            print(f"  - Current device: {torch.cuda.current_device()}")
            print(f"  - Device name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print(f"⚠ CUDA not available - will use CPU (training will be slow)")
            return True  # Not a failure, just a warning
            
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False

def count_data_examples():
    """Count number of examples in each split"""
    print("\n" + "="*80)
    print("Data statistics...")
    print("="*80)
    
    try:
        def count_lines(path):
            with open(path, 'r') as f:
                return len(f.readlines())
        
        train_count = count_lines('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/train.nl')
        dev_count = count_lines('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/dev.nl')
        test_count = count_lines('/content/drive/MyDrive/uni_things/NLP/part-2-code/data/test.nl')
        
        print(f"✓ Training examples: {train_count}")
        print(f"✓ Dev examples: {dev_count}")
        print(f"✓ Test examples: {test_count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error counting examples: {e}")
        return False

def main():
    """Run all validation checks"""
    print("="*80)
    print("SETUP VALIDATION FOR HW4 PART 2")
    print("="*80)
    
    checks = [
        ("Package imports", check_imports),
        ("Data files", check_data_files),
        ("Code files", check_code_files),
        ("GPU availability", check_gpu),
        ("Data examples count", count_data_examples),
        ("Data loading", test_data_loading),
        ("Model initialization", test_model_initialization),
        ("Tokenization", test_tokenization),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    failed_checks = [name for name, passed in results.items() if not passed]
    
    if all_passed:
        print("\n✓ All checks passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. Run: python compute_statistics.py")
        print("  2. Run: python train_t5.py --finetune --experiment_name test")
    else:
        print(f"\n✗ {len(failed_checks)} check(s) failed:")
        for name in failed_checks:
            print(f"  - {name}")
        print("\nPlease fix the issues above before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()