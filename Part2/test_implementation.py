"""
Quick test script to verify the implementation works correctly
Run this before starting full training
"""

import torch
from transformers import T5TokenizerFast
import os

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    try:
        from t5_utils import initialize_model
        from load_data import load_t5_data, T5Dataset
        from utils import compute_metrics
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loading():
    """Test that data can be loaded"""
    print("\nTesting data loading...")
    try:
        from load_data import T5Dataset
        
        # Check if data files exist
        for split in ['train', 'dev', 'test']:
            nl_path = f'data/{split}.nl'
            if not os.path.exists(nl_path):
                print(f"✗ Missing file: {nl_path}")
                return False
        
        # Try loading dataset
        dataset = T5Dataset('data', 'train')
        print(f"✓ Loaded {len(dataset)} training examples")
        
        # Check a sample
        sample = dataset[0]
        print(f"✓ Sample data keys: {sample.keys()}")
        print(f"  - Encoder input shape: {sample['encoder_input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False

def test_dataloader():
    """Test that dataloaders work"""
    print("\nTesting dataloaders...")
    try:
        from load_data import load_t5_data
        
        train_loader, dev_loader, test_loader = load_t5_data(batch_size=4, test_batch_size=4)
        
        # Test train loader
        batch = next(iter(train_loader))
        print(f"✓ Train batch loaded: {len(batch)} components")
        print(f"  - Encoder input shape: {batch[0].shape}")
        print(f"  - Encoder mask shape: {batch[1].shape}")
        print(f"  - Decoder input shape: {batch[2].shape}")
        print(f"  - Decoder target shape: {batch[3].shape}")
        
        # Test test loader
        test_batch = next(iter(test_loader))
        print(f"✓ Test batch loaded: {len(test_batch)} components")
        
        return True
    except Exception as e:
        print(f"✗ Dataloader error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    try:
        from t5_utils import initialize_model
        import argparse
        
        # Create mock args
        args = argparse.Namespace(finetune=True)
        
        model = initialize_model(args)
        print(f"✓ Model initialized")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Device: {next(model.parameters()).device}")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forward_pass():
    """Test forward pass through model"""
    print("\nTesting forward pass...")
    try:
        from t5_utils import initialize_model
        from load_data import load_t5_data
        import argparse
        
        args = argparse.Namespace(finetune=True)
        model = initialize_model(args)
        model.eval()
        
        # Get a batch
        train_loader, _, _ = load_t5_data(batch_size=2, test_batch_size=2)
        encoder_input, encoder_mask, decoder_input, decoder_targets, _ = next(iter(train_loader))
        
        device = next(model.parameters()).device
        encoder_input = encoder_input.to(device)
        encoder_mask = encoder_mask.to(device)
        decoder_input = decoder_input.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input
            )
        
        print(f"✓ Forward pass successful")
        print(f"  - Logits shape: {outputs.logits.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test text generation"""
    print("\nTesting generation...")
    try:
        from t5_utils import initialize_model
        from transformers import T5TokenizerFast
        import argparse
        
        args = argparse.Namespace(finetune=True)
        model = initialize_model(args)
        model.eval()
        
        tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        
        # Test input
        test_input = "Show me all flights from Boston to New York"
        inputs = tokenizer(test_input, return_tensors='pt')
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50
            )
        
        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"✓ Generation successful")
        print(f"  - Input: {test_input}")
        print(f"  - Output: {output_text}")
        
        return True
    except Exception as e:
        print(f"✗ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*80)
    print("IMPLEMENTATION VERIFICATION TESTS")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("DataLoaders", test_dataloader),
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Generation", test_generation)
    ]
    
    results = []
    for name, test_func in tests:
        success = test_func()
        results.append((name, success))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to start training.")
        print("\nTo start training, run:")
        print("  bash run_training.sh")
        print("or")
        print("  python train_t5.py --finetune --experiment_name ft_experiment")
    else:
        print("\n✗ Some tests failed. Please fix the issues before training.")
    
    print("="*80)

if __name__ == "__main__":
    main()