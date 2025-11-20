from load_data import load_t5_data
from transformers import T5TokenizerFast

tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

train_loader, dev_loader, test_loader = load_t5_data(2, 2)

print("Checking first batch...")
for batch in train_loader:
    encoder_ids, encoder_mask, decoder_input, decoder_target, _ = batch
    
    print(f"\nBatch shapes:")
    print(f"  Encoder input: {encoder_ids.shape}")
    print(f"  Decoder input: {decoder_input.shape}")
    print(f"  Decoder target: {decoder_target.shape}")
    
    print(f"\nFirst example:")
    print(f"  Encoder (NL): {tokenizer.decode(encoder_ids[0], skip_special_tokens=True)}")
    print(f"  Decoder input: {tokenizer.decode(decoder_input[0], skip_special_tokens=True)}")
    print(f"  Decoder target: {tokenizer.decode(decoder_target[0], skip_special_tokens=True)}")
    
    print(f"\nTokens:")
    print(f"  Encoder IDs: {encoder_ids[0][:20].tolist()}")
    print(f"  Decoder input IDs: {decoder_input[0][:20].tolist()}")
    print(f"  Decoder target IDs: {decoder_target[0][:20].tolist()}")
    
    break
    