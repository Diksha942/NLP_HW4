"""
Script to compute data statistics for Q4 of the assignment.
Run this to generate the tables needed in your report.
"""

import os
from transformers import T5TokenizerFast
from collections import Counter

def load_lines(path):
    """Load lines from a file"""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_statistics(data_folder='/content/drive/MyDrive/uni_things/NLP/part-2-code/data', splits=['train', 'dev']):
    """
    Compute statistics before and after preprocessing.
    
    For this assignment, "preprocessing" mainly refers to tokenization,
    so we'll show statistics on raw text vs tokenized text.
    """
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    print("=" * 80)
    print("TABLE 1: Data Statistics BEFORE Preprocessing (Raw Text)")
    print("=" * 80)
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        print("-" * 40)
        
        # Load data
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_texts = load_lines(nl_path)
        sql_texts = load_lines(sql_path)
        
        num_examples = len(nl_texts)
        
        # Compute mean lengths (in words/tokens)
        nl_word_lengths = [len(text.split()) for text in nl_texts]
        sql_word_lengths = [len(text.split()) for text in sql_texts]
        
        mean_nl_length = sum(nl_word_lengths) / len(nl_word_lengths)
        mean_sql_length = sum(sql_word_lengths) / len(sql_word_lengths)
        
        # Vocabulary size
        nl_vocab = set()
        sql_vocab = set()
        
        for text in nl_texts:
            nl_vocab.update(text.split())
        
        for text in sql_texts:
            sql_vocab.update(text.split())
        
        print(f"Number of examples: {num_examples}")
        print(f"Mean NL sentence length (words): {mean_nl_length:.2f}")
        print(f"Mean SQL query length (words): {mean_sql_length:.2f}")
        print(f"Vocabulary size (natural language): {len(nl_vocab)}")
        print(f"Vocabulary size (SQL): {len(sql_vocab)}")
    
    print("\n" + "=" * 80)
    print("TABLE 2: Data Statistics AFTER Preprocessing (Tokenized)")
    print("=" * 80)
    print(f"Model name: google-t5/t5-small")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        print("-" * 40)
        
        # Load data
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_texts = load_lines(nl_path)
        sql_texts = load_lines(sql_path)
        
        # Tokenize and compute statistics
        nl_token_lengths = []
        sql_token_lengths = []
        
        nl_tokens_set = set()
        sql_tokens_set = set()
        
        for text in nl_texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            nl_token_lengths.append(len(tokens))
            nl_tokens_set.update(tokens)
        
        for text in sql_texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            sql_token_lengths.append(len(tokens))
            sql_tokens_set.update(tokens)
        
        mean_nl_token_length = sum(nl_token_lengths) / len(nl_token_lengths)
        mean_sql_token_length = sum(sql_token_lengths) / len(sql_token_lengths)
        
        print(f"Mean NL sentence length (tokens): {mean_nl_token_length:.2f}")
        print(f"Mean SQL query length (tokens): {mean_sql_token_length:.2f}")
        print(f"Unique tokens used (NL): {len(nl_tokens_set)}")
        print(f"Unique tokens used (SQL): {len(sql_tokens_set)}")
        print(f"Max NL length (tokens): {max(nl_token_lengths)}")
        print(f"Max SQL length (tokens): {max(sql_token_lengths)}")
    
    print("\n" + "=" * 80)
    print("ADDITIONAL STATISTICS")
    print("=" * 80)
    
    # Show some example tokenizations
    nl_path = os.path.join(data_folder, 'train.nl')
    sql_path = os.path.join(data_folder, 'train.sql')
    
    nl_texts = load_lines(nl_path)
    sql_texts = load_lines(sql_path)
    
    print("\nExample 1:")
    print(f"NL: {nl_texts[0]}")
    print(f"SQL: {sql_texts[0]}")
    nl_tokens = tokenizer.tokenize(nl_texts[0])
    sql_tokens = tokenizer.tokenize(sql_texts[0])
    print(f"NL tokens: {nl_tokens}")
    print(f"SQL tokens: {sql_tokens}")
    
    print("\nExample 2:")
    print(f"NL: {nl_texts[10]}")
    print(f"SQL: {sql_texts[10]}")
    nl_tokens = tokenizer.tokenize(nl_texts[10])
    sql_tokens = tokenizer.tokenize(sql_texts[10])
    print(f"NL tokens: {nl_tokens}")
    print(f"SQL tokens: {sql_tokens}")

if __name__ == "__main__":
    compute_statistics()