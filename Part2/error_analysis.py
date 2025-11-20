"""
Error Analysis Script for Text-to-SQL models
This helps with qualitative error analysis for Q6
"""

import os
import pickle
from collections import Counter
import difflib

def load_queries(sql_path):
    """Load SQL queries from file"""
    with open(sql_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_records(pkl_path):
    """Load database records from pickle file"""
    with open(pkl_path, 'rb') as f:
        records, error_msgs = pickle.load(f)
    return records, error_msgs

def load_nl_queries(nl_path):
    """Load natural language queries"""
    with open(nl_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def analyze_errors(gt_sql_path, pred_sql_path, gt_record_path, pred_record_path, nl_path, output_file='error_analysis.txt'):
    """
    Perform detailed error analysis comparing ground truth and predictions
    """
    
    # Load all data
    gt_queries = load_queries(gt_sql_path)
    pred_queries = load_queries(pred_sql_path)
    gt_records, _ = load_records(gt_record_path)
    pred_records, pred_errors = load_records(pred_record_path)
    nl_queries = load_nl_queries(nl_path)
    
    print(f"Analyzing {len(gt_queries)} examples...")
    
    # Error categories
    errors = {
        'syntax_error': [],
        'wrong_table': [],
        'wrong_column': [],
        'wrong_condition': [],
        'wrong_aggregation': [],
        'wrong_join': [],
        'extra_results': [],
        'missing_results': [],
        'correct': []
    }
    
    # Analyze each example
    for i, (nl, gt_sql, pred_sql, gt_rec, pred_rec, pred_err) in enumerate(
        zip(nl_queries, gt_queries, pred_queries, gt_records, pred_records, pred_errors)):
        
        # Check if there was a syntax error
        if pred_err:
            errors['syntax_error'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql,
                'error': pred_err
            })
            continue
        
        # Check if results match
        if set(gt_rec) == set(pred_rec):
            errors['correct'].append(i)
            continue
        
        # Analyze type of error
        gt_sql_lower = gt_sql.lower()
        pred_sql_lower = pred_sql.lower()
        
        # Check for table errors
        if 'from' in pred_sql_lower:
            gt_tables = extract_tables(gt_sql_lower)
            pred_tables = extract_tables(pred_sql_lower)
            if gt_tables != pred_tables:
                errors['wrong_table'].append({
                    'idx': i,
                    'nl': nl,
                    'gt_sql': gt_sql,
                    'pred_sql': pred_sql,
                    'gt_tables': gt_tables,
                    'pred_tables': pred_tables
                })
                continue
        
        # Check for aggregation errors
        gt_has_agg = any(agg in gt_sql_lower for agg in ['count', 'sum', 'avg', 'max', 'min'])
        pred_has_agg = any(agg in pred_sql_lower for agg in ['count', 'sum', 'avg', 'max', 'min'])
        if gt_has_agg != pred_has_agg or (gt_has_agg and extract_aggregation(gt_sql_lower) != extract_aggregation(pred_sql_lower)):
            errors['wrong_aggregation'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql
            })
            continue
        
        # Check for join errors
        gt_has_join = 'join' in gt_sql_lower
        pred_has_join = 'join' in pred_sql_lower
        if gt_has_join != pred_has_join:
            errors['wrong_join'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql
            })
            continue
        
        # Check result differences
        gt_set = set(gt_rec)
        pred_set = set(pred_rec)
        
        if len(pred_set) > len(gt_set):
            errors['extra_results'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql,
                'extra_count': len(pred_set) - len(gt_set)
            })
        elif len(pred_set) < len(gt_set):
            errors['missing_results'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql,
                'missing_count': len(gt_set) - len(pred_set)
            })
        else:
            errors['wrong_condition'].append({
                'idx': i,
                'nl': nl,
                'gt_sql': gt_sql,
                'pred_sql': pred_sql
            })
    
    # Print summary
    total = len(gt_queries)
    print("\n" + "="*80)
    print("ERROR ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total examples: {total}")
    print(f"Correct: {len(errors['correct'])} ({len(errors['correct'])/total*100:.2f}%)")
    print(f"\nError breakdown:")
    for error_type in ['syntax_error', 'wrong_table', 'wrong_column', 'wrong_condition', 
                       'wrong_aggregation', 'wrong_join', 'extra_results', 'missing_results']:
        count = len(errors[error_type])
        if count > 0:
            print(f"  {error_type}: {count} ({count/total*100:.2f}%)")
    
    # Write detailed report
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED ERROR ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        for error_type in ['syntax_error', 'wrong_table', 'wrong_aggregation', 
                          'wrong_join', 'wrong_condition', 'extra_results', 'missing_results']:
            error_list = errors[error_type]
            if not error_list:
                continue
            
            f.write(f"\n{'='*80}\n")
            f.write(f"{error_type.upper().replace('_', ' ')} ({len(error_list)} examples)\n")
            f.write(f"{'='*80}\n\n")
            
            # Show up to 5 examples
            for example in error_list[:5]:
                f.write(f"Example {example['idx'] + 1}:\n")
                f.write(f"Natural Language: {example['nl']}\n")
                f.write(f"Ground Truth SQL: {example['gt_sql']}\n")
                f.write(f"Predicted SQL:    {example['pred_sql']}\n")
                if 'error' in example:
                    f.write(f"Error Message: {example['error']}\n")
                if 'gt_tables' in example:
                    f.write(f"GT Tables: {example['gt_tables']}\n")
                    f.write(f"Pred Tables: {example['pred_tables']}\n")
                f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\nDetailed error report saved to {output_file}")
    return errors

def extract_tables(sql):
    """Extract table names from SQL query"""
    tables = []
    sql = sql.lower()
    
    # Simple extraction - look for FROM and JOIN clauses
    if 'from' in sql:
        parts = sql.split('from')[1].split()
        if parts:
            table = parts[0].strip(',').strip(';')
            if table and table not in ['(', 'select']:
                tables.append(table)
    
    if 'join' in sql:
        parts = sql.split('join')
        for part in parts[1:]:
            words = part.split()
            if words:
                table = words[0].strip(',').strip(';')
                if table and table not in ['(', 'select']:
                    tables.append(table)
    
    return sorted(set(tables))

def extract_aggregation(sql):
    """Extract aggregation function from SQL"""
    sql = sql.lower()
    for agg in ['count', 'sum', 'avg', 'max', 'min']:
        if agg in sql:
            return agg
    return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python error_analysis.py <gt_sql> <pred_sql> <gt_records> <pred_records> [nl_queries]")
        sys.exit(1)
    
    gt_sql_path = sys.argv[1]
    pred_sql_path = sys.argv[2]
    gt_record_path = sys.argv[3]
    pred_record_path = sys.argv[4]
    nl_path = sys.argv[5] if len(sys.argv) > 5 else 'data/dev.nl'
    
    analyze_errors(gt_sql_path, pred_sql_path, gt_record_path, pred_record_path, nl_path)