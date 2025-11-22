#!/usr/bin/env python3
"""
Evaluation script for QA system predictions
"""

import argparse
import os
from typing import List, Tuple
import difflib


def load_tsv(filepath: str) -> List[str]:
    """Load predictions or answers from TSV file"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                data.append(parts[0])  # Only take the prediction/answer
    return data


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison"""
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove common variations
    text = text.replace("'", "").replace('"', '')
    text = ' '.join(text.split())  # Normalize whitespace
    
    return text


def exact_match(pred: str, gold: str) -> bool:
    """Check if prediction exactly matches gold answer"""
    return normalize_answer(pred) == normalize_answer(gold)


def partial_match(pred: str, gold: str) -> bool:
    """Check if prediction partially matches gold answer"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    # Check if one contains the other
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True
    
    # Check token overlap for lists
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return False
    
    # Jaccard similarity > 0.5
    intersection = len(pred_tokens & gold_tokens)
    union = len(pred_tokens | gold_tokens)
    
    return (intersection / union) > 0.5


def evaluate_predictions(predictions: List[str], answers: List[str], 
                         questions: List[str] = None) -> dict:
    """Evaluate predictions against gold answers"""
    
    if len(predictions) != len(answers):
        print(f"Warning: Number of predictions ({len(predictions)}) != "
              f"number of answers ({len(answers)})")
        min_len = min(len(predictions), len(answers))
        predictions = predictions[:min_len]
        answers = answers[:min_len]
    
    exact_matches = 0
    partial_matches = 0
    total = len(predictions)
    
    errors = []
    
    for i, (pred, gold) in enumerate(zip(predictions, answers)):
        if exact_match(pred, gold):
            exact_matches += 1
        elif partial_match(pred, gold):
            partial_matches += 1
        else:
            q = questions[i] if questions and i < len(questions) else f"Question {i+1}"
            errors.append({
                'question': q,
                'predicted': pred,
                'gold': gold,
                'similarity': difflib.SequenceMatcher(None, 
                    normalize_answer(pred), 
                    normalize_answer(gold)).ratio()
            })
    
    results = {
        'total': total,
        'exact_match': exact_matches,
        'partial_match': partial_matches,
        'no_match': total - exact_matches - partial_matches,
        'exact_match_rate': exact_matches / total if total > 0 else 0,
        'partial_match_rate': partial_matches / total if total > 0 else 0,
        'any_match_rate': (exact_matches + partial_matches) / total if total > 0 else 0,
        'errors': errors[:10]  # Show first 10 errors
    }
    
    return results


def load_questions(filepath: str) -> List[str]:
    """Load questions from TSV file"""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                questions.append(parts[0])
    return questions


def main():
    parser = argparse.ArgumentParser(description='Evaluate QA predictions')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions TSV file')
    parser.add_argument('--answers', type=str, default='answer.tsv',
                        help='Path to gold answers TSV file')
    parser.add_argument('--questions', type=str, default='question.tsv',
                        help='Path to questions TSV file (optional)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}...")
    predictions = load_tsv(args.predictions)
    
    print(f"Loading answers from {args.answers}...")
    answers = load_tsv(args.answers)
    
    questions = None
    if os.path.exists(args.questions):
        print(f"Loading questions from {args.questions}...")
        questions = load_questions(args.questions)
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_predictions(predictions, answers, questions)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total questions: {results['total']}")
    print(f"Exact matches: {results['exact_match']} ({results['exact_match_rate']:.2%})")
    print(f"Partial matches: {results['partial_match']} ({results['partial_match_rate']:.2%})")
    print(f"No matches: {results['no_match']}")
    print(f"Any match rate: {results['any_match_rate']:.2%}")
    print("="*60)
    
    # Show some errors
    if results['errors']:
        print("\nSample Errors (first 10):")
        print("-"*60)
        for i, error in enumerate(results['errors'], 1):
            print(f"\n{i}. Question: {error['question'][:80]}...")
            print(f"   Predicted: {error['predicted'][:80]}")
            print(f"   Gold:      {error['gold'][:80]}")
            print(f"   Similarity: {error['similarity']:.2f}")
    
    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
