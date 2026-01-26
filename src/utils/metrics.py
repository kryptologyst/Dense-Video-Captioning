"""Evaluation metrics for video captioning."""

import re
from typing import List, Dict, Any
import numpy as np
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


def normalize_text(text: str) -> str:
    """Normalize text for evaluation.
    
    Args:
        text: Input text.
        
    Returns:
        Normalized text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation at the end
    text = text.strip('.,!?;:')
    
    return text.strip()


def calculate_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BLEU scores.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Dictionary containing BLEU scores.
    """
    if not NLTK_AVAILABLE:
        return {"bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0}
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    smoothing = SmoothingFunction().method4
    
    bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(normalize_text(pred))
        ref_tokens = word_tokenize(normalize_text(ref))
        
        # Calculate BLEU scores for different n-grams
        for n in range(1, 5):
            bleu_score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                weights=tuple([1.0/n] * n + [0.0] * (4-n)),
                smoothing_function=smoothing,
            )
            bleu_scores[f"bleu_{n}"].append(bleu_score)
    
    # Calculate average scores
    avg_scores = {}
    for key, scores in bleu_scores.items():
        avg_scores[key] = np.mean(scores)
    
    return avg_scores


def calculate_meteor(predictions: List[str], references: List[str]) -> float:
    """Calculate METEOR score.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Average METEOR score.
    """
    if not NLTK_AVAILABLE:
        return 0.0
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    meteor_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(normalize_text(pred))
        ref_tokens = word_tokenize(normalize_text(ref))
        
        meteor_score = meteor_score([ref_tokens], pred_tokens)
        meteor_scores.append(meteor_score)
    
    return np.mean(meteor_scores)


def calculate_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Dictionary containing ROUGE scores.
    """
    if not ROUGE_AVAILABLE:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {"rouge_1": [], "rouge_2": [], "rouge_l": []}
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        
        rouge_scores["rouge_1"].append(scores['rouge1'].fmeasure)
        rouge_scores["rouge_2"].append(scores['rouge2'].fmeasure)
        rouge_scores["rouge_l"].append(scores['rougeL'].fmeasure)
    
    # Calculate average scores
    avg_scores = {}
    for key, scores in rouge_scores.items():
        avg_scores[key] = np.mean(scores)
    
    return avg_scores


def calculate_bert_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate BERTScore.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Dictionary containing BERTScore metrics.
    """
    if not BERT_SCORE_AVAILABLE:
        return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}
    
    try:
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }
    except Exception:
        return {"bert_score_precision": 0.0, "bert_score_recall": 0.0, "bert_score_f1": 0.0}


def calculate_cider(predictions: List[str], references: List[str]) -> float:
    """Calculate CIDEr score (simplified version).
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Average CIDEr score.
    """
    def get_ngrams(tokens, n):
        """Get n-grams from tokens."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams
    
    def get_cider_score(pred_tokens, ref_tokens_list):
        """Calculate CIDEr score for a single prediction."""
        scores = []
        
        for n in range(1, 5):  # 1-gram to 4-gram
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            
            # Calculate precision and recall for each reference
            precisions = []
            recalls = []
            
            for ref_tokens in ref_tokens_list:
                ref_ngrams = Counter(get_ngrams(ref_tokens, n))
                
                # Precision
                overlap = sum((pred_ngrams & ref_ngrams).values())
                precision = overlap / max(len(pred_ngrams), 1)
                precisions.append(precision)
                
                # Recall
                recall = overlap / max(len(ref_ngrams), 1)
                recalls.append(recall)
            
            # Average precision and recall
            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            
            # F1 score
            if avg_precision + avg_recall > 0:
                f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            else:
                f1 = 0.0
            
            scores.append(f1)
        
        # Average across n-grams
        return np.mean(scores)
    
    cider_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = normalize_text(pred).split()
        ref_tokens = normalize_text(ref).split()
        
        cider_score = get_cider_score(pred_tokens, [ref_tokens])
        cider_scores.append(cider_score)
    
    return np.mean(cider_scores)


def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Calculate exact match accuracy.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Exact match accuracy.
    """
    exact_matches = 0
    
    for pred, ref in zip(predictions, references):
        if normalize_text(pred) == normalize_text(ref):
            exact_matches += 1
    
    return exact_matches / len(predictions)


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate all evaluation metrics.
    
    Args:
        predictions: List of predicted captions.
        references: List of reference captions.
        
    Returns:
        Dictionary containing all metrics.
    """
    metrics = {}
    
    # Exact match
    metrics["exact_match"] = calculate_exact_match(predictions, references)
    
    # BLEU scores
    bleu_scores = calculate_bleu(predictions, references)
    metrics.update(bleu_scores)
    
    # METEOR score
    metrics["meteor"] = calculate_meteor(predictions, references)
    
    # ROUGE scores
    rouge_scores = calculate_rouge(predictions, references)
    metrics.update(rouge_scores)
    
    # CIDEr score
    metrics["cider"] = calculate_cider(predictions, references)
    
    # BERTScore
    bert_scores = calculate_bert_score(predictions, references)
    metrics.update(bert_scores)
    
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    """Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing metrics.
    """
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    
    # Group metrics by category
    exact_match = metrics.get("exact_match", 0.0)
    bleu_scores = {k: v for k, v in metrics.items() if k.startswith("bleu_")}
    rouge_scores = {k: v for k, v in metrics.items() if k.startswith("rouge_")}
    bert_scores = {k: v for k, v in metrics.items() if k.startswith("bert_score_")}
    other_scores = {k: v for k, v in metrics.items() if k not in 
                   ["exact_match"] | set(bleu_scores.keys()) | 
                   set(rouge_scores.keys()) | set(bert_scores.keys())}
    
    print(f"Exact Match: {exact_match:.4f}")
    
    if bleu_scores:
        print("\nBLEU Scores:")
        for key, value in sorted(bleu_scores.items()):
            print(f"  {key}: {value:.4f}")
    
    if rouge_scores:
        print("\nROUGE Scores:")
        for key, value in sorted(rouge_scores.items()):
            print(f"  {key}: {value:.4f}")
    
    if bert_scores:
        print("\nBERTScore:")
        for key, value in sorted(bert_scores.items()):
            print(f"  {key}: {value:.4f}")
    
    if other_scores:
        print("\nOther Scores:")
        for key, value in sorted(other_scores.items()):
            print(f"  {key}: {value:.4f}")
    
    print("="*50)
