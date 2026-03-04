"""Post-processing utilities for OCR decoding."""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch
from collections import defaultdict


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding.
    
    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        # Group consecutive identical characters and filter blanks
        # groupby returns (key, group_iterator) pairs
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                # Get maximum probability from this group
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))
    
    return results



def apply_layout_rules(text: str) -> str:
    """
    Forces the predicted text to strictly adhere to Brazilian/Mercosur LP layouts.
    Fixes common OCR character confusions based on expected character types.
    """
    # Common confusions mapping
    letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0', 'D': '0'}
    digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
    
    # If the network completely failed and didn't output 7 characters, just return it
    if len(text) != 7:
        return text
        
    corrected = list(text)
    
    # Helper functions
    def force_letter(idx):
        if corrected[idx].isdigit():
            corrected[idx] = digit_to_letter.get(corrected[idx], corrected[idx])
            
    def force_digit(idx):
        if corrected[idx].isalpha():
            corrected[idx] = letter_to_digit.get(corrected[idx], corrected[idx])

    # Rule 1: Positions 0, 1, 2 are ALWAYS Letters
    for i in range(3):
        force_letter(i)
        
    # Rule 2: Position 3 is ALWAYS a Digit
    force_digit(3)
    
    # Position 4 is variable (Letter or Digit), so we leave it alone!
    
    # Rule 3: Positions 5 and 6 are ALWAYS Digits
    force_digit(5)
    force_digit(6)
    
    return "".join(corrected)


def decode_with_layout_rules(preds: torch.Tensor, idx2char: Dict[int, str]) -> List[Tuple[str, float]]:
    """Greedy decode WITH strict layout rules applied."""
    # Run the normal greedy decoding
    base_results = decode_with_confidence(preds, idx2char)
    
    final_results = []
    for pred_str, conf in base_results:
        # Apply the paper's Layout constraints!
        corrected_str = apply_layout_rules(pred_str)
        final_results.append((corrected_str, conf))
        
    return final_results