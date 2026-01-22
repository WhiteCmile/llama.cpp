#!/usr/bin/env python3

import os
import sys

def analyze_mask_data(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Assuming each line contains 36 comma-separated values for one token's mask
    num_layers = 36
    masks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            values = [float(x.strip()) for x in line.split(',')]
            if len(values) != num_layers:
                print(f"Warning: Line has {len(values)} values, expected {num_layers}. Skipping.")
                continue
            masks.append(values)
        except ValueError as e:
            print(f"Error parsing line: {line} - {e}")
            continue

    num_tokens = len(masks)
    print(f"Total tokens: {num_tokens}")
    print(f"Total layers: {num_layers}")

    if num_tokens == 0:
        print("No valid data found.")
        return

    # Calculate activation frequency per layer
    layer_activation_freq = []
    for layer in range(num_layers):
        activated_count = sum(1 for token in masks if token[layer] == 1.0)
        freq = activated_count / num_tokens if num_tokens > 0 else 0
        layer_activation_freq.append(freq)

    print("\nActivation frequency per layer (0 to 35):")
    for layer, freq in enumerate(layer_activation_freq):
        print(f"Layer {layer}: {freq:.3f}")

    # Analyze changes over different lengths (token positions)
    # Group by token position ranges
    group_size = max(100, num_tokens // 10)  # Divide into 10 groups or at least 1
    groups = []
    for start in range(0, num_tokens, group_size):
        end = min(start + group_size, num_tokens)
        group_masks = masks[start:end]
        group_freq = []
        for layer in range(num_layers):
            activated = sum(1 for token in group_masks if token[layer] == 1.0)
            freq = activated / len(group_masks) if group_masks else 0
            group_freq.append(freq)
        groups.append((start, end, group_freq))

    print(f"\nActivation frequency changes over token position groups (group size ~{group_size}):")
    for start, end, freqs in groups:
        print(f"Tokens {start}-{end-1}:")
        for layer in range(num_layers):
            print(f"  Layer {layer}: {freqs[layer]:.3f}")
        # Optionally, show average change or something
        avg_freq = sum(freqs) / num_layers
        print(f"  Average activation: {avg_freq:.3f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_mask.py <mask_data.txt>")
        sys.exit(1)
    file_path = sys.argv[1]
    analyze_mask_data(file_path)