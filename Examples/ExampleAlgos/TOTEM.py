import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Define an extended set of small patterns with only 3 data points each
patterns = {
    "A": [0.1, 0.3, 0.5],  # Steady up trend
    "B": [0.5, 0.3, 0.1],  # Steady down trend
    "C": [0.2, 0.4, 0.3],  # Up, then slight drop
    "D": [0.4, 0.2, 0.4],  # Down then up
    "E": [0.1, 0.3, 0.2],  # Up, small peak, then decline
    "F": [0.3, 0.5, 0.5],  # Strong up then plateau
    "G": [0.5, 0.2, 0.4],  # Decline then up trend
    "H": [0.2, 0.3, 0.2],  # Smooth rise and fall
    "I": [0.4, 0.2, 0.4],  # Down, up, down
    "J": [0.1, 0.5, 0.3],  # Sharp peak in center
    "K": [0.3, 0.37, 0.35],  # Subtle up plateau
    "L": [0.5, 0.54, 0.52],  # Subtle down plateau
    "M": [0.2, 0.4, 0.4],  # Up then stable
    "N": [0.5, 0.3, 0.3],  # Down then stable
    "O": [0.2, 0.3, 0.35],  # Gradual uptrend
    "P": [0.6, 0.5, 0.45],  # Gradual downtrend
    "Q": [0.3, 0.5, 0.6],  # Steep up then plateau
    "R": [0.7, 0.5, 0.4],  # Steep down then plateau
}

# Define the sequence of patterns to stitch together
sequence = ["A", "C", "E", "G", "J", "M", "Q"]

# Generate a list of unique colors
colors = list(mcolors.TABLEAU_COLORS.values())[:len(patterns)]

# Plot the reduced trend patterns with unique colors
fig, axes = plt.subplots(3, 3, figsize=(12, 15))
axes = axes.flatten()

for ax, (label, values), color in zip(axes, patterns.items(), colors):
    x = np.arange(len(values))  # Generate x-values (index positions)
    ax.plot(x, values, 'o-', label=label, linewidth=2, markersize=6, color=color)
    ax.set_xticks(x)  # Show integer x-values
    ax.set_title(label)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

def rescale_pattern(pattern, start_val, end_val):
    """
    Rescales a pattern so that its first and last values match given start and end values.
    """
    old_start, old_end = pattern[0], pattern[-1]
    scale_factor = (end_val - start_val) / (old_end - old_start) if old_end != old_start else 1
    rescaled_pattern = [(x - old_start) * scale_factor + start_val for x in pattern]
    return rescaled_pattern

def stitch_patterns(pattern_dict, sequence):
    """
    Stitches together a sequence of patterns smoothly.
    """
    stitched_sequence = []
    
    if not sequence:
        return stitched_sequence
    
    prev_end = pattern_dict[sequence[0]][0]  # Start with the first pattern's first value
    for idx, pattern_key in enumerate(sequence):
        pattern = pattern_dict[pattern_key]
        
        # Rescale pattern to match previous ending point
        rescaled_pattern = rescale_pattern(pattern, prev_end, pattern[-1]) if idx > 0 else pattern
        
        # Append to the full sequence, ensuring no duplicate points at junctions
        if stitched_sequence:
            stitched_sequence.extend(rescaled_pattern[1:])
        else:
            stitched_sequence.extend(rescaled_pattern)
        
        prev_end = rescaled_pattern[-1]
    
    return stitched_sequence

# Define the sequence of patterns to stitch together
sequence = ["A", "B", "B", "E", "D", "E", "A"]

# Get the stitched pattern
stitched_pattern = stitch_patterns(patterns, sequence)

# Plot the result
plt.plot(stitched_pattern, marker='o', linestyle='-')
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Stitched Pattern Sequence")
plt.show()
