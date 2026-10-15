#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import color_for_file, pretty_label

def main():
    # Check if files were provided
    if len(sys.argv) < 2:
        print("Usage: python plot_hmmer_roc.py <file1.xy> <file2.xy> ...")
        sys.exit(1)

    # Use a clean, modern plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6))

    files = sys.argv[1:]

    for file in files:
        data = []
        # Parse the XMGRACE-formatted output from rocplot
        with open(file, 'r') as f:
            for line in f:
                # Ignore xmgrace formatting directives and comments
                if line.startswith('@') or line.startswith('#') or line.startswith('&'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    data.append([float(p) for p in parts[:4]])
        
        data = np.array(data)
        if len(data) == 0:
            print(f"Warning: No valid data found in {file}")
            continue
        
        # Extract columns
        # rocplot xydydy format: X (EPQ), Y (Coverage), Y_lower (CI), Y_upper (CI)
        X = data[:, 0]  # False Positive Rate (FPR)
        Y = data[:, 1]  # True Positive Rate (TPR)
        dy_down = data[:, 2] 
        dy_up = data[:, 3] 

        # Filter data to keep only points where Y <= 1.0
        mask = X <= 1.0
        X = X[mask]
        Y = Y[mask]
        dy_down = dy_down[mask]
        dy_up = dy_up[mask]

        # Calculate the absolute min and max for the shading
        Y_min = Y - dy_down
        Y_max = Y + dy_up

        # Calculate AUC
        roc_auc = auc(X, Y)
        
        # Create a clean label from the filename
        tool_label = pretty_label(file)
        label = f"{tool_label}" # (AUC = {roc_auc:.3f})

        # Plot the main line in the tool's associated color.
        ax.plot(X, Y, label=label, linewidth=2, color=color_for_file(file))
        
        # Add the modern shaded confidence intervals (replacing the dydy error bars)
        ax.fill_between(X, Y_min, Y_max, alpha=0.2, color=color_for_file(file))

    # Format the axes
    ax.set_xlabel('Mean number of false positives per profile', fontsize=12, fontweight='bold')
    ax.set_ylabel('True positives recovered', fontsize=12, fontweight='bold')
    ax.set_title('BATH transmark performance', fontsize=14)
    
    # Set default rocplot limits (0 to 10 EPQ, 0 to 100% coverage)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add the legend
    ax.legend(frameon=True, shadow=True)
    
    # Tight layout for better spacing
    plt.tight_layout()

    # Save as a high-quality vector and PNG
    output_filename = 'modern_rocplot.png'
    plt.savefig(output_filename, dpi=300)
    #plt.savefig('modern_rocplot.pdf') # Great for publications
    
    print(f"Plot saved successfully as {output_filename}")

if __name__ == "__main__":
    main()
