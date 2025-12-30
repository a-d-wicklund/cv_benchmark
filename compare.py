#!/usr/bin/env python3
"""
Benchmark Comparison and Visualization Tool
Compares CV benchmark results from different systems and generates visualizations.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class BenchmarkComparison:
    def __init__(self, file1: str, file2: str):
        """Load and compare two benchmark result files."""
        self.file1 = file1
        self.file2 = file2

        # Load the benchmark data
        with open(file1, 'r') as f:
            self.data1 = json.load(f)
        with open(file2, 'r') as f:
            self.data2 = json.load(f)

        # Extract metadata
        self.meta1 = self.data1['metadata']
        self.meta2 = self.data2['metadata']

        # Create a mapping of operation name to results
        self.results1 = {r['name']: r for r in self.data1['results']}
        self.results2 = {r['name']: r for r in self.data2['results']}

        # Find common operations
        self.common_ops = set(self.results1.keys()) & set(self.results2.keys())

        print(f"Loaded results from:")
        print(f"  System 1: {file1}")
        print(f"    - Platform: {self.meta1['platform']}")
        print(f"    - Processor: {self.meta1['processor']}")
        print(f"    - Python: {self.meta1['python_version']}")
        print(f"    - OpenCV: {self.meta1['opencv_version']}")
        print(f"\n  System 2: {file2}")
        print(f"    - Platform: {self.meta2['platform']}")
        print(f"    - Processor: {self.meta2['processor']}")
        print(f"    - Python: {self.meta2['python_version']}")
        print(f"    - OpenCV: {self.meta2['opencv_version']}")
        print(f"\n  Common operations: {len(self.common_ops)}")

    def get_category_operations(self) -> Dict[str, List[str]]:
        """Group operations by category based on naming patterns."""
        categories = {
            "Resize": [],
            "Color Conversion": [],
            "Gaussian Blur": [],
            "Bilateral Filter": [],
            "Median Blur": [],
            "Edge Detection": [],
            "Corner Detection": [],
            "Morphological": [],
            "Feature Detection": [],
            "Pyramids": []
        }

        for op in sorted(self.common_ops):
            if "Resize" in op:
                categories["Resize"].append(op)
            elif "BGR to" in op or "Grayscale" in op:
                categories["Color Conversion"].append(op)
            elif "Gaussian Blur" in op:
                categories["Gaussian Blur"].append(op)
            elif "Bilateral" in op:
                categories["Bilateral Filter"].append(op)
            elif "Median Blur" in op:
                categories["Median Blur"].append(op)
            elif "Canny" in op or "Sobel" in op:
                categories["Edge Detection"].append(op)
            elif "Corner" in op:
                categories["Corner Detection"].append(op)
            elif any(x in op for x in ["Erosion", "Dilation", "Opening", "Closing"]):
                categories["Morphological"].append(op)
            elif any(x in op for x in ["ORB", "AKAZE"]):
                categories["Feature Detection"].append(op)
            elif "Pyramid" in op:
                categories["Pyramids"].append(op)

        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

    def calculate_speedup(self) -> List[Tuple[str, float, float, float]]:
        """Calculate speedup factors (System2 / System1)."""
        speedups = []

        for op in self.common_ops:
            time1 = self.results1[op]['mean_ms']
            time2 = self.results2[op]['mean_ms']
            speedup = time1 / time2  # >1 means system2 is faster
            speedups.append((op, time1, time2, speedup))

        # Sort by absolute speedup difference
        speedups.sort(key=lambda x: abs(x[3] - 1.0), reverse=True)
        return speedups

    def print_summary(self):
        """Print a text summary of the comparison."""
        speedups = self.calculate_speedup()

        print("\n" + "=" * 80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 80)

        # Overall statistics
        speedup_values = [s[3] for s in speedups]
        avg_speedup = np.mean(speedup_values)
        median_speedup = np.median(speedup_values)

        system2_faster = sum(1 for s in speedup_values if s > 1.0)
        system1_faster = sum(1 for s in speedup_values if s < 1.0)

        print(f"\nOverall Statistics:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Median speedup: {median_speedup:.2f}x")
        print(f"  System 2 faster: {system2_faster}/{len(speedups)} operations ({system2_faster/len(speedups)*100:.1f}%)")
        print(f"  System 1 faster: {system1_faster}/{len(speedups)} operations ({system1_faster/len(speedups)*100:.1f}%)")

        # Top speedups for each system
        print(f"\n{'Top 10 - System 2 Faster'}")
        print("-" * 80)
        system2_wins = [s for s in speedups if s[3] > 1.0][:10]
        for op, t1, t2, speedup in system2_wins:
            print(f"  {speedup:.2f}x  {op[:60]:<60} ({t1:.2f}ms → {t2:.2f}ms)")

        print(f"\n{'Top 10 - System 1 Faster'}")
        print("-" * 80)
        system1_wins = [s for s in speedups if s[3] < 1.0][:10]
        for op, t1, t2, speedup in system1_wins:
            print(f"  {1/speedup:.2f}x  {op[:60]:<60} ({t1:.2f}ms → {t2:.2f}ms)")

    def create_visualizations(self, output_prefix: str = "comparison"):
        """Create comprehensive visualization charts."""
        categories = self.get_category_operations()

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))

        # Plot 1: Top 20 operations comparison (bar chart)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_top_operations(ax1)

        # Plot 2: Speedup distribution (histogram)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_speedup_distribution(ax2)

        # Plot 3: Category-wise comparison (grouped bar)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_category_comparison(ax3, categories)

        # Plot 4: Scatter plot (correlation)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_correlation(ax4)

        # Plot 5: Speedup by operation (horizontal bar)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_speedup_bars(ax5)

        # Plot 6: Slowest operations on each system
        ax6 = plt.subplot(2, 3, 6)
        self._plot_slowest_operations(ax6)

        plt.tight_layout()

        # Save the figure
        output_file = f"{output_prefix}_full.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved comprehensive visualization to {output_file}")

        # Create individual detailed charts
        self._create_category_charts(categories, output_prefix)

        plt.show()

    def _plot_top_operations(self, ax):
        """Plot top 20 slowest operations comparison."""
        # Get top 20 slowest operations (by max of either system)
        op_times = [(op, max(self.results1[op]['mean_ms'], self.results2[op]['mean_ms']))
                    for op in self.common_ops]
        op_times.sort(key=lambda x: x[1], reverse=True)
        top_ops = [x[0] for x in op_times[:20]]

        times1 = [self.results1[op]['mean_ms'] for op in top_ops]
        times2 = [self.results2[op]['mean_ms'] for op in top_ops]

        x = np.arange(len(top_ops))
        width = 0.35

        ax.barh(x - width/2, times1, width, label=f'System 1 ({self.meta1["processor"]})', alpha=0.8)
        ax.barh(x + width/2, times2, width, label=f'System 2 ({self.meta2["processor"]})', alpha=0.8)

        ax.set_xlabel('Time (ms)')
        ax.set_title('Top 20 Slowest Operations Comparison')
        ax.set_yticks(x)
        ax.set_yticklabels([op[:40] for op in top_ops], fontsize=8)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    def _plot_speedup_distribution(self, ax):
        """Plot distribution of speedup factors."""
        speedups = self.calculate_speedup()
        speedup_values = [s[3] for s in speedups]

        # Use log scale for speedup
        log_speedups = np.log2(speedup_values)

        ax.hist(log_speedups, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Equal performance')
        ax.set_xlabel('Speedup (log2 scale)')
        ax.set_ylabel('Number of operations')
        ax.set_title('Speedup Distribution\n(Right=System2 faster, Left=System1 faster)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Add text labels
        ax.text(0.02, 0.98, f'Median: {np.median(speedup_values):.2f}x',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _plot_category_comparison(self, ax, categories):
        """Plot average performance by category."""
        cat_names = []
        cat_times1 = []
        cat_times2 = []

        for cat_name, ops in categories.items():
            cat_names.append(cat_name)
            cat_times1.append(np.mean([self.results1[op]['mean_ms'] for op in ops]))
            cat_times2.append(np.mean([self.results2[op]['mean_ms'] for op in ops]))

        x = np.arange(len(cat_names))
        width = 0.35

        ax.bar(x - width/2, cat_times1, width, label=f'System 1', alpha=0.8)
        ax.bar(x + width/2, cat_times2, width, label=f'System 2', alpha=0.8)

        ax.set_ylabel('Average Time (ms)')
        ax.set_title('Average Performance by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    def _plot_correlation(self, ax):
        """Plot correlation between system performances."""
        times1 = [self.results1[op]['mean_ms'] for op in self.common_ops]
        times2 = [self.results2[op]['mean_ms'] for op in self.common_ops]

        # Use log scale for better visualization
        log_times1 = np.log10(times1)
        log_times2 = np.log10(times2)

        ax.scatter(log_times1, log_times2, alpha=0.6, s=30)

        # Add diagonal line (perfect correlation)
        min_val = min(min(log_times1), min(log_times2))
        max_val = max(max(log_times1), max(log_times2))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal performance')

        ax.set_xlabel(f'System 1 Time (log10 ms)')
        ax.set_ylabel(f'System 2 Time (log10 ms)')
        ax.set_title('Performance Correlation\n(Below line = System2 faster)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Calculate and display correlation
        corr = np.corrcoef(times1, times2)[0, 1]
        ax.text(0.02, 0.98, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _plot_speedup_bars(self, ax):
        """Plot speedup factors as horizontal bars."""
        speedups = self.calculate_speedup()[:20]  # Top 20 by speedup difference

        ops = [s[0][:40] for s in speedups]
        speedup_vals = [s[3] for s in speedups]

        # Color bars: green if system2 faster, red if system1 faster
        colors = ['green' if s > 1.0 else 'red' for s in speedup_vals]

        y = np.arange(len(ops))
        bars = ax.barh(y, speedup_vals, color=colors, alpha=0.6)

        ax.axvline(1.0, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Speedup Factor (System1 / System2)')
        ax.set_title('Top 20 Speedup Differences\n(Green=System2 faster, Red=System1 faster)')
        ax.set_yticks(y)
        ax.set_yticklabels(ops, fontsize=8)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, speedup_vals)):
            label = f'{val:.2f}x'
            ax.text(val, bar.get_y() + bar.get_height()/2, label,
                   ha='left' if val > 1 else 'right', va='center', fontsize=7)

    def _plot_slowest_operations(self, ax):
        """Plot top 10 slowest operations on each system."""
        # Get top 10 from each system
        ops1_sorted = sorted(self.results1.items(), key=lambda x: x[1]['mean_ms'], reverse=True)[:10]
        ops2_sorted = sorted(self.results2.items(), key=lambda x: x[1]['mean_ms'], reverse=True)[:10]

        # Combine and deduplicate
        all_slow_ops = list(dict.fromkeys([x[0] for x in ops1_sorted] + [x[0] for x in ops2_sorted]))[:15]

        times1 = [self.results1[op]['mean_ms'] for op in all_slow_ops]
        times2 = [self.results2[op]['mean_ms'] for op in all_slow_ops]

        x = np.arange(len(all_slow_ops))
        width = 0.35

        ax.barh(x - width/2, times1, width, label='System 1', alpha=0.8, color='#1f77b4')
        ax.barh(x + width/2, times2, width, label='System 2', alpha=0.8, color='#ff7f0e')

        ax.set_xlabel('Time (ms)')
        ax.set_title('Top Slowest Operations (Combined)')
        ax.set_yticks(x)
        ax.set_yticklabels([op[:40] for op in all_slow_ops], fontsize=8)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    def _create_category_charts(self, categories, output_prefix):
        """Create detailed charts for each category."""
        for cat_name, ops in categories.items():
            if len(ops) < 2:
                continue

            fig, ax = plt.subplots(figsize=(12, max(6, len(ops) * 0.4)))

            times1 = [self.results1[op]['mean_ms'] for op in ops]
            times2 = [self.results2[op]['mean_ms'] for op in ops]

            x = np.arange(len(ops))
            width = 0.35

            ax.barh(x - width/2, times1, width, label=f'System 1 ({self.meta1["processor"]})', alpha=0.8)
            ax.barh(x + width/2, times2, width, label=f'System 2 ({self.meta2["processor"]})', alpha=0.8)

            ax.set_xlabel('Time (ms)')
            ax.set_title(f'{cat_name} - Detailed Comparison')
            ax.set_yticks(x)
            ax.set_yticklabels(ops, fontsize=9)
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()

            plt.tight_layout()
            output_file = f"{output_prefix}_{cat_name.replace(' ', '_').lower()}.png"
            plt.savefig(output_file, dpi=120, bbox_inches='tight')
            print(f"Saved {cat_name} chart to {output_file}")
            plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare CV benchmark results from two systems")
    parser.add_argument("file1", help="First benchmark results JSON file")
    parser.add_argument("file2", help="Second benchmark results JSON file")
    parser.add_argument("--output", "-o", default="comparison", help="Output file prefix (default: comparison)")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots interactively")

    args = parser.parse_args()

    # Validate files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        sys.exit(1)
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        sys.exit(1)

    # Create comparison
    comparison = BenchmarkComparison(args.file1, args.file2)

    # Print text summary
    comparison.print_summary()

    # Create visualizations
    print("\nGenerating visualizations...")
    comparison.create_visualizations(args.output)

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
