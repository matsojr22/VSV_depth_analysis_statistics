#!/usr/bin/env python3
"""
Professional Pairwise Comparison Plots

This script creates clean, professional plots showing all pairwise comparisons
between age groups with box plots, individual points, means, SEM, and p-values.

Author: Generated for VSV depth analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_analysis_results(results_file):
    """Load the results from vsv_depth_analysis.py"""
    return pd.read_csv(results_file)

def perform_pairwise_tests(results_df, metric='weighted_mean_depth'):
    """
    Perform pairwise comparisons for a specific metric
    """
    
    # Get unique age groups
    age_groups = results_df['age_group'].unique()
    
    # Initialize results
    comparisons = []
    
    # Get all pairwise combinations
    for group1, group2 in itertools.combinations(age_groups, 2):
        data1 = results_df[results_df['age_group'] == group1][metric].values
        data2 = results_df[results_df['age_group'] == group2][metric].values
        
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        # Calculate means and SEM
        mean1, mean2 = np.mean(data1), np.mean(data2)
        sem1, sem2 = stats.sem(data1), stats.sem(data2)
        
        comparison = {
            'group1': group1,
            'group2': group2,
            'test': 'Mann-Whitney U test',
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_name': 'Rank-biserial correlation',
            'significant': p_value < 0.05,
            'n1': len(data1),
            'n2': len(data2),
            'mean1': mean1,
            'mean2': mean2,
            'sem1': sem1,
            'sem2': sem2,
            'std1': np.std(data1),
            'std2': np.std(data2)
        }
        
        comparisons.append(comparison)
    
    # Apply multiple comparison corrections
    p_values = [comp['p_value'] for comp in comparisons]
    
    if len(p_values) > 1:
        # Bonferroni correction
        bonferroni_corrected = multipletests(p_values, method='bonferroni')
        # FDR correction (Benjamini-Hochberg)
        fdr_corrected = multipletests(p_values, method='fdr_bh')
        
        # Add corrected p-values to comparisons
        for i, comp in enumerate(comparisons):
            comp['p_value_bonferroni'] = bonferroni_corrected[1][i]
            comp['p_value_fdr'] = fdr_corrected[1][i]
            comp['significant_bonferroni'] = bonferroni_corrected[0][i]
            comp['significant_fdr'] = fdr_corrected[0][i]
    
    return comparisons

def create_professional_pairwise_plot(results_df, comparisons, metric, title_suffix):
    """
    Create a professional pairwise comparison plot
    """
    
    # Set the order for age groups: p3, p12, p20, adult
    age_group_order = ['p3', 'p12', 'p20', 'adult']
    results_df['age_group'] = pd.Categorical(results_df['age_group'], categories=age_group_order, ordered=True)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create box plot
    sns.boxplot(data=results_df, x='age_group', y=metric, ax=ax, 
                boxprops=dict(alpha=0.7),
                whiskerprops=dict(alpha=0.7),
                capprops=dict(alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    
    # Add individual points
    sns.stripplot(data=results_df, x='age_group', y=metric, 
                  color='black', alpha=0.6, size=6, ax=ax)
    
    # Add individual point labels
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax.annotate(row['animal'].split('_')[1],  # Just the animal ID part
                        (i, row[metric]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, fontweight='bold')
    
    # Add means and SEM as error bars
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        mean_val = group_data[metric].mean()
        sem_val = stats.sem(group_data[metric])
        
        # Add mean point
        ax.scatter(i, mean_val, color='red', s=100, marker='D', 
                  zorder=10, edgecolors='black', linewidth=1)
        
        # Add SEM error bars
        ax.errorbar(i, mean_val, yerr=sem_val, color='red', 
                   capsize=5, capthick=2, elinewidth=2, zorder=10)
    
    # Add pairwise comparison annotations
    y_max = results_df[metric].max()
    y_min = results_df[metric].min()
    y_range = y_max - y_min
    
    # Create comparison lines and p-values
    comparison_y_positions = []
    line_height = y_range * 0.05
    
    for i, comp in enumerate(comparisons):
        group1_idx = list(results_df['age_group'].unique()).index(comp['group1'])
        group2_idx = list(results_df['age_group'].unique()).index(comp['group2'])
        
        # Calculate y position for this comparison
        base_y = y_max + (i + 1) * line_height
        comparison_y_positions.append(base_y)
        
        # Draw comparison line
        ax.plot([group1_idx, group2_idx], [base_y, base_y], 
                color='black', linewidth=1, alpha=0.7)
        
        # Add p-value text
        p_val = comp['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        # Position text above the line
        text_x = (group1_idx + group2_idx) / 2
        text_y = base_y + line_height * 0.3
        
        ax.text(text_x, text_y, f'p = {p_val:.4f} {significance}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_title(f'Pairwise Comparisons - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel(f'{title_suffix}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # Adjust y-axis limits to accommodate comparison lines
    if comparison_y_positions:
        ax.set_ylim(y_max + len(comparisons) * line_height * 1.5, y_min - y_range * 0.1)
    
    # Add sample size annotations
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        n = len(group_data)
        ax.text(i, y_min - y_range * 0.05, f'n={n}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'outputs/pairwise_comparisons_{metric.replace("weighted_", "").replace("_depth", "")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Professional pairwise plot saved to: {plot_filename}")
    
    return fig

def create_summary_statistics_plot(results_df, metric, title_suffix):
    """
    Create a summary statistics plot showing means and SEM
    """
    
    # Set the order for age groups: p3, p12, p20, adult
    age_group_order = ['p3', 'p12', 'p20', 'adult']
    results_df['age_group'] = pd.Categorical(results_df['age_group'], categories=age_group_order, ordered=True)
    
    # Calculate summary statistics
    summary_stats = results_df.groupby('age_group')[metric].agg(['mean', 'sem', 'std', 'count']).reset_index()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create bar plot with error bars
    x_pos = np.arange(len(summary_stats))
    bars = ax.bar(x_pos, summary_stats['mean'], 
                  yerr=summary_stats['sem'], 
                  capsize=5, alpha=0.7, 
                  color=sns.color_palette("husl", len(summary_stats)))
    
    # Add individual points
    for i, age_group in enumerate(summary_stats['age_group']):
        group_data = results_df[results_df['age_group'] == age_group]
        y_vals = group_data[metric].values
        x_vals = np.random.normal(i, 0.1, len(y_vals))  # Jitter the x positions
        ax.scatter(x_vals, y_vals, color='black', alpha=0.6, s=30)
    
    # Add mean values on bars
    for i, (bar, mean_val, sem_val) in enumerate(zip(bars, summary_stats['mean'], summary_stats['sem'])):
        ax.text(bar.get_x() + bar.get_width()/2, mean_val + sem_val + 1, 
                f'{mean_val:.2f}±{sem_val:.2f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Customize plot
    ax.set_title(f'Summary Statistics - {title_suffix}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel(f'{title_suffix}', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary_stats['age_group'])
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # Add sample sizes
    for i, n in enumerate(summary_stats['count']):
        ax.text(i, ax.get_ylim()[0] + 1, f'n={int(n)}', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'outputs/summary_statistics_{metric.replace("weighted_", "").replace("_depth", "")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Summary statistics plot saved to: {plot_filename}")
    
    return fig

def generate_professional_summary_csv(comparisons_mean, comparisons_median, metric_mean, metric_median):
    """
    Generate professional summary CSV
    """
    
    # Combine both metrics
    all_comparisons = []
    
    for comp in comparisons_mean:
        comp['metric'] = metric_mean
        all_comparisons.append(comp)
    
    for comp in comparisons_median:
        comp['metric'] = metric_median
        all_comparisons.append(comp)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_comparisons)
    
    # Save to CSV
    output_filename = 'outputs/professional_pairwise_summary.csv'
    summary_df.to_csv(output_filename, index=False)
    print(f"Professional pairwise summary CSV saved to: {output_filename}")
    
    return summary_df

def main():
    """Main function to run professional pairwise analysis"""
    
    # Load results
    results_file = 'outputs/vsv_depth_analysis_results.csv'
    
    try:
        results_df = load_analysis_results(results_file)
        print("Loaded VSV depth analysis results")
        print(f"Total replicates: {len(results_df)}")
        print(f"Age groups: {', '.join(results_df['age_group'].unique())}")
        print()
        
        # Perform pairwise tests for both metrics
        print("PERFORMING PAIRWISE TESTS:")
        print("-" * 40)
        
        # Weighted mean depth
        print("Weighted Mean Depth comparisons:")
        comparisons_mean = perform_pairwise_tests(results_df, 'weighted_mean_depth')
        
        # Weighted median depth
        print("Weighted Median Depth comparisons:")
        comparisons_median = perform_pairwise_tests(results_df, 'weighted_median_depth')
        
        # Create professional plots
        print(f"\nCREATING PROFESSIONAL PLOTS:")
        print("-" * 40)
        
        # Weighted mean depth plots
        create_professional_pairwise_plot(results_df, comparisons_mean, 
                                        'weighted_mean_depth', 'Weighted Mean Depth')
        create_summary_statistics_plot(results_df, 'weighted_mean_depth', 'Weighted Mean Depth')
        
        # Weighted median depth plots
        create_professional_pairwise_plot(results_df, comparisons_median, 
                                        'weighted_median_depth', 'Weighted Median Depth')
        create_summary_statistics_plot(results_df, 'weighted_median_depth', 'Weighted Median Depth')
        
        # Generate summary CSV
        print(f"\nGENERATING SUMMARY CSV:")
        print("-" * 40)
        summary_df = generate_professional_summary_csv(comparisons_mean, comparisons_median,
                                                    'weighted_mean_depth', 'weighted_median_depth')
        
        # Print final summary
        print(f"\n" + "="*80)
        print("FINAL SUMMARY - PROFESSIONAL PAIRWISE COMPARISONS")
        print("="*80)
        
        print(f"Weighted Mean Depth comparisons: {len(comparisons_mean)}")
        significant_mean = sum(1 for comp in comparisons_mean if comp['significant'])
        print(f"Significant comparisons (uncorrected): {significant_mean}/{len(comparisons_mean)}")
        
        print(f"\nWeighted Median Depth comparisons: {len(comparisons_median)}")
        significant_median = sum(1 for comp in comparisons_median if comp['significant'])
        print(f"Significant comparisons (uncorrected): {significant_median}/{len(comparisons_median)}")
        
        print(f"\nFiles generated:")
        print(f"  - Weighted mean depth plot: outputs/pairwise_comparisons_mean.png")
        print(f"  - Weighted median depth plot: outputs/pairwise_comparisons_median.png")
        print(f"  - Summary statistics plots: outputs/summary_statistics_*.png")
        print(f"  - Summary CSV: outputs/professional_pairwise_summary.csv")
        
        print(f"\n✓ Professional pairwise analysis completed")
        
    except FileNotFoundError:
        print(f"Error: Could not find results file: {results_file}")
        print("Please run vsv_depth_analysis.py first to generate the results file.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


