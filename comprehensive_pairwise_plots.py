#!/usr/bin/env python3
"""
Comprehensive Pairwise Comparison Plots

This script creates detailed plots showing all pairwise comparisons between age groups
with individual data points, p-values, and effect sizes clearly displayed.

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

def perform_comprehensive_pairwise_tests(results_df):
    """
    Perform comprehensive pairwise comparisons between age groups
    """
    
    # Get unique age groups
    age_groups = results_df['age_group'].unique()
    
    print("="*80)
    print("COMPREHENSIVE PAIRWISE COMPARISONS BETWEEN AGE GROUPS")
    print("="*80)
    print(f"Age groups: {', '.join(age_groups)}")
    print(f"Using: Non-parametric tests (Mann-Whitney U)")
    print(f"Depth metric: Weighted Median Depth")
    
    # Initialize results
    all_comparisons = []
    
    # Get all pairwise combinations
    for group1, group2 in itertools.combinations(age_groups, 2):
        data1 = results_df[results_df['age_group'] == group1]['weighted_median_depth'].values
        data2 = results_df[results_df['age_group'] == group2]['weighted_median_depth'].values
        
        # Perform Mann-Whitney U test
        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(data1), len(data2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
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
            'mean1': np.mean(data1),
            'mean2': np.mean(data2),
            'std1': np.std(data1),
            'std2': np.std(data2)
        }
        
        all_comparisons.append(comparison)
        
        # Print results
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"\n{group1.upper()} vs {group2.upper()}:")
        print(f"  Mann-Whitney U test: statistic = {statistic:.4f}, p = {p_value:.4f} {significance}")
        print(f"  Effect size (Rank-biserial correlation): {effect_size:.4f}")
        print(f"  Mean depths: {group1} = {np.mean(data1):.2f} ± {np.std(data1):.2f}, {group2} = {np.mean(data2):.2f} ± {np.std(data2):.2f}")
        print(f"  Sample sizes: {group1} = {len(data1)}, {group2} = {len(data2)}")
    
    # Apply multiple comparison corrections
    print(f"\n" + "="*60)
    print("MULTIPLE COMPARISON CORRECTIONS")
    print("="*60)
    
    p_values = [comp['p_value'] for comp in all_comparisons]
    
    if len(p_values) > 1:
        # Bonferroni correction
        bonferroni_corrected = multipletests(p_values, method='bonferroni')
        # FDR correction (Benjamini-Hochberg)
        fdr_corrected = multipletests(p_values, method='fdr_bh')
        
        print(f"Bonferroni correction:")
        print(f"  Original p-values: {[f'{p:.4f}' for p in p_values]}")
        print(f"  Corrected p-values: {[f'{p:.4f}' for p in bonferroni_corrected[1]]}")
        print(f"  Significant after correction: {sum(bonferroni_corrected[0])}")
        
        print(f"\nFDR correction (Benjamini-Hochberg):")
        print(f"  Corrected p-values: {[f'{p:.4f}' for p in fdr_corrected[1]]}")
        print(f"  Significant after correction: {sum(fdr_corrected[0])}")
        
        # Add corrected p-values to comparisons
        for i, comp in enumerate(all_comparisons):
            comp['p_value_bonferroni'] = bonferroni_corrected[1][i]
            comp['p_value_fdr'] = fdr_corrected[1][i]
            comp['significant_bonferroni'] = bonferroni_corrected[0][i]
            comp['significant_fdr'] = fdr_corrected[0][i]
    
    return all_comparisons

def create_comprehensive_pairwise_plots(results_df, comparisons):
    """
    Create comprehensive plots showing all pairwise comparisons
    """
    
    # Set the order for age groups: p3, p12, p20, adult
    age_group_order = ['p3', 'p12', 'p20', 'adult']
    results_df['age_group'] = pd.Categorical(results_df['age_group'], categories=age_group_order, ordered=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Pairwise Comparisons Between Age Groups', fontsize=16, fontweight='bold')
    
    # 1. Overall box plot by age group
    ax1 = axes[0, 0]
    sns.boxplot(data=results_df, x='age_group', y='weighted_median_depth', ax=ax1)
    sns.stripplot(data=results_df, x='age_group', y='weighted_median_depth', 
                  color='red', alpha=0.7, size=6, ax=ax1)
    
    # Add individual point labels
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax1.annotate(row['animal'].split('_')[1],  # Just the animal ID part
                        (i, row['weighted_median_depth']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    ax1.set_title('Depth Distribution by Age Group')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Weighted Median Depth')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 2. Individual replicates plot
    ax2 = axes[0, 1]
    sns.stripplot(data=results_df, x='age_group', y='weighted_median_depth', 
                  hue='age_group', size=8, ax=ax2)
    
    # Add individual point labels
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax2.annotate(row['animal'], 
                        (i, row['weighted_median_depth']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, rotation=45)
    
    ax2.set_title('Individual Replicates by Age Group')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Weighted Median Depth')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 3. Violin plot
    ax3 = axes[0, 2]
    sns.violinplot(data=results_df, x='age_group', y='weighted_median_depth', ax=ax3)
    sns.stripplot(data=results_df, x='age_group', y='weighted_median_depth', 
                  color='white', alpha=0.8, size=4, ax=ax3)
    ax3.set_title('Depth Distribution (Violin Plot)')
    ax3.set_xlabel('Age Group')
    ax3.set_ylabel('Weighted Median Depth')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 4. Pairwise comparison 1: P3 vs P12
    ax4 = axes[1, 0]
    p3_data = results_df[results_df['age_group'] == 'p3']['weighted_median_depth']
    p12_data = results_df[results_df['age_group'] == 'p12']['weighted_median_depth']
    
    # Find the comparison
    comp = next((c for c in comparisons if c['group1'] == 'p3' and c['group2'] == 'p12'), None)
    if comp:
        p_val = comp['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        # Create box plot
        data_to_plot = [p3_data.values, p12_data.values]
        bp = ax4.boxplot(data_to_plot, labels=['P3', 'P12'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        # Add individual points
        ax4.scatter([1]*len(p3_data), p3_data, color='blue', alpha=0.7, s=50)
        ax4.scatter([2]*len(p12_data), p12_data, color='green', alpha=0.7, s=50)
        
        ax4.set_title(f'P3 vs P12\np = {p_val:.4f} {significance}')
        ax4.set_ylabel('Weighted Median Depth')
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 5. Pairwise comparison 2: P3 vs P20
    ax5 = axes[1, 1]
    p20_data = results_df[results_df['age_group'] == 'p20']['weighted_median_depth']
    
    comp = next((c for c in comparisons if c['group1'] == 'p3' and c['group2'] == 'p20'), None)
    if comp:
        p_val = comp['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        # Create box plot
        data_to_plot = [p3_data.values, p20_data.values]
        bp = ax5.boxplot(data_to_plot, labels=['P3', 'P20'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('orange')
        
        # Add individual points
        ax5.scatter([1]*len(p3_data), p3_data, color='blue', alpha=0.7, s=50)
        ax5.scatter([2]*len(p20_data), p20_data, color='orange', alpha=0.7, s=50)
        
        ax5.set_title(f'P3 vs P20\np = {p_val:.4f} {significance}')
        ax5.set_ylabel('Weighted Median Depth')
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 6. Pairwise comparison 3: P12 vs P20
    ax6 = axes[1, 2]
    
    comp = next((c for c in comparisons if c['group1'] == 'p12' and c['group2'] == 'p20'), None)
    if comp:
        p_val = comp['p_value']
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        
        # Create box plot
        data_to_plot = [p12_data.values, p20_data.values]
        bp = ax6.boxplot(data_to_plot, labels=['P12', 'P20'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('orange')
        
        # Add individual points
        ax6.scatter([1]*len(p12_data), p12_data, color='green', alpha=0.7, s=50)
        ax6.scatter([2]*len(p20_data), p20_data, color='orange', alpha=0.7, s=50)
        
        ax6.set_title(f'P12 vs P20\np = {p_val:.4f} {significance}')
        ax6.set_ylabel('Weighted Median Depth')
        ax6.grid(True, alpha=0.3)
        ax6.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'outputs/comprehensive_pairwise_comparisons.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comprehensive pairwise comparison plots saved to: {plot_filename}")
    
    return fig

def create_detailed_pairwise_plots(results_df, comparisons):
    """
    Create detailed plots for each pairwise comparison
    """
    
    # Set the order for age groups: p3, p12, p20, adult
    age_group_order = ['p3', 'p12', '20', 'adult']
    results_df['age_group'] = pd.Categorical(results_df['age_group'], categories=age_group_order, ordered=True)
    
    # Create figure with subplots for all comparisons
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Pairwise Comparisons Between Age Groups', fontsize=16, fontweight='bold')
    
    # Get all pairwise combinations
    age_groups = results_df['age_group'].unique()
    comparisons_list = list(itertools.combinations(age_groups, 2))
    
    # Create a plot for each comparison
    for i, (group1, group2) in enumerate(comparisons_list):
        if i >= 6:  # Only show first 6 comparisons
            break
            
        ax = axes[i//3, i%3]
        
        # Get data for both groups
        data1 = results_df[results_df['age_group'] == group1]['weighted_median_depth']
        data2 = results_df[results_df['age_group'] == group2]['weighted_median_depth']
        
        # Find the comparison
        comp = next((c for c in comparisons if c['group1'] == group1 and c['group2'] == group2), None)
        if comp:
            p_val = comp['p_value']
            effect_size = comp['effect_size']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            # Create box plot
            data_to_plot = [data1.values, data2.values]
            bp = ax.boxplot(data_to_plot, labels=[group1.upper(), group2.upper()], patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
            bp['boxes'][0].set_facecolor(colors[i % len(colors)])
            bp['boxes'][1].set_facecolor(colors[(i+1) % len(colors)])
            
            # Add individual points
            ax.scatter([1]*len(data1), data1, color='blue', alpha=0.7, s=50)
            ax.scatter([2]*len(data2), data2, color='red', alpha=0.7, s=50)
            
            # Add statistics text
            stats_text = f'p = {p_val:.4f} {significance}\nEffect size = {effect_size:.3f}'
            ax.text(0.5, 0.95, stats_text, transform=ax.transAxes, 
                   ha='center', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{group1.upper()} vs {group2.upper()}')
            ax.set_ylabel('Weighted Median Depth')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # Hide unused subplots
    for i in range(len(comparisons_list), 6):
        axes[i//3, i%3].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'outputs/detailed_pairwise_comparisons.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Detailed pairwise comparison plots saved to: {plot_filename}")
    
    return fig

def generate_comprehensive_summary_csv(comparisons):
    """
    Generate comprehensive summary CSV for pairwise comparisons
    """
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(comparisons)
    
    # Save to CSV
    output_filename = 'outputs/comprehensive_pairwise_summary.csv'
    summary_df.to_csv(output_filename, index=False)
    print(f"Comprehensive pairwise summary CSV saved to: {output_filename}")
    
    return summary_df

def main():
    """Main function to run comprehensive pairwise analysis"""
    
    # Load results
    results_file = 'outputs/vsv_depth_analysis_results.csv'
    
    try:
        results_df = load_analysis_results(results_file)
        print("Loaded VSV depth analysis results")
        print(f"Total replicates: {len(results_df)}")
        print(f"Age groups: {', '.join(results_df['age_group'].unique())}")
        print()
        
        # Perform comprehensive pairwise comparisons
        comparisons = perform_comprehensive_pairwise_tests(results_df)
        
        # Create comprehensive plots
        print(f"\nCREATING COMPREHENSIVE PLOTS:")
        print("-" * 40)
        create_comprehensive_pairwise_plots(results_df, comparisons)
        create_detailed_pairwise_plots(results_df, comparisons)
        
        # Generate summary CSV
        print(f"\nGENERATING SUMMARY CSV:")
        print("-" * 40)
        summary_df = generate_comprehensive_summary_csv(comparisons)
        
        # Print final summary
        print(f"\n" + "="*80)
        print("FINAL SUMMARY - COMPREHENSIVE PAIRWISE COMPARISONS")
        print("="*80)
        print(f"Total comparisons: {len(comparisons)}")
        
        significant_uncorrected = sum(1 for comp in comparisons if comp['significant'])
        print(f"Significant comparisons (uncorrected): {significant_uncorrected}/{len(comparisons)}")
        
        if 'significant_bonferroni' in comparisons[0]:
            significant_bonferroni = sum(1 for comp in comparisons if comp.get('significant_bonferroni', False))
            print(f"Significant comparisons (Bonferroni): {significant_bonferroni}/{len(comparisons)}")
        
        if 'significant_fdr' in comparisons[0]:
            significant_fdr = sum(1 for comp in comparisons if comp.get('significant_fdr', False))
            print(f"Significant comparisons (FDR): {significant_fdr}/{len(comparisons)}")
        
        print(f"\nFiles generated:")
        print(f"  - Comprehensive plots: outputs/comprehensive_pairwise_comparisons.png")
        print(f"  - Detailed plots: outputs/detailed_pairwise_comparisons.png")
        print(f"  - Summary CSV: outputs/comprehensive_pairwise_summary.csv")
        
        print(f"\n✓ Comprehensive pairwise analysis completed")
        
    except FileNotFoundError:
        print(f"Error: Could not find results file: {results_file}")
        print("Please run vsv_depth_analysis.py first to generate the results file.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
