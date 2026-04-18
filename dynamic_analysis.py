#!/usr/bin/env python3
"""
Dynamic VSV Depth Analysis with Automatic Test Selection

This script automatically determines the appropriate statistical tests based on
data characteristics and runs only the appropriate analysis.

Author: Generated for VSV depth analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind, shapiro, levene
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

def load_raw_data(input_file):
    """Load the raw VSV depth data"""
    return pd.read_csv(input_file)

def extract_age_group(animal_name):
    """Extract age group from animal name"""
    if animal_name.startswith('adult'):
        return 'adult'
    elif animal_name.startswith('p3'):
        return 'p3'
    elif animal_name.startswith('p12'):
        return 'p12'
    elif animal_name.startswith('p20'):
        return 'p20'
    else:
        return 'unknown'

def assess_data_assumptions(results_df):
    """
    Assess statistical assumptions and recommend appropriate tests
    
    Returns:
    --------
    dict: Test recommendations and assumption results
    """
    
    print("="*80)
    print("STATISTICAL ASSUMPTION ASSESSMENT")
    print("="*80)
    
    # 1. Normality assessment
    print("\n1. NORMALITY ASSESSMENT:")
    print("-" * 40)
    
    normal_count = results_df['is_normal'].sum()
    total_count = results_df['is_normal'].count()
    normality_percentage = (normal_count / total_count) * 100
    
    print(f"Replicates with normal distributions: {normal_count}/{total_count} ({normality_percentage:.1f}%)")
    
    if normality_percentage >= 70:
        normality_assumption = "met"
        print("✓ Normality assumption: MET (≥70% normal)")
    else:
        normality_assumption = "violated"
        print("✗ Normality assumption: VIOLATED (<70% normal)")
    
    # 2. Equal variance assessment
    print("\n2. EQUAL VARIANCE ASSESSMENT:")
    print("-" * 40)
    
    equal_variances = results_df['equal_variances'].iloc[0]
    levene_p = results_df['levene_p_value'].iloc[0]
    
    print(f"Levene's test p-value: {levene_p:.4f}")
    
    if levene_p > 0.05:
        variance_assumption = "met"
        print("✓ Equal variance assumption: MET (p > 0.05)")
    else:
        variance_assumption = "violated"
        print("✗ Equal variance assumption: VIOLATED (p ≤ 0.05)")
    
    # 3. Test recommendation
    print("\n3. AUTOMATIC TEST SELECTION:")
    print("-" * 40)
    
    if normality_assumption == "met" and variance_assumption == "met":
        recommended_test = "parametric"
        test_name = "t-tests and ANOVA"
        print("✓ RECOMMENDATION: Parametric tests (t-tests, ANOVA)")
        print("  - Independent t-tests for pairwise comparisons")
        print("  - One-way ANOVA for omnibus tests")
    else:
        recommended_test = "nonparametric"
        test_name = "Mann-Whitney U and Kruskal-Wallis"
        print("✓ RECOMMENDATION: Non-parametric tests (Mann-Whitney U, Kruskal-Wallis)")
        print("  - Mann-Whitney U tests for pairwise comparisons")
        print("  - Kruskal-Wallis test for omnibus tests")
    
    # 4. Summary
    print("\n4. ASSUMPTION SUMMARY:")
    print("-" * 40)
    print(f"Normality: {normality_assumption.upper()}")
    print(f"Equal variances: {variance_assumption.upper()}")
    print(f"Recommended approach: {recommended_test.upper()}")
    print(f"Test type: {test_name}")
    
    return {
        'normality_assumption': normality_assumption,
        'variance_assumption': variance_assumption,
        'recommended_test': recommended_test,
        'test_name': test_name,
        'normal_count': normal_count,
        'total_count': total_count,
        'normality_percentage': normality_percentage,
        'levene_p': levene_p
    }

def perform_dynamic_pairwise_tests(results_df, test_type):
    """
    Perform pairwise comparisons using the automatically selected test type
    """
    
    # Get unique age groups
    age_groups = results_df['age_group'].unique()
    
    print(f"\n{'='*80}")
    print(f"DYNAMIC PAIRWISE COMPARISONS - {test_type.upper()} TESTS")
    print(f"{'='*80}")
    print(f"Using: {test_type} approach")
    print(f"Age groups: {', '.join(age_groups)}")
    
    # Choose the appropriate depth metric
    if test_type == 'parametric':
        depth_col = 'weighted_mean_depth'
        ylabel = 'Weighted Mean Depth'
    else:
        depth_col = 'weighted_median_depth'
        ylabel = 'Weighted Median Depth'
    
    print(f"Depth metric: {ylabel}")
    
    # Initialize results
    all_comparisons = []
    
    # Across-age group comparisons
    print(f"\nACROSS-AGE GROUP COMPARISONS:")
    print("-" * 40)
    
    for group1, group2 in itertools.combinations(age_groups, 2):
        data1 = results_df[results_df['age_group'] == group1][depth_col].values
        data2 = results_df[results_df['age_group'] == group2][depth_col].values
        
        if test_type == 'parametric':
            # Use t-test
            statistic, p_value = ttest_ind(data1, data2)
            test_name = "Independent t-test"
        else:
            # Use Mann-Whitney U test
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        
        # Calculate effect size
        if test_type == 'parametric':
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
            effect_size_name = "Cohen's d"
        else:
            # Rank-biserial correlation
            n1, n2 = len(data1), len(data2)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            effect_size_name = "Rank-biserial correlation"
        
        comparison = {
            'comparison_type': 'across_group',
            'group1': group1,
            'group2': group2,
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'significant': p_value < 0.05,
            'n1': len(data1),
            'n2': len(data2),
            'mean1': np.mean(data1),
            'mean2': np.mean(data2)
        }
        
        all_comparisons.append(comparison)
        
        # Print results
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"{group1} vs {group2}:")
        print(f"  {test_name}: statistic = {statistic:.4f}, p = {p_value:.4f} {significance}")
        print(f"  Effect size ({effect_size_name}): {effect_size:.4f}")
        print(f"  Mean depths: {group1} = {np.mean(data1):.2f}, {group2} = {np.mean(data2):.2f}")
        print()
    
    # Apply multiple comparison corrections
    print("MULTIPLE COMPARISON CORRECTIONS:")
    print("-" * 40)
    
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
        
        print(f"\nFDR correction:")
        print(f"  Corrected p-values: {[f'{p:.4f}' for p in fdr_corrected[1]]}")
        print(f"  Significant after correction: {sum(fdr_corrected[0])}")
        
        # Add corrected p-values to comparisons
        for i, comp in enumerate(all_comparisons):
            comp['p_value_bonferroni'] = bonferroni_corrected[1][i]
            comp['p_value_fdr'] = fdr_corrected[1][i]
            comp['significant_bonferroni'] = bonferroni_corrected[0][i]
            comp['significant_fdr'] = fdr_corrected[0][i]
    
    return all_comparisons

def create_dynamic_plots(results_df, comparisons, test_type):
    """
    Create plots using the automatically selected test type
    """
    
    # Choose the appropriate depth metric
    if test_type == 'parametric':
        depth_col = 'weighted_mean_depth'
        ylabel = 'Weighted Mean Depth'
    else:
        depth_col = 'weighted_median_depth'
        ylabel = 'Weighted Median Depth'
    
    # Set the order for age groups: p3, p12, p20, adult
    age_group_order = ['p3', 'p12', 'p20', 'adult']
    results_df['age_group'] = pd.Categorical(results_df['age_group'], categories=age_group_order, ordered=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'VSV Depth Analysis - {ylabel} ({test_type.title()} Tests)', fontsize=16, fontweight='bold')
    
    # 1. Overall box plot by age group
    ax1 = axes[0, 0]
    sns.boxplot(data=results_df, x='age_group', y=depth_col, ax=ax1)
    sns.stripplot(data=results_df, x='age_group', y=depth_col, 
                  color='red', alpha=0.7, size=6, ax=ax1)
    
    # Add individual point labels
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax1.annotate(row['animal'].split('_')[1],  # Just the animal ID part
                        (i, row[depth_col]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
    
    ax1.set_title('Depth Distribution by Age Group')
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 2. Individual replicates plot
    ax2 = axes[0, 1]
    sns.stripplot(data=results_df, x='age_group', y=depth_col, 
                  hue='age_group', size=8, ax=ax2)
    
    # Add individual point labels
    for i, age_group in enumerate(results_df['age_group'].unique()):
        group_data = results_df[results_df['age_group'] == age_group]
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax2.annotate(row['animal'], 
                        (i, row[depth_col]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8, rotation=45)
    
    ax2.set_title('Individual Replicates by Age Group')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel(ylabel)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 3. Violin plot
    ax3 = axes[1, 0]
    sns.violinplot(data=results_df, x='age_group', y=depth_col, ax=ax3)
    sns.stripplot(data=results_df, x='age_group', y=depth_col, 
                  color='white', alpha=0.8, size=4, ax=ax3)
    ax3.set_title('Depth Distribution (Violin Plot)')
    ax3.set_xlabel('Age Group')
    ax3.set_ylabel(ylabel)
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    summary_stats = results_df.groupby('age_group')[depth_col].agg(['mean', 'std', 'count']).reset_index()
    
    x_pos = np.arange(len(summary_stats))
    bars = ax4.bar(x_pos, summary_stats['mean'], 
                   yerr=summary_stats['std'], 
                   capsize=5, alpha=0.7)
    
    # Color bars by age group
    colors = sns.color_palette("husl", len(summary_stats))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax4.set_title('Mean Depth by Age Group (±SD)')
    ax4.set_xlabel('Age Group')
    ax4.set_ylabel(ylabel)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(summary_stats['age_group'])
    ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # Add sample sizes
    for i, (_, row) in enumerate(summary_stats.iterrows()):
        ax4.text(i, row['mean'] + row['std'] + 1, f'n={int(row["count"])}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'outputs/vsv_depth_dynamic_analysis_{test_type}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Dynamic analysis plots saved to: {plot_filename}")
    
    return fig

def generate_dynamic_summary_csv(comparisons, test_type):
    """
    Generate summary CSV for dynamic analysis
    """
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(comparisons)
    
    # Save to CSV
    output_filename = f'outputs/vsv_depth_dynamic_summary_{test_type}.csv'
    summary_df.to_csv(output_filename, index=False)
    print(f"Dynamic analysis summary CSV saved to: {output_filename}")
    
    return summary_df

def main():
    """Main function to run dynamic analysis"""
    
    # Load results
    results_file = 'outputs/vsv_depth_analysis_results.csv'
    
    try:
        results_df = load_analysis_results(results_file)
        print("Loaded VSV depth analysis results")
        print(f"Total replicates: {len(results_df)}")
        print(f"Age groups: {', '.join(results_df['age_group'].unique())}")
        print()
        
        # Assess assumptions and get test recommendation
        assumptions = assess_data_assumptions(results_df)
        
        # Use the recommended test type
        test_type = assumptions['recommended_test']
        
        print(f"\n{'='*80}")
        print(f"EXECUTING DYNAMIC ANALYSIS - {test_type.upper()} TESTS")
        print(f"{'='*80}")
        
        # Perform pairwise comparisons
        comparisons = perform_dynamic_pairwise_tests(results_df, test_type)
        
        # Create plots
        print(f"\nCREATING DYNAMIC PLOTS:")
        print("-" * 40)
        create_dynamic_plots(results_df, comparisons, test_type)
        
        # Generate summary CSV
        print(f"\nGENERATING SUMMARY CSV:")
        print("-" * 40)
        summary_df = generate_dynamic_summary_csv(comparisons, test_type)
        
        # Print final summary
        print(f"\n" + "="*80)
        print("FINAL SUMMARY - DYNAMIC ANALYSIS")
        print("="*80)
        print(f"Test type: {test_type}")
        print(f"Test name: {assumptions['test_name']}")
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
        print(f"  - Dynamic plots: outputs/vsv_depth_dynamic_analysis_{test_type}.png")
        print(f"  - Summary CSV: outputs/vsv_depth_dynamic_summary_{test_type}.csv")
        
        print(f"\n✓ Analysis completed using {test_type} tests based on data characteristics")
        
    except FileNotFoundError:
        print(f"Error: Could not find results file: {results_file}")
        print("Please run vsv_depth_analysis.py first to generate the results file.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


