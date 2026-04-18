#!/usr/bin/env python3
"""
Distribution Comparison Analysis for Individual Replicates

This script compares the full intensity distributions (0-100 depth bins) 
between individual replicates within each age group using statistical tests
that can handle distribution comparisons.

Author: Generated for VSV depth analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

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

def perform_distribution_comparisons(raw_df, age_group, test_type='nonparametric'):
    """
    Perform statistical comparisons between intensity distributions of individual replicates
    within an age group using the full 0-100 depth bin data
    """
    
    # Filter data for the specific age group
    group_animals = [animal for animal in raw_df['animal'] if extract_age_group(animal) == age_group]
    group_data = raw_df[raw_df['animal'].isin(group_animals)].copy()
    
    if len(group_data) < 2:
        print(f"Not enough replicates in {age_group} group for comparisons")
        return None, None
    
    print(f"\n{age_group.upper()} GROUP DISTRIBUTION ANALYSIS:")
    print("=" * 60)
    print(f"Replicates: {', '.join(group_data['animal'].tolist())}")
    print(f"Using {test_type} tests on full intensity distributions")
    
    # Get all pairwise combinations of replicates
    replicates = group_data['animal'].tolist()
    comparisons = []
    
    for i, (rep1, rep2) in enumerate(itertools.combinations(replicates, 2)):
        # Get intensity data for both replicates (all depth bins)
        data1 = group_data[group_data['animal'] == rep1].iloc[0, 1:].values  # Skip 'animal' column
        data2 = group_data[group_data['animal'] == rep2].iloc[0, 1:].values  # Skip 'animal' column
        
        # Convert to float and remove any NaN values
        data1 = np.array(data1, dtype=float)
        data2 = np.array(data2, dtype=float)
        data1 = data1[~np.isnan(data1)]
        data2 = data2[~np.isnan(data2)]
        
        if test_type == 'parametric':
            # Use t-test on the distributions
            statistic, p_value = ttest_ind(data1, data2)
            test_name = "Independent t-test (distributions)"
        else:
            # Use Kolmogorov-Smirnov test for distribution comparison
            statistic, p_value = ks_2samp(data1, data2)
            test_name = "Kolmogorov-Smirnov test"
        
        # Calculate effect size
        if test_type == 'parametric':
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            effect_size = (np.mean(data1) - np.mean(data2)) / pooled_std
            effect_size_name = "Cohen's d"
        else:
            # For KS test, use the statistic as effect size
            effect_size = statistic
            effect_size_name = "KS statistic"
        
        # Calculate weighted mean depths for comparison
        depth_bins = np.arange(0, len(data1))
        weighted_mean1 = np.sum(depth_bins * data1) / np.sum(data1) if np.sum(data1) > 0 else 0
        weighted_mean2 = np.sum(depth_bins * data2) / np.sum(data2) if np.sum(data2) > 0 else 0
        
        comparison = {
            'age_group': age_group,
            'replicate1': rep1,
            'replicate2': rep2,
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'effect_size': effect_size,
            'effect_size_name': effect_size_name,
            'significant': p_value < 0.05,
            'n1': len(data1),
            'n2': len(data2),
            'mean_intensity1': np.mean(data1),
            'mean_intensity2': np.mean(data2),
            'weighted_mean_depth1': weighted_mean1,
            'weighted_mean_depth2': weighted_mean2,
            'depth_difference': weighted_mean1 - weighted_mean2
        }
        
        comparisons.append(comparison)
        
        # Print results
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        print(f"  {rep1} vs {rep2}:")
        print(f"    {test_name}: statistic = {statistic:.4f}, p = {p_value:.4f} {significance}")
        print(f"    Effect size ({effect_size_name}): {effect_size:.4f}")
        print(f"    Weighted mean depths: {rep1} = {weighted_mean1:.2f}, {rep2} = {weighted_mean2:.2f}")
        print(f"    Mean intensities: {rep1} = {np.mean(data1):.3f}, {rep2} = {np.mean(data2):.3f}")
        print()
    
    # Apply multiple comparison corrections
    if len(comparisons) > 1:
        p_values = [comp['p_value'] for comp in comparisons]
        
        # Bonferroni correction
        bonferroni_corrected = multipletests(p_values, method='bonferroni')
        # FDR correction (Benjamini-Hochberg)
        fdr_corrected = multipletests(p_values, method='fdr_bh')
        
        print(f"Multiple comparison corrections:")
        print(f"  Bonferroni correction:")
        print(f"    Original p-values: {[f'{p:.4f}' for p in p_values]}")
        print(f"    Corrected p-values: {[f'{p:.4f}' for p in bonferroni_corrected[1]]}")
        print(f"    Significant after correction: {sum(bonferroni_corrected[0])}")
        
        print(f"  FDR correction (Benjamini-Hochberg):")
        print(f"    Corrected p-values: {[f'{p:.4f}' for p in fdr_corrected[1]]}")
        print(f"    Significant after correction: {sum(fdr_corrected[0])}")
        
        # Add corrected p-values to comparisons
        for i, comp in enumerate(comparisons):
            comp['p_value_bonferroni'] = bonferroni_corrected[1][i]
            comp['p_value_fdr'] = fdr_corrected[1][i]
            comp['significant_bonferroni'] = bonferroni_corrected[0][i]
            comp['significant_fdr'] = fdr_corrected[0][i]
    else:
        print("Only one comparison - no multiple comparison correction needed")
        for comp in comparisons:
            comp['p_value_bonferroni'] = comp['p_value']
            comp['p_value_fdr'] = comp['p_value']
            comp['significant_bonferroni'] = comp['significant']
            comp['significant_fdr'] = comp['significant']
    
    return comparisons, group_data

def create_distribution_comparison_plot(group_data, comparisons, age_group, test_type='nonparametric'):
    """
    Create comprehensive plot comparing intensity distributions between replicates
    """
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{age_group.upper()} Group - Distribution Comparison Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Get depth bins
    depth_bins = np.arange(0, 101)
    
    # 1. Individual intensity profiles
    ax1 = axes[0, 0]
    for i, (_, row) in enumerate(group_data.iterrows()):
        animal = row['animal']
        intensities = row.iloc[1:].values  # Skip 'animal' column
        
        ax1.plot(depth_bins, intensities, label=animal, linewidth=2, alpha=0.8)
    
    ax1.set_title('Individual Intensity Profiles')
    ax1.set_xlabel('Depth (0-100)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()  # Flip x-axis so 0 is at right, 100 at left (surface to deep)
    
    # 2. Bar plot of weighted mean depths
    ax2 = axes[0, 1]
    replicates = group_data['animal'].tolist()
    weighted_means = []
    
    for _, row in group_data.iterrows():
        intensities = row.iloc[1:].values
        weighted_mean = np.sum(depth_bins * intensities) / np.sum(intensities) if np.sum(intensities) > 0 else 0
        weighted_means.append(weighted_mean)
    
    bars = ax2.bar(range(len(replicates)), weighted_means, alpha=0.7, 
                   color=sns.color_palette("husl", len(replicates)))
    
    # Add individual point labels
    for i, (bar, replicate, depth) in enumerate(zip(bars, replicates, weighted_means)):
        ax2.text(bar.get_x() + bar.get_width()/2, depth + 1, 
                f'{depth:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Weighted Mean Depths')
    ax2.set_xlabel('Replicates')
    ax2.set_ylabel('Weighted Mean Depth')
    ax2.set_xticks(range(len(replicates)))
    ax2.set_xticklabels(replicates, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Flip y-axis so 0 is at top, 100 at bottom
    
    # 3. Heatmap of intensity distributions
    ax3 = axes[1, 0]
    intensity_matrix = []
    for _, row in group_data.iterrows():
        intensities = row.iloc[1:].values
        # Convert to float array
        intensities = np.array(intensities, dtype=float)
        intensity_matrix.append(intensities)
    
    # Convert to numpy array
    intensity_matrix = np.array(intensity_matrix)
    
    im = ax3.imshow(intensity_matrix, aspect='auto', cmap='viridis', 
                    extent=[0, 100, len(replicates), 0])
    ax3.set_title('Intensity Distribution Heatmap')
    ax3.set_xlabel('Depth (0-100)')
    ax3.set_ylabel('Replicates')
    ax3.set_yticks(range(len(replicates)))
    ax3.set_yticklabels(replicates)
    ax3.invert_xaxis()  # Flip x-axis so 0 is at right, 100 at left
    plt.colorbar(im, ax=ax3, label='Normalized Intensity')
    
    # 4. Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if comparisons:
        # Create summary text
        summary_text = f"Statistical Summary:\n\n"
        summary_text += f"Total comparisons: {len(comparisons)}\n"
        summary_text += f"Significant (uncorrected): {sum(1 for comp in comparisons if comp['significant'])}\n"
        
        if 'significant_bonferroni' in comparisons[0]:
            summary_text += f"Significant (Bonferroni): {sum(1 for comp in comparisons if comp.get('significant_bonferroni', False))}\n"
            summary_text += f"Significant (FDR): {sum(1 for comp in comparisons if comp.get('significant_fdr', False))}\n"
        
        summary_text += f"\nTest: {comparisons[0]['test']}\n"
        summary_text += f"Effect size: {comparisons[0]['effect_size_name']}\n\n"
        
        # Add individual comparison results
        summary_text += "Pairwise Comparisons:\n"
        for comp in comparisons:
            p_val = comp['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            summary_text += f"{comp['replicate1']} vs {comp['replicate2']}: p = {p_val:.4f} {significance}\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f'outputs/{age_group}_distribution_comparison_{test_type}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison plot saved to: {plot_filename}")
    
    return fig

def generate_distribution_summary_csv(all_comparisons, test_type='nonparametric'):
    """
    Generate summary CSV for distribution comparisons
    """
    
    if not all_comparisons:
        print("No comparisons to summarize")
        return None
    
    # Flatten all comparisons
    all_comparison_data = []
    for age_group_comparisons in all_comparisons.values():
        all_comparison_data.extend(age_group_comparisons)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_comparison_data)
    
    # Save to CSV
    output_filename = f'outputs/distribution_comparisons_{test_type}.csv'
    summary_df.to_csv(output_filename, index=False)
    print(f"Distribution comparison summary CSV saved to: {output_filename}")
    
    return summary_df

def main():
    """Main function to run distribution comparison analysis"""
    
    # Load raw data
    input_file = 'data/visp_raw_normalized_data.csv'
    
    try:
        raw_df = load_raw_data(input_file)
        print("Loaded VSV raw depth data")
        print(f"Total replicates: {len(raw_df)}")
        print(f"Depth bins: {len(raw_df.columns) - 1} (0-100)")
        print()
        
        # Determine test type based on data characteristics
        test_type = 'nonparametric'  # KS test is more appropriate for distribution comparison
        print(f"Using {test_type} tests (Kolmogorov-Smirnov) for distribution comparison")
        print()
        
        # Analyze each age group
        age_groups = ['p3', 'p12', 'p20', 'adult']
        all_comparisons = {}
        
        for age_group in age_groups:
            # Check if age group exists in data
            group_animals = [animal for animal in raw_df['animal'] if extract_age_group(animal) == age_group]
            if not group_animals:
                print(f"Age group {age_group} not found in data, skipping...")
                continue
            
            # Perform distribution comparisons
            comparisons, group_data = perform_distribution_comparisons(raw_df, age_group, test_type)
            
            if comparisons is not None:
                all_comparisons[age_group] = comparisons
                
                # Create distribution comparison plot
                create_distribution_comparison_plot(group_data, comparisons, age_group, test_type)
        
        # Generate summary CSV
        print("\n" + "="*80)
        print("GENERATING SUMMARY CSV")
        print("="*80)
        summary_df = generate_distribution_summary_csv(all_comparisons, test_type)
        
        # Print final summary
        print(f"\n" + "="*80)
        print("FINAL SUMMARY - DISTRIBUTION COMPARISON ANALYSIS")
        print("="*80)
        print(f"Test type: {test_type}")
        
        total_comparisons = sum(len(comps) for comps in all_comparisons.values())
        print(f"Total comparisons: {total_comparisons}")
        
        for age_group, comparisons in all_comparisons.items():
            significant_uncorrected = sum(1 for comp in comparisons if comp['significant'])
            significant_bonferroni = sum(1 for comp in comparisons if comp.get('significant_bonferroni', False))
            significant_fdr = sum(1 for comp in comparisons if comp.get('significant_fdr', False))
            
            print(f"\n{age_group.upper()} group:")
            print(f"  Replicates: {len(set([comp['replicate1'] for comp in comparisons] + [comp['replicate2'] for comp in comparisons]))}")
            print(f"  Comparisons: {len(comparisons)}")
            print(f"  Significant (uncorrected): {significant_uncorrected}")
            print(f"  Significant (Bonferroni): {significant_bonferroni}")
            print(f"  Significant (FDR): {significant_fdr}")
        
        print(f"\nFiles generated:")
        for age_group in age_groups:
            if age_group in all_comparisons:
                print(f"  - {age_group} plot: outputs/{age_group}_distribution_comparison_{test_type}.png")
        print(f"  - Summary CSV: outputs/distribution_comparisons_{test_type}.csv")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file: {input_file}")
        print("Please ensure the raw data file exists.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
