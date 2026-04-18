#!/usr/bin/env python3
"""
VSV Depth Analysis Script

This script reads the visp_raw_normalized_data.csv file and produces comprehensive
per-replicate summary statistics including:
- Mean and median depth of signal
- Standard deviation and standard error of the mean
- Normality tests (Shapiro-Wilk)
- Equal variance tests (Levene's test)
- Data structured for downstream pairwise comparisons

Author: Generated for VSV depth analysis
Date: 2024
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, levene
import warnings
warnings.filterwarnings('ignore')

def extract_age_group(animal_name):
    """Extract age group from animal name (e.g., 'adult_M608' -> 'adult')"""
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

def calculate_weighted_mean_depth(data_row):
    """Calculate weighted mean depth using intensity values as weights"""
    depth_bins = np.arange(0, 101)  # 0 to 100
    intensities = data_row.values
    
    # Calculate weighted mean depth
    weighted_sum = np.sum(depth_bins * intensities)
    total_intensity = np.sum(intensities)
    
    if total_intensity > 0:
        return weighted_sum / total_intensity
    else:
        return 0

def calculate_weighted_median_depth(data_row):
    """Calculate weighted median depth"""
    depth_bins = np.arange(0, 101)
    intensities = data_row.values
    
    # Create cumulative distribution
    cumulative_intensity = np.cumsum(intensities)
    total_intensity = cumulative_intensity[-1]
    
    if total_intensity == 0:
        return 0
    
    # Find median point (50% of total intensity)
    median_point = total_intensity / 2
    
    # Find depth bin where cumulative intensity crosses median
    median_idx = np.searchsorted(cumulative_intensity, median_point)
    
    if median_idx == 0:
        return 0
    elif median_idx >= len(depth_bins):
        return 100
    else:
        # Linear interpolation for more precise median
        if median_idx > 0:
            prev_cumsum = cumulative_intensity[median_idx - 1]
            curr_cumsum = cumulative_intensity[median_idx]
            
            # Linear interpolation
            weight = (median_point - prev_cumsum) / (curr_cumsum - prev_cumsum)
            return depth_bins[median_idx - 1] + weight * (depth_bins[median_idx] - depth_bins[median_idx - 1])
        else:
            return depth_bins[median_idx]

def test_normality(data_row):
    """Test normality of intensity distribution using Shapiro-Wilk test"""
    intensities = data_row.values
    
    # Remove zeros to avoid issues with Shapiro-Wilk test
    non_zero_intensities = intensities[intensities > 0]
    
    if len(non_zero_intensities) < 3:
        return np.nan, np.nan
    
    try:
        statistic, p_value = shapiro(non_zero_intensities)
        return statistic, p_value
    except:
        return np.nan, np.nan

def analyze_vsv_depth_data(input_file, output_file=None):
    """
    Main function to analyze VSV depth data and generate comprehensive statistics
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the output CSV file
    """
    
    print("Loading VSV depth data...")
    # Read the data
    df = pd.read_csv(input_file)
    
    print(f"Loaded data with {len(df)} replicates")
    print(f"Depth bins: {len(df.columns) - 1} (0-100)")
    
    # Initialize results list
    results = []
    
    print("\nCalculating per-replicate statistics...")
    
    for idx, row in df.iterrows():
        animal = row['animal']
        age_group = extract_age_group(animal)
        
        # Get intensity data (all columns except 'animal')
        intensity_data = row.iloc[1:]
        
        # Calculate basic statistics
        mean_intensity = intensity_data.mean()
        median_intensity = intensity_data.median()
        std_intensity = intensity_data.std()
        sem_intensity = std_intensity / np.sqrt(len(intensity_data))
        min_intensity = intensity_data.min()
        max_intensity = intensity_data.max()
        variance_intensity = intensity_data.var()
        
        # Calculate weighted mean and median depths
        weighted_mean_depth = calculate_weighted_mean_depth(intensity_data)
        weighted_median_depth = calculate_weighted_median_depth(intensity_data)
        
        # Test normality
        shapiro_stat, shapiro_p = test_normality(intensity_data)
        
        # Store results
        result = {
            'animal': animal,
            'age_group': age_group,
            'mean_intensity': mean_intensity,
            'median_intensity': median_intensity,
            'std_intensity': std_intensity,
            'sem_intensity': sem_intensity,
            'variance_intensity': variance_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'weighted_mean_depth': weighted_mean_depth,
            'weighted_median_depth': weighted_median_depth,
            'variance_intensity': variance_intensity,
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else np.nan
        }
        
        results.append(result)
        
        print(f"  {animal}: Mean depth = {weighted_mean_depth:.2f}, Median depth = {weighted_median_depth:.2f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Test for equal variances across age groups
    print("\nTesting for equal variances across age groups...")
    age_groups = results_df['age_group'].unique()
    
    # Group data by age group for Levene's test
    group_data = []
    group_labels = []
    
    for age_group in age_groups:
        group_intensities = results_df[results_df['age_group'] == age_group]['mean_intensity'].values
        if len(group_intensities) > 1:  # Need at least 2 groups for Levene's test
            group_data.append(group_intensities)
            group_labels.append(age_group)
    
    if len(group_data) >= 2:
        try:
            levene_stat, levene_p = levene(*group_data)
            print(f"Levene's test for equal variances: statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
            print(f"Equal variances assumption: {'met' if levene_p > 0.05 else 'violated'} (α = 0.05)")
        except Exception as e:
            print(f"Could not perform Levene's test: {e}")
            levene_stat, levene_p = np.nan, np.nan
    else:
        print("Insufficient groups for Levene's test")
        levene_stat, levene_p = np.nan, np.nan
    
    # Add summary statistics
    print("\nSummary by age group:")
    summary_stats = results_df.groupby('age_group').agg({
        'weighted_mean_depth': ['mean', 'std', 'sem'],
        'weighted_median_depth': ['mean', 'std', 'sem'],
        'mean_intensity': ['mean', 'std', 'sem'],
        'is_normal': 'sum'
    }).round(4)
    
    print(summary_stats)
    
    # Add Levene's test results to the dataframe
    results_df['levene_statistic'] = levene_stat
    results_df['levene_p_value'] = levene_p
    results_df['equal_variances'] = levene_p > 0.05 if not np.isnan(levene_p) else np.nan
    
    # Save results
    if output_file is None:
        output_file = 'vsv_depth_analysis_results.csv'
    
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary for downstream analysis
    print("\n" + "="*60)
    print("SUMMARY FOR DOWNSTREAM PAIRWISE COMPARISONS")
    print("="*60)
    print(f"Total replicates: {len(results_df)}")
    print(f"Age groups: {', '.join(age_groups)}")
    print(f"Replicates per age group:")
    for age_group in age_groups:
        count = len(results_df[results_df['age_group'] == age_group])
        print(f"  {age_group}: {count} replicates")
    
    print(f"\nNormality assumptions:")
    normal_count = results_df['is_normal'].sum()
    total_count = results_df['is_normal'].count()
    print(f"  {normal_count}/{total_count} replicates have normal distributions (Shapiro-Wilk p > 0.05)")
    
    print(f"\nEqual variance assumptions:")
    if not np.isnan(levene_p):
        print(f"  Equal variances: {'met' if levene_p > 0.05 else 'violated'} (Levene's test p = {levene_p:.4f})")
    else:
        print("  Could not test equal variances")
    
    print(f"\nRecommended statistical tests:")
    if levene_p > 0.05 and normal_count / total_count > 0.7:
        print("  - Parametric tests (ANOVA, t-tests) are appropriate")
        print("  - Use mean depth values for comparisons")
    else:
        print("  - Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) are recommended")
        print("  - Use median depth values for comparisons")
    
    return results_df

def main():
    """Main function to run the analysis"""
    input_file = 'data/visp_raw_normalized_data.csv'
    output_file = 'outputs/vsv_depth_analysis_results.csv'
    
    try:
        results = analyze_vsv_depth_data(input_file, output_file)
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file: {input_file}")
        print("Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
