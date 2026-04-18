# VSV Depth Analysis

This repository contains scripts for analyzing VSV (Vesicular Stomatitis Virus) depth data from normalized intensity measurements across depth bins (0-100).

## Files

- `vsv_depth_analysis.py` - Main analysis script
- `pairwise_comparison_example.py` - Example script for pairwise comparisons
- `requirements.txt` - Python dependencies
- `data/visp_raw_normalized_data.csv` - Input data file
- `data/visp_statistics_summary.csv` - Original summary statistics
- `outputs/` - Directory for analysis results

## Analysis Overview

The analysis script performs the following:

1. **Data Loading**: Reads normalized intensity data across 101 depth bins (0-100)
2. **Per-Replicate Statistics**: Calculates comprehensive statistics for each replicate
3. **Normality Testing**: Tests distribution normality using Shapiro-Wilk test
4. **Variance Testing**: Tests equal variances across age groups using Levene's test
5. **Depth Calculations**: Computes weighted mean and median depths

## Key Features

### Per-Replicate Statistics
- Mean and median intensity values
- Standard deviation and standard error of the mean
- Variance and range (min/max)
- Weighted mean and median depths
- Normality test results

### Statistical Assumptions Testing
- **Shapiro-Wilk Test**: Tests if intensity distributions are normal
- **Levene's Test**: Tests if variances are equal across age groups
- **Recommendations**: Suggests appropriate statistical tests based on assumptions

### Output Structure
The results are structured for downstream pairwise comparisons with:
- Replicate identification (animal name, age group)
- All necessary statistics for comparisons
- Assumption testing results
- Clear recommendations for statistical tests

## Usage

### Basic Analysis
```bash
python vsv_depth_analysis.py
```

### Pairwise Comparisons
```bash
python pairwise_comparison_example.py
```

## Results Interpretation

### Statistical Assumptions
- **Normal Distributions**: If >70% of replicates are normal, parametric tests are recommended
- **Equal Variances**: If Levene's test p > 0.05, equal variances assumption is met
- **Test Selection**: 
  - Parametric: ANOVA, t-tests (if assumptions met)
  - Non-parametric: Kruskal-Wallis, Mann-Whitney U (if assumptions violated)

### Key Metrics
- **Weighted Mean Depth**: Average depth weighted by intensity
- **Weighted Median Depth**: Median depth weighted by intensity
- **Effect Sizes**: Cohen's d (parametric) or rank-biserial correlation (non-parametric)

## Age Groups
- **adult**: Adult animals (3 replicates)
- **p3**: Postnatal day 3 (5 replicates)  
- **p12**: Postnatal day 12 (4 replicates)
- **p20**: Postnatal day 20 (4 replicates)

## Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0

## Output Files
- `outputs/vsv_depth_analysis_results.csv` - Comprehensive per-replicate statistics
- `outputs/pairwise_comparisons.csv` - Pairwise comparison results

## Example Results

Based on the current analysis:
- **Total replicates**: 16
- **Age groups**: adult, p3, p12, p20
- **Normality**: 0/16 replicates have normal distributions
- **Equal variances**: Met (Levene's test p = 0.406)
- **Recommendation**: Use non-parametric tests (Kruskal-Wallis, Mann-Whitney U)
- **Omnibus test**: Kruskal-Wallis p = 0.216 (not significant)
- **Pairwise comparisons**: No significant differences between age groups

This suggests that VSV depth patterns do not significantly differ across the tested age groups in this dataset.