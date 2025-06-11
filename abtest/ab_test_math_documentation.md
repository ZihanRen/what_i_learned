# A/B Test Power Analysis: Mathematical Foundation

## Overview
This document explains the mathematical equations behind power analysis and sample size calculations for A/B tests with proportions (conversion rates).

## Key Concepts

### Statistical Hypotheses
- **Null Hypothesis (H₀)**: p₁ = p₂ (no difference between groups)
- **Alternative Hypothesis (H₁)**: p₁ ≠ p₂ (there is a difference between groups)

Where:
- p₁ = true proportion in control group
- p₂ = true proportion in treatment group

### Parameters
- **α (alpha)**: Type I error rate (probability of false positive)
- **β (beta)**: Type II error rate (probability of false negative)  
- **Power**: 1 - β (probability of detecting a true effect)
- **n**: Sample size per group (assuming equal group sizes)

## Standard Error Calculations

### Under Null Hypothesis (H₀: p₁ = p₂)
The pooled proportion estimate:
```
p̂_pooled = (p₁ + p₂) / 2
```

Standard error under null hypothesis:
```
SE_null = √[p̂_pooled × (1 - p̂_pooled) × (2/n)]
```

### Under Alternative Hypothesis (H₁: p₁ ≠ p₂)
Standard error under alternative hypothesis:
```
SE_alt = √[p₁×(1-p₁)/n + p₂×(1-p₂)/n]
```

## Power Calculation

### Critical Value
For a two-sided test with significance level α:
```
z_α/2 = Φ⁻¹(1 - α/2)
```
Where Φ⁻¹ is the inverse standard normal CDF.

### Power Formula
The power of the test is calculated as:
```
z_β = (|p₁ - p₂| - z_α/2 × SE_null) / SE_alt
Power = Φ(z_β)
```

## Sample Size Calculation

### Analytical Formula
To find the required sample size for a desired power:

```
z_β = Φ⁻¹(Power)
```

Sample size per group:
```
n = [z_α/2 × √(2 × p̂_pooled × (1 - p̂_pooled)) + z_β × √(p₁×(1-p₁) + p₂×(1-p₂))]² / (p₁ - p₂)²
```

## Issues in Original Implementation

### Problems Identified
1. **Incorrect Standard Error**: The original code used `√(v₁ + v₂)` where v₁ = p₁(1-p₁) and v₂ = p₂(1-p₂), but this doesn't account for sample size.

2. **Missing Sample Size Factor**: Standard errors should include the sample size factor (1/n or 2/n).

3. **Unused Pooled Variance**: The code calculated pooled variance but didn't use it properly.

4. **Non-standard Power Formula**: The two-part power calculation didn't match standard statistical formulas.

### Original Code (INCORRECT)
```python
def get_power(n, p1, p2, cl):
    alpha = 1 - cl
    qu = stats.norm.ppf(1 - alpha/2)
    diff = abs(p2-p1)
    bp = (p1+p2) / 2
    
    v1 = p1 * (1-p1)
    v2 = p2 * (1-p2)
    
    power_part_one = stats.norm.cdf((n**0.5 * diff - qu * (v1+v2)**0.5) / (v1+v2)**0.5)
    power_part_two = 1 - stats.norm.cdf((n**0.5 * diff + qu * (v1+v2)**0.5) / (v1+v2)**0.5)
    
    power = power_part_one + power_part_two
    return power
```

### Corrected Implementation
```python
def get_power(n, p1, p2, cl):
    alpha = 1 - cl
    z_alpha_2 = stats.norm.ppf(1 - alpha/2)
    
    # Pooled proportion under null hypothesis
    p_pooled = (p1 + p2) / 2
    
    # Standard errors
    se_null = np.sqrt(p_pooled * (1 - p_pooled) * (2/n))
    se_alt = np.sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
    
    # Effect size
    effect = abs(p2 - p1)
    
    # Power calculation
    z_beta = (effect - z_alpha_2 * se_null) / se_alt
    power = stats.norm.cdf(z_beta)
    
    return max(0, min(1, power))
```

## Example Calculation

Given:
- p₁ = 0.10 (10% baseline conversion rate)
- p₂ = 0.11 (11% treatment conversion rate, 10% relative increase)
- α = 0.05 (5% significance level)
- Desired power = 0.80 (80%)

### Step-by-step:
1. **Effect size**: |p₂ - p₁| = |0.11 - 0.10| = 0.01
2. **Pooled proportion**: p̂_pooled = (0.10 + 0.11)/2 = 0.105
3. **Critical value**: z₀.₀₂₅ = 1.96
4. **Required z_β for 80% power**: z₀.₂₀ = 0.84

### Sample size calculation:
```
numerator = [1.96 × √(2 × 0.105 × 0.895) + 0.84 × √(0.10×0.90 + 0.11×0.89)]²
numerator = [1.96 × 0.433 + 0.84 × 0.432]²
numerator = [0.849 + 0.363]²
numerator = 1.469²
numerator = 2.158

n = 2.158 / (0.01)² = 2.158 / 0.0001 = 21,580 per group
```

## Validation Methods

### Comparison with R
The corrected implementation should match R's `power.prop.test()`:
```r
power.prop.test(p1=0.10, p2=0.11, power=0.80, sig.level=0.05)
```

### Online Calculators
Results should also match online power calculators from:
- Evan's Awesome A/B Tools
- Optimizely's Sample Size Calculator
- AB Tasty's Statistical Calculator

## Practical Considerations

### Minimum Detectable Effect (MDE)
The smallest effect size that can be reliably detected:
```
MDE = (z_α/2 + z_β) × SE_alt / √n
```

### Relative vs Absolute Effects
- **Absolute effect**: p₂ - p₁ = 0.01 (1 percentage point)
- **Relative effect**: (p₂ - p₁)/p₁ = 0.01/0.10 = 10% relative increase

### Sample Size Factors
Sample size increases with:
- Smaller effect sizes (harder to detect)
- Higher desired power
- Lower baseline conversion rates
- Higher confidence levels (lower α)

## Common Pitfalls

1. **Confusing relative vs absolute effects**
   - 10% relative increase from 10% baseline = 11% final rate
   - 10% absolute increase from 10% baseline = 20% final rate

2. **Not accounting for multiple testing**
   - Testing multiple metrics requires Bonferroni correction or FDR control

3. **Assuming normal distribution for small samples**
   - Use exact tests for small samples or low conversion rates

4. **Ignoring practical significance**
   - Statistical significance ≠ business significance

## References
- Lachin, J.M. (1981). Introduction to sample size determination and power analysis
- Fleiss, J.L., Levin, B., & Paik, M.C. (2003). Statistical methods for rates and proportions
- Cohen, J. (1988). Statistical power analysis for the behavioral sciences
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data 