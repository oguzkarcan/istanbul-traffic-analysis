"""
Statistical Analysis Module for Istanbul Traffic Analysis

This module performs:
- Hypothesis testing
- Statistical significance tests
- Correlation analysis
- Distribution analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, pearsonr, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Perform statistical analysis on traffic and weather data"""
    
    def __init__(self, data_path='data/processed/enriched_data.csv'):
        self.data = pd.read_csv(data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.results = {}
        
    def test_weekday_vs_weekend_density(self, alpha=0.05):
        """
        H1: There is a significant difference in traffic density between weekdays and weekends
        """
        print("Testing H1: Weekday vs Weekend Traffic Density")
        
        weekday_density = self.data[~self.data['is_weekend']]['traffic_density']
        weekend_density = self.data[self.data['is_weekend']]['traffic_density']
        
        # Perform independent t-test
        statistic, p_value = ttest_ind(weekday_density, weekend_density)
        
        result = {
            'hypothesis': 'Weekday vs Weekend Traffic Density',
            'test_type': 'Independent t-test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'weekday_mean': weekday_density.mean(),
            'weekend_mean': weekend_density.mean(),
            'effect_size': abs(weekday_density.mean() - weekend_density.mean()) / np.sqrt(
                ((len(weekday_density)-1)*weekday_density.var() + (len(weekend_density)-1)*weekend_density.var()) / 
                (len(weekday_density) + len(weekend_density) - 2)
            )
        }
        
        self.results['weekday_weekend_density'] = result
        self._print_test_result(result)
        return result
        
    def test_weather_effect_on_speed(self, alpha=0.05):
        """
        H2: Weather conditions have a significant effect on average traffic speed
        """
        print("\nTesting H2: Weather Effect on Average Speed")
        
        weather_groups = []
        for category in self.data['weather_category'].unique():
            if pd.notna(category):
                group_data = self.data[self.data['weather_category'] == category]['average_speed']
                weather_groups.append(group_data)
        
        # Perform one-way ANOVA
        statistic, p_value = f_oneway(*weather_groups)
        
        # Calculate means for each group
        group_means = {}
        for category in self.data['weather_category'].unique():
            if pd.notna(category):
                group_means[category] = self.data[self.data['weather_category'] == category]['average_speed'].mean()
        
        result = {
            'hypothesis': 'Weather Effect on Average Speed',
            'test_type': 'One-way ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'group_means': group_means,
            'categories_tested': list(self.data['weather_category'].unique())
        }
        
        self.results['weather_speed'] = result
        self._print_test_result(result)
        return result
        
    def test_precipitation_density_correlation(self, alpha=0.05):
        """
        H3: There is a significant correlation between precipitation and traffic density
        """
        print("\nTesting H3: Precipitation vs Traffic Density Correlation")
        
        # Remove any NaN values
        clean_data = self.data[['precipitation', 'traffic_density']].dropna()
        
        # Perform Pearson correlation test
        correlation, p_value = pearsonr(clean_data['precipitation'], clean_data['traffic_density'])
        
        result = {
            'hypothesis': 'Precipitation vs Traffic Density Correlation',
            'test_type': 'Pearson correlation',
            'correlation': correlation,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'sample_size': len(clean_data),
            'interpretation': self._interpret_correlation(correlation)
        }
        
        self.results['precipitation_density'] = result
        self._print_test_result(result)
        return result
        
    def test_rush_hour_congestion(self, alpha=0.05):
        """
        H4: Rush hours show significantly different congestion patterns compared to non-rush hours
        """
        print("\nTesting H4: Rush Hour vs Non-Rush Hour Congestion")
        
        rush_density = self.data[self.data['time_period'] == 'Rush']['traffic_density']
        non_rush_density = self.data[self.data['time_period'] != 'Rush']['traffic_density']
        
        # Perform independent t-test
        statistic, p_value = ttest_ind(rush_density, non_rush_density)
        
        result = {
            'hypothesis': 'Rush Hour vs Non-Rush Hour Congestion',
            'test_type': 'Independent t-test',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'rush_mean': rush_density.mean(),
            'non_rush_mean': non_rush_density.mean(),
            'effect_size': abs(rush_density.mean() - non_rush_density.mean()) / np.sqrt(
                ((len(rush_density)-1)*rush_density.var() + (len(non_rush_density)-1)*non_rush_density.var()) / 
                (len(rush_density) + len(non_rush_density) - 2)
            )
        }
        
        self.results['rush_hour_congestion'] = result
        self._print_test_result(result)
        return result
        
    def test_district_variations(self, alpha=0.05):
        """
        H5: Different districts show significantly different traffic patterns
        """
        print("\nTesting H5: District Traffic Variations")
        
        district_groups = []
        districts = self.data['district'].unique()
        
        for district in districts:
            if pd.notna(district):
                group_data = self.data[self.data['district'] == district]['traffic_density']
                district_groups.append(group_data)
        
        # Perform one-way ANOVA
        statistic, p_value = f_oneway(*district_groups)
        
        # Calculate means for each district
        district_means = {}
        for district in districts:
            if pd.notna(district):
                district_means[district] = self.data[self.data['district'] == district]['traffic_density'].mean()
        
        result = {
            'hypothesis': 'District Traffic Variations',
            'test_type': 'One-way ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'significant': p_value < alpha,
            'district_means': district_means,
            'districts_tested': list(districts)
        }
        
        self.results['district_variations'] = result
        self._print_test_result(result)
        return result
        
    def perform_correlation_matrix(self):
        """Perform comprehensive correlation analysis"""
        print("\nPerforming Correlation Matrix Analysis")
        
        numerical_cols = ['traffic_density', 'average_speed', 'vehicle_count', 
                         'temperature', 'humidity', 'precipitation', 'wind_speed', 
                         'hour', 'day_of_week', 'month', 'traffic_efficiency']
        
        # Select only columns that exist in the data
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        correlation_data = self.data[available_cols]
        
        correlation_matrix = correlation_data.corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.3:  # Consider correlations > 0.3 as notable
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'correlation': corr_value,
                            'strength': self._interpret_correlation(corr_value)
                        })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        print("Strong Correlations Found (|r| > 0.3):")
        for corr in strong_correlations[:10]:  # Top 10
            print(f"  {corr['variable1']} <-> {corr['variable2']}: {corr['correlation']:.3f} ({corr['strength']})")
        
        self.results['correlation_matrix'] = {
            'matrix': correlation_matrix,
            'strong_correlations': strong_correlations
        }
        
        return correlation_matrix, strong_correlations
        
    def _interpret_correlation(self, correlation):
        """Interpret correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "Very weak"
        elif abs_corr < 0.3:
            return "Weak"
        elif abs_corr < 0.5:
            return "Moderate"
        elif abs_corr < 0.7:
            return "Strong"
        else:
            return "Very strong"
            
    def _print_test_result(self, result):
        """Print formatted test results"""
        print(f"  Test: {result['test_type']}")
        print(f"  Statistic: {result.get('statistic', result.get('correlation', 'N/A')):.4f}")
        print(f"  P-value: {result['p_value']:.4f}")
        print(f"  Significant at α={result['alpha']}: {'Yes' if result['significant'] else 'No'}")
        
        if 'correlation' in result:
            print(f"  Correlation strength: {result['interpretation']}")
        if 'effect_size' in result:
            print(f"  Effect size (Cohen's d): {result['effect_size']:.4f}")
            
        print()
        
    def run_all_tests(self):
        """Run all hypothesis tests"""
        print("=" * 60)
        print("STATISTICAL ANALYSIS - HYPOTHESIS TESTING")
        print("=" * 60)
        
        # Run all hypothesis tests
        self.test_weekday_vs_weekend_density()
        self.test_weather_effect_on_speed()
        self.test_precipitation_density_correlation()
        self.test_rush_hour_congestion()
        self.test_district_variations()
        self.perform_correlation_matrix()
        
        # Summary
        self.print_summary()
        
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "=" * 60)
        print("SUMMARY OF HYPOTHESIS TESTS")
        print("=" * 60)
        
        significant_tests = []
        non_significant_tests = []
        
        for test_name, result in self.results.items():
            if test_name != 'correlation_matrix':
                if result['significant']:
                    significant_tests.append((test_name, result))
                else:
                    non_significant_tests.append((test_name, result))
        
        print(f"\nSignificant Results ({len(significant_tests)}):")
        for test_name, result in significant_tests:
            print(f"  ✓ {result['hypothesis']} (p = {result['p_value']:.4f})")
            
        print(f"\nNon-Significant Results ({len(non_significant_tests)}):")
        for test_name, result in non_significant_tests:
            print(f"  ✗ {result['hypothesis']} (p = {result['p_value']:.4f})")
            
        print("\nNote: Significance level α = 0.05")

def main():
    """Main statistical analysis pipeline"""
    print("Starting Statistical Analysis...\n")
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Run all tests
    analyzer.run_all_tests()
    
    print("\nStatistical analysis completed!")

if __name__ == "__main__":
    main() 