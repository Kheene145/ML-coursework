"""
Loan Approval Dataset - Step 4: Distribution & Bias Analysis
=============================================================

This script performs comprehensive analysis of:
1. Probabilistic distributions for numerical features
2. Class balance analysis
3. Bias detection in categorical features
4. Feature relationships with target variable
5. Generates detailed visualizations and report

Author: Data Analysis Lecture Series
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import normaltest, shapiro, skew, kurtosis
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class DistributionBiasAnalyzer:
    """Main class for distribution and bias analysis"""
    
    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Load the dataset"""
        print("\n" + "=" * 70)
        print("LOADING DATASET")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        
        self.results['dataset_info'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': list(self.df.columns)
        }
    
    def analyze_target_balance(self):
        """Analyze class balance in target variable"""
        print("\n" + "=" * 70)
        print("CLASS BALANCE ANALYSIS")
        print("=" * 70)
        
        # Identify target column (loan_approval_status)
        target_col = 'loan_approval_status'
        
        if target_col not in self.df.columns:
            print(f"✗ Target column '{target_col}' not found!")
            return
        
        # Calculate class distribution
        class_counts = self.df[target_col].value_counts()
        class_percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        print(f"\nTarget Variable: {target_col}")
        print("\nClass Distribution:")
        for cls, count in class_counts.items():
            pct = class_percentages[cls]
            print(f"  {cls}: {count} ({pct:.2f}%)")
        
        # Calculate imbalance ratio
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print("  ⚠ Dataset shows class imbalance")
        else:
            print("  ✓ Dataset is reasonably balanced")
        
        # Store results
        self.results['class_balance'] = {
            'target_column': target_col,
            'class_counts': class_counts.to_dict(),
            'class_percentages': class_percentages.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
        
        # Visualize class balance
        self._plot_class_balance(class_counts, target_col)
    
    def _plot_class_balance(self, class_counts, target_col):
        """Create visualization for class balance"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        colors = ['#2ecc71' if 'Approve' in str(x) else '#e74c3c' 
                  for x in class_counts.index]
        axes[0].bar(class_counts.index, class_counts.values, color=colors, 
                    edgecolor='black', linewidth=1.5)
        axes[0].set_title('Class Distribution - Bar Chart', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Loan Status', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(class_counts.items()):
            axes[0].text(i, val, f'{val}\n({val/class_counts.sum()*100:.1f}%)', 
                        ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title('Class Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_class_balance.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: 01_class_balance.png")
        plt.close()
    
    def analyze_numerical_distributions(self):
        """Analyze probability distributions of numerical features"""
        print("\n" + "=" * 70)
        print("NUMERICAL FEATURE DISTRIBUTIONS")
        print("=" * 70)
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns
        numerical_cols = [col for col in numerical_cols if 'id' not in col.lower()]
        
        print(f"\nAnalyzing {len(numerical_cols)} numerical features:")
        print(", ".join(numerical_cols))
        
        self.results['numerical_distributions'] = {}
        
        for col in numerical_cols:
            print(f"\n{'-'*60}")
            print(f"Feature: {col}")
            print(f"{'-'*60}")
            
            data = self.df[col].dropna()
            
            # Calculate statistics
            dist_stats = {
                'mean': data.mean(),
                'median': data.median(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'skewness': skew(data),
                'kurtosis': kurtosis(data),
                'missing_pct': (self.df[col].isnull().sum() / len(self.df)) * 100
            }
            
            print(f"  Mean: {dist_stats['mean']:.2f}")
            print(f"  Median: {dist_stats['median']:.2f}")
            print(f"  Std Dev: {dist_stats['std']:.2f}")
            print(f"  Skewness: {dist_stats['skewness']:.3f}", end='')
            
            if abs(dist_stats['skewness']) < 0.5:
                print(" (Symmetric)")
            elif dist_stats['skewness'] > 0:
                print(" (Right-skewed)")
            else:
                print(" (Left-skewed)")
            
            # Normality test
            if len(data) >= 8:
                _, p_value = normaltest(data)
                dist_stats['normality_p_value'] = p_value
                dist_stats['is_normal'] = p_value > 0.05
                
                print(f"  Normality Test p-value: {p_value:.4f}", end='')
                if p_value > 0.05:
                    print(" ✓ Appears Normal")
                else:
                    print(" ✗ Not Normal")
            
            # Identify distribution type
            dist_type = self._identify_distribution_type(dist_stats)
            dist_stats['distribution_type'] = dist_type
            print(f"  Likely Distribution: {dist_type}")
            
            self.results['numerical_distributions'][col] = dist_stats
        
        # Create visualizations
        self._plot_numerical_distributions(numerical_cols)
    
    def _identify_distribution_type(self, stats):
        """Identify the likely probability distribution"""
        skewness = stats['skewness']
        kurtosis_val = stats['kurtosis']
        is_normal = stats.get('is_normal', False)
        
        if is_normal and abs(skewness) < 0.5 and abs(kurtosis_val) < 3:
            return "Normal (Gaussian)"
        elif abs(skewness) < 0.5:
            return "Approximately Normal"
        elif skewness > 1:
            return "Right-skewed (Log-normal or Exponential)"
        elif skewness < -1:
            return "Left-skewed"
        else:
            return "Moderately Skewed"
    
    def _plot_numerical_distributions(self, numerical_cols):
        """Create comprehensive distribution visualizations"""
        print(f"\n  Creating distribution plots...")
        
        for col in numerical_cols:
            data = self.df[col].dropna()
            
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(3, 3, figure=fig)
            
            # Title
            fig.suptitle(f'Distribution Analysis: {col}', fontsize=16, fontweight='bold')
            
            # 1. Histogram with KDE
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            data.plot(kind='kde', ax=ax1, color='red', linewidth=2)
            ax1.set_title('Histogram with KDE', fontweight='bold')
            ax1.set_xlabel(col)
            ax1.set_ylabel('Density')
            ax1.grid(alpha=0.3)
            
            # 2. Box Plot
            ax2 = fig.add_subplot(gs[0, 2])
            ax2.boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
            ax2.set_title('Box Plot', fontweight='bold')
            ax2.set_ylabel(col)
            ax2.grid(alpha=0.3)
            
            # 3. Q-Q Plot
            ax3 = fig.add_subplot(gs[1, 0])
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # 4. Violin Plot
            ax4 = fig.add_subplot(gs[1, 1])
            parts = ax4.violinplot([data], positions=[1], showmeans=True, showmedians=True)
            ax4.set_title('Violin Plot', fontweight='bold')
            ax4.set_ylabel(col)
            ax4.grid(alpha=0.3)
            
            # 5. ECDF (Empirical Cumulative Distribution)
            ax5 = fig.add_subplot(gs[1, 2])
            sorted_data = np.sort(data)
            ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax5.plot(sorted_data, ecdf, linewidth=2, color='purple')
            ax5.set_title('ECDF', fontweight='bold')
            ax5.set_xlabel(col)
            ax5.set_ylabel('Cumulative Probability')
            ax5.grid(alpha=0.3)
            
            # 6. Statistics Table
            ax6 = fig.add_subplot(gs[2, :])
            ax6.axis('off')
            
            stats_data = [
                ['Statistic', 'Value'],
                ['Mean', f'{data.mean():.2f}'],
                ['Median', f'{data.median():.2f}'],
                ['Std Dev', f'{data.std():.2f}'],
                ['Min', f'{data.min():.2f}'],
                ['Max', f'{data.max():.2f}'],
                ['Skewness', f'{skew(data):.3f}'],
                ['Kurtosis', f'{kurtosis(data):.3f}']
            ]
            
            table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                            colWidths=[0.3, 0.3])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style header row
            for i in range(2):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.tight_layout()
            safe_col_name = col.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{self.output_dir}/dist_{safe_col_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Created {len(numerical_cols)} distribution plots")
    
    def analyze_categorical_bias(self):
        """Analyze bias in categorical features"""
        print("\n" + "=" * 70)
        print("BIAS ANALYSIS IN CATEGORICAL FEATURES")
        print("=" * 70)
        
        # Select categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Exclude target from bias analysis
        target_col = 'loan_approval_status'
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        print(f"\nAnalyzing {len(categorical_cols)} categorical features for bias:")
        
        self.results['bias_analysis'] = {}
        
        for col in categorical_cols:
            print(f"\n{'-'*60}")
            print(f"Feature: {col}")
            print(f"{'-'*60}")
            
            # Overall distribution
            value_counts = self.df[col].value_counts()
            print(f"  Unique values: {len(value_counts)}")
            print(f"  Top 3 values:")
            for val, count in value_counts.head(3).items():
                pct = (count / len(self.df)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")
            
            # Bias analysis with target variable
            if target_col in self.df.columns:
                crosstab = pd.crosstab(self.df[col], self.df[target_col], normalize='index') * 100
                
                print(f"\n  Approval Rate by {col}:")
                approval_col = 'Approved' if 'Approved' in crosstab.columns else crosstab.columns[0]
                
                bias_detected = False
                approval_rates = {}
                
                for category in crosstab.index:
                    rate = crosstab.loc[category, approval_col]
                    approval_rates[str(category)] = rate
                    print(f"    {category}: {rate:.1f}%")
                
                # Check for significant bias (>15% difference)
                max_rate = max(approval_rates.values())
                min_rate = min(approval_rates.values())
                bias_diff = max_rate - min_rate
                
                if bias_diff > 15:
                    print(f"\n  ⚠ BIAS DETECTED: {bias_diff:.1f}% difference in approval rates")
                    bias_detected = True
                else:
                    print(f"\n  ✓ No significant bias detected")
                
                self.results['bias_analysis'][col] = {
                    'value_counts': value_counts.to_dict(),
                    'approval_rates': approval_rates,
                    'bias_difference': bias_diff,
                    'bias_detected': bias_detected
                }
        
        # Create visualizations
        self._plot_categorical_bias(categorical_cols)
    
    def _plot_categorical_bias(self, categorical_cols):
        """Visualize bias in categorical features"""
        print(f"\n  Creating bias visualization plots...")
        
        target_col = 'loan_approval_status'
        
        for col in categorical_cols:
            # Limit to top 10 categories for readability
            top_categories = self.df[col].value_counts().head(10).index
            filtered_df = self.df[self.df[col].isin(top_categories)]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Bias Analysis: {col}', fontsize=16, fontweight='bold')
            
            # 1. Overall distribution
            value_counts = filtered_df[col].value_counts()
            axes[0, 0].barh(value_counts.index, value_counts.values, color='steelblue', edgecolor='black')
            axes[0, 0].set_title('Overall Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Count')
            axes[0, 0].grid(axis='x', alpha=0.3)
            
            # 2. Approval rate by category
            crosstab = pd.crosstab(filtered_df[col], filtered_df[target_col], normalize='index') * 100
            approval_col = 'Approved' if 'Approved' in crosstab.columns else crosstab.columns[0]
            approval_rates = crosstab[approval_col].sort_values()
            
            colors_approval = ['#e74c3c' if x < 50 else '#2ecc71' for x in approval_rates.values]
            axes[0, 1].barh(approval_rates.index, approval_rates.values, color=colors_approval, edgecolor='black')
            axes[0, 1].axvline(x=50, color='red', linestyle='--', linewidth=2, label='50% baseline')
            axes[0, 1].set_title('Approval Rate by Category', fontweight='bold')
            axes[0, 1].set_xlabel('Approval Rate (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='x', alpha=0.3)
            
            # 3. Stacked bar chart
            crosstab_counts = pd.crosstab(filtered_df[col], filtered_df[target_col])
            crosstab_counts.plot(kind='barh', stacked=True, ax=axes[1, 0], 
                                color=['#e74c3c', '#2ecc71'], edgecolor='black')
            axes[1, 0].set_title('Approval vs Rejection Counts', fontweight='bold')
            axes[1, 0].set_xlabel('Count')
            axes[1, 0].legend(title='Status')
            axes[1, 0].grid(axis='x', alpha=0.3)
            
            # 4. Percentage stacked bar
            crosstab_pct = pd.crosstab(filtered_df[col], filtered_df[target_col], normalize='index') * 100
            crosstab_pct.plot(kind='barh', stacked=True, ax=axes[1, 1],
                             color=['#e74c3c', '#2ecc71'], edgecolor='black')
            axes[1, 1].set_title('Approval vs Rejection (%)', fontweight='bold')
            axes[1, 1].set_xlabel('Percentage (%)')
            axes[1, 1].legend(title='Status')
            axes[1, 1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            safe_col_name = col.replace('/', '_').replace(' ', '_')
            plt.savefig(f'{self.output_dir}/bias_{safe_col_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Created {len(categorical_cols)} bias analysis plots")
    
    def generate_report(self):
        """Generate comprehensive Markdown report"""
        print("\n" + "=" * 70)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 70)
        
        report_path = f'{self.output_dir}/DISTRIBUTION_BIAS_ANALYSIS_REPORT.md'
        
        with open(report_path, 'w') as f:
            f.write(self._create_report_content())
        
        print(f"✓ Report saved: {report_path}")
        return report_path
    
    def _create_report_content(self):
        """Create the Markdown report content"""
        # This will be expanded in the separate detailed report file
        content = f"""# Distribution and Bias Analysis Report

## Dataset Overview
- **Total Rows:** {self.results['dataset_info']['rows']:,}
- **Total Columns:** {self.results['dataset_info']['columns']}
- **Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Class Balance Analysis

### Target Variable: {self.results['class_balance']['target_column']}

"""
        
        for cls, count in self.results['class_balance']['class_counts'].items():
            pct = self.results['class_balance']['class_percentages'][cls]
            content += f"- **{cls}:** {count:,} ({pct:.2f}%)\n"
        
        content += f"\n**Imbalance Ratio:** {self.results['class_balance']['imbalance_ratio']:.2f}:1\n\n"
        
        if self.results['class_balance']['imbalance_ratio'] > 1.5:
            content += "⚠️ **Warning:** Dataset shows class imbalance. Consider using techniques like:\n"
            content += "- Oversampling minority class (SMOTE)\n"
            content += "- Undersampling majority class\n"
            content += "- Class weights in models\n\n"
        
        content += "![Class Balance](01_class_balance.png)\n\n---\n\n"
        
        # Numerical distributions
        content += "## Numerical Feature Distributions\n\n"
        
        for col, stats in self.results['numerical_distributions'].items():
            content += f"### {col}\n\n"
            content += f"- **Distribution Type:** {stats['distribution_type']}\n"
            content += f"- **Mean:** {stats['mean']:.2f}\n"
            content += f"- **Median:** {stats['median']:.2f}\n"
            content += f"- **Skewness:** {stats['skewness']:.3f}\n"
            if 'is_normal' in stats:
                content += f"- **Normal Distribution:** {'Yes ✓' if stats['is_normal'] else 'No ✗'}\n"
            content += f"\n![Distribution](dist_{col.replace('/', '_').replace(' ', '_')}.png)\n\n"
        
        content += "---\n\n## Bias Analysis\n\n"
        
        for col, analysis in self.results['bias_analysis'].items():
            content += f"### {col}\n\n"
            if analysis['bias_detected']:
                content += f"⚠️ **Bias Detected:** {analysis['bias_difference']:.1f}% difference in approval rates\n\n"
            else:
                content += f"✓ **No Significant Bias Detected**\n\n"
            
            content += "**Approval Rates:**\n"
            for category, rate in analysis['approval_rates'].items():
                content += f"- {category}: {rate:.1f}%\n"
            
            content += f"\n![Bias Analysis](bias_{col.replace('/', '_').replace(' ', '_')}.png)\n\n"
        
        return content
    
    def run_analysis(self):
        """Execute complete analysis pipeline"""
        print("\n" + "=" * 70)
        print("STARTING DISTRIBUTION & BIAS ANALYSIS")
        print("=" * 70)
        
        self.load_data()
        self.analyze_target_balance()
        self.analyze_numerical_distributions()
        self.analyze_categorical_bias()
        report_path = self.generate_report()
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\n✓ All visualizations saved in: {self.output_dir}/")
        print(f"✓ Full report available at: {report_path}")
        print("\nNext steps:")
        print("1. Review the generated report (DISTRIBUTION_BIAS_ANALYSIS_REPORT.md)")
        print("2. Examine all visualization files")
        print("3. Discuss findings with the class")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Configuration
    data_file = "../data/loan_approval.csv"
    output_dir = "../outputs/distribution_bias_analysis"
    
    # Run analysis
    analyzer = DistributionBiasAnalyzer(data_file, output_dir)
    analyzer.run_analysis()
