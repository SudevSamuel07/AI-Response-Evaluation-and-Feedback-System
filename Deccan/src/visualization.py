"""
Visualization module for creating plots and charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import logging

from .config import PLOTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Create visualizations for evaluation results"""
    
    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_score_distribution(self, 
                               df: pd.DataFrame, 
                               dimension: str = "accuracy_score",
                               save_name: Optional[str] = None):
        """
        Plot distribution of scores for a dimension
        
        Args:
            df: DataFrame with annotations
            dimension: Score dimension to plot
            save_name: Filename to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(df[dimension], bins=6, range=(0, 5), edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{dimension.replace("_", " ").title()} Distribution')
        axes[0].set_xticks(range(6))
        
        # Box plot
        axes[1].boxplot(df[dimension], vert=True)
        axes[1].set_ylabel('Score')
        axes[1].set_title(f'{dimension.replace("_", " ").title()} Box Plot')
        axes[1].set_ylim(-0.5, 5.5)
        axes[1].set_xticks([])
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_all_dimensions(self, 
                           df: pd.DataFrame,
                           save_name: Optional[str] = None):
        """
        Plot distributions for all score dimensions
        
        Args:
            df: DataFrame with annotations
            save_name: Filename to save plot (optional)
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, dim in enumerate(dimensions):
            if dim in df.columns:
                axes[i].hist(df[dim], bins=6, range=(0, 5), 
                           edgecolor='black', alpha=0.7, color=f'C{i}')
                axes[i].set_xlabel('Score')
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(dim.replace('_', ' ').title())
                axes[i].set_xticks(range(6))
                
                # Add mean line
                mean_val = df[dim].mean()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_score_comparison(self,
                             df: pd.DataFrame,
                             save_name: Optional[str] = None):
        """
        Compare scores across dimensions using bar chart
        
        Args:
            df: DataFrame with annotations
            save_name: Filename to save plot (optional)
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        means = [df[dim].mean() for dim in dimensions if dim in df.columns]
        stds = [df[dim].std() for dim in dimensions if dim in df.columns]
        labels = [dim.replace('_score', '').title() for dim in dimensions if dim in df.columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7, 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores Across Dimensions')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 5.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Threshold (3.0)')
        ax.legend()
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self,
                                df: pd.DataFrame,
                                save_name: Optional[str] = None):
        """
        Plot correlation heatmap between dimensions
        
        Args:
            df: DataFrame with annotations
            save_name: Filename to save plot (optional)
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        corr_matrix = df[dimensions].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        ax.set_title('Correlation Between Score Dimensions')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_question_type_analysis(self,
                                   df: pd.DataFrame,
                                   save_name: Optional[str] = None):
        """
        Plot score analysis by question type
        
        Args:
            df: DataFrame with annotations and question_type column
            save_name: Filename to save plot (optional)
        """
        if 'question_type' not in df.columns:
            logger.warning("question_type column not found. Skipping plot.")
            return
        
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        # Prepare data
        grouped = df.groupby('question_type')[dimensions].mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        grouped.plot(kind='bar', ax=ax, alpha=0.8)
        
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Question Type')
        ax.legend(title='Dimension', labels=['Accuracy', 'Helpfulness', 'Tone'])
        ax.set_ylim(0, 5.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_scatter_matrix(self,
                           df: pd.DataFrame,
                           save_name: Optional[str] = None):
        """
        Create scatter plot matrix for all dimensions
        
        Args:
            df: DataFrame with annotations
            save_name: Filename to save plot (optional)
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        fig = plt.figure(figsize=(12, 10))
        
        # Create pairplot
        plot_df = df[dimensions].copy()
        plot_df.columns = ['Accuracy', 'Helpfulness', 'Tone']
        
        sns.pairplot(plot_df, diag_kind='hist', plot_kws={'alpha': 0.6})
        
        plt.suptitle('Scatter Matrix of Score Dimensions', y=1.02)
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_annotator_comparison(self,
                                 df: pd.DataFrame,
                                 dimension: str = 'accuracy_score',
                                 save_name: Optional[str] = None):
        """
        Compare scores between annotators
        
        Args:
            df: DataFrame with annotations from multiple annotators
            dimension: Score dimension to compare
            save_name: Filename to save plot (optional)
        """
        if 'annotator_id' not in df.columns:
            logger.warning("annotator_id column not found. Skipping plot.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot by annotator
        df.boxplot(column=dimension, by='annotator_id', ax=ax)
        
        ax.set_xlabel('Annotator')
        ax.set_ylabel('Score')
        ax.set_title(f'{dimension.replace("_", " ").title()} by Annotator')
        plt.suptitle('')  # Remove default title
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def create_summary_dashboard(self, df: pd.DataFrame, save_name: str = "summary_dashboard.png"):
        """
        Create comprehensive summary dashboard
        
        Args:
            df: DataFrame with annotations
            save_name: Filename to save plot
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Score distributions
        for i, dim in enumerate(dimensions):
            ax = fig.add_subplot(gs[0, i])
            ax.hist(df[dim], bins=6, range=(0, 5), edgecolor='black', alpha=0.7, color=f'C{i}')
            ax.set_title(dim.replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            mean_val = df[dim].mean()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'μ={mean_val:.2f}')
            ax.legend()
        
        # 2. Comparison bar chart
        ax = fig.add_subplot(gs[1, :2])
        means = [df[dim].mean() for dim in dimensions]
        stds = [df[dim].std() for dim in dimensions]
        x = np.arange(len(dimensions))
        ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace('_score', '').title() for d in dimensions])
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores with Standard Deviation')
        ax.set_ylim(0, 5.5)
        
        # 3. Correlation heatmap
        ax = fig.add_subplot(gs[1, 2])
        corr = df[dimensions].corr()
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['Acc', 'Help', 'Tone'], rotation=45)
        ax.set_yticklabels(['Acc', 'Help', 'Tone'])
        ax.set_title('Correlation')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center')
        
        # 4. Summary statistics table
        ax = fig.add_subplot(gs[2, :])
        ax.axis('off')
        
        stats_data = []
        for dim in dimensions:
            stats_data.append([
                dim.replace('_score', '').title(),
                f"{df[dim].mean():.2f}",
                f"{df[dim].median():.2f}",
                f"{df[dim].std():.2f}",
                f"{df[dim].min():.0f}",
                f"{df[dim].max():.0f}"
            ])
        
        table = ax.table(cellText=stats_data,
                        colLabels=['Dimension', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.suptitle('Evaluation Summary Dashboard', fontsize=16, fontweight='bold')
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()


def main():
    """Example usage of visualization module"""
    # Create sample data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'question': ['Q' + str(i) for i in range(100)],
        'model_response': ['R' + str(i) for i in range(100)],
        'accuracy_score': np.random.randint(2, 6, 100),
        'helpfulness_score': np.random.randint(2, 6, 100),
        'tone_score': np.random.randint(3, 6, 100),
        'annotator_id': ['A1'] * 50 + ['A2'] * 50
    })
    
    # Create visualizer
    viz = Visualizer()
    
    # Create various plots
    print("Creating visualizations...")
    
    viz.plot_all_dimensions(sample_data, "all_dimensions.png")
    viz.plot_score_comparison(sample_data, "score_comparison.png")
    viz.plot_correlation_heatmap(sample_data, "correlation_heatmap.png")
    viz.create_summary_dashboard(sample_data, "summary_dashboard.png")
    
    print(f"\nAll visualizations saved to: {viz.output_dir}")


if __name__ == "__main__":
    main()
