"""
Statistical analysis module for inter-annotator agreement and quality patterns
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import logging

from .config import PROCESSED_DATA_DIR, AGREEMENT_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgreementAnalyzer:
    """Analyze inter-annotator agreement"""
    
    def __init__(self):
        pass
    
    def compute_agreement_percentage(self, 
                                    annotator1_scores: List[int], 
                                    annotator2_scores: List[int]) -> float:
        """
        Compute simple agreement percentage between two annotators
        
        Args:
            annotator1_scores: Scores from first annotator
            annotator2_scores: Scores from second annotator
            
        Returns:
            Agreement percentage (0-100)
        """
        if len(annotator1_scores) != len(annotator2_scores):
            raise ValueError("Score lists must have same length")
        
        agreements = sum(1 for a, b in zip(annotator1_scores, annotator2_scores) if a == b)
        percentage = (agreements / len(annotator1_scores)) * 100
        
        return percentage
    
    def compute_cohen_kappa(self, 
                           annotator1_scores: List[int], 
                           annotator2_scores: List[int]) -> float:
        """
        Compute Cohen's Kappa for inter-annotator agreement
        
        Args:
            annotator1_scores: Scores from first annotator
            annotator2_scores: Scores from second annotator
            
        Returns:
            Cohen's Kappa score (-1 to 1)
        """
        if len(annotator1_scores) != len(annotator2_scores):
            raise ValueError("Score lists must have same length")
        
        kappa = cohen_kappa_score(annotator1_scores, annotator2_scores)
        return kappa
    
    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Cohen's Kappa value
        
        Args:
            kappa: Kappa score
            
        Returns:
            Interpretation string
        """
        if kappa < 0:
            return "Poor (Less than chance agreement)"
        elif kappa < 0.20:
            return "Slight"
        elif kappa < 0.40:
            return "Fair"
        elif kappa < 0.60:
            return "Moderate"
        elif kappa < 0.80:
            return "Substantial"
        else:
            return "Almost Perfect"
    
    def analyze_multi_annotator(self, df: pd.DataFrame, dimension: str = "accuracy_score") -> Dict:
        """
        Analyze agreement between multiple annotators
        
        Args:
            df: DataFrame with annotations from multiple annotators
            dimension: Score dimension to analyze
            
        Returns:
            Dictionary with agreement metrics
        """
        # Pivot to get annotator scores side by side
        pivot = df.pivot_table(
            index=['question', 'model_response'],
            columns='annotator_id',
            values=dimension
        )
        
        annotators = pivot.columns.tolist()
        
        if len(annotators) < 2:
            logger.warning("Need at least 2 annotators for agreement analysis")
            return {}
        
        results = {
            'dimension': dimension,
            'num_annotators': len(annotators),
            'pairwise_agreements': {}
        }
        
        # Compute pairwise agreements
        for i, ann1 in enumerate(annotators):
            for ann2 in annotators[i+1:]:
                # Remove NaN values
                valid_pairs = pivot[[ann1, ann2]].dropna()
                
                if len(valid_pairs) == 0:
                    continue
                
                scores1 = valid_pairs[ann1].astype(int).tolist()
                scores2 = valid_pairs[ann2].astype(int).tolist()
                
                agreement_pct = self.compute_agreement_percentage(scores1, scores2)
                kappa = self.compute_cohen_kappa(scores1, scores2)
                
                pair_key = f"{ann1}_vs_{ann2}"
                results['pairwise_agreements'][pair_key] = {
                    'agreement_percentage': agreement_pct,
                    'cohen_kappa': kappa,
                    'kappa_interpretation': self.interpret_kappa(kappa),
                    'num_samples': len(scores1)
                }
        
        return results
    
    def print_agreement_report(self, results: Dict):
        """Print formatted agreement analysis report"""
        print("=" * 80)
        print(f"INTER-ANNOTATOR AGREEMENT ANALYSIS: {results['dimension']}")
        print("=" * 80)
        print(f"\nNumber of annotators: {results['num_annotators']}")
        print(f"\nPairwise Agreements:")
        print("-" * 80)
        
        for pair, metrics in results['pairwise_agreements'].items():
            print(f"\n{pair}:")
            print(f"  Samples: {metrics['num_samples']}")
            print(f"  Agreement %: {metrics['agreement_percentage']:.2f}%")
            print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
            print(f"  Interpretation: {metrics['kappa_interpretation']}")
            
            if metrics['cohen_kappa'] < AGREEMENT_THRESHOLD:
                print(f"  ⚠️  WARNING: Kappa below threshold ({AGREEMENT_THRESHOLD})")
        
        print("=" * 80)


class QualityAnalyzer:
    """Analyze quality patterns in annotations"""
    
    def __init__(self, output_dir: Path = PROCESSED_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute summary statistics for each dimension
        
        Args:
            df: DataFrame with annotations
            
        Returns:
            Dictionary with summary statistics
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        stats = {}
        for dim in dimensions:
            if dim in df.columns:
                stats[dim] = {
                    'mean': df[dim].mean(),
                    'median': df[dim].median(),
                    'std': df[dim].std(),
                    'min': df[dim].min(),
                    'max': df[dim].max(),
                    'q25': df[dim].quantile(0.25),
                    'q75': df[dim].quantile(0.75)
                }
        
        return stats
    
    def identify_low_quality_responses(self, 
                                      df: pd.DataFrame, 
                                      threshold: float = 2.5) -> pd.DataFrame:
        """
        Identify responses with low quality scores
        
        Args:
            df: DataFrame with annotations
            threshold: Score threshold for low quality (default: 2.5)
            
        Returns:
            DataFrame with low quality responses
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        # Find responses where any dimension is below threshold
        low_quality = df[
            (df['accuracy_score'] < threshold) |
            (df['helpfulness_score'] < threshold) |
            (df['tone_score'] < threshold)
        ].copy()
        
        # Add average score
        low_quality['avg_score'] = low_quality[dimensions].mean(axis=1)
        
        # Sort by average score
        low_quality = low_quality.sort_values('avg_score')
        
        logger.info(f"Found {len(low_quality)} low quality responses (threshold: {threshold})")
        return low_quality
    
    def analyze_by_question_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze scores by question type/category
        
        Args:
            df: DataFrame with annotations
            
        Returns:
            DataFrame with grouped statistics
        """
        # Simple question type classification based on starting words
        def classify_question(question: str) -> str:
            question_lower = question.lower().strip()
            if question_lower.startswith(('what', 'which')):
                return 'Factual'
            elif question_lower.startswith(('how', 'can you')):
                return 'Procedural'
            elif question_lower.startswith(('why', 'explain')):
                return 'Explanatory'
            elif question_lower.startswith(('should', 'would', 'could')):
                return 'Opinion/Advice'
            else:
                return 'Other'
        
        df['question_type'] = df['question'].apply(classify_question)
        
        # Group by question type
        grouped = df.groupby('question_type').agg({
            'accuracy_score': ['mean', 'std', 'count'],
            'helpfulness_score': ['mean', 'std'],
            'tone_score': ['mean', 'std']
        }).round(2)
        
        return grouped
    
    def find_score_correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find correlations between different score dimensions
        
        Args:
            df: DataFrame with annotations
            
        Returns:
            Correlation matrix
        """
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        correlation_matrix = df[dimensions].corr()
        
        return correlation_matrix
    
    def save_analysis(self, data: pd.DataFrame, filename: str):
        """Save analysis results to CSV"""
        output_path = self.output_dir / filename
        data.to_csv(output_path)
        logger.info(f"Analysis saved to {output_path}")
        return output_path


def main():
    """Example usage of analysis module"""
    # Create sample annotated data
    np.random.seed(42)
    
    sample_data = {
        'question': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'] * 2,
        'model_response': ['R1', 'R2', 'R3', 'R4', 'R5'] * 2,
        'annotator_id': ['A1']*5 + ['A2']*5,
        'accuracy_score': [5, 4, 3, 4, 5, 5, 4, 2, 4, 5],
        'helpfulness_score': [5, 4, 3, 5, 4, 4, 4, 3, 5, 4],
        'tone_score': [5, 5, 4, 5, 5, 5, 5, 4, 5, 5]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Agreement analysis
    print("\n" + "="*80)
    print("AGREEMENT ANALYSIS")
    print("="*80)
    
    agreement_analyzer = AgreementAnalyzer()
    results = agreement_analyzer.analyze_multi_annotator(df, 'accuracy_score')
    agreement_analyzer.print_agreement_report(results)
    
    # Quality analysis
    print("\n" + "="*80)
    print("QUALITY ANALYSIS")
    print("="*80)
    
    quality_analyzer = QualityAnalyzer()
    
    # Summary statistics
    stats = quality_analyzer.compute_summary_statistics(df)
    print("\nSummary Statistics:")
    for dim, values in stats.items():
        print(f"\n{dim}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.2f}")
    
    # Correlations
    print("\nScore Correlations:")
    print(quality_analyzer.find_score_correlations(df))


if __name__ == "__main__":
    main()
