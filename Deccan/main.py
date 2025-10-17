"""
Main entry point for running the evaluation pipeline
"""
import argparse
from pathlib import Path
import logging

from src.data_collection import DatasetCollector
from src.annotation import Annotator, AnnotationManager
from src.analysis import AgreementAnalyzer, QualityAnalyzer
from src.visualization import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='AI Response Evaluation System')
    parser.add_argument('--collect', action='store_true', help='Collect dataset')
    parser.add_argument('--annotate', action='store_true', help='Annotate dataset')
    parser.add_argument('--analyze', action='store_true', help='Analyze annotations')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--dataset', type=str, help='Path to dataset file')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples to collect')
    
    args = parser.parse_args()
    
    if args.collect:
        logger.info("Starting data collection...")
        collector = DatasetCollector()
        df = collector.collect_from_alpaca(num_samples=args.samples)
        collector.save_dataset(df, "qa_dataset.csv")
        logger.info(f"Collected {len(df)} samples")
    
    if args.annotate:
        if not args.dataset:
            logger.error("Please provide --dataset path")
            return
        
        logger.info(f"Starting annotation of {args.dataset}...")
        import pandas as pd
        df = pd.read_csv(args.dataset)
        
        annotator = Annotator("annotator_1")
        annotated_df = annotator.annotate_batch(df)
        output_path = annotator.save_annotations(annotated_df)
        logger.info(f"Annotations saved to {output_path}")
    
    if args.analyze:
        if not args.dataset:
            logger.error("Please provide --dataset path")
            return
        
        logger.info(f"Starting analysis of {args.dataset}...")
        import pandas as pd
        df = pd.read_csv(args.dataset)
        
        # Quality analysis
        quality = QualityAnalyzer()
        stats = quality.compute_summary_statistics(df)
        logger.info("Summary Statistics:")
        for dim, values in stats.items():
            logger.info(f"{dim}: mean={values['mean']:.2f}, std={values['std']:.2f}")
        
        # Agreement analysis (if multiple annotators)
        if 'annotator_id' in df.columns and df['annotator_id'].nunique() >= 2:
            agreement = AgreementAnalyzer()
            results = agreement.analyze_multi_annotator(df, 'accuracy_score')
            agreement.print_agreement_report(results)
    
    if args.visualize:
        if not args.dataset:
            logger.error("Please provide --dataset path")
            return
        
        logger.info(f"Creating visualizations for {args.dataset}...")
        import pandas as pd
        df = pd.read_csv(args.dataset)
        
        viz = Visualizer()
        viz.plot_all_dimensions(df, "all_dimensions.png")
        viz.plot_score_comparison(df, "score_comparison.png")
        viz.create_summary_dashboard(df, "summary_dashboard.png")
        logger.info(f"Visualizations saved to outputs/plots/")


if __name__ == "__main__":
    main()
