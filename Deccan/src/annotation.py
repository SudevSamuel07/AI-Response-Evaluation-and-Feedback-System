"""
Annotation module for evaluating LLM responses
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

from .config import ANNOTATED_DATA_DIR, NUM_ANNOTATORS
from .rubrics import EvaluationRubrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Annotator:
    """Handle annotation of LLM responses"""
    
    def __init__(self, annotator_id: str, output_dir: Path = ANNOTATED_DATA_DIR):
        self.annotator_id = annotator_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rubrics = EvaluationRubrics()
    
    def annotate_single(self, 
                       question: str, 
                       response: str,
                       accuracy: int,
                       helpfulness: int,
                       tone: int,
                       notes: str = "") -> Dict:
        """
        Annotate a single QA pair
        
        Args:
            question: The question text
            response: The model's response
            accuracy: Accuracy score (0-5)
            helpfulness: Helpfulness score (0-5)
            tone: Tone score (0-5)
            notes: Optional notes or justification
            
        Returns:
            Dictionary with annotation data
        """
        # Validate scores
        for dimension, score in [("accuracy", accuracy), 
                                 ("helpfulness", helpfulness), 
                                 ("tone", tone)]:
            if not self.rubrics.validate_score(dimension, score):
                raise ValueError(f"Invalid {dimension} score: {score}")
        
        annotation = {
            'question': question,
            'model_response': response,
            'accuracy_score': accuracy,
            'helpfulness_score': helpfulness,
            'tone_score': tone,
            'annotator_id': self.annotator_id,
            'notes': notes,
            'timestamp': datetime.now().isoformat()
        }
        
        return annotation
    
    def annotate_batch(self, qa_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate a batch of QA pairs (simulated or manual)
        
        Args:
            qa_pairs: DataFrame with 'question' and 'model_response' columns
            
        Returns:
            DataFrame with added annotation columns
        """
        logger.info(f"Starting annotation of {len(qa_pairs)} QA pairs by {self.annotator_id}")
        
        annotations = []
        
        for idx, row in qa_pairs.iterrows():
            print(f"\n{'='*80}")
            print(f"QA Pair {idx + 1}/{len(qa_pairs)}")
            print(f"{'='*80}")
            print(f"\nQuestion: {row['question'][:200]}...")
            print(f"\nResponse: {row['model_response'][:300]}...")
            print(f"\n{'-'*80}")
            
            # Display rubrics
            print("\nRubric Guidelines:")
            print("Accuracy: 0=Wrong, 1=Mostly wrong, 2=Some errors, 3=Partial, 4=Mostly right, 5=Perfect")
            print("Helpfulness: 0=Not helpful, 1=Minimal, 2=Somewhat, 3=Moderate, 4=Very, 5=Extremely")
            print("Tone: 0=Inappropriate, 1=Noticeably poor, 2=Slight issues, 3=Generally ok, 4=Good, 5=Perfect")
            
            # Get scores (in real use, this would be interactive input)
            # For now, we'll use a simple heuristic
            accuracy = self._simulate_accuracy_score(row['question'], row['model_response'])
            helpfulness = self._simulate_helpfulness_score(row['question'], row['model_response'])
            tone = self._simulate_tone_score(row['model_response'])
            
            print(f"\nScores assigned: Accuracy={accuracy}, Helpfulness={helpfulness}, Tone={tone}")
            
            annotation = self.annotate_single(
                question=row['question'],
                response=row['model_response'],
                accuracy=accuracy,
                helpfulness=helpfulness,
                tone=tone,
                notes=""
            )
            
            annotations.append(annotation)
        
        df = pd.DataFrame(annotations)
        logger.info(f"Annotation complete: {len(df)} QA pairs annotated")
        return df
    
    def _simulate_accuracy_score(self, question: str, response: str) -> int:
        """Simulate accuracy scoring (for demonstration)"""
        # Simple heuristic based on response length and keywords
        if len(response) < 20:
            return 1
        elif len(response) < 50:
            return 2
        elif len(response) < 100:
            return 3
        elif len(response) < 200:
            return 4
        else:
            return 5
    
    def _simulate_helpfulness_score(self, question: str, response: str) -> int:
        """Simulate helpfulness scoring (for demonstration)"""
        # Simple heuristic
        keywords = ['how', 'why', 'what', 'explain', 'example', 'steps']
        question_lower = question.lower()
        response_lower = response.lower()
        
        matches = sum(1 for kw in keywords if kw in question_lower and kw in response_lower)
        
        if matches == 0:
            return 2
        elif matches <= 2:
            return 3
        elif matches <= 4:
            return 4
        else:
            return 5
    
    def _simulate_tone_score(self, response: str) -> int:
        """Simulate tone scoring (for demonstration)"""
        # Simple heuristic checking for inappropriate words
        inappropriate = ['stupid', 'dumb', 'idiot', 'hate']
        response_lower = response.lower()
        
        if any(word in response_lower for word in inappropriate):
            return 1
        
        # Default to good tone
        return 4
    
    def save_annotations(self, df: pd.DataFrame, filename: Optional[str] = None):
        """
        Save annotations to CSV
        
        Args:
            df: DataFrame with annotations
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"annotations_{self.annotator_id}_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Annotations saved to {output_path}")
        return output_path


class AnnotationManager:
    """Manage multiple annotators and consolidate annotations"""
    
    def __init__(self, output_dir: Path = ANNOTATED_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.annotators = []
    
    def add_annotator(self, annotator_id: str) -> Annotator:
        """Add a new annotator"""
        annotator = Annotator(annotator_id, self.output_dir)
        self.annotators.append(annotator)
        return annotator
    
    def merge_annotations(self, annotation_files: List[Path]) -> pd.DataFrame:
        """
        Merge annotations from multiple annotators
        
        Args:
            annotation_files: List of annotation CSV files
            
        Returns:
            Merged DataFrame with all annotations
        """
        dfs = []
        for file in annotation_files:
            df = pd.read_csv(file)
            dfs.append(df)
        
        merged = pd.concat(dfs, ignore_index=True)
        logger.info(f"Merged {len(annotation_files)} annotation files: {len(merged)} total annotations")
        return merged
    
    def compute_average_scores(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute average scores across annotators
        
        Args:
            merged_df: DataFrame with annotations from multiple annotators
            
        Returns:
            DataFrame with averaged scores
        """
        # Group by question and response, compute means
        grouped = merged_df.groupby(['question', 'model_response']).agg({
            'accuracy_score': 'mean',
            'helpfulness_score': 'mean',
            'tone_score': 'mean',
            'annotator_id': lambda x: ','.join(x)
        }).reset_index()
        
        grouped.rename(columns={'annotator_id': 'annotators'}, inplace=True)
        
        logger.info(f"Computed average scores for {len(grouped)} unique QA pairs")
        return grouped


def main():
    """Example usage of annotation module"""
    # Load dataset
    from .data_collection import DatasetCollector
    
    collector = DatasetCollector()
    
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'question': [
            "What is the capital of France?",
            "Explain machine learning in simple terms.",
            "How do I bake chocolate chip cookies?"
        ],
        'model_response': [
            "The capital of France is Paris. Paris is the largest city in France and serves as its political, economic, and cultural center.",
            "Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed.",
            "To bake chocolate chip cookies, you'll need flour, butter, sugar, eggs, chocolate chips, and baking soda. Mix ingredients, form dough balls, and bake at 350°F for 10-12 minutes."
        ]
    })
    
    # Create annotator
    annotator = Annotator("annotator_1")
    
    # Annotate
    annotated_df = annotator.annotate_batch(sample_data)
    
    # Save
    output_path = annotator.save_annotations(annotated_df)
    
    print(f"\nAnnotation complete!")
    print(f"Output saved to: {output_path}")
    print(f"\nSample annotations:")
    print(annotated_df[['question', 'accuracy_score', 'helpfulness_score', 'tone_score']].head())


if __name__ == "__main__":
    main()
