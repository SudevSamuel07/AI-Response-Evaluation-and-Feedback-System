"""
Module for collecting and preparing QA datasets from various sources
"""
import pandas as pd
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

from .config import RAW_DATA_DIR, TARGET_DATASET_SIZE, OPENAI_API_KEY, OPENAI_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetCollector:
    """Collect QA pairs from various sources"""
    
    def __init__(self, output_dir: Path = RAW_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_from_alpaca(self, num_samples: int = TARGET_DATASET_SIZE) -> pd.DataFrame:
        """
        Collect data from Alpaca dataset
        
        Args:
            num_samples: Number of samples to collect
            
        Returns:
            DataFrame with columns: question, model_response
        """
        logger.info(f"Collecting {num_samples} samples from Alpaca dataset...")
        
        # Download Alpaca dataset
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            samples = []
            for item in data[:num_samples]:
                question = item.get('instruction', '')
                if item.get('input'):
                    question += f"\n\nInput: {item['input']}"
                
                samples.append({
                    'question': question,
                    'model_response': item.get('output', '')
                })
            
            df = pd.DataFrame(samples)
            logger.info(f"Successfully collected {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting Alpaca data: {e}")
            return pd.DataFrame()
    
    def collect_from_openai(self, 
                           questions: List[str],
                           model: str = OPENAI_MODEL) -> pd.DataFrame:
        """
        Generate responses using OpenAI API
        
        Args:
            questions: List of questions to ask
            model: OpenAI model to use
            
        Returns:
            DataFrame with columns: question, model_response
        """
        if not OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set. Skipping OpenAI collection.")
            return pd.DataFrame()
        
        logger.info(f"Generating responses for {len(questions)} questions using {model}...")
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            samples = []
            for question in tqdm(questions):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": question}],
                        max_tokens=500
                    )
                    
                    samples.append({
                        'question': question,
                        'model_response': response.choices[0].message.content
                    })
                except Exception as e:
                    logger.error(f"Error generating response: {e}")
                    continue
            
            df = pd.DataFrame(samples)
            logger.info(f"Successfully generated {len(df)} responses")
            return df
            
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
            return pd.DataFrame()
    
    def load_custom_dataset(self, filepath: Path) -> pd.DataFrame:
        """
        Load custom dataset from CSV/JSON
        
        Args:
            filepath: Path to dataset file
            
        Returns:
            DataFrame with columns: question, model_response
        """
        logger.info(f"Loading custom dataset from {filepath}...")
        
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix == '.json':
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Validate required columns
            required_cols = ['question', 'model_response']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dataset must contain columns: {required_cols}")
            
            logger.info(f"Successfully loaded {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "qa_dataset.csv"):
        """
        Save collected dataset to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    
    def create_sample_questions(self, num_questions: int = 50) -> List[str]:
        """
        Create sample questions for testing
        
        Args:
            num_questions: Number of sample questions to create
            
        Returns:
            List of sample questions
        """
        sample_questions = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "How do I make chocolate chip cookies?",
            "What are the benefits of regular exercise?",
            "Explain the theory of relativity.",
            "How does photosynthesis work?",
            "What is machine learning?",
            "How do I learn Python programming?",
            "What causes climate change?",
            "Explain blockchain technology.",
            "How does the human immune system work?",
            "What is the difference between AI and ML?",
            "How do solar panels generate electricity?",
            "What are the main causes of World War II?",
            "How does the stock market work?",
            "What is the best way to learn a new language?",
            "Explain how vaccines work.",
            "What is cryptocurrency?",
            "How do computers process information?",
            "What is the theory of evolution?",
        ]
        
        # Repeat to reach desired number
        questions = (sample_questions * (num_questions // len(sample_questions) + 1))[:num_questions]
        return questions


def main():
    """Example usage of DatasetCollector"""
    collector = DatasetCollector()
    
    # Option 1: Collect from Alpaca dataset
    df = collector.collect_from_alpaca(num_samples=500)
    
    # Option 2: Generate using OpenAI (requires API key)
    # questions = collector.create_sample_questions(50)
    # df = collector.collect_from_openai(questions)
    
    # Option 3: Load custom dataset
    # df = collector.load_custom_dataset(Path("your_dataset.csv"))
    
    if not df.empty:
        collector.save_dataset(df, "qa_dataset.csv")
        print(f"\nDataset collected: {len(df)} QA pairs")
        print(f"\nSample data:")
        print(df.head())


if __name__ == "__main__":
    main()
