"""
Configuration settings for the evaluation system
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ANNOTATED_DATA_DIR = DATA_DIR / "annotated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure directories exist
for directory in [RAW_DATA_DIR, ANNOTATED_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Evaluation rubrics configuration
RUBRIC_DIMENSIONS = {
    "accuracy": {
        "min_score": 0,
        "max_score": 5,
        "description": "Factual correctness and relevance of the response"
    },
    "helpfulness": {
        "min_score": 0,
        "max_score": 5,
        "description": "How useful the response is in addressing the question"
    },
    "tone": {
        "min_score": 0,
        "max_score": 5,
        "description": "Appropriateness of tone and absence of bias"
    }
}

# Dataset configuration
TARGET_DATASET_SIZE = 500
MIN_DATASET_SIZE = 300

# OpenAI API configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"

# Annotation configuration
NUM_ANNOTATORS = 2
AGREEMENT_THRESHOLD = 0.7  # Cohen's Kappa threshold for good agreement
