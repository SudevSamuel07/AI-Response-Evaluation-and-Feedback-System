# AI Response Evaluation and Feedback System for LLMs

A comprehensive system for systematically evaluating AI-generated responses using predefined quality rubrics, analyzing inter-annotator agreement, and identifying patterns in model performance.

## 📋 Project Overview

This project enables you to:
- Collect and prepare QA datasets from various sources (Alpaca, OpenAI API, custom datasets)
- Evaluate responses using standardized rubrics (Accuracy, Helpfulness, Tone)
- Compute inter-annotator agreement using Cohen's Kappa
- Analyze quality patterns and identify model weaknesses
- Visualize evaluation results
- Monitor progress through an interactive Streamlit dashboard

## 🏗️ Project Structure

```
Deccan/
├── data/
│   ├── raw/                    # Original QA datasets
│   ├── annotated/              # Annotated datasets with scores
│   └── processed/              # Processed analysis results
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── data_collection.py     # Dataset collection module
│   ├── rubrics.py             # Evaluation rubrics definitions
│   ├── annotation.py          # Annotation workflow
│   ├── analysis.py            # Statistical analysis (Cohen's Kappa)
│   └── visualization.py       # Plotting and charts
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_annotation.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_visualization.ipynb
├── outputs/
│   ├── plots/                 # Generated visualizations
│   └── reports/               # Analysis reports
├── docs/
│   └── rubrics.md             # Detailed rubrics documentation
├── dashboard.py               # Streamlit interactive dashboard
├── requirements.txt           # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone or navigate to the project directory
cd C:\Users\sudev\Deccan

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

#### Option A: Use Alpaca Dataset (Recommended for testing)

```python
from src.data_collection import DatasetCollector

collector = DatasetCollector()
df = collector.collect_from_alpaca(num_samples=500)
collector.save_dataset(df, "qa_dataset.csv")
```

#### Option B: Use OpenAI API

```python
# Set your OpenAI API key
# In PowerShell: $env:OPENAI_API_KEY="your-key-here"

collector = DatasetCollector()
questions = collector.create_sample_questions(50)
df = collector.collect_from_openai(questions)
collector.save_dataset(df, "qa_dataset.csv")
```

#### Option C: Load Custom Dataset

```python
collector = DatasetCollector()
df = collector.load_custom_dataset(Path("your_dataset.csv"))
```

### 3. View Evaluation Rubrics

```python
from src.rubrics import EvaluationRubrics

rubrics = EvaluationRubrics()
rubrics.print_rubrics()
```

### 4. Annotate Responses

```python
from src.annotation import Annotator
import pandas as pd

# Load your dataset
df = pd.read_csv("data/raw/qa_dataset.csv")

# Create annotator
annotator = Annotator("annotator_1")

# Annotate batch (simulated for now)
annotated_df = annotator.annotate_batch(df)

# Save annotations
annotator.save_annotations(annotated_df)
```

### 5. Analyze Results

```python
from src.analysis import AgreementAnalyzer, QualityAnalyzer

# Inter-annotator agreement
agreement = AgreementAnalyzer()
results = agreement.analyze_multi_annotator(df, 'accuracy_score')
agreement.print_agreement_report(results)

# Quality analysis
quality = QualityAnalyzer()
stats = quality.compute_summary_statistics(df)
low_quality = quality.identify_low_quality_responses(df, threshold=2.5)
```

### 6. Create Visualizations

```python
from src.visualization import Visualizer

viz = Visualizer()
viz.plot_all_dimensions(df, "all_dimensions.png")
viz.plot_score_comparison(df, "score_comparison.png")
viz.create_summary_dashboard(df, "summary_dashboard.png")
```

### 7. Launch Dashboard

```powershell
streamlit run dashboard.py
```

Then open your browser to `http://localhost:8501`

## 📊 Evaluation Rubrics

### Accuracy (0-5)
Measures factual correctness and relevance of the response.

- **5**: Completely accurate and factually correct
- **4**: Mostly accurate with minor imperfections
- **3**: Partially accurate with notable gaps or minor errors
- **2**: Some correct information but significant errors
- **1**: Mostly incorrect with minimal accurate content
- **0**: Completely incorrect or irrelevant information

### Helpfulness (0-5)
Measures how useful the response is in addressing the question.

- **5**: Extremely helpful, comprehensive and actionable response
- **4**: Very helpful, addresses the question well with minor gaps
- **3**: Moderately helpful, provides useful but incomplete information
- **2**: Somewhat helpful but lacks important details
- **1**: Minimally helpful, barely addresses the question
- **0**: Not helpful at all, does not address the question

### Tone (0-5)
Measures appropriateness of tone and absence of bias.

- **5**: Perfect tone, completely unbiased and professional
- **4**: Professional and appropriate tone with no significant bias
- **3**: Generally appropriate with minor tone issues
- **2**: Somewhat inappropriate tone or slight bias
- **1**: Noticeably inappropriate or biased
- **0**: Highly inappropriate tone or significant bias

## 📈 Analysis Features

### Inter-Annotator Agreement
- **Simple Agreement**: Percentage of exact matches
- **Cohen's Kappa**: Accounts for chance agreement
- **Interpretation**: Automatically interprets Kappa scores
- **Threshold Warnings**: Alerts when agreement is below 0.7

### Quality Pattern Analysis
- **Summary Statistics**: Mean, median, std dev for each dimension
- **Low Quality Detection**: Identifies responses below threshold
- **Question Type Analysis**: Groups by question categories
- **Score Correlations**: Examines relationships between dimensions

### Visualizations
- Score distributions (histograms, box plots)
- Comparison charts (bar charts with error bars)
- Correlation heatmaps
- Question type analysis
- Annotator comparison plots
- Comprehensive summary dashboards

## 🎯 Workflow Example

### Complete End-to-End Example

```python
# 1. Collect data
from src.data_collection import DatasetCollector
collector = DatasetCollector()
df = collector.collect_from_alpaca(500)
collector.save_dataset(df, "qa_dataset.csv")

# 2. Annotate (with 2 annotators for agreement analysis)
from src.annotation import Annotator, AnnotationManager
import pandas as pd

df = pd.read_csv("data/raw/qa_dataset.csv")

# Annotator 1
ann1 = Annotator("annotator_1")
df1 = ann1.annotate_batch(df)
ann1.save_annotations(df1, "annotations_1.csv")

# Annotator 2
ann2 = Annotator("annotator_2")
df2 = ann2.annotate_batch(df)
ann2.save_annotations(df2, "annotations_2.csv")

# Merge annotations
manager = AnnotationManager()
merged = manager.merge_annotations([
    Path("data/annotated/annotations_1.csv"),
    Path("data/annotated/annotations_2.csv")
])

# 3. Analyze agreement
from src.analysis import AgreementAnalyzer
agreement = AgreementAnalyzer()
results = agreement.analyze_multi_annotator(merged, 'accuracy_score')
agreement.print_agreement_report(results)

# 4. Quality analysis
from src.analysis import QualityAnalyzer
quality = QualityAnalyzer()
low_quality = quality.identify_low_quality_responses(merged, 2.5)
by_type = quality.analyze_by_question_type(merged)

# 5. Visualize
from src.visualization import Visualizer
viz = Visualizer()
viz.create_summary_dashboard(merged, "dashboard.png")

# 6. Launch interactive dashboard
# In terminal: streamlit run dashboard.py
```

## 📓 Jupyter Notebooks

Interactive notebooks are provided for step-by-step exploration:

1. **01_data_collection.ipynb**: Data collection from various sources
2. **02_annotation.ipynb**: Annotation workflow and rubrics
3. **03_analysis.ipynb**: Statistical analysis and agreement metrics
4. **04_visualization.ipynb**: Creating visualizations

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
# Dataset size
TARGET_DATASET_SIZE = 500
MIN_DATASET_SIZE = 300

# Agreement threshold
AGREEMENT_THRESHOLD = 0.7  # Cohen's Kappa

# OpenAI settings (if using API)
OPENAI_MODEL = "gpt-3.5-turbo"
```

## 📦 Dependencies

- **Data Processing**: pandas, numpy
- **Statistical Analysis**: scipy, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: streamlit
- **LLM API** (optional): openai
- **Development**: jupyter, ipykernel

## 🎓 Interpreting Cohen's Kappa

| Kappa Range | Interpretation |
|-------------|----------------|
| < 0.00 | Poor (Less than chance) |
| 0.00 - 0.20 | Slight |
| 0.21 - 0.40 | Fair |
| 0.41 - 0.60 | Moderate |
| 0.61 - 0.80 | Substantial |
| 0.81 - 1.00 | Almost Perfect |

**Recommendation**: Aim for Kappa > 0.70 for reliable annotations.

## 🚧 Troubleshooting

### Import Errors
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### OpenAI API Issues
```powershell
# Set API key in environment
$env:OPENAI_API_KEY="your-key-here"
```

### Dashboard Not Loading
```powershell
# Check Streamlit installation
pip install streamlit --upgrade

# Run dashboard
streamlit run dashboard.py
```

## 📝 Next Steps

1. **Refine Rubrics**: Based on initial annotations, adjust guidelines
2. **Increase Dataset**: Collect more QA pairs for robust analysis
3. **Multiple Annotators**: Recruit 2-3 annotators for agreement analysis
4. **Iterative Refinement**: Re-evaluate after rubric adjustments
5. **Model Comparison**: Evaluate multiple LLMs side-by-side

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Add new data sources
- Implement additional metrics
- Create new visualizations
- Improve the annotation interface

## 📄 License

MIT License - Feel free to use and modify for your research.

## 📞 Support

For questions or issues:
1. Check the documentation in `/docs`
2. Review example notebooks in `/notebooks`
3. Examine sample data in `/data`

---

**Happy Evaluating! 🎯📊**
