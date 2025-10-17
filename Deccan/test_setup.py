"""
Quick Start Guide - Test the AI Response Evaluation System
"""
import pandas as pd
from pathlib import Path

print("=" * 80)
print("AI RESPONSE EVALUATION SYSTEM - QUICK START TEST")
print("=" * 80)

# Test 1: Load sample data
print("\n1. Loading sample annotated data...")
sample_file = Path("data/annotated/sample_annotations.csv")
if sample_file.exists():
    df = pd.read_csv(sample_file)
    print(f"   ✓ Loaded {len(df)} sample annotations")
    print(f"   Columns: {list(df.columns)}")
else:
    print("   ✗ Sample file not found")

# Test 2: Import and test rubrics
print("\n2. Testing rubrics module...")
try:
    from src.rubrics import EvaluationRubrics
    rubrics = EvaluationRubrics()
    print(f"   ✓ Rubrics module loaded successfully")
    print(f"   Dimensions: {list(rubrics.dimensions.keys())}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Test configuration
print("\n3. Testing configuration...")
try:
    from src.config import RUBRIC_DIMENSIONS, DATA_DIR
    print(f"   ✓ Configuration loaded successfully")
    print(f"   Data directory: {DATA_DIR}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Test analysis module
print("\n4. Testing analysis module...")
try:
    from src.analysis import QualityAnalyzer
    analyzer = QualityAnalyzer()
    if sample_file.exists():
        stats = analyzer.compute_summary_statistics(df)
        print(f"   ✓ Analysis module works!")
        for dim, values in stats.items():
            print(f"   {dim}: mean={values['mean']:.2f}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Test visualization module
print("\n5. Testing visualization module...")
try:
    from src.visualization import Visualizer
    viz = Visualizer()
    print(f"   ✓ Visualization module loaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 80)
print("NEXT STEPS:")
print("=" * 80)
print("\n1. To view rubrics:")
print("   python -m src.rubrics")
print("\n2. To launch the Streamlit dashboard:")
print("   streamlit run dashboard.py")
print("\n3. To run data collection:")
print("   python main.py --collect --samples 100")
print("\n4. To see full documentation:")
print("   Open README.md or docs/rubrics.md")
print("\n" + "=" * 80)
