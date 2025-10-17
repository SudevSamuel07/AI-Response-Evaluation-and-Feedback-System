"""
Streamlit Dashboard for AI Response Evaluation System
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.config import ANNOTATED_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR
from src.analysis import AgreementAnalyzer, QualityAnalyzer
from src.visualization import Visualizer
from src.rubrics import EvaluationRubrics

# Page configuration
st.set_page_config(
    page_title="AI Response Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

# Sidebar
st.sidebar.title("📊 AI Response Evaluation")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigate to:",
    ["Overview", "Data Upload", "Evaluation Rubrics", "Analysis", "Quality Patterns", "Visualizations"]
)

# Initialize analyzers
@st.cache_resource
def get_analyzers():
    return {
        'agreement': AgreementAnalyzer(),
        'quality': QualityAnalyzer(),
        'visualizer': Visualizer(),
        'rubrics': EvaluationRubrics()
    }

analyzers = get_analyzers()

# Helper function to load data
@st.cache_data
def load_data(file):
    """Load CSV data"""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==============================================================================
# PAGE: OVERVIEW
# ==============================================================================
if page == "Overview":
    st.title("🎯 AI Response Evaluation Dashboard")
    st.markdown("### Systematic Evaluation of LLM Responses")
    
    st.markdown("""
    This dashboard helps you:
    - 📥 Upload and manage QA datasets
    - 📋 View evaluation rubrics
    - 📊 Analyze inter-annotator agreement (Cohen's Kappa)
    - 🔍 Identify quality patterns and weaknesses
    - 📈 Visualize evaluation results
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Project Status", "Active", delta="Ready")
    
    with col2:
        # Count files in annotated directory
        annotated_files = list(ANNOTATED_DATA_DIR.glob("*.csv"))
        st.metric("Annotated Datasets", len(annotated_files))
    
    with col3:
        # Count plots
        plot_files = list(PLOTS_DIR.glob("*.png"))
        st.metric("Generated Plots", len(plot_files))
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    1. **Data Upload**: Upload your annotated dataset (CSV format)
    2. **Evaluation Rubrics**: Review the scoring guidelines
    3. **Analysis**: Compute inter-annotator agreement and statistics
    4. **Quality Patterns**: Identify low-quality responses
    5. **Visualizations**: Generate charts and plots
    """)

# ==============================================================================
# PAGE: DATA UPLOAD
# ==============================================================================
elif page == "Data Upload":
    st.title("📥 Data Upload")
    
    st.markdown("""
    Upload your annotated dataset with the following columns:
    - `question`: The question text
    - `model_response`: The AI-generated response
    - `accuracy_score`: Score 0-5
    - `helpfulness_score`: Score 0-5
    - `tone_score`: Score 0-5
    - `annotator_id` (optional): Annotator identifier
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Loaded {len(df)} records")
            
            # Display basic info
            st.markdown("### Dataset Preview")
            st.dataframe(df.head(10))
            
            # Display column info
            st.markdown("### Dataset Info")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Columns:**")
                st.write(df.columns.tolist())
            
            with col2:
                st.write("**Shape:**")
                st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")
            
            # Check for required columns
            required_cols = ['question', 'model_response', 'accuracy_score', 
                           'helpfulness_score', 'tone_score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"⚠️ Missing required columns: {missing_cols}")
            else:
                st.success("✅ All required columns present")
                
                # Store in session state
                st.session_state['data'] = df
                st.info("💾 Data saved to session. Navigate to other pages to analyze.")

# ==============================================================================
# PAGE: EVALUATION RUBRICS
# ==============================================================================
elif page == "Evaluation Rubrics":
    st.title("📋 Evaluation Rubrics")
    
    rubrics = analyzers['rubrics']
    all_guidelines = rubrics.get_all_guidelines()
    
    for dimension, guidelines in all_guidelines.items():
        st.markdown(f"## {dimension.upper()}")
        st.markdown(f"*{rubrics.dimensions[dimension]['description']}*")
        
        # Create table
        data = [[score, description] for score, description in guidelines.items()]
        df_rubric = pd.DataFrame(data, columns=['Score', 'Description'])
        st.table(df_rubric)
        st.markdown("---")
    
    st.markdown("""
    ### 📝 Scoring Instructions
    
    - **Read carefully**: Review both the question and response thoroughly
    - **Independent evaluation**: Score each dimension separately
    - **Consistency**: Apply the same standards across all evaluations
    - **Justification**: Document reasons for edge cases
    - **Calibration**: Periodically review past annotations for consistency
    """)

# ==============================================================================
# PAGE: ANALYSIS
# ==============================================================================
elif page == "Analysis":
    st.title("📊 Statistical Analysis")
    
    # Check if data is loaded
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        st.stop()
    
    df = st.session_state['data']
    
    # Summary Statistics
    st.markdown("## 📈 Summary Statistics")
    
    quality_analyzer = analyzers['quality']
    stats = quality_analyzer.compute_summary_statistics(df)
    
    # Display as table
    stats_data = []
    for dim, values in stats.items():
        stats_data.append({
            'Dimension': dim.replace('_score', '').title(),
            'Mean': f"{values['mean']:.2f}",
            'Median': f"{values['median']:.2f}",
            'Std Dev': f"{values['std']:.2f}",
            'Min': f"{values['min']:.0f}",
            'Max': f"{values['max']:.0f}"
        })
    
    st.table(pd.DataFrame(stats_data))
    
    # Inter-Annotator Agreement
    st.markdown("---")
    st.markdown("## 🤝 Inter-Annotator Agreement")
    
    if 'annotator_id' in df.columns and df['annotator_id'].nunique() >= 2:
        agreement_analyzer = analyzers['agreement']
        
        dimension_choice = st.selectbox(
            "Select dimension to analyze:",
            ['accuracy_score', 'helpfulness_score', 'tone_score']
        )
        
        results = agreement_analyzer.analyze_multi_annotator(df, dimension_choice)
        
        if results:
            st.markdown(f"**Number of annotators:** {results['num_annotators']}")
            
            for pair, metrics in results['pairwise_agreements'].items():
                st.markdown(f"### {pair.replace('_', ' ')}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Agreement %", f"{metrics['agreement_percentage']:.2f}%")
                
                with col2:
                    st.metric("Cohen's Kappa", f"{metrics['cohen_kappa']:.3f}")
                
                with col3:
                    st.metric("Interpretation", metrics['kappa_interpretation'])
                
                if metrics['cohen_kappa'] < 0.7:
                    st.warning("⚠️ Agreement below recommended threshold (0.7)")
                else:
                    st.success("✅ Good agreement")
    else:
        st.info("ℹ️ Inter-annotator agreement requires multiple annotators in the dataset.")
    
    # Score Correlations
    st.markdown("---")
    st.markdown("## 🔗 Score Correlations")
    
    corr_matrix = quality_analyzer.find_score_correlations(df)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, ax=ax)
    ax.set_title('Correlation Between Dimensions')
    st.pyplot(fig)

# ==============================================================================
# PAGE: QUALITY PATTERNS
# ==============================================================================
elif page == "Quality Patterns":
    st.title("🔍 Quality Pattern Analysis")
    
    # Check if data is loaded
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        st.stop()
    
    df = st.session_state['data']
    quality_analyzer = analyzers['quality']
    
    # Low Quality Responses
    st.markdown("## 🚨 Low Quality Responses")
    
    threshold = st.slider("Score threshold", 0.0, 5.0, 2.5, 0.5)
    
    low_quality = quality_analyzer.identify_low_quality_responses(df, threshold)
    
    st.markdown(f"**Found {len(low_quality)} responses below threshold {threshold}**")
    
    if len(low_quality) > 0:
        # Display top 10 lowest
        st.markdown("### Lowest Scoring Responses")
        display_cols = ['question', 'accuracy_score', 'helpfulness_score', 
                       'tone_score', 'avg_score']
        st.dataframe(low_quality[display_cols].head(10))
        
        # Download button
        csv = low_quality.to_csv(index=False)
        st.download_button(
            label="📥 Download Low Quality Responses",
            data=csv,
            file_name="low_quality_responses.csv",
            mime="text/csv"
        )
    
    # Analysis by Question Type
    st.markdown("---")
    st.markdown("## 📝 Analysis by Question Type")
    
    grouped = quality_analyzer.analyze_by_question_type(df)
    
    st.markdown("### Average Scores by Question Type")
    st.dataframe(grouped)
    
    # Identify weakest category
    avg_scores = grouped.mean(axis=1).sort_values()
    weakest_type = avg_scores.index[0]
    weakest_score = avg_scores.iloc[0]
    
    st.warning(f"⚠️ **Weakest Category**: {weakest_type} (avg score: {weakest_score:.2f})")

# ==============================================================================
# PAGE: VISUALIZATIONS
# ==============================================================================
elif page == "Visualizations":
    st.title("📈 Visualizations")
    
    # Check if data is loaded
    if 'data' not in st.session_state:
        st.warning("⚠️ Please upload data first from the 'Data Upload' page.")
        st.stop()
    
    df = st.session_state['data']
    
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Score Distributions", "Score Comparison", "Correlation Heatmap", 
         "Question Type Analysis", "Summary Dashboard"]
    )
    
    if viz_type == "Score Distributions":
        st.markdown("## 📊 Score Distributions")
        
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, dim in enumerate(dimensions):
            axes[i].hist(df[dim], bins=6, range=(0, 5), 
                       edgecolor='black', alpha=0.7, color=f'C{i}')
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(dim.replace('_', ' ').title())
            axes[i].set_xticks(range(6))
            
            mean_val = df[dim].mean()
            axes[i].axvline(mean_val, color='red', linestyle='--', 
                          label=f'Mean: {mean_val:.2f}')
            axes[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_type == "Score Comparison":
        st.markdown("## 📊 Score Comparison")
        
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        means = [df[dim].mean() for dim in dimensions]
        stds = [df[dim].std() for dim in dimensions]
        labels = [dim.replace('_score', '').title() for dim in dimensions]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=10, alpha=0.7,
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores Across Dimensions')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 5.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_type == "Correlation Heatmap":
        st.markdown("## 🔗 Correlation Heatmap")
        
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        corr_matrix = df[dimensions].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Between Score Dimensions')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_type == "Question Type Analysis":
        st.markdown("## 📝 Question Type Analysis")
        
        quality_analyzer = analyzers['quality']
        grouped = quality_analyzer.analyze_by_question_type(df)
        
        # Extract means
        grouped_means = grouped.xs('mean', axis=1, level=1)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        grouped_means.plot(kind='bar', ax=ax, alpha=0.8)
        
        ax.set_xlabel('Question Type')
        ax.set_ylabel('Average Score')
        ax.set_title('Average Scores by Question Type')
        ax.legend(title='Dimension', labels=['Accuracy', 'Helpfulness', 'Tone'])
        ax.set_ylim(0, 5.5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    elif viz_type == "Summary Dashboard":
        st.markdown("## 📊 Summary Dashboard")
        
        visualizer = analyzers['visualizer']
        
        # Generate dashboard
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        dimensions = ['accuracy_score', 'helpfulness_score', 'tone_score']
        
        # Score distributions
        for i, dim in enumerate(dimensions):
            ax = fig.add_subplot(gs[0, i])
            ax.hist(df[dim], bins=6, range=(0, 5), edgecolor='black', alpha=0.7, color=f'C{i}')
            ax.set_title(dim.replace('_', ' ').title())
            ax.set_xlabel('Score')
            ax.set_ylabel('Count')
            mean_val = df[dim].mean()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'μ={mean_val:.2f}')
            ax.legend()
        
        # Comparison
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
        
        # Correlation
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
        
        plt.suptitle('Evaluation Summary Dashboard', fontsize=16, fontweight='bold')
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 Resources
- [Documentation](docs/)
- [Rubrics Guidelines](docs/rubrics.md)
""")
