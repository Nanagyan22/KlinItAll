import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re  # For more robust text cleaning

# -----------------------------
# System Knowledge Database (Data Science Topics)
# -----------------------------

system_knowledge = {
    "general": {
        "about_system": "KlinItAll is an advanced data preprocessing and analysis system. It automates tasks like missing value imputation, outlier detection, feature scaling, categorical encoding, and advanced data transformations to prepare data for analysis and modeling.",
        "how_it_works": "KlinItAll automatically detects data issues such as missing values, outliers, skewed distributions, and inappropriate data types. The system then applies the best preprocessing techniques like imputation, scaling, and encoding, with full automation.",
        "core_features": [
            "Handling missing values",
            "Outlier detection and treatment",
            "Feature scaling and normalization",
            "Categorical encoding (one-hot, label, etc.)",
            "Feature engineering (binning, interaction features)",
            "Advanced batch processing and scheduling",
            "Data visualization and profiling",
            "Fuzzy matching and duplicate detection"
        ]
    },
    "data_preprocessing": {
        "missing_values": {
            "description": "Missing values occur when some data points are not available or recorded. The system detects columns with missing values and suggests imputation strategies.",
            "methods": [
                "Drop rows/columns with excessive missing values.",
                "Impute with mean, median, or mode for small percentages of missing data.",
                "Use KNN imputation or model-based imputation for larger missing values."
            ]
        },
        "outliers": {
            "description": "Outliers are extreme data points that deviate significantly from other observations in a dataset. The system uses methods like IQR, Z-score, or model-based techniques to detect and treat outliers.",
            "methods": [
                "Trim outliers (remove rows with extreme values).",
                "Winsorize (capping the extreme values to a fixed range).",
                "Replace outliers with the median or mean value."
            ]
        },
        "scaling": {
            "description": "Scaling involves adjusting the feature values so they are on the same scale, often necessary for algorithms sensitive to feature magnitudes, such as gradient descent or distance-based models.",
            "methods": [
                "StandardScaler (z-score normalization).",
                "MinMaxScaler (scales data to a specific range, e.g., [0,1]).",
                "RobustScaler (scales data based on median and IQR).",
                "PowerTransformer (Yeo-Johnson, Box-Cox) to handle skewed data."
            ]
        },
        "encoding": {
            "description": "Encoding transforms categorical variables into a numerical format. The system suggests methods based on the cardinality of the categories.",
            "methods": [
                "One-hot encoding for columns with fewer than 15 unique categories.",
                "Label encoding for ordinal categories.",
                "Frequency encoding or target encoding for high cardinality categories."
            ]
        },
        "feature_engineering": {
            "description": "Feature engineering is the process of creating new features or transforming existing ones to improve the performance of machine learning models.",
            "methods": [
                "Binning (equal-width or equal-frequency binning).",
                "Interaction features (e.g., product, ratio, difference between columns).",
                "Dimensionality reduction using PCA, t-SNE, or UMAP."
            ]
        }
    },
    "statistics": {
        "description": "Statistics is a critical aspect of data science. It involves collecting, analyzing, interpreting, presenting, and organizing data.",
        "core_concepts": [
            "Mean, Median, Mode",
            "Variance, Standard Deviation",
            "Skewness, Kurtosis",
            "Correlation, Regression",
            "Hypothesis Testing"
        ],
        "common_questions": {
            "mean": "The **mean** is the average of all values in a dataset.",
            "variance": "The **variance** measures the spread of data points around the mean.",
            "skewness": "Skewness measures the asymmetry of the distribution of data. Positive skewness means the right tail is longer, while negative skewness means the left tail is longer.",
            "correlation": "Correlation is a measure of the relationship between two variables. Values range from -1 (perfect negative) to +1 (perfect positive).",
            "regression": "Regression is used to model relationships between variables, typically for prediction."
        }
    }
}

# -----------------------------
# Helper Functions for Data Operations
# -----------------------------

def dataset_summary(df):
    """Provides a comprehensive summary of the dataset."""
    summary = f"Your dataset has **{df.shape[0]} rows** and **{df.shape[1]} columns**."
    summary += "\nHere are the column names:\n" + ", ".join(df.columns)
    return summary

def detailed_stats(df):
    """Provides detailed descriptive statistics and insights on numerical columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        return "It seems like there are no numerical columns in your dataset, so I can't calculate statistics like mean or standard deviation."

    stats = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        mean = col_data.mean()
        median = col_data.median()
        std_dev = col_data.std()
        skewness = skew(col_data)
        kurt = kurtosis(col_data)
        
        stats.append(f"### {col} Statistics:\n")
        stats.append(f"  - **Mean**: {mean:.2f} (The average value)")
        stats.append(f"  - **Median**: {median:.2f} (The middle value when sorted)")
        stats.append(f"  - **Standard Deviation**: {std_dev:.2f} (How spread out the values are)")
        stats.append(f"  - **Skewness**: {skewness:.2f} (How asymmetric the data is)")
        stats.append(f"  - **Kurtosis**: {kurt:.2f} (How peaked the data distribution is)")

        # Interpretation of skewness and kurtosis
        if skewness > 0:
            stats.append(f"  - **Skewness Interpretation**: The data is positively skewed (tail on the right).")
        elif skewness < 0:
            stats.append(f"  - **Skewness Interpretation**: The data is negatively skewed (tail on the left).")
        else:
            stats.append(f"  - **Skewness Interpretation**: The data is symmetrical.")

        if kurt > 3:
            stats.append(f"  - **Kurtosis Interpretation**: The data has heavy tails (leptokurtic).")
        elif kurt < 3:
            stats.append(f"  - **Kurtosis Interpretation**: The data is light-tailed (platykurtic).")
        else:
            stats.append(f"  - **Kurtosis Interpretation**: The data has a normal peak (mesokurtic).")
    
    return "\n".join(stats)

def data_visualizations(df):
    """Generates and returns visualizations for the dataset."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(df[numeric_cols[0]].dropna(), kde=True, ax=ax[0])
        ax[0].set_title(f'Distribution of {numeric_cols[0]}')
        sns.boxplot(data=df[numeric_cols[0]].dropna(), ax=ax[1])
        ax[1].set_title(f'Boxplot of {numeric_cols[0]}')
        st.pyplot(fig)
    else:
        st.write("There are no numeric columns available for visualization.")

# -----------------------------
# Enhanced Answer Mapping with Clarification
# -----------------------------
def ask_for_clarity(question, df):
    """Ask for clarification on the user's question."""
    if "missing" in question or "outlier" in question or "scaling" in question:
        return f"Did you mean to ask about **{', '.join([word for word in ['missing', 'outlier', 'scaling'] if word in question])}**? I can provide detailed information on this."

    # Clarifying common terms
    elif "data preprocessing" in question:
        return "Are you referring to preprocessing steps like missing value handling, outlier detection, or scaling?"

    return "I'm not sure I understand. Could you clarify your question? Are you asking about a specific aspect of the data or a method?"

def answer_question(question, df=None):
    """Answers the question based on the provided DataFrame and system knowledge."""
    q_lower = question.lower()

    # Handle yes/no clarification
    if "yes" in q_lower:
        return "Great! Let me provide you with the relevant information."

    elif "no" in q_lower:
        return "Okay, I will need more details to help you with that. Could you clarify your question?"

    # Clarifying ambiguous questions
    if any(keyword in q_lower for keyword in ['missing', 'outlier', 'scaling', 'data science', 'machine learning']):
        return ask_for_clarity(question, df)

    # Answering general Data Science questions (without needing data)
    if "data science" in q_lower:
        return "Data science involves extracting insights and knowledge from structured and unstructured data. It combines techniques from statistics, machine learning, and data mining to analyze and interpret complex data."

    elif "machine learning" in q_lower:
        return "Machine learning is a field of AI that focuses on building algorithms that allow computers to learn from data and make predictions or decisions based on that data."

    elif "statistics" in q_lower:
        return "Statistics is the science of collecting, analyzing, and interpreting data. Key concepts include mean, median, variance, correlation, and hypothesis testing."

    elif "outliers" in q_lower:
        return "Outliers are extreme values that differ significantly from other observations in a dataset. They can be detected using statistical methods such as the IQR method or Z-scores."

    elif "missing values" in q_lower:
        return "Missing values occur when no data is recorded for a particular feature. Handling missing data can be done through methods like imputation (mean, median, mode) or deletion (drop rows/columns)."

    elif "scaling" in q_lower:
        return "Scaling refers to the process of adjusting feature values to a standard range, often necessary for algorithms that are sensitive to feature magnitudes, such as gradient descent or distance-based models."

    elif "data preprocessing" in q_lower:
        return "Data preprocessing involves cleaning and transforming raw data into a format suitable for analysis. Common steps include handling missing values, encoding categorical data, scaling features, and detecting outliers."

    # Handle specific dataset questions (only if data is available)
    if df is not None:
        if "dataset summary" in q_lower:
            return dataset_summary(df)
        elif "describe" in q_lower or "statistics" in q_lower or "statistical summary" in q_lower:
            return detailed_stats(df)

    # Fallback response
    return "Hmm, I couldn't quite catch that. Could you ask something related to the data or the system's functionality?"

# -----------------------------
# Main Streamlit Page
# -----------------------------
def main():
    st.set_page_config(page_title="KlinItAll AI Data Assistant", page_icon="🤖")
    st.title("🤖 KlinItAll AI-Powered Data Assistant")

    df = st.session_state.get("current_dataset", None)
    if df is None:
        st.warning("Please upload a dataset first! 🤖💡")
        return
    st.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask me anything about your data or the system..."):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the response
        response = answer_question(prompt, df)

        # Display assistant response in chat message container
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
