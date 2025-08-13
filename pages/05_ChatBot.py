import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import re  # For more robust text cleaning

# -----------------------------
# Helper Functions
# -----------------------------

def dataset_summary(df):
    """Provides a comprehensive summary of the dataset."""
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "duplicate_rows": df.duplicated().sum()
    }
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        summary["descriptive_stats"] = df[numeric_cols].describe().to_dict()
    else:
        summary["descriptive_stats"] = "No numerical columns found."
    categorical_cols = df.select_dtypes(include="object").columns
    value_counts = {}
    for col in categorical_cols:
        value_counts[col] = df[col].value_counts(dropna=False).head(5).to_dict()
    summary["value_counts"] = value_counts
    return summary

def missing_info(df):
    """Analyzes missing values and suggests imputation strategies."""
    info = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_percentage = (missing_count / len(df)) * 100
            data_type = df[col].dtype
            if np.issubdtype(data_type, np.number):
                if missing_percentage < 5:
                    suggestion = "Impute with mean/median. Consider KNN imputation if data is related."
                elif missing_percentage < 20:
                    suggestion = "Impute with median or KNN imputation. Model-based imputation is also an option."
                else:
                    suggestion = "Model-based imputation or multiple imputation. Consider missingness informativeness."
            elif np.issubdtype(data_type, np.object_):
                if missing_percentage < 5:
                    suggestion = "Impute with mode. Consider creating a 'Missing' category."
                elif missing_percentage < 20:
                    suggestion = "Impute with mode or predict the missing category."
                else:
                    suggestion = "Consider missingness informativeness. Create a 'Missing' category or use model-based imputation."
            else:
                suggestion = "Impute appropriately based on the specific data type. Investigate the nature of missingness."
            if missing_percentage > 50:
                missingness_note = "High percentage of missing values. Consider dropping or advanced imputation."
            else:
                missingness_note = ""
            info[col] = {
                "missing": missing_count,
                "percentage": f"{missing_percentage:.2f}%",
                "suggestion": suggestion,
                "missingness_note": missingness_note
            }
    return info

def outlier_info(df):
    """Identifies outliers using the IQR method and suggests handling strategies."""
    info = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        num_outliers = outliers.shape[0]
        outlier_percentage = (num_outliers / len(df)) * 100
        if num_outliers > 0:
            if outlier_percentage < 1:
                action = "Consider trimming or winsorizing. Evaluate the impact on the distribution."
            elif outlier_percentage < 5:
                action = "Winsorizing or capping is recommended. Transform if skewed."
            else:
                action = "Investigate carefully. Consider genuine data points or errors. Robust methods may be necessary."
            info[col] = {
                "num_outliers": num_outliers,
                "percentage": f"{outlier_percentage:.2f}%",
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "action": action
            }
    return info

def scaling_advice(df):
    """Provides advice on scaling and normalization techniques."""
    advice = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        data = df[col].dropna()
        skewness = skew(data)
        abs_skew = abs(skewness)
        if abs_skew > 1:
            if all(data > 0):
                transform = "Log transformation (if all values are positive) or Box-Cox transformation."
            else:
                transform = "Box-Cox or Yeo-Johnson transformation."
            scaler = "After transformation, consider StandardScaler or MinMaxScaler."
            advice[col] = f"Skew={skewness:.2f}: {transform} {scaler}"
        elif abs_skew > 0.5:
            transform = "Consider a square root or Box-Cox/Yeo-Johnson transformation."
            scaler = "StandardScaler or MinMaxScaler can be used after transformation."
            advice[col] = f"Skew={skewness:.2f}: {transform} {scaler}"
        else:
            scaler = "StandardScaler is generally suitable. MinMaxScaler can be used if a specific range is required."
            advice[col] = f"Skew={skewness:.2f}: {scaler}"
    return advice

def encoding_advice(df):
    """Provides advice on encoding categorical features."""
    advice = {}
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique < 5:
            advice[col] = f"One-hot encoding is recommended. Consider ordinal encoding if categories have a natural order."
        elif n_unique < 15:
            advice[col] = f"One-hot encoding is feasible. Consider target encoding or frequency encoding."
        else:
            advice[col] = f"Target encoding, frequency encoding, or WOE encoding are recommended. Be mindful of overfitting."
    return advice

def text_cleaning_advice(df):
    """Provides advice on cleaning text columns."""
    advice = {}
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        has_special = df[col].str.contains(r"[^a-zA-Z0-9\s]", regex=True).any()
        has_encoding_issues = df[col].str.contains(r"[\u0080-\uffff]").any()
        if has_special:
            advice[col] = "Contains special characters. Remove or replace them. Consider stemming/lemmatization."
        elif has_encoding_issues:
            advice[col] = "Potential encoding issues detected. Ensure consistent UTF-8 encoding."
        else:
            advice[col] = "Text appears clean. Consider lowercasing and removing punctuation."
        advice[col] += " General steps: Lowercasing, punctuation removal, stop word removal, stemming/lemmatization."
    return advice

def feature_engineering_advice(df):
    """Suggests potential feature engineering opportunities."""
    advice = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            advice[f"{col1}*{col2}"] = f"Consider interaction feature between {col1} and {col2}."
            advice[f"{col1}/{col2}"] = f"Consider ratio feature between {col1} and {col2} (if appropriate). Handle division by zero."
    for col in datetime_cols:
        advice[f"Extract features from {col}"] = f"Extract year, month, day, day of week, and hour from {col}."
    if len(numeric_cols) > 0:
        advice["Polynomial Features"] = "Consider creating polynomial features for numerical columns."
    return advice

# -----------------------------
# Answer Mapping
# -----------------------------
def answer_question(question, df):
    """Answers the question based on the provided DataFrame."""
    q_lower = question.lower()
    if "dataset summary" in q_lower:
        return dataset_summary(df)
    elif "missing" in q_lower:
        return missing_info(df)
    elif "outlier" in q_lower:
        return outlier_info(df)
    elif "scale" in q_lower or "normalize" in q_lower:
        return scaling_advice(df)
    elif "categorical" in q_lower or "encoding" in q_lower:
        return encoding_advice(df)
    elif "text" in q_lower or "clean" in q_lower:
        return text_cleaning_advice(df)
    elif "feature engineering" in q_lower:
        return feature_engineering_advice(df)
    elif "total sales" in q_lower:
        sales_column = next((col for col in df.columns if "sales" in col.lower() and np.issubdtype(df[col].dtype, np.number)), None)
        if sales_column:
            return f"Total sales: {df[sales_column].sum():.2f}"
        else:
            return "No sales column found."
    elif "unique regions" in q_lower:
        region_column = next((col for col in df.columns if "region" in col.lower()), None)
        if region_column:
            return f"Number of unique regions: {df[region_column].nunique()}"
        else:
            return "No 'Region' column found."
    elif "unique cities" in q_lower:
        city_column = next((col for col in df.columns if "city" in col.lower()), None)
        if city_column:
            return f"Number of unique cities: {df[city_column].nunique()}"
        else:
            return "No 'City' column found."
    else:
        return "Sorry, I cannot answer that question directly."

# -----------------------------
# Main Streamlit Page
# -----------------------------
def main():
    st.set_page_config(page_title="KlinItAll AI Data Assistant", page_icon="🤖")
    st.title("🤖 KlinItAll AI-Powered Data Assistant")

    df = st.session_state.get("current_dataset", None)
    if df is None:
        st.warning("Please upload a dataset first!")
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
    if prompt := st.chat_input("Ask me anything about your data..."):
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
