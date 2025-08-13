import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Helper Functions
# -----------------------------
def dataset_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict()
    }

def missing_info(df):
    info = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            if np.issubdtype(df[col].dtype, np.number):
                suggestion = "Impute with mean/median or model-based imputation"
            elif np.issubdtype(df[col].dtype, np.object_):
                suggestion = "Impute with mode or semantic text imputation"
            else:
                suggestion = "Impute appropriately based on type"
            info[col] = {"missing": missing_count, "suggestion": suggestion}
    return info

def outlier_info(df):
    info = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)]
        if not outliers.empty:
            info[col] = {
                "num_outliers": outliers.shape[0],
                "action": "Consider capping, winsorizing, or removing"
            }
    return info

def scaling_advice(df):
    advice = {}
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        skew = df[col].skew()
        if abs(skew) > 1:
            advice[col] = f"Skew={skew:.2f}: Consider log or power transform"
        else:
            advice[col] = f"Skew={skew:.2f}: Standard scaling may be sufficient"
    return advice

def encoding_advice(df):
    advice = {}
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique < 10:
            advice[col] = f"One-hot encoding recommended (unique={n_unique})"
        else:
            advice[col] = f"Consider target/frequency encoding (unique={n_unique})"
    return advice

def text_cleaning_advice(df):
    advice = {}
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        has_special = df[col].str.contains(r"\W").any()
        advice[col] = "Contains special characters or emojis" if has_special else "Text appears clean"
    return advice

def feature_engineering_advice(df):
    advice = {}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            advice[f"{col1}*{col2}"] = "Consider creating interaction feature"
    return advice

# -----------------------------
# Page Mapping
# -----------------------------
page_links = {
    "dataset summary": "Data Overview",
    "missing": "Missing Values",
    "outlier": "Outlier Detection",
    "scale": "Scaling Normalization",
    "categorical": "Categorical Encoding",
    "encoding": "Categorical Encoding",
    "text": "Text Cleaning",
    "clean": "Text Cleaning",
    "feature engineering": "Feature Engineering",
    "total sales": "Data Overview",
    "unique regions": "Data Overview",
    "unique cities": "Data Overview"
}

# Suggested Questions
def suggested_questions(df):
    questions = [
        "Show dataset summary (rows, columns, column names, data types)",
        "Show missing value summary and imputation suggestions",
        "Show numeric outliers and handling suggestions",
        "Which columns should be scaled or normalized?",
        "Which categorical columns need encoding?",
        "Which text columns need cleaning?",
        "Feature engineering suggestions for numeric columns"
    ]
    lower_cols = [c.lower() for c in df.columns]
    if any("sales" in c for c in lower_cols):
        questions.append("Total sales in the dataset?")
    if any("region" in c for c in lower_cols):
        questions.append("Number of unique regions?")
    if any("city" in c for c in lower_cols):
        questions.append("Number of unique cities?")
    return questions

# -----------------------------
# Answer Mapping
# -----------------------------
def answer_question(question, df):
    q_lower = question.lower()
    answer = ""
    page = "Unknown"
    if "dataset summary" in q_lower:
        answer = dataset_summary(df)
        page = page_links["dataset summary"]
    elif "missing" in q_lower:
        answer = missing_info(df)
        page = page_links["missing"]
    elif "outlier" in q_lower:
        answer = outlier_info(df)
        page = page_links["outlier"]
    elif "scale" in q_lower or "normalize" in q_lower:
        answer = scaling_advice(df)
        page = page_links["scale"]
    elif "categorical" in q_lower or "encoding" in q_lower:
        answer = encoding_advice(df)
        page = page_links["encoding"]
    elif "text" in q_lower or "clean" in q_lower:
        answer = text_cleaning_advice(df)
        page = page_links["text"]
    elif "feature engineering" in q_lower:
        answer = feature_engineering_advice(df)
        page = page_links["feature engineering"]
    elif "total sales" in q_lower:
        for col in df.columns:
            if "sales" in col.lower() and np.issubdtype(df[col].dtype, np.number):
                answer = df[col].sum()
        page = page_links["total sales"]
        if not answer:
            answer = "No sales column found."
    elif "unique regions" in q_lower:
        if "Region" in df.columns:
            answer = df["Region"].nunique()
        else:
            answer = "No 'Region' column found."
        page = page_links["unique regions"]
    elif "unique cities" in q_lower:
        if "City" in df.columns:
            answer = df["City"].nunique()
        else:
            answer = "No 'City' column found."
        page = page_links["unique cities"]
    else:
        answer = "Sorry, I cannot answer that question directly."
    return answer, page

# -----------------------------
# Main Streamlit Page
# -----------------------------
def main():
    st.title("🤖 KlinItAll AI-Powered Data Assistant with Clickable Page Links")
    df = st.session_state.get("current_dataset", None)
    if df is None:
        st.warning("Please upload a dataset first!")
        return
    st.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Suggested Questions
    st.subheader("💡 Suggested Questions")
    sel_q = st.selectbox("Select a question:", options=suggested_questions(df))
    if st.button("Ask Suggested Question"):
        answer, page = answer_question(sel_q, df)
        st.write(answer)
        # clickable link using query params
        st.markdown(f"[🔗 Go to Related Page: {page}](?page={page.replace(' ', '_')})")

    # Free Text Questions
    st.subheader("✏️ Ask a Custom Question")
    custom_q = st.text_input("Type your question here...")
    if st.button("Ask Custom Question"):
        if custom_q.strip():
            answer, page = answer_question(custom_q, df)
            st.write(answer)
            st.markdown(f"[🔗 Go to Related Page: {page}](?page={page.replace(' ', '_')})")
        else:
            st.warning("Please enter a valid question.")

if __name__ == "__main__":
    main()
