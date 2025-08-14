import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re  # For more robust text cleaning
import nltk  # For NLP tasks
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # For sentiment analysis
from nltk.corpus import stopwords  # For stopword removal
from nltk.tokenize import word_tokenize  # For tokenization

# Download necessary NLTK data (only needs to be done once)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')

# -----------------------------
# System Knowledge Database (Data Science Topics)
# -----------------------------

system_knowledge = {
    "general": {
        "about_system": "KlinIt-Bot is an advanced AI-powered data assistant designed to help you with all aspects of data science, from understanding basic concepts to performing complex data preprocessing and analysis tasks. I can answer questions, provide explanations, generate visualizations, and even suggest code snippets to help you work with your data more effectively.",
        "how_it_works": "I use a combination of natural language processing (NLP) and a comprehensive knowledge base to understand your questions and provide relevant answers. If you upload a dataset, I can also analyze it and provide insights specific to your data.",
        "core_features": [
            "Answering questions about data science, machine learning, and AI",
            "Providing explanations of data preprocessing techniques",
            "Generating visualizations of data",
            "Suggesting code snippets for data manipulation",
            "Analyzing uploaded datasets and providing insights",
            "Performing sentiment analysis on text data",
            "Identifying and suggesting ways to handle missing values and outliers",
            "Recommending appropriate feature scaling and encoding methods",
            "Suggesting feature engineering opportunities"
        ],
        "suggested_questions": [
            "What is data science?",
            "How do I handle missing values in my data?",
            "What are some common data preprocessing techniques?",
            "Can you give me a summary of my dataset?",
            "What are some good feature engineering ideas?",
            "How does reinforcement learning work?",
            "What are generative models used for?",
            "Explain the concept of transfer learning.",
            "What is the difference between CNNs and RNNs?",
            "How do transformers work in NLP?"
        ]
    },
    "data_preprocessing": {
        "missing_values": {
            "description": "Missing values occur when some data points are not available or recorded. I can help you identify columns with missing values and suggest appropriate imputation strategies.",
            "methods": [
                "Dropping rows/columns with excessive missing values (use with caution!).",
                "Imputing with the mean, median, or mode (simple but can introduce bias).",
                "Using KNN imputation (more sophisticated, considers relationships between features).",
                "Employing model-based imputation (e.g., using a regression model to predict missing values)."
            ]
        },
        "outliers": {
            "description": "Outliers are extreme data points that deviate significantly from other observations. I can help you detect and handle outliers using various methods.",
            "methods": [
                "Trimming outliers (removing rows with extreme values).",
                "Winsorizing (capping extreme values to a specified percentile).",
                "Transforming the data (e.g., using a log transformation to reduce the impact of outliers).",
                "Using robust statistical methods (less sensitive to outliers)."
            ]
        },
        "scaling": {
            "description": "Scaling involves adjusting feature values to a similar range. This is often necessary for algorithms sensitive to feature magnitudes.",
            "methods": [
                "StandardScaler (z-score normalization): Centers the data around zero with unit variance.",
                "MinMaxScaler: Scales data to a specific range (e.g., [0, 1]).",
                "RobustScaler: Uses median and interquartile range, making it robust to outliers.",
                "PowerTransformer (Yeo-Johnson, Box-Cox): Handles skewed data by applying a power transformation."
            ]
        },
        "encoding": {
            "description": "Encoding transforms categorical variables into a numerical format suitable for machine learning models. I can suggest appropriate encoding methods based on the characteristics of your categorical features.",
            "methods": [
                "One-hot encoding: Creates binary columns for each category (suitable for low-cardinality features).",
                "Label encoding: Assigns a unique integer to each category (suitable for ordinal features).",
                "Frequency encoding: Replaces categories with their frequency in the dataset.",
                "Target encoding: Replaces categories with the mean target value for that category (be careful of overfitting!)."
            ]
        },
        "feature_engineering": {
            "description": "Feature engineering is the art of creating new features or transforming existing ones to improve model performance. I can suggest various feature engineering techniques based on your data.",
            "methods": [
                "Binning: Grouping continuous values into discrete bins.",
                "Interaction features: Creating new features by combining existing ones (e.g., multiplication, division).",
                "Polynomial features: Creating new features by raising existing ones to a power (e.g., squaring, cubing).",
                "Date/time features: Extracting meaningful information from date/time columns (e.g., year, month, day of week)."
            ]
        }
    },
    "statistics": {
        "description": "Statistics is the foundation of data science. I can help you understand key statistical concepts and apply them to your data.",
        "core_concepts": [
            "Mean, Median, Mode",
            "Variance, Standard Deviation",
            "Skewness, Kurtosis",
            "Correlation, Regression",
            "Hypothesis Testing"
        ],
        "common_questions": {
            "mean": "The **mean** is the average of all values in a dataset. It's sensitive to outliers.",
            "variance": "The **variance** measures the spread of data points around the mean. A higher variance indicates greater variability.",
            "skewness": "Skewness measures the asymmetry of the data distribution. Positive skewness indicates a longer tail on the right, while negative skewness indicates a longer tail on the left.",
            "correlation": "Correlation measures the strength and direction of the linear relationship between two variables. Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).",
            "regression": "Regression is used to model the relationship between a dependent variable and one or more independent variables. It's commonly used for prediction."
        }
    },
    "advanced_ml_ai": {
        "reinforcement_learning": {
            "description": "Reinforcement Learning (RL) involves training agents to make decisions in an environment to maximize a reward. Techniques like Q-Learning and Deep Q Networks (DQN) are used."
        },
        "generative_models": {
            "description": "Generative models, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), learn the underlying distribution of data and can generate new samples that resemble the original data."
        },
        "transfer_learning": {
            "description": "Transfer learning leverages knowledge gained from solving one problem to solve a different but related problem. This can significantly reduce training time and improve performance, especially when data is limited."
        },
        "meta_learning": {
            "description": "Meta-learning, or 'learning to learn,' focuses on developing algorithms that can quickly adapt to new tasks with limited data. It aims to improve the learning process itself."
        },
        "deep_learning": {
            "description": "Deep learning uses neural networks with multiple layers to learn complex patterns from data. Common architectures include Convolutional Neural Networks (CNNs) for images and Recurrent Neural Networks (RNNs) for sequences."
        },
        "transformers": {
            "description": "Transformers are a powerful type of neural network architecture that has revolutionized natural language processing (NLP). They use attention mechanisms to process sequential data and have achieved state-of-the-art results on various NLP tasks."
        },
        "cnn": {
            "description": "Convolutional Neural Networks (CNNs) are particularly well-suited for image-related tasks. They use convolutional layers to automatically learn spatial hierarchies of features from images."
        },
        "gan": {
            "description": "Generative Adversarial Networks (GANs) consist of two neural networks, a generator and a discriminator, that compete against each other. The generator tries to create realistic data, while the discriminator tries to distinguish between real and generated data."
        },
        "nlp": {
            "description": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language. Techniques like sentiment analysis, machine translation, and text summarization fall under NLP."
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

def perform_sentiment_analysis(text):
    """Performs sentiment analysis on the given text using VADER."""
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    return scores

def remove_stopwords(text):
    """Removes stopwords from the given text."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

# -----------------------------
# Enhanced Answer Mapping with NLP and Knowledge Access
# -----------------------------

def answer_question(question, df=None):
    """Answers the question based on the provided DataFrame and system knowledge."""
    q_lower = question.lower().strip()

    # --- Friendly Greetings and Introductions ---
    if any(greet in q_lower for greet in ["hello", "hi", "hey", "you"]):
        introduction = "Hey there! 👋 I'm KlinItAll, your AI-powered data assistant. I can help you with a wide range of data science tasks, including:\n\n"
        introduction += "- Answering questions about data science, machine learning, and AI\n"
        introduction += "- Providing explanations of data preprocessing techniques\n"
        introduction += "- Generating visualizations of data (if you upload a dataset)\n"
        introduction += "- Suggesting code snippets for data manipulation\n"
        introduction += "- Analyzing uploaded datasets and providing insights\n"
        introduction += "- Performing sentiment analysis on text data\n\n"
        introduction += "To get started, try asking me a question or uploading a dataset. Here are some suggested questions:\n\n"
        introduction += "\n".join([f"- {q}" for q in system_knowledge["general"]["suggested_questions"]])
        return introduction

    # --- General system info ---
    if "what is" in q_lower:
        return "Ah, looks like you're asking for an explanation. Could you clarify which topic you're referring to? Data science, data preprocessing, or maybe something else?"

    # --- Data Science / Machine Learning ---
    if "data science" in q_lower:
        return system_knowledge["general"]["about_system"]

    if "machine learning" in q_lower:
        return "Machine learning allows systems to learn from data and make predictions or decisions. It’s a core part of AI."

    # --- Advanced AI & Machine Learning ---
    if "reinforcement learning" in q_lower:
        return system_knowledge["advanced_ml_ai"]["reinforcement_learning"]["description"]

    if "generative models" in q_lower:
        return system_knowledge["advanced_ml_ai"]["generative_models"]["description"]

    if "transfer learning" in q_lower:
        return system_knowledge["advanced_ml_ai"]["transfer_learning"]["description"]

    if "meta learning" in q_lower:
        return system_knowledge["advanced_ml_ai"]["meta_learning"]["description"]

    if "deep learning" in q_lower:
        return system_knowledge["advanced_ml_ai"]["deep_learning"]["description"]

    if "transformers" in q_lower:
        return system_knowledge["advanced_ml_ai"]["transformers"]["description"]

    # --- Data Preprocessing ---
    if "data preprocessing" in q_lower or "data cleaning" in q_lower:
        return "Data preprocessing is about preparing raw data for analysis. It includes cleaning, transforming, handling missing values, and encoding."

    if "missing values" in q_lower or "imputation" in q_lower:
        return system_knowledge["data_preprocessing"]["missing_values"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["missing_values"]["methods"]])

    if "outliers" in q_lower:
        return system_knowledge["data_preprocessing"]["outliers"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["outliers"]["methods"]])

    if "scaling" in q_lower:
        return system_knowledge["data_preprocessing"]["scaling"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["scaling"]["methods"]])

    # --- NLP Tasks ---
    if "sentiment analysis" in q_lower:
        if df is not None:
            text_cols = df.select_dtypes(include='object').columns
            if not text_cols.empty:
                # Perform sentiment analysis on the first text column
                first_text_col = text_cols[0]
                sentiment_scores = df[first_text_col].astype(str).apply(perform_sentiment_analysis)
                st.write(f"Sentiment analysis scores for the first text column ('{first_text_col}'):")
                st.write(sentiment_scores)
                return "I've performed sentiment analysis on the first text column of your dataset.  See the results above!"
            else:
                return "There are no text columns in your dataset to perform sentiment analysis on."
        else:
            return "Sentiment analysis is a technique used to determine the emotional tone of a piece of text.  Please upload a dataset with text columns to perform sentiment analysis."

    if "remove stopwords" in q_lower:
        if df is not None:
            text_cols = df.select_dtypes(include='object').columns
            if not text_cols.empty:
                # Remove stopwords from the first text column
                first_text_col = text_cols[0]
                df['cleaned_text'] = df[first_text_col].astype(str).apply(remove_stopwords)
                st.write(f"Stopwords removed from the first text column ('{first_text_col}'):")
                st.dataframe(df[['cleaned_text']].head())
                return "I've removed stopwords from the first text column of your dataset.  See the cleaned text above!"
            else:
                return "There are no text columns in your dataset to remove stopwords from."
        else:
            return "Stopwords are common words (e.g., 'the', 'a', 'is') that are often removed from text to improve NLP tasks.  Please upload a dataset with text columns to remove stopwords."

    # --- Clarifying Vague Queries ---
    if any(keyword in q_lower for keyword in ["data", "value", "row", "column", "shape"]):
        if df is not None:
            return dataset_summary(df)
        else:
            return "You don’t have a dataset loaded yet, but I can explain terms like rows, columns, or values. For example, rows are records, and columns are features."

    # --- Dataset-Specific Queries ---
    if df is not None:
        if "describe" in q_lower or "statistics" in q_lower or "summary" in q_lower:
            return detailed_stats(df)
        if "visualization" in q_lower or "plot" in q_lower:
            data_visualizations(df)
            return "Here’s a visualization of your dataset. Does this help clarify things?"
        if "shape" in q_lower:
            return f"Your dataset has {df.shape[0]} rows and {df.shape[1]} columns."

    # --- General System Information ---
    if "how it works" in q_lower or "about" in q_lower:
        return system_knowledge["general"]["how_it_works"]

    # --- Unrecognized Input with Friendly Tone ---
    return "Hmm, I didn’t quite catch that. Could you clarify or ask something specific related to data science, machine learning, or data preprocessing? I’m here to help! 😄"

# -----------------------------
# Main Streamlit Page
# -----------------------------

def main():
    st.set_page_config(page_title="KlinIt-Bot AI Data Assistant", page_icon="🤖")
    st.title("🤖 KlinIt-Bot AI-Powered Data Assistant")
    st.markdown("Hi there! I'm KlinIt-Bot, your data mentor. Ask me anything about data science, machine learning, or data preprocessing! ✨")

    # --- Dataset Loading (Optional) ---
    df = st.session_state.get("current_dataset", None)
    # Removed file uploader

    # --- Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display Chat Messages ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input and Response ---
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = answer_question(prompt, df)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
