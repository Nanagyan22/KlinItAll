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
import requests  # For fetching data from the internet
from bs4 import BeautifulSoup  # For web scraping
import random  # For varied responses
import time  # For simulating human-like delays

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
        "about_system": "KlinItAll is your AI-powered data science assistant, designed to provide information, guidance, and support throughout your data journey. I leverage NLP, a comprehensive knowledge base, and real-time web access to assist you.",
        "how_it_works": "My architecture combines natural language understanding (NLU) to interpret your queries, a structured knowledge base for core concepts, and web scraping capabilities for up-to-date information. This allows me to provide contextually relevant and informative responses.",
        "core_features": [
            "Answering questions on data science, ML, and AI",
            "Explaining data preprocessing techniques",
            "Generating data visualizations (with data input)",
            "Suggesting code snippets for data manipulation",
            "Analyzing datasets and providing insights",
            "Performing sentiment analysis on text data",
            "Fetching real-time information from the web",
            "Adapting response style based on user input"
        ],
        "suggested_questions": [
            "What are the fundamental principles of data science?",
            "How can I effectively address missing data in my dataset?",
            "What are the key considerations when choosing a data preprocessing technique?",
            "Can you provide a statistical overview of my dataset?",
            "What are some innovative approaches to feature engineering?",
            "Explain the underlying mechanisms of reinforcement learning.",
            "What are the applications of generative models in data science?",
            "How does transfer learning improve model efficiency?",
            "Compare and contrast CNNs and RNNs in the context of deep learning.",
            "What are the advantages of using transformers in NLP tasks?"
        ],
        "follow_up_prompts": [
            "Would you like a more detailed explanation of that concept?",
            "Are there any specific aspects of this topic you'd like to explore further?",
            "Would you benefit from a practical code example?",
            "Shall we consider alternative approaches to this problem?",
            "Are you ready to move on to the next stage of your data analysis?"
        ],
        "acknowledgement_phrases": [
            "Understood.",
            "Acknowledged.",
            "Noted.",
            "Affirmative.",
            "Duly noted."
        ]
    },
    "data_preprocessing": {
        "missing_values": {
            "description": "Missing data can introduce bias and reduce the statistical power of your analysis. I can guide you through various strategies for handling missing values.",
            "methods": [
                "Deletion (removing rows or columns with missing data).",
                "Imputation with statistical measures (mean, median, mode).",
                "K-Nearest Neighbors (KNN) imputation.",
                "Model-based imputation (e.g., regression models)."
            ]
        },
        "outliers": {
            "description": "Outliers can distort statistical analyses and negatively impact model performance. I can assist you in identifying and mitigating the effects of outliers.",
            "methods": [
                "Trimming (removing data points beyond a specified threshold).",
                "Winsorizing (capping extreme values at a predefined percentile).",
                "Data transformation (e.g., logarithmic transformation).",
                "Robust statistical methods (less sensitive to outliers)."
            ]
        },
        "scaling": {
            "description": "Scaling ensures that all features contribute equally to the analysis, preventing features with larger magnitudes from dominating the results.",
            "methods": [
                "StandardScaler (z-score normalization).",
                "MinMaxScaler (scaling to a specific range, e.g., [0, 1]).",
                "RobustScaler (using median and interquartile range).",
                "PowerTransformer (Yeo-Johnson, Box-Cox transformations)."
            ]
        },
        "encoding": {
            "description": "Encoding converts categorical variables into a numerical format suitable for machine learning algorithms. The choice of encoding method depends on the nature of the categorical variable.",
            "methods": [
                "One-hot encoding (creating binary columns for each category).",
                "Label encoding (assigning a unique integer to each category).",
                "Frequency encoding (replacing categories with their frequency).",
                "Target encoding (replacing categories with the mean target value)."
            ]
        },
        "feature_engineering": {
            "description": "Feature engineering involves creating new features or transforming existing ones to improve model accuracy and interpretability.",
            "methods": [
                "Binning (grouping continuous values into discrete intervals).",
                "Interaction features (combining two or more features).",
                "Polynomial features (creating higher-order terms of existing features).",
                "Date/time feature extraction (extracting relevant information from date/time variables)."
            ]
        }
    },
    "statistics": {
        "description": "Statistical analysis provides a framework for understanding and interpreting data. I can assist you in applying statistical methods to your data.",
        "core_concepts": [
            "Measures of central tendency (mean, median, mode)",
            "Measures of dispersion (variance, standard deviation)",
            "Skewness and kurtosis",
            "Correlation and regression analysis",
            "Hypothesis testing"
        ],
        "common_questions": {
            "mean": "The **mean** represents the average value of a dataset. It is sensitive to extreme values.",
            "variance": "The **variance** quantifies the spread of data points around the mean.",
            "skewness": "Skewness measures the asymmetry of the data distribution. Positive skewness indicates a longer tail on the right, while negative skewness indicates a longer tail on the left.",
            "correlation": "Correlation measures the strength and direction of the linear relationship between two variables. Values range from -1 to +1.",
            "regression": "Regression analysis models the relationship between a dependent variable and one or more independent variables."
        }
    },
    "advanced_ml_ai": {
        "reinforcement_learning": {
            "description": "Reinforcement learning (RL) is a type of machine learning where an agent learns to make decisions in an environment to maximize a reward signal."
        },
        "generative_models": {
            "description": "Generative models, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), learn the underlying distribution of data and can generate new samples that resemble the original data."
        },
        "transfer_learning": {
            "description": "Transfer learning leverages knowledge gained from solving one problem to solve a different but related problem, improving efficiency and performance."
        },
        "meta_learning": {
            "description": "Meta-learning, or 'learning to learn,' focuses on developing algorithms that can quickly adapt to new tasks with limited data."
        },
        "deep_learning": {
            "description": "Deep learning utilizes artificial neural networks with multiple layers to extract complex patterns from data."
        },
        "transformers": {
            "description": "Transformers are a powerful neural network architecture that has revolutionized natural language processing (NLP) tasks."
        },
        "cnn": {
            "description": "Convolutional Neural Networks (CNNs) are particularly effective for image-related tasks, automatically learning spatial hierarchies of features."
        },
        "gan": {
            "description": "Generative Adversarial Networks (GANs) consist of two neural networks, a generator and a discriminator, that compete against each other to generate realistic data."
        },
        "nlp": {
            "description": "Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language."
        }
    }
}

# -----------------------------
# Helper Functions for Data Operations
# -----------------------------

def dataset_summary(df):
    """Provides a comprehensive summary of the dataset."""
    summary = f"{random.choice(system_knowledge['general']['acknowledgement_phrases'])}. Here's a summary of your dataset:\n\n"
    summary += f"The dataset comprises **{df.shape[0]} rows** and **{df.shape[1]} columns**."
    summary += "\n\nThe column names are as follows:\n" + ", ".join(df.columns)
    return summary

def detailed_stats(df):
    """Provides detailed descriptive statistics and insights on numerical columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        return "It appears that there are no numerical columns in your dataset, precluding the calculation of descriptive statistics."

    stats = ["Here's a detailed statistical overview of your numerical data:\n"]
    for col in numeric_cols:
        col_data = df[col].dropna()
        mean = col_data.mean()
        median = col_data.median()
        std_dev = col_data.std()
        skewness = skew(col_data)
        kurt = kurtosis(col_data)
        
        stats.append(f"### {col} Statistics:\n")
        stats.append(f"  - **Mean**: {mean:.2f} (The arithmetic average of the values)")
        stats.append(f"  - **Median**: {median:.2f} (The central value when the data is ordered)")
        stats.append(f"  - **Standard Deviation**: {std_dev:.2f} (A measure of the data's dispersion around the mean)")
        stats.append(f"  - **Skewness**: {skewness:.2f} (A measure of the asymmetry of the data distribution)")
        stats.append(f"  - **Kurtosis**: {kurt:.2f} (A measure of the peakedness of the data distribution)")

        # Interpretation of skewness and kurtosis
        if skewness > 0:
            stats.append(f"  - **Skewness Interpretation**: The data exhibits positive skewness, indicating a longer tail on the right.")
        elif skewness < 0:
            stats.append(f"  - **Skewness Interpretation**: The data exhibits negative skewness, indicating a longer tail on the left.")
        else:
            stats.append(f"  - **Skewness Interpretation**: The data distribution is approximately symmetrical.")

        if kurt > 3:
            stats.append(f"  - **Kurtosis Interpretation**: The data distribution is leptokurtic, characterized by heavy tails and a sharper peak.")
        elif kurt < 3:
            stats.append(f"  - **Kurtosis Interpretation**: The data distribution is platykurtic, characterized by lighter tails and a flatter peak.")
        else:
            stats.append(f"  - **Kurtosis Interpretation**: The data distribution is mesokurtic, resembling a normal distribution.")
    
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

def get_realtime_info(query):
    """Fetches information from the internet using web scraping."""
    try:
        search_results = []
        search_url = f"https://www.google.com/search?q={query}"
        response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.text, "html.parser")
        result_stats = soup.find("div", {"id": "result-stats"})
        if result_stats:
            search_results.append(result_stats.text)

        # Extract snippets from search results
        for g in soup.find_all('div', class_='g'):
            anchors = g.find_all('a')
            if anchors:
                link = anchors[0]['href']
                title = g.find('h3').text if g.find('h3') else "No title"
                snippet = g.find('span', class_='st').text if g.find('span', class_='st') else "No snippet"
                search_results.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n")

        if search_results:
            return "\n".join(search_results[:3])  # Limit to top 3 results
        else:
            return "I was unable to retrieve any relevant information from the web regarding that query."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while attempting to access the internet: {e}"
    except Exception as e:
        return f"An unexpected error occurred during the information retrieval process: {e}"

def generate_human_like_response(response_text):
    """Simulates a human-like response by adding pauses and varied sentence structure."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response_text)  # Split into sentences
    modified_sentences = []
    for sentence in sentences:
        # Add slight pauses between sentences
        time.sleep(random.uniform(0.2, 0.8))
        # Add interjections or transitional phrases
        if random.random() < 0.3:
            interjections = ["Indeed,", "Furthermore,", "Moreover,", "In addition,", "As a matter of fact,"]
            sentence = random.choice(interjections) + " " + sentence
        modified_sentences.append(sentence)
    return " ".join(modified_sentences)

# -----------------------------
# Enhanced Answer Mapping with NLP and Knowledge Access
# -----------------------------

def answer_question(question, df=None, conversation_history=None):
    """Answers the question based on the provided DataFrame, system knowledge, and conversation history."""
    q_lower = question.lower().strip()

    # --- Personality and Tone ---
    tone = st.session_state.get("tone", "neutral")  # Get the current tone
    if "funny" in q_lower:
        tone = "funny"
        st.session_state["tone"] = "funny"
        return generate_human_like_response("Very well, prepare for a humorous perspective on data science!")
    elif "formal" in q_lower or "professional" in q_lower:
        tone = "formal"
        st.session_state["tone"] = "formal"
        return generate_human_like_response("Understood. I will adopt a formal and professional tone for our interaction.")
    elif "neutral" in q_lower:
        tone = "neutral"
        st.session_state["tone"] = "neutral"
        return generate_human_like_response("Resetting to a neutral tone. Please proceed.")

    # --- Friendly Greetings and Introductions ---
    if any(greet in q_lower for greet in ["hello", "hi", "hey", "you"]):
        responses = [
            "Greetings! I am KlinItAll, your AI-driven data science assistant. I am prepared to assist you with your data-related inquiries.",
            "Good day. I am KlinItAll, a sophisticated AI designed to support your data science endeavors.",
            "Salutations! I am KlinItAll, at your service for all matters pertaining to data analysis and machine learning.",
            "Hello. I am KlinItAll, your dedicated AI assistant for data science tasks."
        ]
        introduction = random.choice(responses) + "\n\n"
        introduction += "My capabilities include:\n\n"
        introduction += "- Answering questions related to data science, machine learning, and artificial intelligence.\n"
        introduction += "- Providing detailed explanations of data preprocessing methodologies.\n"
        introduction += "- Generating visualizations of data (contingent upon data input).\n"
        introduction += "- Suggesting code snippets for data manipulation tasks.\n"
        introduction += "- Analyzing datasets and providing insightful observations.\n"
        introduction += "- Performing sentiment analysis on textual data.\n"
        introduction += "- Retrieving real-time information from the internet.\n\n"
        introduction += "To commence, please pose a question or provide a dataset for analysis. Consider the following examples:\n\n"
        introduction += "\n".join([f"- {q}" for q in system_knowledge["general"]["suggested_questions"]])
        return generate_human_like_response(introduction)

    # --- General system info ---
    if "what is" in q_lower:
        return generate_human_like_response("Kindly specify the topic you are inquiring about. Are you interested in data science, data preprocessing, or another related subject?")

    # --- Data Science / Machine Learning ---
    if "data science" in q_lower:
        if tone == "funny":
            return generate_human_like_response("Data science is like being a detective, but instead of solving crimes, you're solving business problems with data. And sometimes, the data is the real criminal!")
        else:
            return generate_human_like_response(system_knowledge["general"]["about_system"])

    if "machine learning" in q_lower:
        if tone == "funny":
            return generate_human_like_response("Machine learning is like teaching a computer to learn from its mistakes, except the computer never gets detention. It just gets better at predicting things.")
        else:
            return generate_human_like_response("Machine learning allows systems to learn from data and make predictions or decisions. It’s a core part of AI.")

    # --- Advanced AI & Machine Learning ---
    if "reinforcement learning" in q_lower:
        if tone == "funny":
            return generate_human_like_response("Reinforcement learning is like training a dog with treats, but instead of a dog, it's an AI, and instead of treats, it's rewards. And sometimes, the AI bites back.")
        else:
            return generate_human_like_response(system_knowledge["advanced_ml_ai"]["reinforcement_learning"]["description"])

    if "generative models" in q_lower:
        return generate_human_like_response(system_knowledge["advanced_ml_ai"]["generative_models"]["description"])

    if "transfer learning" in q_lower:
        return generate_human_like_response(system_knowledge["advanced_ml_ai"]["transfer_learning"]["description"])

    if "meta learning" in q_lower:
        return generate_human_like_response(system_knowledge["advanced_ml_ai"]["meta_learning"]["description"])

    if "deep learning" in q_lower:
        return generate_human_like_response(system_knowledge["advanced_ml_ai"]["deep_learning"]["description"])

    if "transformers" in q_lower:
        return generate_human_like_response(system_knowledge["advanced_ml_ai"]["transformers"]["description"])

    # --- Data Preprocessing ---
    if "data preprocessing" in q_lower or "data cleaning" in q_lower:
        return generate_human_like_response("Data preprocessing is like cleaning your room before a party. It's essential to make sure everything looks good and works well.")

    if "missing values" in q_lower or "imputation" in q_lower:
        return generate_human_like_response(system_knowledge["data_preprocessing"]["missing_values"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["missing_values"]["methods"]]))

    if "outliers" in q_lower:
        return generate_human_like_response(system_knowledge["data_preprocessing"]["outliers"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["outliers"]["methods"]]))

    if "scaling" in q_lower:
        return generate_human_like_response(system_knowledge["data_preprocessing"]["scaling"]["description"] + "\n\nHere are some common methods:\n" + "\n".join([f"- {m}" for m in system_knowledge["data_preprocessing"]["scaling"]["methods"]]))

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
                return generate_human_like_response("I have performed sentiment analysis on the first text column of your dataset. The results are displayed above.")
            else:
                return generate_human_like_response("There are no text columns in your dataset to perform sentiment analysis on.")
        else:
            return generate_human_like_response("Sentiment analysis is a technique used to determine the emotional tone of a piece of text. Please upload a dataset with text columns to perform sentiment analysis.")

    if "remove stopwords" in q_lower:
        if df is not None:
            text_cols = df.select_dtypes(include='object').columns
            if not text_cols.empty:
                # Remove stopwords from the first text column
                first_text_col = text_cols[0]
                df['cleaned_text'] = df[first_text_col].astype(str).apply(remove_stopwords)
                st.write(f"Stopwords removed from the first text column ('{first_text_col}'):")
                st.dataframe(df[['cleaned_text']].head())
                return generate_human_like_response("I have removed stopwords from the first text column of your dataset. The cleaned text is displayed above.")
            else:
                return generate_human_like_response("There are no text columns in your dataset to remove stopwords from.")
        else:
            return generate_human_like_response("Stopwords are common words (e.g., 'the', 'a', 'is') that are often removed from text to improve NLP tasks. Please upload a dataset with text columns to remove stopwords.")

    # --- Real-time Information Retrieval ---
    if "internet" in q_lower or "web" in q_lower or "real-time" in q_lower:
        query = question.replace("internet", "").replace("web", "").replace("real-time", "").strip()
        realtime_info = get_realtime_info(query)
        return generate_human_like_response(f"I have retrieved information from the web regarding '{query}':\n\n{realtime_info}")

    # --- Clarifying Vague Queries ---
    if any(keyword in q_lower for keyword in ["data", "value", "row", "column", "shape"]):
        if df is not None:
            return generate_human_like_response(dataset_summary(df))
        else:
            return generate_human_like_response("You have not yet provided a dataset. However, I can explain fundamental concepts such as rows, columns, and values.")

    # --- Dataset-Specific Queries ---
    if df is not None:
        if "describe" in q_lower or "statistics" in q_lower or "summary" in q_lower:
            return generate_human_like_response(detailed_stats(df))
        if "visualization" in q_lower or "plot" in q_lower:
            data_visualizations(df)
            return generate_human_like_response("A visualization of your dataset has been generated. Does this provide further clarity?")
        if "shape" in q_lower:
            return generate_human_like_response(f"Your dataset consists of {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- General System Information ---
    if "how it works" in q_lower or "about" in q_lower:
        return generate_human_like_response(system_knowledge["general"]["how_it_works"])

    # --- Follow-up Prompts ---
    if conversation_history:
        follow_up = random.choice(system_knowledge["general"]["follow_up_prompts"])
        return generate_human_like_response(f"{random.choice(system_knowledge['general']['acknowledgement_phrases'])}. {follow_up}")

    # --- Unrecognized Input with Varied Responses ---
    unrecognized_responses = [
        "I regret to inform you that I did not fully comprehend your query. Could you please rephrase it?",
        "My understanding of your request is incomplete. Would you mind providing additional clarification?",
        "I apologize, but I am unable to process your request at this time. Please try again later.",
        "My current capabilities do not extend to addressing that particular question. Please consider an alternative inquiry.",
        "I am still under development and may not be equipped to handle all requests. Please try a different approach."
    ]
    return generate_human_like_response(random.choice(unrecognized_responses))

# -----------------------------
# Main Streamlit Page
# -----------------------------

def main():
    st.set_page_config(page_title="KlinItAll AI Data Assistant", page_icon="🤖")
    st.title("🤖 KlinItAll AI-Powered Data Assistant")
    st.markdown("Greetings! I am KlinItAll, your AI-driven data science assistant. I am prepared to assist you with your data-related inquiries. ✨")

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
    if prompt := st.chat_input("Please enter your query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the conversation history
        conversation_history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]

        response = answer_question(prompt, df, conversation_history)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
