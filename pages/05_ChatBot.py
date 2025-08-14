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
        "about_system": "KlinItAll is your AI-powered data sidekick! I'm here to help you navigate the world of data science, from basic concepts to advanced techniques. Think of me as your friendly, knowledgeable, and sometimes sassy data guide.",
        "how_it_works": "I use a blend of natural language processing (NLP), a vast knowledge base, and real-time internet access to answer your questions. I can also analyze datasets (if you upload one) and provide tailored insights.",
        "core_features": [
            "Answering data science, ML, and AI questions",
            "Explaining data preprocessing techniques (with examples!)",
            "Generating visualizations (if you provide data)",
            "Suggesting code snippets (because who wants to write everything from scratch?)",
            "Analyzing datasets and providing insights (I'm like a data detective)",
            "Performing sentiment analysis (because emotions matter, even in data)",
            "Fetching real-time info from the web (I'm always learning)",
            "Adapting my tone to your style (I can be formal, funny, or professional)"
        ],
        "suggested_questions": [
            "What's the deal with data science?",
            "How do I wrangle those pesky missing values?",
            "Tell me about common data preprocessing techniques.",
            "Give me the lowdown on my dataset (if I upload one).",
            "What are some cool feature engineering tricks?",
            "Explain reinforcement learning like I'm five.",
            "What are generative models good for?",
            "How does transfer learning save the day?",
            "CNNs vs. RNNs: What's the difference?",
            "Transformers in NLP: Hype or reality?"
        ],
        "follow_up_prompts": [
            "Want to dive deeper into that?",
            "Anything else I can help you with on that topic?",
            "Interested in a code example?",
            "Should we explore a different angle?",
            "Ready for the next challenge?"
        ]
    },
    "data_preprocessing": {
        "missing_values": {
            "description": "Missing values are like potholes in your data road. I can help you fill them in (or avoid them altogether!).",
            "methods": [
                "Dropping rows/columns (the drastic approach!).",
                "Imputing with mean/median/mode (the quick fix).",
                "KNN imputation (the neighborly approach).",
                "Model-based imputation (the predictive approach)."
            ]
        },
        "outliers": {
            "description": "Outliers are the rebels of your dataset. I can help you identify them and decide whether to embrace or exile them.",
            "methods": [
                "Trimming (the ruthless approach).",
                "Winsorizing (the diplomatic approach).",
                "Transforming (the disguise approach).",
                "Robust statistics (the outlier-resistant approach)."
            ]
        },
        "scaling": {
            "description": "Scaling is like putting all your features on the same playing field. It's essential for many machine learning algorithms.",
            "methods": [
                "StandardScaler (the zero-mean, unit-variance approach).",
                "MinMaxScaler (the 0-to-1 approach).",
                "RobustScaler (the outlier-friendly approach).",
                "PowerTransformer (the skewness-busting approach)."
            ]
        },
        "encoding": {
            "description": "Encoding is like translating your categorical data into a language that machines can understand.",
            "methods": [
                "One-hot encoding (the all-or-nothing approach).",
                "Label encoding (the ordinal approach).",
                "Frequency encoding (the popular approach).",
                "Target encoding (the predictive approach)."
            ]
        },
        "feature_engineering": {
            "description": "Feature engineering is where you get creative and craft new features to boost your model's performance.",
            "methods": [
                "Binning (the grouping approach).",
                "Interaction features (the combination approach).",
                "Polynomial features (the power-up approach).",
                "Date/time features (the time-traveling approach)."
            ]
        }
    },
    "statistics": {
        "description": "Statistics is the language of data. I can help you understand key statistical concepts and apply them to your data.",
        "core_concepts": [
            "Mean, Median, Mode",
            "Variance, Standard Deviation",
            "Skewness, Kurtosis",
            "Correlation, Regression",
            "Hypothesis Testing"
        ],
        "common_questions": {
            "mean": "The **mean** is the average. It's easily swayed by outliers.",
            "variance": "Variance measures data spread. High variance = more variability.",
            "skewness": "Skewness tells you about data asymmetry. Positive = right tail, negative = left tail.",
            "correlation": "Correlation measures the relationship between variables. -1 to +1, baby!",
            "regression": "Regression models relationships for prediction. It's like fortune-telling, but with data."
        }
    },
    "advanced_ml_ai": {
        "reinforcement_learning": {
            "description": "Reinforcement Learning (RL) trains agents to make decisions in an environment to maximize rewards. Think of it as teaching a robot to play video games."
        },
        "generative_models": {
            "description": "Generative models (GANs, VAEs) learn data distributions and create new, similar samples. They're like AI artists."
        },
        "transfer_learning": {
            "description": "Transfer learning uses knowledge from one problem to solve another. It's like reusing a cheat sheet for a slightly different test."
        },
        "meta_learning": {
            "description": "Meta-learning ('learning to learn') helps algorithms adapt quickly to new tasks. It's like giving your AI a learning superpower."
        },
        "deep_learning": {
            "description": "Deep learning uses multi-layered neural networks to learn complex patterns. It's the engine behind many AI breakthroughs."
        },
        "transformers": {
            "description": "Transformers are a powerful NLP architecture using attention mechanisms. They've revolutionized machine translation and text generation."
        },
        "cnn": {
            "description": "Convolutional Neural Networks (CNNs) excel at image-related tasks. They learn spatial hierarchies of features from images."
        },
        "gan": {
            "description": "Generative Adversarial Networks (GANs) pit a generator against a discriminator. They're used to create realistic images and other data."
        },
        "nlp": {
            "description": "Natural Language Processing (NLP) enables computers to understand and generate human language. It's the key to chatbots and machine translation."
        }
    }
}

# -----------------------------
# Helper Functions for Data Operations
# -----------------------------

def dataset_summary(df):
    """Provides a comprehensive summary of the dataset."""
    summary = f"Alright, here's the scoop on your dataset:\n\n"
    summary += f"It's got **{df.shape[0]} rows** and **{df.shape[1]} columns**."
    summary += "\n\nColumn names:\n" + ", ".join(df.columns)
    return summary

def detailed_stats(df):
    """Provides detailed descriptive statistics and insights on numerical columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        return "Hold on... it looks like there aren't any numerical columns in your dataset. I can't calculate stats without numbers!"

    stats = ["Okay, buckle up! Here's the statistical breakdown:\n"]
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
        st.write("Hmm, it seems like there are no numeric columns available for visualization.  I need numbers to make pretty pictures!")

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
            return "I couldn't find any relevant information on the web for that query."
    except requests.exceptions.RequestException as e:
        return f"Sorry, I encountered an error while trying to access the internet: {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching information: {e}"

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
        return "Alright, buckle up for some data-driven humor! What's your question?"
    elif "formal" in q_lower or "professional" in q_lower:
        tone = "formal"
        st.session_state["tone"] = "formal"
        return "Understood. I will respond in a formal and professional manner. Please proceed with your inquiry."
    elif "neutral" in q_lower:
        tone = "neutral"
        st.session_state["tone"] = "neutral"
        return "Back to neutral! Ask away."

    # --- Friendly Greetings and Introductions ---
    if any(greet in q_lower for greet in ["hello", "hi", "hey", "you"]):
        responses = [
            "Hey there! 👋 I'm KlinIt-Bot, your AI-powered data assistant. Ready to dive into the dataverse?",
            "Greetings! I'm KlinItAll, at your service for all things data science.",
            "Hi! KlinItAll here, ready to help you make sense of your data.",
            "Hey! I'm KlinItAll, your friendly neighborhood data assistant."
        ]
        introduction = random.choice(responses) + "\n\n"
        introduction += "I can help you with a wide range of data science tasks, including:\n\n"
        introduction += "- Answering questions about data science, machine learning, and AI\n"
        introduction += "- Explaining data preprocessing techniques\n"
        introduction += "- Generating visualizations (if you provide data)\n"
        introduction += "- Suggesting code snippets\n"
        introduction += "- Analyzing datasets and providing insights\n"
        introduction += "- Performing sentiment analysis on text data\n"
        introduction += "- Fetching real-time information from the web\n\n"
        introduction += "To get started, ask me a question or upload a dataset. Here are some ideas:\n\n"
        introduction += "\n".join([f"- {q}" for q in system_knowledge["general"]["suggested_questions"]])
        return introduction

    # --- General system info ---
    if "what is" in q_lower:
        return "Ah, looks like you're asking for an explanation. Could you clarify which topic you're referring to? Data science, data preprocessing, or maybe something else?"

    # --- Data Science / Machine Learning ---
    if "data science" in q_lower:
        if tone == "funny":
            return "Data science is like being a detective, but instead of solving crimes, you're solving business problems with data. And sometimes, the data is the real criminal!"
        else:
            return system_knowledge["general"]["about_system"]

    if "machine learning" in q_lower:
        if tone == "funny":
            return "Machine learning is like teaching a computer to learn from its mistakes, except the computer never gets detention. It just gets better at predicting things."
        else:
            return "Machine learning allows systems to learn from data and make predictions or decisions. It’s a core part of AI."

    # --- Advanced AI & Machine Learning ---
    if "reinforcement learning" in q_lower:
        if tone == "funny":
            return "Reinforcement learning is like training a dog with treats, but instead of a dog, it's an AI, and instead of treats, it's rewards. And sometimes, the AI bites back."
        else:
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
        return "Data preprocessing is like cleaning your room before a party. It's essential to make sure everything looks good and works well."

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
                return "I've performed sentiment analysis on the first text column of your dataset. See the results above!"
            else:
                return "There are no text columns in your dataset to perform sentiment analysis on."
        else:
            return "Sentiment analysis is a technique used to determine the emotional tone of a piece of text. Please upload a dataset with text columns to perform sentiment analysis."

    if "remove stopwords" in q_lower:
        if df is not None:
            text_cols = df.select_dtypes(include='object').columns
            if not text_cols.empty:
                # Remove stopwords from the first text column
                first_text_col = text_cols[0]
                df['cleaned_text'] = df[first_text_col].astype(str).apply(remove_stopwords)
                st.write(f"Stopwords removed from the first text column ('{first_text_col}'):")
                st.dataframe(df[['cleaned_text']].head())
                return "I've removed stopwords from the first text column of your dataset. See the cleaned text above!"
            else:
                return "There are no text columns in your dataset to remove stopwords from."
        else:
            return "Stopwords are common words (e.g., 'the', 'a', 'is') that are often removed from text to improve NLP tasks. Please upload a dataset with text columns to remove stopwords."

    # --- Real-time Information Retrieval ---
    if "internet" in q_lower or "web" in q_lower or "real-time" in q_lower:
        query = question.replace("internet", "").replace("web", "").replace("real-time", "").strip()
        realtime_info = get_realtime_info(query)
        return f"Okay, I've scoured the web for information on '{query}':\n\n{realtime_info}"

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

    # --- Follow-up Prompts ---
    if conversation_history:
        follow_up = random.choice(system_knowledge["general"]["follow_up_prompts"])
        return "Got it! " + follow_up

    # --- Unrecognized Input with Varied Responses ---
    unrecognized_responses = [
        "Hmm, I'm not quite sure I understand. Could you rephrase your question?",
        "I'm a little lost. Can you ask me something else?",
        "Sorry, I didn't catch that. What else can I help you with?",
        "My circuits are a bit fuzzy. Could you try asking again?",
        "I'm still under development. Can you ask me something different?"
    ]
    return random.choice(unrecognized_responses)

# -----------------------------
# Main Streamlit Page
# -----------------------------

def main():
    st.set_page_config(page_title="KlinItAll AI Data Assistant", page_icon="🤖")
    st.title("🤖 KlinItAll AI-Powered Data Assistant")
    st.markdown("Hi there! I'm KlinItAll, your data mentor. Ask me anything about data science, machine learning, or data preprocessing! ✨")

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

        # Get the conversation history
        conversation_history = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]

        response = answer_question(prompt, df, conversation_history)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
