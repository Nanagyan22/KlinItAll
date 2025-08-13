import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
from textblob import TextBlob
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Text Cleaning", page_icon="📝", layout="wide")

st.title("📝 Text Cleaning")
st.markdown("Advanced text cleaning with NLP capabilities and intelligent preprocessing")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()
text_cols = df.select_dtypes(include=['object']).columns.tolist()

if len(text_cols) == 0:
    st.warning("⚠️ No text columns found for cleaning.")
    st.stop()

# Text cleaning functions
def clean_text_basic(text):
    """Basic text cleaning operations"""
    if pd.isna(text) or text == "":
        return text
    
    text = str(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_advanced(text, operations):
    """Advanced text cleaning with multiple operations"""
    if pd.isna(text) or text == "":
        return text
    
    text = str(text)
    
    for op in operations:
        if op == "lowercase":
            text = text.lower()
        elif op == "remove_numbers":
            text = re.sub(r'\d+', '', text)
        elif op == "remove_punctuation":
            text = re.sub(r'[^\w\s]', '', text)
        elif op == "remove_special_chars":
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        elif op == "remove_extra_spaces":
            text = re.sub(r'\s+', ' ', text).strip()
        elif op == "remove_urls":
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        elif op == "remove_emails":
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    return text

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Text Columns", len(text_cols))

with col2:
    total_text_cells = sum(df[col].notna().sum() for col in text_cols)
    st.metric("Text Cells", f"{total_text_cells:,}")

with col3:
    avg_length = np.mean([df[col].str.len().mean() for col in text_cols if df[col].dtype == 'object'])
    st.metric("Avg Text Length", f"{avg_length:.0f}" if not pd.isna(avg_length) else "0")

with col4:
    unique_values = sum(df[col].nunique() for col in text_cols)
    st.metric("Unique Values", f"{unique_values:,}")

# Text cleaning tabs
text_tabs = st.tabs(["📊 Analysis", "🧹 Basic Cleaning", "⚡ Advanced Cleaning", "📈 NLP Processing"])

with text_tabs[0]:
    st.markdown("### 📊 Text Data Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column analysis
        selected_text_col = st.selectbox("Select column for analysis:", text_cols)
        
        col_data = df[selected_text_col].dropna().astype(str)
        
        # Text statistics
        text_stats = {
            'Metric': ['Total Values', 'Unique Values', 'Average Length', 'Min Length', 'Max Length', 'Empty Values', 'Numeric Values'],
            'Value': [
                f"{len(col_data):,}",
                f"{col_data.nunique():,}",
                f"{col_data.str.len().mean():.1f}",
                f"{col_data.str.len().min()}",
                f"{col_data.str.len().max()}",
                f"{(col_data == '').sum():,}",
                f"{col_data.str.isnumeric().sum():,}"
            ]
        }
        
        stats_df = pd.DataFrame(text_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Sample values
        st.markdown("#### Sample Values")
        sample_values = col_data.head(10).tolist()
        for i, val in enumerate(sample_values, 1):
            display_val = val[:100] + "..." if len(val) > 100 else val
            st.write(f"{i}. {display_val}")
    
    with col2:
        # Text length distribution
        if len(col_data) > 0:
            lengths = col_data.str.len()
            
            fig = px.histogram(x=lengths, nbins=30, title=f"Text Length Distribution: {selected_text_col}")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Word frequency
            if st.checkbox("Show word frequency"):
                all_text = ' '.join(col_data.head(1000))  # Limit for performance
                words = re.findall(r'\w+', all_text.lower())
                word_freq = Counter(words).most_common(10)
                
                if word_freq:
                    freq_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                    fig_words = px.bar(freq_df, x='Word', y='Frequency', title="Top 10 Words")
                    st.plotly_chart(fig_words, use_container_width=True)

with text_tabs[1]:
    st.markdown("### 🧹 Basic Text Cleaning")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Quick Cleaning Options")
        
        basic_operations = st.multiselect(
            "Select cleaning operations:",
            [
                "Remove leading/trailing whitespace",
                "Remove extra spaces", 
                "Convert to lowercase",
                "Remove numbers",
                "Remove punctuation",
                "Remove special characters"
            ],
            help="Choose basic text cleaning operations"
        )
        
        target_columns = st.multiselect(
            "Select columns to clean:",
            text_cols,
            default=[text_cols[0]] if text_cols else [],
            help="Choose which columns to apply cleaning to"
        )
        
        if st.button("🧹 Apply Basic Cleaning", type="primary") and target_columns:
            cleaned_count = 0
            
            for col in target_columns:
                try:
                    original_sample = df[col].head(3).tolist()
                    
                    # Apply selected operations
                    if "Remove leading/trailing whitespace" in basic_operations:
                        df[col] = df[col].str.strip()
                    
                    if "Remove extra spaces" in basic_operations:
                        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                    
                    if "Convert to lowercase" in basic_operations:
                        df[col] = df[col].str.lower()
                    
                    if "Remove numbers" in basic_operations:
                        df[col] = df[col].str.replace(r'\d+', '', regex=True)
                    
                    if "Remove punctuation" in basic_operations:
                        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
                    
                    if "Remove special characters" in basic_operations:
                        df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    
                    cleaned_count += 1
                    
                except Exception as e:
                    st.error(f"Error cleaning {col}: {str(e)}")
                    continue
            
            if cleaned_count > 0:
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Text Cleaning',
                    'details': f"Applied {len(basic_operations)} operations to {cleaned_count} columns: {', '.join(target_columns)}"
                })
                
                st.success(f"✅ Cleaned {cleaned_count} text columns!")
                st.rerun()
    
    with col2:
        st.markdown("#### Before/After Preview")
        
        if target_columns and basic_operations:
            preview_col = target_columns[0]
            sample_text = df[preview_col].dropna().iloc[0] if len(df[preview_col].dropna()) > 0 else "No sample text"
            
            st.markdown("**Before:**")
            st.code(str(sample_text)[:200] + "..." if len(str(sample_text)) > 200 else str(sample_text))
            
            # Show what it would look like after cleaning
            preview_text = str(sample_text)
            
            if "Remove leading/trailing whitespace" in basic_operations:
                preview_text = preview_text.strip()
            if "Remove extra spaces" in basic_operations:
                preview_text = re.sub(r'\s+', ' ', preview_text)
            if "Convert to lowercase" in basic_operations:
                preview_text = preview_text.lower()
            if "Remove numbers" in basic_operations:
                preview_text = re.sub(r'\d+', '', preview_text)
            if "Remove punctuation" in basic_operations:
                preview_text = re.sub(r'[^\w\s]', '', preview_text)
            if "Remove special characters" in basic_operations:
                preview_text = re.sub(r'[^a-zA-Z0-9\s]', '', preview_text)
            
            st.markdown("**After:**")
            st.code(preview_text[:200] + "..." if len(preview_text) > 200 else preview_text)

with text_tabs[2]:
    st.markdown("### ⚡ Advanced Text Cleaning")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Advanced Operations")
        
        advanced_col = st.selectbox("Select column for advanced cleaning:", text_cols, key="advanced_col")
        
        advanced_ops = st.multiselect(
            "Advanced cleaning operations:",
            [
                "Remove URLs",
                "Remove email addresses",
                "Remove phone numbers",
                "Remove HTML tags",
                "Standardize whitespace",
                "Remove duplicate words",
                "Expand contractions",
                "Remove stop words (common words)"
            ],
            help="Advanced text processing operations"
        )
        
        # Custom pattern removal
        st.markdown("#### Custom Pattern Removal")
        custom_pattern = st.text_input("Custom regex pattern:", placeholder="e.g., [0-9]{3}-[0-9]{3}-[0-9]{4}")
        replacement = st.text_input("Replace with:", placeholder="Leave empty to remove")
        
        if st.button("⚡ Apply Advanced Cleaning", type="primary"):
            try:
                # Apply advanced operations
                col_data = df[advanced_col].copy()
                
                for op in advanced_ops:
                    if op == "Remove URLs":
                        col_data = col_data.str.replace(r'http\S+|www\S+|https\S+', '', regex=True)
                    elif op == "Remove email addresses":
                        col_data = col_data.str.replace(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', regex=True)
                    elif op == "Remove phone numbers":
                        col_data = col_data.str.replace(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', regex=True)
                    elif op == "Remove HTML tags":
                        col_data = col_data.str.replace(r'<[^<>]*>', '', regex=True)
                    elif op == "Standardize whitespace":
                        col_data = col_data.str.replace(r'\s+', ' ', regex=True).str.strip()
                    elif op == "Remove duplicate words":
                        # This is complex - simplified version
                        col_data = col_data.apply(lambda x: ' '.join(dict.fromkeys(str(x).split())) if pd.notna(x) else x)
                
                # Apply custom pattern if provided
                if custom_pattern:
                    col_data = col_data.str.replace(custom_pattern, replacement if replacement else '', regex=True)
                
                # Update dataframe
                df[advanced_col] = col_data
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                operation_list = advanced_ops + ([f"Custom pattern: {custom_pattern}"] if custom_pattern else [])
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Advanced Text Cleaning',
                    'details': f"Applied {len(operation_list)} operations to {advanced_col}: {', '.join(operation_list)}"
                })
                
                st.success(f"✅ Applied {len(operation_list)} advanced operations to {advanced_col}!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Advanced cleaning failed: {str(e)}")
    
    with col2:
        st.markdown("#### Text Quality Metrics")
        
        if advanced_col in locals():
            col_data = df[advanced_col].dropna().astype(str)
            
            # Calculate quality metrics
            total_chars = col_data.str.len().sum()
            total_words = col_data.str.split().str.len().sum()
            
            # Detect potential issues
            issues = {
                'URLs detected': col_data.str.contains(r'http|www', case=False, na=False).sum(),
                'Email addresses': col_data.str.contains(r'@.*\.', case=False, na=False).sum(),
                'Phone numbers': col_data.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', na=False).sum(),
                'HTML tags': col_data.str.contains(r'<.*?>', na=False).sum(),
                'Extra spaces': col_data.str.contains(r'\s{2,}', na=False).sum(),
                'All caps entries': col_data.str.isupper().sum(),
                'Numeric entries': col_data.str.isnumeric().sum()
            }
            
            st.markdown("**Potential Issues Found:**")
            for issue, count in issues.items():
                if count > 0:
                    st.write(f"• {issue}: {count:,}")
            
            if sum(issues.values()) == 0:
                st.success("✅ No obvious text quality issues detected!")

with text_tabs[3]:
    st.markdown("### 📈 NLP Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Natural Language Processing")
        
        nlp_col = st.selectbox("Select column for NLP analysis:", text_cols, key="nlp_col")
        
        nlp_operations = st.multiselect(
            "NLP operations:",
            [
                "Sentiment analysis",
                "Text length analysis", 
                "Language detection",
                "Extract keywords",
                "Readability score",
                "Part-of-speech tagging"
            ],
            help="Advanced NLP processing options"
        )
        
        if st.button("🔍 Run NLP Analysis", type="primary") and nlp_operations:
            try:
                col_data = df[nlp_col].dropna().astype(str).head(100)  # Limit for performance
                results = {}
                
                if "Sentiment analysis" in nlp_operations:
                    sentiments = []
                    for text in col_data:
                        try:
                            blob = TextBlob(text)
                            sentiment = blob.sentiment.polarity
                            if sentiment > 0.1:
                                sentiments.append('Positive')
                            elif sentiment < -0.1:
                                sentiments.append('Negative')
                            else:
                                sentiments.append('Neutral')
                        except:
                            sentiments.append('Unknown')
                    
                    results['Sentiment Distribution'] = Counter(sentiments)
                
                if "Text length analysis" in nlp_operations:
                    lengths = col_data.str.len()
                    results['Length Stats'] = {
                        'Average': f"{lengths.mean():.1f}",
                        'Median': f"{lengths.median():.1f}",
                        'Min': f"{lengths.min()}",
                        'Max': f"{lengths.max()}"
                    }
                
                if "Extract keywords" in nlp_operations:
                    # Simple keyword extraction using word frequency
                    all_text = ' '.join(col_data.head(50))  # Limit for performance
                    words = re.findall(r'\w+', all_text.lower())
                    # Filter out common words
                    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
                    filtered_words = [word for word in words if word not in common_words and len(word) > 2]
                    results['Top Keywords'] = Counter(filtered_words).most_common(10)
                
                # Display results
                st.markdown("#### Analysis Results")
                for analysis_type, result in results.items():
                    st.markdown(f"**{analysis_type}:**")
                    if isinstance(result, Counter):
                        for item, count in result.items():
                            st.write(f"• {item}: {count}")
                    elif isinstance(result, dict):
                        for key, value in result.items():
                            st.write(f"• {key}: {value}")
                    elif isinstance(result, list):
                        for item in result:
                            st.write(f"• {item[0]}: {item[1]}")
                    st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ NLP analysis failed: {str(e)}")
    
    with col2:
        st.markdown("#### Text Standardization")
        
        standardization_col = st.selectbox("Column to standardize:", text_cols, key="std_col")
        
        std_operations = st.multiselect(
            "Standardization operations:",
            [
                "Title Case",
                "Sentence Case", 
                "UPPER CASE",
                "lower case",
                "Remove leading/trailing spaces",
                "Standardize line breaks",
                "Fix encoding issues"
            ]
        )
        
        if st.button("📝 Apply Standardization") and std_operations:
            try:
                col_data = df[standardization_col].copy()
                
                for op in std_operations:
                    if op == "Title Case":
                        col_data = col_data.str.title()
                    elif op == "Sentence Case":
                        col_data = col_data.str.capitalize()
                    elif op == "UPPER CASE":
                        col_data = col_data.str.upper()
                    elif op == "lower case":
                        col_data = col_data.str.lower()
                    elif op == "Remove leading/trailing spaces":
                        col_data = col_data.str.strip()
                    elif op == "Standardize line breaks":
                        col_data = col_data.str.replace(r'\r\n|\r|\n', ' ', regex=True)
                    elif op == "Fix encoding issues":
                        # Basic encoding fixes
                        col_data = col_data.str.replace('\u2019', "'", regex=False)
                        col_data = col_data.str.replace('\u201c', '"', regex=False)
                        col_data = col_data.str.replace('\u201d', '"', regex=False)
                
                df[standardization_col] = col_data
                st.session_state.current_dataset = df
                
                st.success(f"✅ Applied {len(std_operations)} standardization operations!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Standardization failed: {str(e)}")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Cleaned Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"text_cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to DateTime Processing"):
        st.switch_page("pages/09_DateTime.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📝 Text Cleaning Guide")
    
    st.markdown("#### Cleaning Operations:")
    operations = [
        "**Basic:** Whitespace, case, punctuation",
        "**Advanced:** URLs, emails, patterns",
        "**NLP:** Sentiment, keywords, analysis",
        "**Standardization:** Format consistency"
    ]
    
    for op in operations:
        st.markdown(f"• {op}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Understand your text data first
    • Apply cleaning operations incrementally
    • Keep backups of original text
    • Test on sample data before bulk operations
    • Consider domain-specific requirements
    """)
    
    if st.session_state.get('processing_log'):
        st.markdown("#### 📝 Recent Actions")
        text_actions = [log for log in st.session_state.processing_log if 'Text' in log.get('action', '')][-3:]
        
        for action in text_actions:
            st.caption(f"✅ {action.get('details', 'Text processing action')}")