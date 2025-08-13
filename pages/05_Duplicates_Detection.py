import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Duplicates Detection", page_icon="🔄", layout="wide")

st.title("🔄 Duplicates Detection") 
st.markdown("Advanced duplicate detection with fuzzy matching and intelligent removal strategies")

# Check if data exists
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Initialize processing log
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

def log_action(action, details):
    """Log processing actions"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_log.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })

# Duplicate Analysis Functions
def analyze_duplicates(df, subset_cols=None):
    """Comprehensive duplicate analysis"""
    analysis = {}
    
    # Exact duplicates
    if subset_cols:
        exact_dups = df.duplicated(subset=subset_cols)
    else:
        exact_dups = df.duplicated()
    
    analysis['exact_count'] = exact_dups.sum()
    analysis['exact_percentage'] = (exact_dups.sum() / len(df)) * 100
    
    # Duplicate rows details
    if analysis['exact_count'] > 0:
        if subset_cols:
            duplicate_rows = df[df.duplicated(subset=subset_cols, keep=False)]
        else:
            duplicate_rows = df[df.duplicated(keep=False)]
        analysis['duplicate_groups'] = len(duplicate_rows.drop_duplicates(subset=subset_cols if subset_cols else df.columns))
    else:
        analysis['duplicate_groups'] = 0
    
    return analysis

def find_fuzzy_duplicates(df, column, threshold=80):
    """Find fuzzy duplicates in a text column using similarity matching"""
    try:
        from fuzzywuzzy import fuzz, process
        
        fuzzy_groups = []
        processed_values = set()
        
        unique_values = df[column].dropna().unique()
        
        for value in unique_values:
            if value in processed_values:
                continue
                
            # Find similar values
            matches = process.extractBests(str(value), [str(v) for v in unique_values], 
                                         score_cutoff=threshold, limit=None)
            
            if len(matches) > 1:  # Found fuzzy matches
                group_values = [match[0] for match in matches]
                fuzzy_groups.append({
                    'representative': value,
                    'similar_values': group_values,
                    'count': sum(df[column] == v for v in group_values),
                    'similarity_scores': [match[1] for match in matches]
                })
                
                processed_values.update(group_values)
        
        return fuzzy_groups
    
    except ImportError:
        st.warning("📝 Fuzzy matching requires fuzzywuzzy package. Install with: pip install fuzzywuzzy")
        return []

# Main Analysis Section
st.markdown("### 📊 Duplicate Analysis Overview")

# Quick metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    exact_duplicates = df.duplicated().sum()
    st.metric("Exact Duplicates", f"{exact_duplicates:,}", 
              f"{(exact_duplicates/len(df)*100):.1f}% of data")

with col2:
    unique_rows = len(df.drop_duplicates())
    st.metric("Unique Rows", f"{unique_rows:,}", 
              f"{((len(df) - unique_rows)/len(df)*100):.1f}% reduction")

with col3:
    total_rows = len(df)
    st.metric("Total Rows", f"{total_rows:,}")

with col4:
    columns_with_dups = sum(1 for col in df.columns if df[col].duplicated().any())
    st.metric("Columns with Duplicates", f"{columns_with_dups}")

# Detailed Analysis Tabs
analysis_tabs = st.tabs(["🔍 Exact Duplicates", "🎯 Fuzzy Matching", "🔧 Advanced Detection", "📊 Column Analysis"])

# ==================== EXACT DUPLICATES TAB ====================
with analysis_tabs[0]:
    st.markdown("#### 🔍 Exact Duplicate Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column selection for duplicate checking
        st.markdown("##### Select Columns for Duplicate Checking")
        all_columns = df.columns.tolist()
        
        duplicate_check_options = st.multiselect(
            "Columns to consider (leave empty for all columns):",
            all_columns,
            help="Select specific columns to check for duplicates, or leave empty to check entire rows"
        )
        
        # Use all columns if none selected
        check_columns = duplicate_check_options if duplicate_check_options else all_columns
        
        # Analyze duplicates based on selected columns
        dup_analysis = analyze_duplicates(df, check_columns)
        
        if dup_analysis['exact_count'] > 0:
            st.markdown(f"""
            <div class="warning-box">
                <strong>⚠️ Duplicates Found:</strong><br>
                • {dup_analysis['exact_count']} duplicate rows ({dup_analysis['exact_percentage']:.1f}% of data)<br>
                • {dup_analysis['duplicate_groups']} unique duplicate groups<br>
                • Based on columns: {', '.join(check_columns)}
            </div>
            """, unsafe_allow_html=True)
            
            # Show sample duplicates
            st.markdown("##### Sample Duplicate Rows")
            if duplicate_check_options:
                sample_dups = df[df.duplicated(subset=check_columns, keep=False)].head(10)
            else:
                sample_dups = df[df.duplicated(keep=False)].head(10)
            
            # Convert to strings to avoid Arrow issues
            for col in sample_dups.columns:
                sample_dups[col] = sample_dups[col].astype(str)
            
            st.dataframe(sample_dups, use_container_width=True)
            
        else:
            st.markdown("""
            <div class="success-box">
                <strong>✅ No Exact Duplicates Found!</strong><br>
                Your dataset appears to be free of exact duplicate rows.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Duplicate removal options
        st.markdown("##### Removal Options")
        
        keep_option = st.selectbox(
            "Which duplicate to keep:",
            ["first", "last", "none"],
            help="first: keep first occurrence, last: keep last occurrence, none: remove all duplicates"
        )
        
        if dup_analysis['exact_count'] > 0:
            if st.button("🗑️ Remove Exact Duplicates", type="primary"):
                initial_rows = len(df)
                
                if duplicate_check_options:
                    if keep_option == "none":
                        df_cleaned = df.drop_duplicates(subset=check_columns, keep=False)
                    else:
                        df_cleaned = df.drop_duplicates(subset=check_columns, keep=keep_option)
                else:
                    if keep_option == "none":
                        df_cleaned = df.drop_duplicates(keep=False)
                    else:
                        df_cleaned = df.drop_duplicates(keep=keep_option)
                
                removed_count = initial_rows - len(df_cleaned)
                
                # Update session state
                st.session_state.current_dataset = df_cleaned
                
                # Log the action
                log_action("Duplicate Removal", 
                          f"Removed {removed_count} exact duplicates (kept={keep_option}, columns={check_columns})")
                
                st.success(f"✅ Removed {removed_count} duplicate rows!")
                st.rerun()
        
        # Visualization
        if dup_analysis['exact_count'] > 0:
            st.markdown("##### Duplicate Distribution")
            
            fig = go.Figure(data=[
                go.Bar(x=['Unique Rows', 'Duplicate Rows'], 
                      y=[len(df) - dup_analysis['exact_count'], dup_analysis['exact_count']],
                      marker_color=['#2E8B57', '#CD5C5C'])
            ])
            fig.update_layout(title="Duplicate vs Unique Rows", height=300)
            st.plotly_chart(fig, use_container_width=True)

# ==================== FUZZY MATCHING TAB ====================
with analysis_tabs[1]:
    st.markdown("#### 🎯 Fuzzy Duplicate Detection")
    
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not text_columns:
        st.info("ℹ️ No text columns available for fuzzy matching.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Fuzzy matching configuration
            fuzzy_column = st.selectbox(
                "Select column for fuzzy matching:",
                text_columns,
                help="Choose a text column to find similar values"
            )
            
            similarity_threshold = st.slider(
                "Similarity threshold:",
                min_value=60,
                max_value=95,
                value=80,
                help="Higher values = more strict matching"
            )
            
            if st.button("🔍 Find Fuzzy Duplicates", type="primary"):
                with st.spinner("Analyzing fuzzy duplicates..."):
                    fuzzy_groups = find_fuzzy_duplicates(df, fuzzy_column, similarity_threshold)
                    
                    if fuzzy_groups:
                        st.success(f"✅ Found {len(fuzzy_groups)} groups of similar values!")
                        
                        for i, group in enumerate(fuzzy_groups[:5]):  # Show top 5 groups
                            with st.expander(f"Group {i+1}: '{group['representative'][:50]}...' ({group['count']} occurrences)"):
                                
                                similar_df = pd.DataFrame({
                                    'Similar Value': group['similar_values'],
                                    'Similarity Score': [f"{score}%" for score in group['similarity_scores']],
                                    'Occurrences': [sum(df[fuzzy_column] == val) for val in group['similar_values']]
                                })
                                
                                st.dataframe(similar_df, use_container_width=True)
                                
                                # Option to standardize this group
                                standard_value = st.text_input(
                                    f"Standardize to:",
                                    value=group['representative'],
                                    key=f"standard_{i}"
                                )
                                
                                if st.button(f"🔧 Standardize Group {i+1}", key=f"standardize_{i}"):
                                    # Replace all similar values with the standard value
                                    for similar_val in group['similar_values']:
                                        df.loc[df[fuzzy_column] == similar_val, fuzzy_column] = standard_value
                                    
                                    st.session_state.current_dataset = df
                                    log_action("Fuzzy Duplicate Standardization", 
                                              f"Standardized {len(group['similar_values'])} similar values in {fuzzy_column}")
                                    st.success(f"✅ Standardized group to '{standard_value}'")
                                    st.rerun()
                    
                    else:
                        st.info("ℹ️ No fuzzy duplicates found with current threshold.")
        
        with col2:
            st.markdown("##### Fuzzy Matching Info")
            
            st.markdown("""
            <div class="duplicate-card">
                <strong>🎯 How it works:</strong><br>
                • Compares text similarity using advanced algorithms<br>
                • Finds variations like typos, different formats<br>
                • Allows standardization of similar values<br><br>
                <strong>💡 Use cases:</strong><br>
                • Company names with variations<br>
                • Product names with typos<br>
                • Location names in different formats<br>
                • Customer names with inconsistencies
            </div>
            """, unsafe_allow_html=True)
            
            # Sample analysis
            if text_columns:
                sample_col = fuzzy_column if 'fuzzy_column' in locals() else text_columns[0]
                unique_count = df[sample_col].nunique()
                total_count = df[sample_col].count()
                
                st.metric("Unique Values", f"{unique_count:,}")
                st.metric("Total Values", f"{total_count:,}")
                st.metric("Potential Reduction", f"{((total_count - unique_count)/total_count*100):.1f}%")

# ==================== ADVANCED DETECTION TAB ====================
with analysis_tabs[2]:
    st.markdown("#### 🔧 Advanced Duplicate Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Conditional Duplicates")
        st.info("Detect duplicates based on specific conditions or business rules")
        
        # Conditional duplicate detection
        condition_column = st.selectbox(
            "Primary column for grouping:",
            df.columns,
            help="Group records by this column first"
        )
        
        comparison_columns = st.multiselect(
            "Columns to compare within groups:",
            [col for col in df.columns if col != condition_column],
            help="Find duplicates within each group based on these columns"
        )
        
        if condition_column and comparison_columns:
            if st.button("🔍 Find Conditional Duplicates"):
                # Group by condition column and find duplicates within each group
                conditional_dups = []
                
                for group_value in df[condition_column].unique():
                    group_data = df[df[condition_column] == group_value]
                    
                    if len(group_data) > 1:
                        # Check for duplicates within this group
                        group_dups = group_data.duplicated(subset=comparison_columns)
                        
                        if group_dups.any():
                            dup_count = group_dups.sum()
                            conditional_dups.append({
                                'Group': str(group_value)[:30],
                                'Total Records': len(group_data),
                                'Duplicates': dup_count,
                                'Duplicate %': f"{(dup_count/len(group_data)*100):.1f}%"
                            })
                
                if conditional_dups:
                    st.success(f"✅ Found conditional duplicates in {len(conditional_dups)} groups!")
                    
                    cond_df = pd.DataFrame(conditional_dups)
                    st.dataframe(cond_df, use_container_width=True)
                    
                    # Option to remove conditional duplicates
                    if st.button("🗑️ Remove Conditional Duplicates", key="remove_conditional"):
                        initial_rows = len(df)
                        
                        # Remove duplicates within each group
                        df_cleaned = df.groupby(condition_column).apply(
                            lambda x: x.drop_duplicates(subset=comparison_columns)
                        ).reset_index(drop=True)
                        
                        removed_count = initial_rows - len(df_cleaned)
                        
                        st.session_state.current_dataset = df_cleaned
                        log_action("Conditional Duplicate Removal", 
                                  f"Removed {removed_count} conditional duplicates grouped by {condition_column}")
                        
                        st.success(f"✅ Removed {removed_count} conditional duplicates!")
                        st.rerun()
                
                else:
                    st.info("ℹ️ No conditional duplicates found.")
    
    with col2:
        st.markdown("##### Near-Duplicate Detection")
        st.info("Find records that are nearly identical with small differences")
        
        # Near-duplicate threshold
        similarity_columns = st.multiselect(
            "Columns for similarity comparison:",
            df.columns,
            help="Compare these columns for near-duplicates"
        )
        
        if similarity_columns:
            difference_threshold = st.number_input(
                "Maximum allowed differences:",
                min_value=1,
                max_value=len(similarity_columns),
                value=1,
                help="Number of columns that can differ"
            )
            
            if st.button("🎯 Find Near-Duplicates"):
                near_duplicates = []
                
                # Compare each row with every other row
                for i in range(len(df)):
                    for j in range(i + 1, len(df)):
                        differences = 0
                        
                        for col in similarity_columns:
                            if df.iloc[i][col] != df.iloc[j][col]:
                                differences += 1
                        
                        if differences <= difference_threshold and differences > 0:
                            near_duplicates.append({
                                'Row 1': i,
                                'Row 2': j,
                                'Differences': differences,
                                'Similarity': f"{((len(similarity_columns) - differences) / len(similarity_columns) * 100):.1f}%"
                            })
                        
                        # Limit to prevent performance issues
                        if len(near_duplicates) > 100:
                            break
                    
                    if len(near_duplicates) > 100:
                        break
                
                if near_duplicates:
                    st.success(f"✅ Found {len(near_duplicates)} near-duplicate pairs!")
                    
                    near_df = pd.DataFrame(near_duplicates[:20])  # Show first 20
                    st.dataframe(near_df, use_container_width=True)
                    
                    if len(near_duplicates) > 20:
                        st.info(f"Showing first 20 of {len(near_duplicates)} near-duplicate pairs")
                
                else:
                    st.info("ℹ️ No near-duplicates found with current threshold.")

# ==================== COLUMN ANALYSIS TAB ====================
with analysis_tabs[3]:
    st.markdown("#### 📊 Per-Column Duplicate Analysis")
    
    # Analyze duplicates in each column
    column_analysis = []
    
    for col in df.columns:
        col_data = df[col]
        total_values = len(col_data)
        unique_values = col_data.nunique()
        duplicate_values = total_values - unique_values
        duplicate_percentage = (duplicate_values / total_values) * 100 if total_values > 0 else 0
        
        # Most frequent duplicates
        if duplicate_values > 0:
            most_frequent = col_data.value_counts().head(1)
            most_freq_value = str(most_frequent.index[0])[:30] + "..." if len(str(most_frequent.index[0])) > 30 else str(most_frequent.index[0])
            most_freq_count = most_frequent.iloc[0]
        else:
            most_freq_value = "N/A"
            most_freq_count = 0
        
        column_analysis.append({
            'Column': col,
            'Total Values': total_values,
            'Unique Values': unique_values,
            'Duplicate Values': duplicate_values,
            'Duplicate %': f"{duplicate_percentage:.1f}%",
            'Most Frequent Value': most_freq_value,
            'Frequency': most_freq_count
        })
    
    # Display analysis
    analysis_df = pd.DataFrame(column_analysis)
    st.dataframe(analysis_df, use_container_width=True)
    
    # Column-specific actions
    st.markdown("##### Column-Specific Duplicate Removal")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_column = st.selectbox(
            "Select column:",
            df.columns,
            help="Choose column for duplicate removal"
        )
    
    with col2:
        removal_strategy = st.selectbox(
            "Removal strategy:",
            ["Keep first occurrence", "Keep last occurrence", "Keep most frequent", "Remove all duplicates"],
            help="How to handle duplicate values"
        )
    
    with col3:
        if st.button("🗑️ Remove Column Duplicates", key="column_duplicates"):
            initial_rows = len(df)
            
            if removal_strategy == "Keep first occurrence":
                df_cleaned = df.drop_duplicates(subset=[target_column], keep='first')
            elif removal_strategy == "Keep last occurrence":
                df_cleaned = df.drop_duplicates(subset=[target_column], keep='last')
            elif removal_strategy == "Remove all duplicates":
                df_cleaned = df.drop_duplicates(subset=[target_column], keep=False)
            else:  # Keep most frequent
                # This is more complex - keep rows with the most frequent value in the target column
                most_frequent_val = df[target_column].mode().iloc[0]
                mask = df.duplicated(subset=[target_column], keep=False)
                df_cleaned = df[~mask | (df[target_column] == most_frequent_val)].drop_duplicates(subset=[target_column])
            
            removed_count = initial_rows - len(df_cleaned)
            
            if removed_count > 0:
                st.session_state.current_dataset = df_cleaned
                log_action("Column Duplicate Removal", 
                          f"Removed {removed_count} rows with duplicate values in {target_column}")
                st.success(f"✅ Removed {removed_count} rows based on {target_column} duplicates!")
                st.rerun()
            else:
                st.info("ℹ️ No duplicates found in the selected column.")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Deduplicated Dataset", type="primary"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"deduplicated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Missing Values"):
        st.switch_page("pages/06_Missing_Values.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🔄 Duplicate Detection Info")
    
    st.markdown("#### Detection Methods:")
    methods = [
        "🎯 **Exact Match:** Identical rows",
        "🔍 **Fuzzy Match:** Similar text values",
        "🔧 **Conditional:** Within groups", 
        "📊 **Near-Duplicate:** Almost identical",
        "🏷️ **Column-Specific:** Single column focus"
    ]
    
    for method in methods:
        st.markdown(method)
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    tips = [
        "Review sample duplicates before removal",
        "Consider business rules for conditional duplicates", 
        "Use fuzzy matching for text inconsistencies",
        "Backup data before removing duplicates",
        "Log all removal actions for audit trail"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    st.markdown("---")
    if st.session_state.processing_log:
        st.markdown("#### 📝 Recent Actions")
        recent_actions = [log for log in st.session_state.processing_log if 'Duplicate' in log.get('action', '')][-3:]
        
        for action in recent_actions:
            st.caption(f"✅ {action.get('details', 'Duplicate processing action')}")
    
    if st.button("🔄 Reset to Original"):
        if 'original_dataset' in st.session_state:
            st.session_state.current_dataset = st.session_state.original_dataset.copy()
            st.success("✅ Dataset reset to original")
            st.rerun()