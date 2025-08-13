import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Types", page_icon="🔍", layout="wide")

st.title("🔍 Data Types")
st.markdown("Intelligent data type detection and conversion with automatic recommendations")

# Check if data exists
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

def auto_detect_column_types(df):
    """Enhanced automatic detection of column types"""
    suggestions = {}
    confidence_scores = {}
    
    for col in df.columns:
        sample_data = df[col].dropna().head(100)  # Use larger sample
        current_type = str(df[col].dtype)
        
        # Initialize with current type
        suggestions[col] = current_type
        confidence_scores[col] = 0.5  # Default confidence
        
        # Skip if already numeric or datetime
        if df[col].dtype in ['int64', 'float64', 'datetime64[ns]']:
            confidence_scores[col] = 0.9
            continue
        
        if df[col].dtype == 'object' and len(sample_data) > 0:
            # Test for datetime
            try:
                parsed_dates = pd.to_datetime(sample_data, infer_datetime_format=True, errors='coerce')
                success_rate = parsed_dates.notna().sum() / len(sample_data)
                if success_rate > 0.8:
                    suggestions[col] = 'datetime64'
                    confidence_scores[col] = success_rate
                    continue
            except:
                pass
            
            # Test for numeric
            try:
                numeric_series = pd.to_numeric(sample_data, errors='coerce')
                success_rate = numeric_series.notna().sum() / len(sample_data)
                if success_rate > 0.8:
                    # Determine if int or float
                    if all(val == int(val) for val in numeric_series.dropna()):
                        suggestions[col] = 'int64'
                    else:
                        suggestions[col] = 'float64'
                    confidence_scores[col] = success_rate
                    continue
            except:
                pass
            
            # Test for boolean
            unique_vals = sample_data.str.lower().unique()
            bool_patterns = ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f']
            if len(unique_vals) <= 3 and all(str(val).lower() in bool_patterns for val in unique_vals):
                suggestions[col] = 'boolean'
                confidence_scores[col] = 0.9
                continue
            
            # Default to category for object types
            suggestions[col] = 'category'
            confidence_scores[col] = 0.7
    
    return suggestions, confidence_scores

# Data Type Analysis
st.markdown("### 📊 Current Data Types Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    # Get type suggestions
    type_suggestions, confidence_scores = auto_detect_column_types(df)
    
    # Create analysis dataframe
    type_analysis = []
    for col in df.columns:
        current_type = str(df[col].dtype)
        suggested_type = type_suggestions.get(col, current_type)
        confidence = confidence_scores.get(col, 0.5)
        
        # Get sample values
        sample_vals = df[col].dropna().head(3).astype(str).tolist()
        sample_str = ', '.join(sample_vals) if sample_vals else 'No data'
        
        type_analysis.append({
            'Column': col,
            'Current Type': current_type,
            'Suggested Type': suggested_type,
            'Confidence': f"{confidence:.1%}",
            'Non-Null Count': str(df[col].count()),
            'Unique Values': str(df[col].nunique()),
            'Sample Values': sample_str[:50] + '...' if len(sample_str) > 50 else sample_str
        })
    
    # Display analysis
    analysis_df = pd.DataFrame(type_analysis)
    st.dataframe(analysis_df, use_container_width=True)

with col2:
    # Type distribution chart
    st.markdown("#### Type Distribution")
    type_counts = df.dtypes.astype(str).value_counts()
    
    fig = px.pie(
        values=type_counts.values, 
        names=type_counts.index,
        title="Current Data Types",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    st.markdown("#### Quick Stats")
    st.metric("Total Columns", len(df.columns))
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    st.metric("Text Columns", len(df.select_dtypes(include=['object']).columns))
    st.metric("DateTime Columns", len(df.select_dtypes(include=['datetime64']).columns))

# Manual Type Conversion
st.markdown("---")
st.markdown("### 🔧 Manual Type Conversion")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_column = st.selectbox(
        "Select Column to Convert:",
        df.columns,
        help="Choose the column you want to convert"
    )

with col2:
    target_type = st.selectbox(
        "Convert To:",
        ["int64", "float64", "object", "datetime64", "category", "boolean"],
        help="Select the target data type"
    )

with col3:
    conversion_method = st.selectbox(
        "Error Handling:",
        ["coerce", "ignore", "raise"],
        help="How to handle conversion errors"
    )

with col4:
    if st.button("🔄 Convert Type", type="primary"):
        try:
            if target_type == "datetime64":
                df[selected_column] = pd.to_datetime(df[selected_column], errors=conversion_method)
            elif target_type == "boolean":
                # Custom boolean conversion
                bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                           'yes': True, 'no': False, 'y': True, 'n': False}
                if df[selected_column].dtype == 'object':
                    df[selected_column] = df[selected_column].str.lower().map(bool_map).fillna(df[selected_column])
                df[selected_column] = df[selected_column].astype('bool')
            elif target_type == "category":
                df[selected_column] = df[selected_column].astype('category')
            elif target_type in ["int64", "float64"]:
                df[selected_column] = pd.to_numeric(df[selected_column], errors=conversion_method)
                if target_type == "int64":
                    df[selected_column] = df[selected_column].astype('int64')
            else:
                df[selected_column] = df[selected_column].astype(target_type)
            
            # Update session state
            st.session_state.current_dataset = df
            st.success(f"✅ Successfully converted {selected_column} to {target_type}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Conversion failed: {str(e)}")

# Auto Conversion Section
st.markdown("---")
st.markdown("### 🤖 Automatic Type Conversion")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Recommended Conversions")
    
    recommendations = []
    for col in df.columns:
        current_type = str(df[col].dtype)
        suggested_type = type_suggestions.get(col, current_type)
        confidence = confidence_scores.get(col, 0.5)
        
        if suggested_type != current_type and confidence > 0.7:
            recommendations.append({
                'Column': col,
                'From': current_type,
                'To': suggested_type,
                'Confidence': f"{confidence:.1%}"
            })
    
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, use_container_width=True)
        
        if st.button("🚀 Apply All Recommendations", type="primary"):
            conversions_made = 0
            failed_conversions = []
            
            for rec in recommendations:
                try:
                    col = rec['Column']
                    target = rec['To']
                    
                    if target == "datetime64":
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    elif target == "boolean":
                        bool_map = {'true': True, 'false': False, '1': True, '0': False, 
                                   'yes': True, 'no': False, 'y': True, 'n': False}
                        if df[col].dtype == 'object':
                            df[col] = df[col].str.lower().map(bool_map).fillna(df[col])
                        df[col] = df[col].astype('bool')
                    elif target in ["int64", "float64"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if target == "int64":
                            df[col] = df[col].astype('int64')
                    elif target == "category":
                        df[col] = df[col].astype('category')
                    
                    conversions_made += 1
                
                except Exception as e:
                    failed_conversions.append(f"{col}: {str(e)}")
            
            # Update session state
            st.session_state.current_dataset = df
            
            # Show results
            if conversions_made > 0:
                st.success(f"✅ Successfully applied {conversions_made} type conversions!")
            
            if failed_conversions:
                st.warning("⚠️ Some conversions failed:")
                for failure in failed_conversions:
                    st.write(f"- {failure}")
            
            st.rerun()
    
    else:
        st.info("✅ No type conversion recommendations. Data types look optimal!")

with col2:
    st.markdown("#### Type Conversion Rules")
    
    rules = {
        "🔢 Numeric": "Columns with numbers are converted to int64 or float64",
        "📅 DateTime": "Date/time strings are converted to datetime64",
        "✅ Boolean": "True/False, Yes/No, 1/0 patterns become boolean",
        "🏷️ Category": "Text with limited unique values become category",
        "📝 Object": "Mixed or text data remains as object"
    }
    
    for rule_type, description in rules.items():
        st.markdown(f"**{rule_type}**")
        st.caption(description)
        st.markdown("")

# Column Details Section
st.markdown("---")
st.markdown("### 🔍 Detailed Column Analysis")

selected_col_detail = st.selectbox("Select column for detailed analysis:", df.columns, key="detail_analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Basic Statistics")
    col_data = df[selected_col_detail]
    
    stats = {
        "Data Type": str(col_data.dtype),
        "Non-Null Count": col_data.count(),
        "Null Count": col_data.isnull().sum(),
        "Unique Values": col_data.nunique(),
        "Duplicate Values": col_data.duplicated().sum()
    }
    
    # Add type-specific stats
    if col_data.dtype in ['int64', 'float64']:
        stats.update({
            "Mean": f"{col_data.mean():.2f}",
            "Median": f"{col_data.median():.2f}",
            "Std Dev": f"{col_data.std():.2f}",
            "Min": f"{col_data.min():.2f}",
            "Max": f"{col_data.max():.2f}"
        })
    
    for key, value in stats.items():
        st.metric(key, str(value))

with col2:
    st.markdown("#### Sample Values")
    
    # Show sample values
    sample_vals = col_data.dropna().head(10).tolist()
    for i, val in enumerate(sample_vals, 1):
        st.write(f"{i}. {val}")
    
    if col_data.dtype == 'object':
        st.markdown("#### Most Common Values")
        top_values = col_data.value_counts().head(5)
        for val, count in top_values.items():
            st.write(f"'{val}': {count} times")

with col3:
    st.markdown("#### Conversion Preview")
    
    # Show what would happen with different conversions
    preview_options = ["numeric", "datetime", "boolean", "category"]
    
    for option in preview_options:
        try:
            if option == "numeric":
                converted = pd.to_numeric(col_data, errors='coerce')
                success_rate = converted.notna().sum() / len(col_data)
                st.metric(f"→ Numeric", f"{success_rate:.1%} success")
                
            elif option == "datetime":
                converted = pd.to_datetime(col_data, errors='coerce')
                success_rate = converted.notna().sum() / len(col_data)
                st.metric(f"→ DateTime", f"{success_rate:.1%} success")
                
            elif option == "boolean":
                bool_patterns = ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n']
                if col_data.dtype == 'object':
                    matches = col_data.str.lower().isin(bool_patterns).sum()
                    success_rate = matches / len(col_data)
                    st.metric(f"→ Boolean", f"{success_rate:.1%} success")
                
        except:
            continue

# Data Quality Assessment
st.markdown("---")
st.markdown("### 📊 Data Quality Assessment")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Type Consistency Score")
    
    # Calculate consistency scores for each column
    consistency_scores = []
    for col in df.columns:
        col_data = df[col]
        
        if col_data.dtype == 'object':
            # For object columns, check how many could be converted to a single type
            numeric_convertible = pd.to_numeric(col_data, errors='coerce').notna().sum()
            datetime_convertible = pd.to_datetime(col_data, errors='coerce').notna().sum()
            
            max_convertible = max(numeric_convertible, datetime_convertible)
            consistency = max_convertible / len(col_data)
        else:
            # For already typed columns, consistency is high
            consistency = 0.95
        
        consistency_scores.append({
            'Column': col,
            'Consistency Score': f"{consistency:.1%}",
            'Recommendation': 'Good' if consistency > 0.8 else 'Review' if consistency > 0.5 else 'Convert'
        })
    
    consistency_df = pd.DataFrame(consistency_scores)
    st.dataframe(consistency_df, use_container_width=True)

with col2:
    st.markdown("#### Type Distribution Over Time")
    
    # Show how types have changed if there's a processing log
    if hasattr(st.session_state, 'processing_log'):
        type_changes = [log for log in st.session_state.processing_log if 'Data Type' in log.get('action', '')]
        
        if type_changes:
            st.success(f"✅ {len(type_changes)} type conversions applied")
            for change in type_changes[-3:]:  # Show last 3 changes
                st.write(f"• {change.get('details', 'Type conversion applied')}")
        else:
            st.info("No type conversions applied yet")
    else:
        st.info("No processing history available")

# Export Section
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Current Dataset", type="primary"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"typed_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Duplicates"):
        st.switch_page("pages/05_Duplicates_Detection.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 Type Detection Info")
    
    st.markdown("#### Auto-Detection Features:")
    features = [
        "📅 Smart date recognition",
        "🔢 Numeric pattern detection", 
        "✅ Boolean value identification",
        "🏷️ Category optimization",
        "📊 Confidence scoring"
    ]
    
    for feature in features:
        st.markdown(f"- {feature}")
    
    st.markdown("---")
    st.markdown("#### 💡 Tips")
    st.info("""
    **Best Practices:**
    - Review auto-suggestions before applying
    - Test conversions on sample data first
    - Handle missing values before type conversion
    - Consider memory usage with large datasets
    """)
    
    if st.button("🔄 Reset Data Types"):
        if 'original_dataset' in st.session_state:
            st.session_state.current_dataset = st.session_state.original_dataset.copy()
            st.success("✅ Data types reset to original")
            st.rerun()