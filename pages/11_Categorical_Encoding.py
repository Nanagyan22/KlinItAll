import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import TargetEncoder
from category_encoders import BinaryEncoder, HashingEncoder, OneHotEncoder as CatOneHotEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Categorical Encoding", page_icon="🏷️", layout="wide")

st.title("🏷️ Categorical Encoding")
st.markdown("Advanced categorical encoding with multiple algorithms and intelligent strategy selection")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(categorical_cols) == 0:
    st.warning("⚠️ No categorical columns found for encoding.")
    st.stop()

# Encoding utility functions
def get_encoding_recommendation(series, max_categories_onehot=10):
    """Recommend encoding strategy based on column characteristics"""
    unique_count = series.nunique()
    total_count = len(series)
    
    if unique_count <= 2:
        return "Binary/Label Encoding", "Low cardinality, simple encoding sufficient"
    elif unique_count <= max_categories_onehot:
        return "One-Hot Encoding", "Moderate cardinality, preserves category independence"
    elif unique_count <= 50:
        return "Target/Ordinal Encoding", "Medium cardinality, structured encoding recommended"
    else:
        return "Hashing/Binary Encoding", "High cardinality, dimension reduction needed"

def analyze_categorical_column(df, column):
    """Comprehensive analysis of a categorical column"""
    series = df[column]
    
    analysis = {
        'unique_count': series.nunique(),
        'missing_count': series.isnull().sum(),
        'most_frequent': series.mode().iloc[0] if len(series.mode()) > 0 else None,
        'most_frequent_count': series.value_counts().iloc[0] if len(series.value_counts()) > 0 else 0,
        'least_frequent': series.value_counts().index[-1] if len(series.value_counts()) > 0 else None,
        'least_frequent_count': series.value_counts().iloc[-1] if len(series.value_counts()) > 0 else 0,
        'value_counts': series.value_counts().head(10),
        'cardinality_ratio': series.nunique() / len(series) if len(series) > 0 else 0
    }
    
    return analysis

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Categorical Columns", len(categorical_cols))

with col2:
    total_categories = sum(df[col].nunique() for col in categorical_cols)
    st.metric("Total Categories", f"{total_categories:,}")

with col3:
    avg_cardinality = total_categories / len(categorical_cols) if len(categorical_cols) > 0 else 0
    st.metric("Avg Cardinality", f"{avg_cardinality:.1f}")

with col4:
    high_cardinality_cols = sum(1 for col in categorical_cols if df[col].nunique() > 20)
    st.metric("High Cardinality Cols", high_cardinality_cols)

# Encoding tabs
encoding_tabs = st.tabs(["📊 Analysis", "🎯 Strategy Selection", "🔧 Encoding Methods", "⚙️ Advanced Encoding"])

with encoding_tabs[0]:
    st.markdown("### 📊 Categorical Column Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column analysis table
        analysis_data = []
        
        for col in categorical_cols:
            analysis = analyze_categorical_column(df, col)
            recommendation, reason = get_encoding_recommendation(df[col])
            
            analysis_data.append({
                'Column': col,
                'Unique Values': analysis['unique_count'],
                'Missing': analysis['missing_count'],
                'Cardinality Ratio': f"{analysis['cardinality_ratio']:.3f}",
                'Most Frequent': str(analysis['most_frequent'])[:20] + "..." if analysis['most_frequent'] and len(str(analysis['most_frequent'])) > 20 else str(analysis['most_frequent']),
                'Frequency': analysis['most_frequent_count'],
                'Recommended Encoding': recommendation
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Convert to strings to avoid Arrow issues
        for col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].astype(str)
        
        st.dataframe(analysis_df, use_container_width=True)
    
    with col2:
        # Cardinality distribution
        cardinalities = [df[col].nunique() for col in categorical_cols]
        
        fig = px.histogram(
            x=cardinalities,
            nbins=20,
            title="Cardinality Distribution",
            labels={'x': 'Unique Values', 'y': 'Number of Columns'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Column selection for detailed analysis
        selected_col = st.selectbox("Select column for detailed analysis:", categorical_cols)
        
        if selected_col:
            analysis = analyze_categorical_column(df, selected_col)
            
            st.markdown(f"#### {selected_col} Details")
            st.write(f"**Unique Values:** {analysis['unique_count']:,}")
            st.write(f"**Missing Values:** {analysis['missing_count']:,}")
            st.write(f"**Cardinality Ratio:** {analysis['cardinality_ratio']:.3f}")
            
            # Top categories
            st.markdown("**Top Categories:**")
            for cat, count in analysis['value_counts'].head(5).items():
                percentage = (count / len(df[selected_col])) * 100
                st.write(f"• {cat}: {count:,} ({percentage:.1f}%)")

with encoding_tabs[1]:
    st.markdown("### 🎯 Encoding Strategy Selection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Smart Strategy Recommendations")
        
        # Generate recommendations for each column
        recommendations = []
        
        for col in categorical_cols:
            analysis = analyze_categorical_column(df, col)
            recommendation, reason = get_encoding_recommendation(df[col])
            
            recommendations.append({
                'column': col,
                'unique_count': analysis['unique_count'],
                'strategy': recommendation,
                'reason': reason,
                'priority': 'High' if analysis['unique_count'] > 50 else 'Medium' if analysis['unique_count'] > 10 else 'Low'
            })
        
        # Display recommendations
        for rec in recommendations:
            priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}[rec['priority']]
            
            st.markdown(f"""
            <div class="encoding-card">
                <strong>{priority_color} {rec['column']}</strong> ({rec['unique_count']} categories)<br>
                <em>Recommended:</em> {rec['strategy']}<br>
                <small>{rec['reason']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Bulk strategy application
        st.markdown("#### Apply Bulk Strategy")
        
        bulk_strategy = st.selectbox(
            "Select strategy to apply to all suitable columns:",
            ["Smart Auto-Selection", "One-Hot Encoding (Low Cardinality)", "Label Encoding (All)", "Target Encoding (with target)"],
            help="Choose a strategy to apply to multiple columns at once"
        )
        
        if bulk_strategy == "Target Encoding (with target)" and numeric_cols:
            target_column = st.selectbox("Select target column:", numeric_cols, help="Numeric column for target encoding")
        
        if st.button("🚀 Apply Bulk Strategy", type="primary"):
            encoded_columns = []
            
            try:
                for col in categorical_cols:
                    unique_count = df[col].nunique()
                    
                    if bulk_strategy == "Smart Auto-Selection":
                        if unique_count <= 2:
                            # Binary encoding for 2 categories
                            le = LabelEncoder()
                            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str).fillna('missing'))
                            encoded_columns.append(f"{col}_encoded")
                        
                        elif unique_count <= 10:
                            # One-hot encoding for low cardinality
                            encoded_df = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                            df = pd.concat([df, encoded_df], axis=1)
                            encoded_columns.extend(encoded_df.columns.tolist())
                        
                        elif unique_count <= 50:
                            # Label encoding for medium cardinality
                            le = LabelEncoder()
                            df[f"{col}_label_encoded"] = le.fit_transform(df[col].astype(str).fillna('missing'))
                            encoded_columns.append(f"{col}_label_encoded")
                        
                        else:
                            # Hashing encoding for high cardinality
                            he = HashingEncoder(n_components=8, cols=[col])
                            encoded_df = he.fit_transform(df[[col]])
                            encoded_df.columns = [f"{col}_hash_{i}" for i in range(len(encoded_df.columns))]
                            df = pd.concat([df, encoded_df], axis=1)
                            encoded_columns.extend(encoded_df.columns.tolist())
                    
                    elif bulk_strategy == "One-Hot Encoding (Low Cardinality)" and unique_count <= 20:
                        encoded_df = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                        df = pd.concat([df, encoded_df], axis=1)
                        encoded_columns.extend(encoded_df.columns.tolist())
                    
                    elif bulk_strategy == "Label Encoding (All)":
                        le = LabelEncoder()
                        df[f"{col}_label_encoded"] = le.fit_transform(df[col].astype(str).fillna('missing'))
                        encoded_columns.append(f"{col}_label_encoded")
                    
                    elif bulk_strategy == "Target Encoding (with target)" and 'target_column' in locals():
                        try:
                            te = TargetEncoder()
                            df[f"{col}_target_encoded"] = te.fit_transform(df[col].astype(str).fillna('missing'), df[target_column])
                            encoded_columns.append(f"{col}_target_encoded")
                        except:
                            # Fallback to label encoding
                            le = LabelEncoder()
                            df[f"{col}_label_encoded"] = le.fit_transform(df[col].astype(str).fillna('missing'))
                            encoded_columns.append(f"{col}_label_encoded")
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Categorical Encoding',
                    'details': f"Applied {bulk_strategy} to {len(categorical_cols)} columns, created {len(encoded_columns)} new encoded columns"
                })
                
                st.success(f"✅ Applied {bulk_strategy}! Created {len(encoded_columns)} new encoded columns.")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Bulk encoding failed: {str(e)}")
    
    with col2:
        st.markdown("#### Encoding Method Comparison")
        
        comparison_data = {
            'Method': ['One-Hot', 'Label', 'Target', 'Binary', 'Hashing'],
            'Best For': ['Low cardinality', 'Ordinal data', 'Predictive tasks', 'Memory efficiency', 'High cardinality'],
            'Output Size': ['Large', 'Single column', 'Single column', 'Log₂(n) columns', 'Fixed size'],
            'Interpretability': ['High', 'Low', 'Medium', 'Low', 'Low'],
            'ML Performance': ['Good', 'Varies', 'Excellent', 'Good', 'Good']
        }
        
        comp_df = pd.DataFrame(comparison_data)
        
        # Convert to strings to avoid Arrow issues
        for col in comp_df.columns:
            comp_df[col] = comp_df[col].astype(str)
        
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("#### Column Priority Analysis")
        
        # Prioritize columns by complexity/importance
        priority_analysis = []
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            missing_ratio = df[col].isnull().sum() / len(df)
            
            # Calculate complexity score
            if unique_count <= 5:
                complexity = "Low"
                priority_score = 1
            elif unique_count <= 20:
                complexity = "Medium" 
                priority_score = 2
            else:
                complexity = "High"
                priority_score = 3
            
            # Adjust for missing values
            if missing_ratio > 0.2:
                priority_score += 1
                complexity += " (Missing Data)"
            
            priority_analysis.append({
                'Column': col,
                'Complexity': complexity,
                'Unique Values': unique_count,
                'Priority Score': priority_score
            })
        
        # Sort by priority score
        priority_analysis.sort(key=lambda x: x['Priority Score'], reverse=True)
        
        st.markdown("**Processing Priority Order:**")
        for i, item in enumerate(priority_analysis, 1):
            st.write(f"{i}. **{item['Column']}** - {item['Complexity']} ({item['Unique Values']} categories)")

with encoding_tabs[2]:
    st.markdown("### 🔧 Individual Encoding Methods")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Choose Column and Method")
        
        encoding_column = st.selectbox("Select column to encode:", categorical_cols, key="encoding_col")
        
        encoding_method = st.selectbox(
            "Encoding method:",
            [
                "One-Hot Encoding",
                "Label Encoding", 
                "Ordinal Encoding",
                "Binary Encoding",
                "Target Encoding",
                "Frequency Encoding",
                "Hashing Encoding"
            ],
            help="Choose the encoding method to apply"
        )
        
        # Method-specific parameters
        if encoding_method == "Ordinal Encoding":
            # Allow user to specify category order
            categories = df[encoding_column].unique().tolist()
            category_order = st.multiselect(
                "Specify category order (optional):",
                categories,
                help="Drag to reorder, or leave empty for automatic ordering"
            )
        
        elif encoding_method == "Target Encoding":
            if numeric_cols:
                target_col = st.selectbox("Select target column:", numeric_cols, help="Numeric target for encoding")
            else:
                st.error("Target encoding requires a numeric target column")
                st.stop()
        
        elif encoding_method == "Hashing Encoding":
            n_components = st.slider("Number of hash components:", 2, 16, 8, help="Number of output columns")
        
        elif encoding_method == "Binary Encoding":
            st.info("Binary encoding will create log₂(n) columns where n is the number of unique categories")
        
        # Preview of encoding
        if st.button("👁️ Preview Encoding", key="preview_encoding"):
            try:
                preview_data = df[encoding_column].head(10)
                
                st.markdown("**Original Values:**")
                for i, val in enumerate(preview_data, 1):
                    st.write(f"{i}. {val}")
                
                # Show what encoding would produce
                st.markdown("**Encoded Preview:**")
                
                if encoding_method == "One-Hot Encoding":
                    encoded_preview = pd.get_dummies(preview_data, prefix=encoding_column)
                    st.dataframe(encoded_preview.head(), use_container_width=True)
                
                elif encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    encoded_vals = le.fit_transform(preview_data.astype(str))
                    for i, (orig, enc) in enumerate(zip(preview_data, encoded_vals), 1):
                        st.write(f"{i}. {orig} → {enc}")
                
                elif encoding_method == "Binary Encoding":
                    be = BinaryEncoder(cols=[encoding_column])
                    encoded_preview = be.fit_transform(pd.DataFrame({encoding_column: preview_data}))
                    st.dataframe(encoded_preview.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")
    
    with col2:
        st.markdown("#### Apply Encoding")
        
        if st.button("🔧 Apply Encoding", type="primary", key="apply_individual"):
            try:
                original_columns = len(df.columns)
                
                if encoding_method == "One-Hot Encoding":
                    encoded_df = pd.get_dummies(df[encoding_column], prefix=encoding_column, dummy_na=True)
                    df = pd.concat([df, encoded_df], axis=1)
                    new_columns = encoded_df.columns.tolist()
                    result_msg = f"Created {len(new_columns)} one-hot encoded columns"
                
                elif encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    df[f"{encoding_column}_label_encoded"] = le.fit_transform(df[encoding_column].astype(str).fillna('missing'))
                    new_columns = [f"{encoding_column}_label_encoded"]
                    result_msg = f"Created label encoded column: {new_columns[0]}"
                
                elif encoding_method == "Ordinal Encoding":
                    oe = OrdinalEncoder()
                    if category_order:
                        # Use specified order
                        oe.set_params(categories=[category_order])
                    df[f"{encoding_column}_ordinal_encoded"] = oe.fit_transform(df[[encoding_column]].fillna('missing'))
                    new_columns = [f"{encoding_column}_ordinal_encoded"]
                    result_msg = f"Created ordinal encoded column: {new_columns[0]}"
                
                elif encoding_method == "Binary Encoding":
                    be = BinaryEncoder(cols=[encoding_column])
                    encoded_df = be.fit_transform(df[[encoding_column]])
                    
                    # Rename columns to be more descriptive
                    encoded_df.columns = [f"{encoding_column}_binary_{i}" for i in range(len(encoded_df.columns))]
                    
                    df = pd.concat([df, encoded_df], axis=1)
                    new_columns = encoded_df.columns.tolist()
                    result_msg = f"Created {len(new_columns)} binary encoded columns"
                
                elif encoding_method == "Target Encoding":
                    te = TargetEncoder()
                    df[f"{encoding_column}_target_encoded"] = te.fit_transform(
                        df[encoding_column].astype(str).fillna('missing'), 
                        df[target_col]
                    )
                    new_columns = [f"{encoding_column}_target_encoded"]
                    result_msg = f"Created target encoded column: {new_columns[0]}"
                
                elif encoding_method == "Frequency Encoding":
                    freq_map = df[encoding_column].value_counts().to_dict()
                    df[f"{encoding_column}_freq_encoded"] = df[encoding_column].map(freq_map)
                    new_columns = [f"{encoding_column}_freq_encoded"]
                    result_msg = f"Created frequency encoded column: {new_columns[0]}"
                
                elif encoding_method == "Hashing Encoding":
                    he = HashingEncoder(n_components=n_components, cols=[encoding_column])
                    encoded_df = he.fit_transform(df[[encoding_column]])
                    encoded_df.columns = [f"{encoding_column}_hash_{i}" for i in range(len(encoded_df.columns))]
                    
                    df = pd.concat([df, encoded_df], axis=1)
                    new_columns = encoded_df.columns.tolist()
                    result_msg = f"Created {len(new_columns)} hash encoded columns"
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Categorical Encoding',
                    'details': f"Applied {encoding_method} to {encoding_column}: {result_msg}"
                })
                
                st.success(f"✅ {result_msg}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Encoding failed: {str(e)}")
        
        # Show current column info
        if encoding_column:
            col_analysis = analyze_categorical_column(df, encoding_column)
            
            st.markdown("#### Column Information")
            st.write(f"**Unique Categories:** {col_analysis['unique_count']:,}")
            st.write(f"**Missing Values:** {col_analysis['missing_count']:,}")
            st.write(f"**Most Frequent:** {col_analysis['most_frequent']} ({col_analysis['most_frequent_count']:,} times)")
            
            # Estimate output size
            if encoding_method == "One-Hot Encoding":
                output_size = col_analysis['unique_count'] + (1 if col_analysis['missing_count'] > 0 else 0)
                st.write(f"**Expected Output Columns:** {output_size}")
            elif encoding_method == "Binary Encoding":
                output_size = max(1, int(np.ceil(np.log2(col_analysis['unique_count']))))
                st.write(f"**Expected Output Columns:** {output_size}")
            elif encoding_method == "Hashing Encoding":
                st.write(f"**Expected Output Columns:** {n_components if 'n_components' in locals() else 8}")
            else:
                st.write("**Expected Output Columns:** 1")

with encoding_tabs[3]:
    st.markdown("### ⚙️ Advanced Encoding Techniques")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Custom Encoding")
        
        custom_col = st.selectbox("Select column for custom encoding:", categorical_cols, key="custom_col")
        
        custom_method = st.selectbox(
            "Custom encoding method:",
            [
                "Manual Category Mapping",
                "Statistical Encoding (Mean/Median)",
                "Rank Encoding",
                "Leave-One-Out Encoding",
                "Weight of Evidence Encoding"
            ]
        )
        
        if custom_method == "Manual Category Mapping":
            st.markdown("#### Define Custom Mappings")
            
            categories = df[custom_col].unique().tolist()
            
            # Create mapping interface
            mappings = {}
            for cat in categories[:10]:  # Limit to 10 for UI purposes
                mappings[cat] = st.number_input(f"Map '{cat}' to:", value=0.0, key=f"map_{cat}")
            
            if len(categories) > 10:
                st.info(f"Showing first 10 categories. Total: {len(categories)}")
        
        elif custom_method == "Statistical Encoding (Mean/Median)":
            if numeric_cols:
                stat_target = st.selectbox("Target column for statistics:", numeric_cols, key="stat_target")
                stat_method = st.selectbox("Statistical method:", ["mean", "median", "std", "min", "max"])
        
        elif custom_method == "Rank Encoding":
            if numeric_cols:
                rank_target = st.selectbox("Target column for ranking:", numeric_cols, key="rank_target")
                rank_method = st.selectbox("Ranking method:", ["mean_rank", "median_rank", "frequency_rank"])
        
        if st.button("🎯 Apply Custom Encoding", key="custom_encoding"):
            try:
                if custom_method == "Manual Category Mapping":
                    df[f"{custom_col}_custom_mapped"] = df[custom_col].map(mappings).fillna(0)
                    result_msg = f"Applied custom mapping to {custom_col}"
                
                elif custom_method == "Statistical Encoding (Mean/Median)":
                    if stat_method == "mean":
                        encoding_map = df.groupby(custom_col)[stat_target].mean().to_dict()
                    elif stat_method == "median":
                        encoding_map = df.groupby(custom_col)[stat_target].median().to_dict()
                    elif stat_method == "std":
                        encoding_map = df.groupby(custom_col)[stat_target].std().to_dict()
                    elif stat_method == "min":
                        encoding_map = df.groupby(custom_col)[stat_target].min().to_dict()
                    else:  # max
                        encoding_map = df.groupby(custom_col)[stat_target].max().to_dict()
                    
                    df[f"{custom_col}_{stat_method}_encoded"] = df[custom_col].map(encoding_map)
                    result_msg = f"Applied {stat_method} encoding to {custom_col}"
                
                elif custom_method == "Rank Encoding":
                    if rank_method == "mean_rank":
                        rank_map = df.groupby(custom_col)[rank_target].mean().rank().to_dict()
                    elif rank_method == "median_rank":
                        rank_map = df.groupby(custom_col)[rank_target].median().rank().to_dict()
                    else:  # frequency_rank
                        rank_map = df[custom_col].value_counts().rank(ascending=False).to_dict()
                    
                    df[f"{custom_col}_rank_encoded"] = df[custom_col].map(rank_map)
                    result_msg = f"Applied {rank_method} encoding to {custom_col}"
                
                # Update session state
                st.session_state.current_dataset = df
                
                st.success(f"✅ {result_msg}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Custom encoding failed: {str(e)}")
    
    with col2:
        st.markdown("#### Encoding Validation")
        
        # Show encoded columns
        encoded_cols = [col for col in df.columns if any(suffix in col for suffix in ['_encoded', '_hash_', '_binary_'])]
        
        if encoded_cols:
            st.markdown("#### Available Encoded Columns")
            
            for col in encoded_cols[:10]:  # Show first 10
                original_col = col.split('_')[0] + '_' + col.split('_')[1] if len(col.split('_')) > 1 else col
                encoding_type = col.split('_')[-1] if '_' in col else 'unknown'
                st.write(f"• **{col}** ({encoding_type} encoding)")
            
            if len(encoded_cols) > 10:
                st.write(f"... and {len(encoded_cols) - 10} more")
            
            # Validation options
            st.markdown("#### Validation Options")
            
            validate_col = st.selectbox("Select encoded column to validate:", encoded_cols[:20], key="validate_encoded")
            
            if st.button("🔍 Validate Encoding"):
                try:
                    # Basic validation
                    col_data = df[validate_col]
                    
                    validation_results = {
                        'Column': validate_col,
                        'Data Type': str(col_data.dtype),
                        'Unique Values': col_data.nunique(),
                        'Missing Values': col_data.isnull().sum(),
                        'Min Value': col_data.min() if col_data.dtype in ['int64', 'float64'] else 'N/A',
                        'Max Value': col_data.max() if col_data.dtype in ['int64', 'float64'] else 'N/A',
                        'Memory Usage': f"{col_data.memory_usage(deep=True) / 1024:.1f} KB"
                    }
                    
                    for key, value in validation_results.items():
                        st.write(f"**{key}:** {value}")
                    
                    # Sample values
                    st.markdown("**Sample Values:**")
                    sample_vals = col_data.dropna().head(10).tolist()
                    for i, val in enumerate(sample_vals, 1):
                        st.write(f"{i}. {val}")
                    
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
        
        else:
            st.info("No encoded columns found. Apply encoding methods first.")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Encoded Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"encoded_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Scaling & Normalization"):
        st.switch_page("pages/12_Scaling_Normalization.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🏷️ Categorical Encoding Guide")
    
    st.markdown("#### Encoding Methods:")
    methods = [
        "**One-Hot:** Creates binary columns",
        "**Label:** Assigns numeric labels",
        "**Target:** Uses target variable correlation", 
        "**Binary:** Efficient binary representation",
        "**Hashing:** Fixed-size hash encoding"
    ]
    
    for method in methods:
        st.markdown(f"• {method}")
    
    st.markdown("---")
    st.markdown("#### Selection Guidelines:")
    
    guidelines = [
        "**Low cardinality (≤10):** One-Hot",
        "**Medium cardinality (≤50):** Target/Label",
        "**High cardinality (>50):** Hashing/Binary",
        "**Ordinal data:** Ordinal encoding",
        "**Memory constraints:** Binary encoding"
    ]
    
    for guideline in guidelines:
        st.markdown(f"• {guideline}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Consider target variable relationship
    • Handle missing values appropriately
    • Validate encoding output
    • Monitor dimensionality impact
    • Test multiple strategies for ML
    """)