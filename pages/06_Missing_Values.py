import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Missing Values", page_icon="🕳️", layout="wide")

st.title("🕳️ Missing Values")
st.markdown("Advanced missing value detection, analysis, and intelligent imputation strategies")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Missing value analysis
missing_analysis = df.isnull().sum()
missing_cols = missing_analysis[missing_analysis > 0].sort_values(ascending=False)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_missing = df.isnull().sum().sum()
    st.metric("Total Missing Values", f"{total_missing:,}")

with col2:
    missing_percentage = (total_missing / (len(df) * len(df.columns))) * 100
    st.metric("Missing Percentage", f"{missing_percentage:.1f}%")

with col3:
    columns_with_missing = len(missing_cols)
    st.metric("Columns with Missing", f"{columns_with_missing}")

with col4:
    complete_rows = len(df.dropna())
    st.metric("Complete Rows", f"{complete_rows:,}")

if len(missing_cols) == 0:
    st.success("✅ No missing values found in the dataset!")
    st.stop()

# Missing value analysis tabs
analysis_tabs = st.tabs(["📊 Overview", "🔍 Pattern Analysis", "🔧 Treatment", "📈 Advanced Imputation"])

with analysis_tabs[0]:
    st.markdown("### 📊 Missing Value Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Missing values by column
        missing_df = pd.DataFrame({
            'Column': missing_cols.index,
            'Missing Count': missing_cols.values,
            'Missing %': (missing_cols.values / len(df) * 100).round(2),
            'Data Type': [str(df[col].dtype) for col in missing_cols.index],
            'Non-Missing Count': [df[col].count() for col in missing_cols.index]
        })
        
        for col in missing_df.columns:
            missing_df[col] = missing_df[col].astype(str)
        
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        # Missing values heatmap
        st.markdown("#### Missing Value Heatmap")
        if len(df) <= 1000:  # Only show for reasonable sizes
            missing_matrix = df.isnull().astype(int)
            fig = px.imshow(missing_matrix.T, aspect="auto", 
                          color_continuous_scale=['white', 'red'],
                          title="Missing Values Pattern")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset too large for heatmap visualization")

with analysis_tabs[1]:
    st.markdown("### 🔍 Missing Value Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Missing Value Correlation")
        
        # Calculate correlation between missing value patterns
        missing_corr = df.isnull().corr()
        
        # Show only columns with missing values
        relevant_cols = [col for col in missing_corr.columns if col in missing_cols.index]
        if len(relevant_cols) > 1:
            subset_corr = missing_corr.loc[relevant_cols, relevant_cols]
            
            fig = px.imshow(subset_corr, 
                          text_auto=True,
                          aspect="auto",
                          title="Missing Value Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Values close to 1 indicate columns tend to be missing together")
        else:
            st.info("Need at least 2 columns with missing values for correlation analysis")
    
    with col2:
        st.markdown("#### Missing Value Distribution")
        
        # Distribution of missing values across rows
        missing_per_row = df.isnull().sum(axis=1)
        
        fig = px.histogram(missing_per_row, 
                         title="Distribution of Missing Values per Row",
                         labels={'value': 'Missing Values per Row', 'count': 'Frequency'})
        st.plotly_chart(fig, use_container_width=True)

with analysis_tabs[2]:
    st.markdown("### 🔧 Missing Value Treatment")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Select Treatment Method")
        
        selected_column = st.selectbox(
            "Choose column to treat:",
            missing_cols.index,
            help="Select a column with missing values"
        )
        
        # Show column info
        col_info = df[selected_column]
        st.write(f"**Column:** {selected_column}")
        st.write(f"**Data Type:** {col_info.dtype}")
        st.write(f"**Missing:** {col_info.isnull().sum()} ({col_info.isnull().sum()/len(df)*100:.1f}%)")
        st.write(f"**Non-Missing:** {col_info.count()}")
        
        # Treatment options based on data type
        if col_info.dtype in ['int64', 'float64']:
            treatment_options = [
                "Drop rows with missing values",
                "Fill with mean",
                "Fill with median", 
                "Fill with mode",
                "Fill with constant value",
                "Forward fill",
                "Backward fill",
                "Interpolate (linear)",
                "KNN Imputation",
                "Iterative Imputation"
            ]
        elif col_info.dtype == 'object':
            treatment_options = [
                "Drop rows with missing values",
                "Fill with mode",
                "Fill with constant value",
                "Forward fill",
                "Backward fill"
            ]
        else:
            treatment_options = [
                "Drop rows with missing values",
                "Fill with constant value",
                "Forward fill",
                "Backward fill"
            ]
        
        treatment_method = st.selectbox(
            "Treatment method:",
            treatment_options,
            help="Choose how to handle missing values"
        )
        
        # Additional parameters
        if "constant value" in treatment_method:
            if col_info.dtype in ['int64', 'float64']:
                constant_value = st.number_input("Constant value:", value=0.0)
            else:
                constant_value = st.text_input("Constant value:", value="Unknown")
        
        if "KNN" in treatment_method:
            n_neighbors = st.slider("Number of neighbors:", 1, 10, 5)
        
        if "Iterative" in treatment_method:
            max_iter = st.slider("Maximum iterations:", 1, 20, 10)
    
    with col2:
        st.markdown("#### Preview & Apply")
        
        # Show sample of current values
        st.markdown("**Current Sample Values:**")
        sample_data = col_info.head(10)
        for i, val in enumerate(sample_data):
            status = "❌ Missing" if pd.isna(val) else f"✅ {val}"
            st.write(f"{i+1}. {status}")
        
        # Apply treatment button
        if st.button("🔧 Apply Treatment", type="primary"):
            try:
                original_missing = col_info.isnull().sum()
                
                if "Drop rows" in treatment_method:
                    df = df.dropna(subset=[selected_column])
                    result_msg = f"Dropped {len(st.session_state.current_dataset) - len(df)} rows"
                
                elif "mean" in treatment_method:
                    fill_value = col_info.mean()
                    df[selected_column].fillna(fill_value, inplace=True)
                    result_msg = f"Filled with mean: {fill_value:.2f}"
                
                elif "median" in treatment_method:
                    fill_value = col_info.median()
                    df[selected_column].fillna(fill_value, inplace=True)
                    result_msg = f"Filled with median: {fill_value:.2f}"
                
                elif "mode" in treatment_method:
                    fill_value = col_info.mode().iloc[0] if len(col_info.mode()) > 0 else "Unknown"
                    df[selected_column].fillna(fill_value, inplace=True)
                    result_msg = f"Filled with mode: {fill_value}"
                
                elif "constant value" in treatment_method:
                    df[selected_column].fillna(constant_value, inplace=True)
                    result_msg = f"Filled with constant: {constant_value}"
                
                elif "Forward fill" in treatment_method:
                    df[selected_column].fillna(method='ffill', inplace=True)
                    result_msg = "Applied forward fill"
                
                elif "Backward fill" in treatment_method:
                    df[selected_column].fillna(method='bfill', inplace=True)
                    result_msg = "Applied backward fill"
                
                elif "Interpolate" in treatment_method:
                    df[selected_column] = df[selected_column].interpolate()
                    result_msg = "Applied linear interpolation"
                
                elif "KNN" in treatment_method:
                    # KNN Imputation for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if selected_column in numeric_cols:
                        imputer = KNNImputer(n_neighbors=n_neighbors)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        result_msg = f"Applied KNN imputation (k={n_neighbors})"
                    else:
                        st.error("KNN imputation only available for numeric columns")
                        st.stop()
                
                elif "Iterative" in treatment_method:
                    # Iterative Imputation for numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if selected_column in numeric_cols:
                        imputer = IterativeImputer(max_iter=max_iter, random_state=42)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                        result_msg = f"Applied iterative imputation (max_iter={max_iter})"
                    else:
                        st.error("Iterative imputation only available for numeric columns")
                        st.stop()
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Missing Value Treatment',
                    'details': f"Applied {treatment_method} to {selected_column}: {result_msg}"
                })
                
                new_missing = df[selected_column].isnull().sum()
                treated_count = original_missing - new_missing
                
                st.success(f"✅ Treatment applied! Handled {treated_count} missing values in {selected_column}")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Treatment failed: {str(e)}")

with analysis_tabs[3]:
    st.markdown("### 📈 Advanced Imputation Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Bulk Treatment Options")
        
        st.markdown("""
        <div class="imputation-option">
            <strong>🎯 Smart Auto-Imputation</strong><br>
            Automatically choose the best imputation method based on data type and distribution
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🤖 Apply Smart Auto-Imputation", type="primary"):
            treatments_applied = []
            
            for col in missing_cols.index:
                col_data = df[col]
                missing_ratio = col_data.isnull().sum() / len(col_data)
                
                try:
                    # Strategy based on data type and missing ratio
                    if missing_ratio > 0.7:
                        # Too many missing - consider dropping column
                        treatments_applied.append(f"⚠️ {col}: {missing_ratio:.1%} missing - consider dropping")
                        continue
                    
                    if col_data.dtype in ['int64', 'float64']:
                        # Numeric columns
                        if missing_ratio < 0.1:
                            # Few missing - use median
                            fill_value = col_data.median()
                            df[col].fillna(fill_value, inplace=True)
                            treatments_applied.append(f"✅ {col}: Filled with median ({fill_value:.2f})")
                        else:
                            # More missing - use KNN if possible
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 1:
                                imputer = KNNImputer(n_neighbors=5)
                                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                                treatments_applied.append(f"✅ {col}: Applied KNN imputation")
                            else:
                                fill_value = col_data.median()
                                df[col].fillna(fill_value, inplace=True)
                                treatments_applied.append(f"✅ {col}: Filled with median ({fill_value:.2f})")
                    
                    elif col_data.dtype == 'object':
                        # Categorical columns
                        mode_values = col_data.mode()
                        if len(mode_values) > 0:
                            fill_value = mode_values.iloc[0]
                            df[col].fillna(fill_value, inplace=True)
                            treatments_applied.append(f"✅ {col}: Filled with mode ('{fill_value}')")
                        else:
                            df[col].fillna("Unknown", inplace=True)
                            treatments_applied.append(f"✅ {col}: Filled with 'Unknown'")
                    
                    else:
                        # Other data types - forward fill
                        df[col].fillna(method='ffill', inplace=True)
                        df[col].fillna(method='bfill', inplace=True)
                        treatments_applied.append(f"✅ {col}: Applied forward/backward fill")
                
                except Exception as e:
                    treatments_applied.append(f"❌ {col}: Failed - {str(e)}")
            
            # Update session state
            st.session_state.current_dataset = df
            
            # Show results
            st.markdown("#### Treatment Results:")
            for treatment in treatments_applied:
                st.markdown(treatment)
            
            st.success(f"✅ Auto-imputation complete! Processed {len(treatments_applied)} columns.")
            st.rerun()
        
        st.markdown("""
        <div class="imputation-option">
            <strong>🗑️ Drop Strategy</strong><br>
            Remove rows or columns with high missing value ratios
        </div>
        """, unsafe_allow_html=True)
        
        drop_threshold = st.slider(
            "Missing value threshold for dropping columns:",
            0.1, 0.9, 0.5,
            help="Columns with missing values above this threshold will be dropped"
        )
        
        if st.button("🗑️ Drop High-Missing Columns"):
            initial_cols = len(df.columns)
            
            # Find columns to drop
            cols_to_drop = []
            for col in df.columns:
                missing_ratio = df[col].isnull().sum() / len(df)
                if missing_ratio > drop_threshold:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                st.session_state.current_dataset = df
                
                st.success(f"✅ Dropped {len(cols_to_drop)} columns with >{drop_threshold:.0%} missing values")
                st.write("Dropped columns:", cols_to_drop)
                st.rerun()
            else:
                st.info(f"No columns found with >{drop_threshold:.0%} missing values")
    
    with col2:
        st.markdown("#### Treatment Recommendations")
        
        recommendations = []
        
        for col in missing_cols.index:
            col_data = df[col]
            missing_ratio = col_data.isnull().sum() / len(col_data)
            
            if missing_ratio > 0.5:
                recommendation = "Consider dropping column"
                priority = "🔴 High"
            elif col_data.dtype in ['int64', 'float64']:
                if missing_ratio < 0.1:
                    recommendation = "Fill with median/mean"
                    priority = "🟢 Low"
                else:
                    recommendation = "Use KNN/Iterative imputation"
                    priority = "🟡 Medium"
            elif col_data.dtype == 'object':
                recommendation = "Fill with mode or 'Unknown'"
                priority = "🟢 Low"
            else:
                recommendation = "Forward/backward fill"
                priority = "🟡 Medium"
            
            recommendations.append({
                'Column': col,
                'Missing %': f"{missing_ratio:.1%}",
                'Priority': priority,
                'Recommendation': recommendation
            })
        
        # Display recommendations
        rec_df = pd.DataFrame(recommendations)
        for col in rec_df.columns:
            rec_df[col] = rec_df[col].astype(str)
        
        st.dataframe(rec_df, use_container_width=True)

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Processed Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"missing_values_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Outlier Detection"):
        st.switch_page("pages/07_Outlier_Detection.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🕳️ Missing Value Guide")
    
    st.markdown("#### Imputation Methods:")
    methods = [
        "**Mean/Median:** For numeric data",
        "**Mode:** For categorical data",
        "**Forward/Backward Fill:** For time series",
        "**KNN:** Uses similar records",
        "**Iterative:** Multiple imputation",
        "**Constant:** Fixed replacement value"
    ]
    
    for method in methods:
        st.markdown(f"• {method}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Understand why data is missing
    • Consider the impact on analysis
    • Test different imputation methods
    • Document all treatments applied
    • Validate results after imputation
    """)
    
    if st.session_state.get('processing_log'):
        st.markdown("#### 📝 Recent Treatments")
        recent_missing = [log for log in st.session_state.processing_log if 'Missing' in log.get('action', '')][-3:]
        
        for action in recent_missing:
            st.caption(f"✅ {action.get('details', 'Missing value treatment applied')}")