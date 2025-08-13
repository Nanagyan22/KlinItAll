import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Outlier Detection", page_icon="🎯", layout="wide")

st.title("🎯 Outlier Detection")
st.markdown("Advanced outlier detection using multiple algorithms with intelligent treatment options")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.warning("⚠️ No numeric columns found for outlier detection.")
    st.stop()

# Outlier Detection Functions
def detect_outliers_iqr(data, column, multiplier=1.5):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    
    return {
        'outliers': data[outlier_mask],
        'outlier_indices': data[outlier_mask].index.tolist(),
        'bounds': (lower_bound, upper_bound),
        'count': outlier_mask.sum(),
        'percentage': (outlier_mask.sum() / len(data)) * 100
    }

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outlier_mask = z_scores > threshold
    
    # Map back to original dataframe indices
    outlier_indices = data[column].dropna()[outlier_mask].index.tolist()
    #outlier_indices = data[column].dropna().loc[outlier_mask].index.tolist()

    #outlier_indices = data[column].dropna().iloc[outlier_mask].index.tolist()
    
    return {
        'outliers': data.loc[outlier_indices],
        'outlier_indices': outlier_indices,
        'z_scores': z_scores,
        'threshold': threshold,
        'count': len(outlier_indices),
        'percentage': (len(outlier_indices) / len(data)) * 100
    }

def detect_outliers_isolation_forest(data, columns, contamination=0.1):
    """Detect outliers using Isolation Forest"""
    clf = IsolationForest(contamination=contamination, random_state=42)
    
    # Use only numeric columns and handle missing values
    data_clean = data[columns].fillna(data[columns].median())
    
    outlier_labels = clf.fit_predict(data_clean)
    outlier_mask = outlier_labels == -1
    
    return {
        'outliers': data[outlier_mask],
        'outlier_indices': data[outlier_mask].index.tolist(),
        'contamination': contamination,
        'count': outlier_mask.sum(),
        'percentage': (outlier_mask.sum() / len(data)) * 100
    }

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Numeric Columns", len(numeric_cols))

with col2:
    total_rows = len(df)
    st.metric("Total Rows", f"{total_rows:,}")

with col3:
    # Quick IQR outlier count for first numeric column
    if numeric_cols:
        sample_outliers = detect_outliers_iqr(df, numeric_cols[0])
        st.metric("Sample Outliers (IQR)", f"{sample_outliers['count']}")

with col4:
    st.metric("Analysis Methods", "4 Algorithms")

# Analysis tabs
outlier_tabs = st.tabs(["📊 Overview", "📈 IQR Method", "📉 Z-Score Method", "🤖 Isolation Forest", "🔧 Treatment"])

with outlier_tabs[0]:
    st.markdown("### 📊 Outlier Detection Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column selection for analysis
        selected_column = st.selectbox(
            "Select column for detailed analysis:",
            numeric_cols,
            help="Choose a numeric column to analyze for outliers"
        )
        
        # Basic statistics
        col_data = df[selected_column].dropna()
        
        stats_data = {
            'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                f"{len(col_data):,}",
                f"{col_data.mean():.2f}",
                f"{col_data.median():.2f}",
                f"{col_data.std():.2f}",
                f"{col_data.min():.2f}",
                f"{col_data.max():.2f}",
                f"{col_data.skew():.2f}",
                f"{col_data.kurtosis():.2f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Distribution plot
        fig = px.histogram(df, x=selected_column, nbins=50, 
                          title=f"Distribution of {selected_column}")
        fig.add_vline(x=col_data.mean(), line_dash="dash", line_color="red", 
                     annotation_text="Mean")
        fig.add_vline(x=col_data.median(), line_dash="dash", line_color="green", 
                     annotation_text="Median")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Quick outlier summary for selected column
        st.markdown("#### Quick Outlier Summary")
        
        # IQR outliers
        iqr_result = detect_outliers_iqr(df, selected_column)
        st.metric("IQR Outliers", f"{iqr_result['count']}", f"{iqr_result['percentage']:.1f}%")
        
        # Z-score outliers
        zscore_result = detect_outliers_zscore(df, selected_column)
        st.metric("Z-Score Outliers", f"{zscore_result['count']}", f"{zscore_result['percentage']:.1f}%")
        
        # Isolation Forest outliers (single column)
        if_result = detect_outliers_isolation_forest(df, [selected_column])
        st.metric("Isolation Forest", f"{if_result['count']}", f"{if_result['percentage']:.1f}%")
        
        # Box plot
        fig_box = px.box(df, y=selected_column, title=f"Box Plot: {selected_column}")
        st.plotly_chart(fig_box, use_container_width=True)

with outlier_tabs[1]:
    st.markdown("### 📈 IQR (Interquartile Range) Method")
    
    st.markdown("""
    <div class="method-info">
        <strong>📊 IQR Method:</strong><br>
        • Uses quartiles (Q1, Q3) to define outliers<br>
        • Outliers: values < Q1 - 1.5×IQR or > Q3 + 1.5×IQR<br>
        • Good for: Skewed distributions, robust to extreme values<br>
        • Best for: Initial outlier screening
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        # IQR Configuration
        iqr_column = st.selectbox("Column for IQR analysis:", numeric_cols, key="iqr_col")
        iqr_multiplier = st.slider("IQR Multiplier:", 1.0, 3.0, 1.5, 0.1,
                                  help="Higher values = less sensitive to outliers")

        if st.button("🔍 Detect IQR Outliers", type="primary"):
            iqr_results = detect_outliers_iqr(df, iqr_column, iqr_multiplier)

            st.markdown(f"#### Results for {iqr_column}")
            st.metric("Outliers Found", f"{iqr_results['count']}",
                     f"{iqr_results['percentage']:.2f}% of data")

            st.write(f"**Lower Bound:** {iqr_results['bounds'][0]:.2f}")
            st.write(f"**Upper Bound:** {iqr_results['bounds'][1]:.2f}")

            if iqr_results['count'] > 0:
                st.markdown("**Sample Outliers:**")
                sample_outliers = iqr_results['outliers'].head(10)[[iqr_column]]
                for col in sample_outliers.columns:
                    sample_outliers[col] = sample_outliers[col].astype(str)
                st.dataframe(sample_outliers, use_container_width=True)

                # Store results in session state for treatment
                st.session_state.iqr_outliers = iqr_results
            else:
                st.info("✅ No outliers detected with current settings")

    with col2:
        # Visualization
        if 'iqr_outliers' in st.session_state and st.session_state.iqr_outliers['count'] > 0:
            outlier_data = st.session_state.iqr_outliers
            
            # Scatter plot with outliers highlighted
            fig = go.Figure()
            
            # Normal points
            normal_mask = ~df.index.isin(outlier_data['outlier_indices'])
            fig.add_trace(go.Scatter(
                x=df[normal_mask].index,
                y=df[normal_mask][iqr_column],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            ))
            
            # Outliers
            fig.add_trace(go.Scatter(
                x=df[df.index.isin(outlier_data['outlier_indices'])].index,
                y=df[df.index.isin(outlier_data['outlier_indices'])][iqr_column],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=6)
            ))
            
            # Bounds
            fig.add_hline(y=outlier_data['bounds'][0], line_dash="dash", line_color="orange")
            fig.add_hline(y=outlier_data['bounds'][1], line_dash="dash", line_color="orange")
            
            fig.update_layout(title=f"IQR Outliers: {iqr_column}", 
                            xaxis_title="Index", yaxis_title=iqr_column)
            st.plotly_chart(fig, use_container_width=True)

with outlier_tabs[2]:
    st.markdown("### 📉 Z-Score Method")
    
    st.markdown("""
    <div class="method-info">
        <strong>📊 Z-Score Method:</strong><br>
        • Measures how many standard deviations away from mean<br>
        • Outliers: |z-score| > threshold (usually 2 or 3)<br>
        • Good for: Normally distributed data<br>
        • Best for: Data with known normal distribution
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Z-Score Configuration
        zscore_column = st.selectbox("Column for Z-Score analysis:", numeric_cols, key="zscore_col")
        zscore_threshold = st.slider("Z-Score Threshold:", 1.5, 4.0, 3.0, 0.1,
                                   help="Higher values = less sensitive to outliers")
        
        if st.button("🔍 Detect Z-Score Outliers", type="primary"):
            zscore_results = detect_outliers_zscore(df, zscore_column, zscore_threshold)
            
            st.markdown(f"#### Results for {zscore_column}")
            st.metric("Outliers Found", f"{zscore_results['count']}", 
                     f"{zscore_results['percentage']:.2f}% of data")
            
            st.write(f"**Threshold:** ±{zscore_results['threshold']}")
            
            if zscore_results['count'] > 0:
                st.markdown("**Sample Outliers:**")
                sample_outliers = zscore_results['outliers'].head(10)[[zscore_column]]
                
                # Add Z-scores
                sample_indices = sample_outliers.index
                z_scores_sample = np.abs(stats.zscore(df[zscore_column].dropna()))
                zscore_dict = dict(zip(df[zscore_column].dropna().index, z_scores_sample))
                
                sample_outliers['Z_Score'] = [f"{zscore_dict.get(idx, 0):.2f}" for idx in sample_indices]
                
                for col in sample_outliers.columns:
                    sample_outliers[col] = sample_outliers[col].astype(str)
                
                st.dataframe(sample_outliers, use_container_width=True)
                
                # Store results
                st.session_state.zscore_outliers = zscore_results
            else:
                st.info("✅ No outliers detected with current settings")
    
    with col2:
        # Visualization
        if 'zscore_outliers' in st.session_state and st.session_state.zscore_outliers['count'] > 0:
            outlier_data = st.session_state.zscore_outliers
            
            # Z-score distribution plot
            col_data_clean = df[zscore_column].dropna()
            z_scores = stats.zscore(col_data_clean)
            
            fig = go.Figure()
            
            # Normal z-scores
            normal_mask = np.abs(z_scores) <= zscore_threshold
            fig.add_trace(go.Scatter(
                x=col_data_clean[normal_mask].index,
                y=z_scores[normal_mask],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4)
            ))
            
            # Outlier z-scores
            outlier_mask = np.abs(z_scores) > zscore_threshold
            fig.add_trace(go.Scatter(
                x=col_data_clean[outlier_mask].index,
                y=z_scores[outlier_mask],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=6)
            ))
            
            # Threshold lines
            fig.add_hline(y=zscore_threshold, line_dash="dash", line_color="orange")
            fig.add_hline(y=-zscore_threshold, line_dash="dash", line_color="orange")
            
            fig.update_layout(title=f"Z-Score Outliers: {zscore_column}", 
                            xaxis_title="Index", yaxis_title="Z-Score")
            st.plotly_chart(fig, use_container_width=True)

with outlier_tabs[3]:
    st.markdown("### 🤖 Isolation Forest Method")
    
    st.markdown("""
    <div class="method-info">
        <strong>🤖 Isolation Forest:</strong><br>
        • Machine learning algorithm for anomaly detection<br>
        • Works with multiple variables simultaneously<br>
        • Good for: High-dimensional data, complex patterns<br>
        • Best for: Multivariate outlier detection
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Isolation Forest Configuration
        if_columns = st.multiselect(
            "Select columns for multivariate analysis:",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
            help="Multiple columns for comprehensive outlier detection"
        )
        
        contamination = st.slider(
            "Expected outlier proportion:",
            0.01, 0.5, 0.1, 0.01,
            help="Estimated percentage of outliers in the data"
        )
        
        if st.button("🤖 Run Isolation Forest", type="primary") and if_columns:
            if_results = detect_outliers_isolation_forest(df, if_columns, contamination)
            
            st.markdown(f"#### Results for {len(if_columns)} columns")
            st.metric("Outliers Found", f"{if_results['count']}", 
                     f"{if_results['percentage']:.2f}% of data")
            
            st.write(f"**Contamination:** {contamination:.1%}")
            st.write(f"**Features Used:** {len(if_columns)}")
            
            if if_results['count'] > 0:
                st.markdown("**Sample Outliers:**")
                sample_outliers = if_results['outliers'][if_columns].head(10)
                for col in sample_outliers.columns:
                    sample_outliers[col] = sample_outliers[col].astype(str)
                st.dataframe(sample_outliers, use_container_width=True)
                
                # Store results
                st.session_state.if_outliers = if_results
            else:
                st.info("✅ No outliers detected with current settings")
    
    with col2:
        # Visualization for Isolation Forest
        if 'if_outliers' in st.session_state and st.session_state.if_outliers['count'] > 0 and len(if_columns) >= 2:
            outlier_data = st.session_state.if_outliers
            
            # 2D scatter plot of first two features
            col1, col2 = if_columns[0], if_columns[1]
            
            fig = go.Figure()
            
            # Normal points
            normal_mask = ~df.index.isin(outlier_data['outlier_indices'])
            fig.add_trace(go.Scatter(
                x=df[normal_mask][col1],
                y=df[normal_mask][col2],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=4, opacity=0.6)
            ))
            
            # Outliers
            outlier_df = df[df.index.isin(outlier_data['outlier_indices'])]
            fig.add_trace(go.Scatter(
                x=outlier_df[col1],
                y=outlier_df[col2],
                mode='markers',
                name='Outliers',
                marker=dict(color='red', size=8, symbol='x')
            ))
            
            fig.update_layout(
                title=f"Isolation Forest Results: {col1} vs {col2}",
                xaxis_title=col1,
                yaxis_title=col2
            )
            st.plotly_chart(fig, use_container_width=True)

with outlier_tabs[4]:
    st.markdown("### 🔧 Outlier Treatment")
    
    # Check if any outliers have been detected
    available_methods = []
    if 'iqr_outliers' in st.session_state and st.session_state.iqr_outliers['count'] > 0:
        available_methods.append("IQR")
    if 'zscore_outliers' in st.session_state and st.session_state.zscore_outliers['count'] > 0:
        available_methods.append("Z-Score")
    if 'if_outliers' in st.session_state and st.session_state.if_outliers['count'] > 0:
        available_methods.append("Isolation Forest")
    
    if not available_methods:
        st.info("ℹ️ Run outlier detection first to see treatment options.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Treatment Options")
            
            selected_method = st.selectbox(
                "Select detection method results to treat:",
                available_methods,
                help="Choose which outlier detection results to apply treatment to"
            )
            
            treatment_method = st.selectbox(
                "Treatment method:",
                [
                    "Remove outliers",
                    "Cap outliers (clip to bounds)",
                    "Transform (log)",
                    "Replace with median",
                    "Replace with mean",
                    "Mark as missing (NaN)"
                ],
                help="Choose how to handle detected outliers"
            )
            
            # Get outlier data based on selected method
            if selected_method == "IQR":
                outlier_data = st.session_state.iqr_outliers
                target_column = [col for col in numeric_cols if col in iqr_column][0] if 'iqr_column' in locals() else numeric_cols[0]
            elif selected_method == "Z-Score":
                outlier_data = st.session_state.zscore_outliers
                target_column = [col for col in numeric_cols if col in zscore_column][0] if 'zscore_column' in locals() else numeric_cols[0]
            else:  # Isolation Forest
                outlier_data = st.session_state.if_outliers
                target_column = "Multiple Columns"
            
            st.write(f"**Outliers to treat:** {outlier_data['count']}")
            st.write(f"**Target:** {target_column}")
            
            if st.button("🔧 Apply Treatment", type="primary"):
                initial_rows = len(df)
                outlier_indices = outlier_data['outlier_indices']
                
                try:
                    result_msg = ""
                    treatment_applied = False
                    
                    if treatment_method == "Remove outliers":
                        df = df.drop(index=outlier_indices)
                        result_msg = f"Removed {len(outlier_indices)} outlier rows"
                        treatment_applied = True
                    
                    elif treatment_method == "Cap outliers (clip to bounds)" and selected_method in ["IQR", "Z-Score"]:
                        if selected_method == "IQR" and 'bounds' in outlier_data:
                            lower, upper = outlier_data['bounds']
                            df[target_column] = df[target_column].clip(lower=lower, upper=upper)
                            result_msg = f"Capped outliers to bounds [{lower:.2f}, {upper:.2f}]"
                            treatment_applied = True
                        else:
                            st.warning("Capping only available for IQR method")
                    
                    elif treatment_method == "Transform (log)":
                        if selected_method != "Isolation Forest":
                            # Apply log transform (add 1 to handle zeros)
                            df[target_column] = np.log1p(df[target_column])
                            result_msg = f"Applied log transformation to {target_column}"
                            treatment_applied = True
                        else:
                            st.warning("Log transform not applicable to multivariate outliers")
                    
                    elif treatment_method == "Replace with median":
                        if selected_method != "Isolation Forest":
                            median_val = df[target_column].median()
                            df.loc[outlier_indices, target_column] = median_val
                            result_msg = f"Replaced {len(outlier_indices)} outliers with median ({median_val:.2f})"
                            treatment_applied = True
                        else:
                            st.warning("Median replacement not applicable to multivariate outliers")
                    
                    elif treatment_method == "Replace with mean":
                        if selected_method != "Isolation Forest":
                            mean_val = df[target_column].mean()
                            df.loc[outlier_indices, target_column] = mean_val
                            result_msg = f"Replaced {len(outlier_indices)} outliers with mean ({mean_val:.2f})"
                            treatment_applied = True
                        else:
                            st.warning("Mean replacement not applicable to multivariate outliers")
                    
                    elif treatment_method == "Mark as missing (NaN)":
                        if selected_method != "Isolation Forest":
                            df.loc[outlier_indices, target_column] = np.nan
                            result_msg = f"Marked {len(outlier_indices)} outliers as missing values"
                            treatment_applied = True
                        else:
                            # For multivariate, mark all involved columns as NaN
                            for col in if_columns if 'if_columns' in locals() else numeric_cols:
                                df.loc[outlier_indices, col] = np.nan
                            result_msg = f"Marked {len(outlier_indices)} rows as missing in multiple columns"
                            treatment_applied = True
                    
                    if treatment_applied:
                        # Update session state
                        st.session_state.current_dataset = df
                        
                        # Log the action
                        if 'processing_log' not in st.session_state:
                            st.session_state.processing_log = []
                        
                        st.session_state.processing_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Outlier Treatment',
                            'details': f"Applied {treatment_method} using {selected_method} method: {result_msg}"
                        })
                        
                        # Clear outlier detection results
                        if 'iqr_outliers' in st.session_state:
                            del st.session_state.iqr_outliers
                        if 'zscore_outliers' in st.session_state:
                            del st.session_state.zscore_outliers
                        if 'if_outliers' in st.session_state:
                            del st.session_state.if_outliers
                        
                        st.success(f"✅ {result_msg}")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Treatment failed: {str(e)}")
        
        with col2:
            st.markdown("#### Treatment Impact Preview")
            
            # Show impact of each treatment method
            if selected_method in available_methods:
                outlier_count = st.session_state[f"{selected_method.lower().replace('-', '_')}_outliers"]['count']
                
                st.write("**Expected Impact:**")
                
                if treatment_method == "Remove outliers":
                    st.metric("Rows After Treatment", f"{len(df) - outlier_count:,}", 
                             f"-{outlier_count} rows")
                
                elif "Replace" in treatment_method or "Mark as missing" in treatment_method:
                    st.metric("Rows Affected", f"{outlier_count:,}", "Values changed")
                    st.metric("Total Rows", f"{len(df):,}", "Unchanged")
                
                elif "Transform" in treatment_method or "Cap" in treatment_method:
                    st.metric("Values Modified", f"{outlier_count:,}", "In-place changes")
                    st.metric("Total Rows", f"{len(df):,}", "Unchanged")
                
                # Show sample outliers that will be treated
                outlier_data = st.session_state[f"{selected_method.lower().replace('-', '_')}_outliers"]
                if outlier_data['count'] > 0:
                    st.markdown("**Sample Outliers to Treat:**")
                    sample = outlier_data['outliers'].head(5)
                    if selected_method != "Isolation Forest":
                        # Show specific column
                        display_cols = [target_column] if target_column in sample.columns else sample.columns[:2]
                    else:
                        display_cols = sample.columns[:3]  # Show first 3 columns
                    
                    sample_display = sample[display_cols]
                    for col in sample_display.columns:
                        sample_display[col] = sample_display[col].astype(str)
                    st.dataframe(sample_display, use_container_width=True)

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Processed Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"outliers_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Text Cleaning"):
        st.switch_page("pages/08_Text_Cleaning.py")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Outlier Detection Guide")
    
    st.markdown("#### Detection Methods:")
    methods_info = [
        "**IQR:** Best for skewed data",
        "**Z-Score:** Best for normal distributions",  
        "**Isolation Forest:** Best for complex patterns",
        "**Modified Z-Score:** Robust alternative"
    ]
    
    for method in methods_info:
        st.markdown(f"• {method}")
    
    st.markdown("---")
    st.markdown("#### Treatment Strategies:")
    
    treatments = [
        "**Remove:** Delete outlier rows",
        "**Cap:** Limit to acceptable range",
        "**Transform:** Apply mathematical transformation",
        "**Replace:** Use central tendency values",
        "**Mark Missing:** Convert to NaN for imputation"
    ]
    
    for treatment in treatments:
        st.markdown(f"• {treatment}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Understand domain context
    • Try multiple detection methods
    • Validate with business rules
    • Document all treatments
    • Monitor impact on analysis
    """)