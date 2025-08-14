import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview")
st.markdown("Comprehensive data profiling, visualization, and AI-powered insights")

# Check if data exists
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Initialize processing log if it doesn't exist
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# Overview tabs
overview_tabs = st.tabs([
    "📋 Dataset Summary", 
    "🧠 AI Insights",
    "🔍 Data Types & Quality", 
    "📈 Visualizations", 
    "🔗 Correlations", 
    "📊 Grouping & Segmentation"
])

with overview_tabs[0]:

    # Get numeric columns for detailed analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Numerical Summary Statistics Section
    st.markdown("#### 📊 Numerical Summary Statistics")
    
    # Calculate comprehensive statistics
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    overall_completeness = ((df.size - df.isnull().sum().sum()) / df.size) * 100
    
    # Numeric-specific calculations
    if numeric_cols:
        numeric_df = df[numeric_cols]
        numeric_completeness = ((numeric_df.size - numeric_df.isnull().sum().sum()) / numeric_df.size) * 100
        total_missing_numeric = numeric_df.isnull().sum().sum()
        
        # Statistical measures
        mean_values = numeric_df.mean()
        median_values = numeric_df.median()
        std_values = numeric_df.std()
        min_value = numeric_df.min().min()
        max_value = numeric_df.max().max()
        
        # Calculate skewness and kurtosis
        skewness_values = numeric_df.skew()
        kurtosis_values = numeric_df.kurtosis()
        avg_skewness = skewness_values.mean()
        avg_kurtosis = kurtosis_values.mean()
        
        # Outlier detection using IQR method
        total_outliers = 0
        for col in numeric_cols:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = numeric_df[(numeric_df[col] < Q1 - 1.5 * IQR) | (numeric_df[col] > Q3 + 1.5 * IQR)]
            total_outliers += len(outliers)
        
        # Correlation analysis
        corr_matrix = numeric_df.corr()
        high_corr_pairs = 0
        threshold = 0.8
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs += 1
    else:
        numeric_completeness = 0
        total_missing_numeric = 0
        avg_skewness = 0
        avg_kurtosis = 0
        total_outliers = 0
        high_corr_pairs = 0
        min_value = "N/A"
        max_value = "N/A"
    
    # Display metrics in organized layout
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("##### 📏 Basic Metrics")
        st.metric("Total Rows", f"{len(df):,}", help="Total number of records in the dataset")
        st.metric("Total Columns", f"{len(df.columns):,}", help="Total number of variables/features")
        st.metric("Memory Usage", f"{memory_usage:.2f} MB", help="Memory footprint of the dataset")
        st.metric("Data Completeness (%)", f"{overall_completeness:.1f}%", help="Percentage of non-missing values across all columns")
        st.metric("Numeric Columns Count", f"{len(numeric_cols)}", help="Number of numerical columns in the dataset")
    
    with metrics_col2:
        st.markdown("##### 📈 Statistical Measures")
        if numeric_cols:
            st.metric("Total Missing Values (Numeric)", f"{total_missing_numeric:,}", help="Count of missing values in numeric columns")
            st.metric("Mean of Numeric Columns", f"{mean_values.mean():.2f}", help="Average of all numeric column means")
            st.metric("Median of Numeric Columns", f"{median_values.median():.2f}", help="Average of all numeric column medians")
            st.metric("Standard Deviation (Avg)", f"{std_values.mean():.2f}", help="Average standard deviation across numeric columns")
            st.metric("Data Completeness (Numeric %)", f"{numeric_completeness:.1f}%", help="Percentage of non-missing values in numeric columns")
        else:
            st.info("No numeric columns available for statistical analysis")
    
    with metrics_col3:
        st.markdown("##### 🔍 Advanced Analytics")
        if numeric_cols:
            st.metric("Minimum Value (Overall)", f"{min_value:.2f}" if isinstance(min_value, (int, float)) else min_value, help="Smallest value across all numeric columns")
            st.metric("Maximum Value (Overall)", f"{max_value:.2f}" if isinstance(max_value, (int, float)) else max_value, help="Largest value across all numeric columns")
            st.metric("Skewness (Average)", f"{avg_skewness:.2f}", help="Average skewness - measures asymmetry of distributions")
            st.metric("Kurtosis (Average)", f"{avg_kurtosis:.2f}", help="Average kurtosis - measures tail heaviness of distributions")
            st.metric("Outliers Detected", f"{total_outliers:,}", help="Total outliers detected using IQR method (1.5 * IQR)")
            st.metric("Highly Correlated Pairs", f"{high_corr_pairs}", help=f"Number of variable pairs with correlation > {threshold}")
        else:
            st.info("No numeric data for advanced analytics")
    
    # Statistical interpretation
    if numeric_cols:
        st.markdown("📝 Statistical Interpretation")
        interpretation_col1, interpretation_col2 = st.columns(2)
        
        with interpretation_col1:
            # Skewness interpretation
            if abs(avg_skewness) < 0.5:
                skew_interpretation = "🟢 Approximately symmetric distributions"
            elif abs(avg_skewness) < 1:
                skew_interpretation = "🟡 Moderately skewed distributions"
            else:
                skew_interpretation = "🔴 Highly skewed distributions"
            
            st.markdown(f"**Distribution Shape:** {skew_interpretation}")
            
            # Outlier interpretation
            outlier_percentage = (total_outliers / (len(df) * len(numeric_cols))) * 100
            if outlier_percentage < 1:
                outlier_interpretation = "🟢 Low outlier presence"
            elif outlier_percentage < 5:
                outlier_interpretation = "🟡 Moderate outlier presence"
            else:
                outlier_interpretation = "🔴 High outlier presence"
            
            st.markdown(f"**Outlier Assessment:** {outlier_interpretation} ({outlier_percentage:.1f}%)")
        
        with interpretation_col2:
            # Correlation interpretation
            if high_corr_pairs == 0:
                corr_interpretation = "🟢 No multicollinearity concerns"
            elif high_corr_pairs <= 2:
                corr_interpretation = "🟡 Some correlated variables detected"
            else:
                corr_interpretation = "🔴 Multiple correlated variables found"
            
            st.markdown(f"**Correlation Status:** {corr_interpretation}")
            
            # Data quality assessment
            if overall_completeness > 95:
                quality_interpretation = "🟢 Excellent data quality"
            elif overall_completeness > 80:
                quality_interpretation = "🟡 Good data quality"
            else:
                quality_interpretation = "🔴 Data quality needs attention"
            
            st.markdown(f"**Data Quality:** {quality_interpretation}")
    
    st.markdown("---")
    
    # Dataset preview
    st.markdown("#### 🔍 Dataset Preview")
    
    preview_options = st.columns([1, 1, 1, 1])
    with preview_options[0]:
        preview_type = st.selectbox("Preview Type", ["Head", "Tail", "Random Sample", "Custom"])
    with preview_options[1]:
        n_rows = st.number_input("Number of rows", min_value=5, max_value=100, value=10)
    with preview_options[2]:
        if st.button("🔄 Refresh Preview"):
            st.rerun()
    
    if preview_type == "Head":
        st.dataframe(df.head(n_rows), use_container_width=True)
    elif preview_type == "Tail":
        st.dataframe(df.tail(n_rows), use_container_width=True)
    elif preview_type == "Random Sample":
        if len(df) >= n_rows:
            st.dataframe(df.sample(n_rows), use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    elif preview_type == "Custom":
        col_custom1, col_custom2 = st.columns(2)
        with col_custom1:
            start_row = st.number_input("Start row", min_value=0, max_value=len(df)-1, value=0)
        with col_custom2:
            end_row = st.number_input("End row", min_value=start_row+1, max_value=len(df), value=min(start_row+n_rows, len(df)))
        st.dataframe(df.iloc[start_row:end_row], use_container_width=True)
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Column Information")
        
        # Create column info dataframe
        column_info = []
        for col in df.columns:
            col_data = df[col]
            column_info.append({
                'Column': col,
                'Data Type': str(col_data.dtype),
                'Non-Null Count': col_data.count(),
                'Null Count': col_data.isnull().sum(),
                'Null %': f"{(col_data.isnull().sum() / len(df)) * 100:.1f}%",
                'Unique Values': col_data.nunique(),
                'Unique %': f"{(col_data.nunique() / len(df)) * 100:.1f}%"
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Data Quality Summary")
        
        # Missing values heatmap
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Missing Count', 'y': 'Column'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")

with overview_tabs[2]:
    st.markdown("### 🔍 Data Types & Quality Analysis")
    
    # Auto type detection
    st.markdown("#### 🤖 Auto Type Detection")
    
    def detect_column_types(df):
        """Enhanced column type detection"""
        type_info = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if len(col_data) == 0:
                type_info[col] = "Empty"
                continue
            
            # Numeric detection
            if pd.api.types.is_numeric_dtype(col_data):
                if col_data.dtype in ['int64', 'int32', 'int16', 'int8']:
                    if col_data.nunique() == 2:
                        type_info[col] = "Boolean (Numeric)"
                    elif col_data.nunique() / len(col_data) < 0.05:
                        type_info[col] = "Categorical (Numeric)"
                    else:
                        type_info[col] = "Integer"
                else:
                    type_info[col] = "Float"
            
            # DateTime detection
            elif col_data.dtype == 'datetime64[ns]':
                type_info[col] = "DateTime"
            
            # String/Object detection
            else:
                # Try to parse as datetime
                try:
                    sample_data = col_data.astype(str).head(100)
                    parsed_dates = pd.to_datetime(sample_data, errors='coerce', infer_datetime_format=True)
                    success_rate = parsed_dates.notna().sum() / len(sample_data)
                    if success_rate > 0.7:
                        type_info[col] = "Potential DateTime"
                        continue
                except:
                    pass
                
                # Check for boolean strings
                unique_vals = set(col_data.astype(str).str.lower().unique())
                if unique_vals.issubset({'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}):
                    type_info[col] = "Boolean (Text)"
                
                # Check for categorical
                elif col_data.nunique() / len(col_data) < 0.05:
                    type_info[col] = "Categorical"
                
                # Check for ID columns
                elif col.lower() in ['id', 'user_id', 'customer_id', 'order_id'] or '_id' in col.lower():
                    type_info[col] = "ID Column"
                
                # Check for geospatial
                elif any(geo_keyword in col.lower() for geo_keyword in ['lat', 'lon', 'latitude', 'longitude', 'address', 'zip', 'postal']):
                    type_info[col] = "Geospatial"
                
                # Default to text
                else:
                    type_info[col] = "Text"
        
        return type_info
    
    detected_types = detect_column_types(df)
    
    # Display detected types
    type_df = pd.DataFrame([
        {'Column': col, 'Current Type': str(df[col].dtype), 'Detected Type': detected_types[col]}
        for col in df.columns
    ])
    
    st.dataframe(type_df, use_container_width=True)
    
    # Data quality issues
    st.markdown("#### ⚠️ Data Quality Issues")
    
    quality_issues = []
    
    for col in df.columns:
        col_data = df[col]
        
        # Missing values
        missing_count = col_data.isnull().sum()
        if missing_count > 0:
            quality_issues.append({
                'Column': col,
                'Issue': 'Missing Values',
                'Count': missing_count,
                'Percentage': f"{(missing_count / len(df)) * 100:.1f}%",
                'Severity': 'High' if missing_count / len(df) > 0.1 else 'Medium'
            })
        
        # Duplicates (for non-numeric columns)
        if not pd.api.types.is_numeric_dtype(col_data):
            duplicate_count = col_data.duplicated().sum()
            if duplicate_count > len(df) * 0.1:  # More than 10% duplicates
                quality_issues.append({
                    'Column': col,
                    'Issue': 'High Duplication',
                    'Count': duplicate_count,
                    'Percentage': f"{(duplicate_count / len(df)) * 100:.1f}%",
                    'Severity': 'Medium'
                })
        
        # Outliers (for numeric columns)
        if pd.api.types.is_numeric_dtype(col_data):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = col_data[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                quality_issues.append({
                    'Column': col,
                    'Issue': 'Outliers (IQR)',
                    'Count': len(outliers),
                    'Percentage': f"{(len(outliers) / len(df)) * 100:.1f}%",
                    'Severity': 'Low' if len(outliers) / len(df) < 0.05 else 'Medium'
                })
    
    if quality_issues:
        issues_df = pd.DataFrame(quality_issues)
        st.dataframe(issues_df, use_container_width=True)
        
        # Quick fix suggestions
        st.markdown("#### 🔧 Quick Fix Suggestions")
        if st.button("🚀 Go to Clean Pipeline"):
            st.switch_page("pages/04_Clean_Pipeline.py")
    else:
        st.success("✅ No significant data quality issues detected!")

with overview_tabs[3]:
    st.markdown("### 📈 Data Visualizations")
    
    # Visualization options
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Distribution Analysis", "Correlation Matrix", "Missing Data Pattern", "Box Plots", "Time Series"]
        )
    
    with viz_col2:
        if viz_type in ["Distribution Analysis", "Box Plots"]:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select Column", numeric_cols)
    
    # Generate visualizations
    if viz_type == "Distribution Analysis" and 'selected_col' in locals():
        col_data = df[selected_col].dropna()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Statistics'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=col_data, name="Distribution", nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=col_data, name="Box Plot"),
            row=1, col=2
        )
        
        # Q-Q plot
        qq_data = stats.probplot(col_data, dist="norm")
        fig.add_trace(
            go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name="Q-Q Plot"),
            row=2, col=1
        )
        
        # Statistics table
        stats_data = {
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
            'Value': [
                f"{col_data.mean():.3f}",
                f"{col_data.median():.3f}",
                f"{col_data.std():.3f}",
                f"{col_data.skew():.3f}",
                f"{col_data.kurtosis():.3f}",
                f"{col_data.min():.3f}",
                f"{col_data.max():.3f}"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys())),
                cells=dict(values=list(stats_data.values()))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title=f"Distribution Analysis: {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Matrix":
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })
            
            if strong_corr:
                st.markdown("#### 🔍 Strong Correlations (|r| > 0.7)")
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    elif viz_type == "Missing Data Pattern":
        if df.isnull().sum().sum() > 0:
            # Missing data heatmap
            missing_matrix = df.isnull()
            
            fig = px.imshow(
                missing_matrix.T,
                title="Missing Data Pattern",
                labels=dict(x="Row Index", y="Column", color="Missing"),
                color_continuous_scale=["lightblue", "red"]
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing data to visualize!")

with overview_tabs[4]:
    st.markdown("### 🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        corr_method = st.selectbox("Correlation Method", ["Pearson", "Spearman", "Kendall"])
        
        if corr_method == "Pearson":
            correlation_matrix = df[numeric_cols].corr(method='pearson')
        elif corr_method == "Spearman":
            correlation_matrix = df[numeric_cols].corr(method='spearman')
        else:
            correlation_matrix = df[numeric_cols].corr(method='kendall')
        
        # Correlation heatmap
        fig = px.imshow(
            correlation_matrix,
            title=f"{corr_method} Correlation Matrix",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            text_auto=True
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # VIF analysis
        st.markdown("#### 📊 Variance Inflation Factor (VIF) Analysis")
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            # Clean data for VIF calculation - remove missing and infinite values
            vif_df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(vif_df) > 0 and len(vif_df.columns) > 1:
                # Calculate VIF for each variable
                vif_data = pd.DataFrame()
                vif_data["Variable"] = vif_df.columns
                
                try:
                    vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) 
                                      for i in range(len(vif_df.columns))]
                    
                    # Color code VIF values
                    def vif_color(val):
                        if val < 5:
                            return 'background-color: lightgreen'
                        elif val < 10:
                            return 'background-color: yellow'
                        else:
                            return 'background-color: lightcoral'
                    
                    styled_vif = vif_data.style.applymap(vif_color, subset=['VIF'])
                    st.dataframe(styled_vif, use_container_width=True)
                    
                    st.markdown("""
                    **##VIF Interpretation:**
                    - 🟢 VIF < 5: No multicollinearity concerns
                    - 🟡 5 ≤ VIF < 10: Moderate multicollinearity
                    - 🔴 VIF ≥ 10: High multicollinearity (consider removal)
                    """)
                except Exception as e:
                    st.warning("Cannot calculate VIF: Data may contain issues preventing analysis")
                    st.info("Consider cleaning the data first to enable multicollinearity analysis")
            else:
                st.warning("Insufficient clean data for VIF calculation")
            
        except ImportError:
            st.info("Install statsmodels for VIF analysis: pip install statsmodels")
        except Exception as e:
            st.warning(f"VIF analysis failed: {str(e)}")
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

with overview_tabs[1]:


    
    # Smart Activity List & Fix Suggestions Section
    st.markdown("#### 🎯 Smart Activity List & Fix Suggestions (Automated)")
    
    st.markdown("##### 🔍 Detected Issues with Severity Rankings")
    
    # Analyze data and suggest fixes
    issues = []
    
    # Check for categorical encoding needs
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        issues.append({
            'severity': '🟡',
            'issue': f'Categorical encoding needed: {len(categorical_cols)} columns',
            'action': 'Apply categorical encoding',
            'priority': 'medium'
        })
    
    # Check for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    total_outliers = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        total_outliers += len(outliers)
    
    if total_outliers > 0:
        issues.append({
            'severity': '🟡',
            'issue': f'Outliers detected: ~{total_outliers} data points',
            'action': 'Apply outlier treatment',
            'priority': 'medium'
        })
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        issues.append({
            'severity': '🔴',
            'issue': f'Missing values found: {missing_count} cells',
            'action': 'Apply missing value imputation',
            'priority': 'high'
        })
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        issues.append({
            'severity': '🟠',
            'issue': f'Duplicate rows found: {duplicate_count} rows',
            'action': 'Remove duplicate rows',
            'priority': 'medium'
        })
    
    # Display issues
    if issues:
        for issue in issues:
            st.markdown(f"{issue['severity']} **{issue['issue']}** - {issue['action']}")
        
        st.markdown("---")
        
        # One-Click Fixes button
        if st.button("⚡ Apply All Suggested Fixes", type="primary", use_container_width=True):
            with st.spinner("🔄 Applying automated fixes..."):
                fixes_applied = []
                
                # Apply missing value imputation
                if missing_count > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    
                    # Impute numeric columns with median
                    for col in numeric_cols:
                        if df[col].isnull().any():
                            df[col].fillna(df[col].median(), inplace=True)
                    
                    # Impute categorical columns with mode
                    for col in categorical_cols:
                        if df[col].isnull().any():
                            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                    
                    fixes_applied.append(f"Missing values imputed: {missing_count} cells")
                
                # Remove duplicate rows
                if duplicate_count > 0:
                    df.drop_duplicates(inplace=True)
                    fixes_applied.append(f"Duplicate rows removed: {duplicate_count} rows")
                
                # Apply outlier treatment (capping method)
                if total_outliers > 0:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    outliers_treated = 0
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Cap outliers
                        outliers_before = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        outliers_treated += outliers_before
                    
                    if outliers_treated > 0:
                        fixes_applied.append(f"Outliers treated: {outliers_treated} data points")
                
                # Update session state with cleaned data
                st.session_state.current_dataset = df.copy()
                
                # Log the operations
                import datetime
                for fix in fixes_applied:
                    st.session_state.processing_log.append({
                        'step': 'AI Auto-Fix',
                        'operation': fix,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'details': 'Automatically applied via AI Insights'
                    })
                
                if fixes_applied:
                    st.success(f"✅ Successfully applied {len(fixes_applied)} fixes:")
                    for fix in fixes_applied:
                        st.info(f"• {fix}")
                    
                    # Trigger balloons celebration first
                    st.balloons()
                    
                    # Trigger milestone reward for AI auto-fix
                    if 'milestone_rewards' in st.session_state:
                        milestone_triggered = st.session_state.milestone_rewards.complete_activity('ai_auto_fix')
                        if milestone_triggered:
                            # Enhanced milestone celebration
                            st.markdown("""
                            <div style="background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4); 
                                        padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                                <h2 style="color: white; margin: 0;">🎉 MILESTONE ACHIEVEMENT UNLOCKED! 🎉</h2>
                                <h3 style="color: white; margin: 10px 0;">🤖 AI Assistant Master</h3>
                                <p style="color: white; margin: 0;">+200 Points! Visit Milestone Dashboard for details!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Regular milestone progress
                            st.success("🎯 Progress updated! Check your Milestone Dashboard!")
                    
                    # Additional celebration with confetti-like effect
                    st.markdown("""
                    <style>
                    @keyframes celebrate {
                        0% { transform: scale(1) rotate(0deg); }
                        25% { transform: scale(1.1) rotate(5deg); }
                        50% { transform: scale(1.2) rotate(-5deg); }
                        75% { transform: scale(1.1) rotate(3deg); }
                        100% { transform: scale(1) rotate(0deg); }
                    }
                    @keyframes sparkle {
                        0%, 100% { opacity: 0; }
                        50% { opacity: 1; }
                    }
                    .celebration {
                        animation: celebrate 3s ease-in-out;
                        text-align: center;
                        color: #28a745;
                        font-size: 1.5em;
                        font-weight: bold;
                        margin: 20px 0;
                    }
                    .sparkles {
                        animation: sparkle 2s infinite;
                        color: #FFD700;
                        font-size: 2em;
                    }
                    </style>
                    <div class="celebration">
                        <span class="sparkles">✨</span>
                        🎊 Data Cleaning Complete! 🎊
                        <span class="sparkles">✨</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.rerun()  # Refresh to show updated data
                else:
                    st.info("No fixes were needed - data is already clean!")
    else:
        st.success("🎉 No critical issues detected! Your data looks good.")
    
    # Generate AI insights
    insights = []
    
    # Data size insights
    if len(df) > 100000:
        insights.append({
            'type': 'info',
            'message': f"Large dataset detected ({len(df):,} rows). Consider sampling for faster processing.",
            'severity': 'Medium'
        })
    
    # Missing data insights
    missing_percentage = (df.isnull().sum().sum() / df.size) * 100
    if missing_percentage > 10:
        insights.append({
            'type': 'warning', 
            'message': f"High missing data rate ({missing_percentage:.1f}%). This may impact analysis quality.",
            'severity': 'High'
        })
    elif missing_percentage > 5:
        insights.append({
            'type': 'warning',
            'message': f"Moderate missing data rate ({missing_percentage:.1f}%). Consider imputation strategies.",
            'severity': 'Medium'
        })
    
    # Duplicate insights
    total_duplicates = df.duplicated().sum()
    if total_duplicates > 0:
        dup_percentage = (total_duplicates / len(df)) * 100
        insights.append({
            'type': 'warning',
            'message': f"Found {total_duplicates:,} duplicate rows ({dup_percentage:.1f}%). Consider deduplication.",
            'severity': 'Medium' if dup_percentage > 5 else 'Low'
        })
    
    # Skewness insights
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) > 2:
            insights.append({
                'type': 'info',
                'message': f"Column '{col}' is highly skewed (skewness: {skewness:.2f}). Consider transformation.",
                'severity': 'Low'
            })
    
    # Cardinality insights
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95 and df[col].nunique() > 10:
            insights.append({
                'type': 'info',
                'message': f"Column '{col}' has very high cardinality ({df[col].nunique():,} unique values). Might be an identifier.",
                'severity': 'Low'
            })
        elif unique_ratio < 0.01 and df[col].nunique() > 1:
            insights.append({
                'type': 'info',
                'message': f"Column '{col}' has very low cardinality ({df[col].nunique()} unique values). Consider as categorical.",
                'severity': 'Low'
            })
    
    # Display insights
    if insights:
        for insight in insights:
            if insight['type'] == 'warning':
                st.warning(f"⚠️ {insight['message']}")
            else:
                st.info(f"💡 {insight['message']}")
    else:
        st.success("✅ No significant data quality issues detected!")
    
    # Smart Activity List
    st.markdown("#### 📋 Smart Activity List")
    
    activities = []
    
    # Add missing value handling
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        activities.append({
            'action': 'Handle Missing Values',
            'description': f'Impute missing values in {len(missing_cols)} columns',
            'severity': 'High' if missing_percentage > 10 else 'Medium',
            'page': 'pages/06_Missing_Values.py'
        })
    
    # Add outlier detection
    if len(numeric_cols) > 0:
        activities.append({
            'action': 'Detect Outliers',
            'description': f'Analyze outliers in {len(numeric_cols)} numeric columns',
            'severity': 'Medium',
            'page': 'pages/07_Outlier_Detection.py'
        })
    
    # Add data type conversion
    activities.append({
        'action': 'Optimize Data Types',
        'description': 'Convert columns to optimal data types for memory efficiency',
        'severity': 'Low',
        'page': 'pages/04_Data_Types.py'
    })
    
    if activities:
        activity_df = pd.DataFrame(activities)
        
        # Color code by severity
        def severity_color(val):
            if val == 'High':
                return 'background-color: #ffebee'
            elif val == 'Medium':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'
        
        styled_activities = activity_df.style.applymap(severity_color, subset=['severity'])
        st.dataframe(styled_activities, use_container_width=True)


        
        # Quick action buttons
        #st.markdown("#### ⚡ Quick Actions")
        #action_cols = st.columns(len(activities))
        #for i, activity in enumerate(activities):
            #with action_cols[i]:
                #if st.button(f"🚀 {activity['action']}", key=f"action_{i}"):
                    #st.switch_page(activity['page'])
                    #st.switch_page(page_map[activity['page']])

with overview_tabs[5]:
    st.markdown("### 📊 Grouping & Segmentation")
    
    # Grouping options
    group_col1, group_col2 = st.columns(2)
    
    with group_col1:
        grouping_type = st.selectbox(
            "Grouping Type",
            ["Single Column", "Multiple Columns", "Date-based", "Custom Conditions"]
        )
    
    if grouping_type == "Single Column":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            group_column = st.selectbox("Select Column", categorical_cols)
            
            if group_column:
                # Group analysis
                group_stats = df.groupby(group_column).agg({
                    df.columns[0]: 'count'  # Count of records
                }).rename(columns={df.columns[0]: 'Count'})
                
                # Add percentage
                group_stats['Percentage'] = (group_stats['Count'] / group_stats['Count'].sum() * 100).round(2)
                
                st.markdown(f"#### Group Analysis: {group_column}")
                st.dataframe(group_stats, use_container_width=True)
                
                # Visualization
                fig = px.pie(
                    values=group_stats['Count'],
                    names=group_stats.index,
                    title=f"Distribution of {group_column}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns found for grouping")
    
    elif grouping_type == "Multiple Columns":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(categorical_cols) >= 2:
            group_columns = st.multiselect("Select Columns", categorical_cols, max_selections=3)
            
            if len(group_columns) >= 2:
                # Multi-level grouping
                group_stats = df.groupby(group_columns).size().reset_index(name='Count')
                group_stats['Percentage'] = (group_stats['Count'] / group_stats['Count'].sum() * 100).round(2)
                
                st.markdown(f"#### Multi-level Group Analysis")
                st.dataframe(group_stats, use_container_width=True)
                
                # Hierarchical visualization for 2 columns
                if len(group_columns) == 2:
                    fig = px.sunburst(
                        group_stats,
                        path=group_columns,
                        values='Count',
                        title=f"Hierarchical Distribution: {' → '.join(group_columns)}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 categorical columns for multi-level grouping")
    
    elif grouping_type == "Date-based":
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also check for potential datetime columns
        potential_datetime_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            try:
                sample_data = df[col].dropna().head(20)
                if len(sample_data) > 0:
                    parsed_dates = pd.to_datetime(sample_data, errors='coerce')
                    success_rate = parsed_dates.notna().sum() / len(sample_data)
                    if success_rate > 0.7:
                        potential_datetime_cols.append(col)
            except:
                continue
        
        all_datetime_cols = datetime_cols + potential_datetime_cols
        
        if all_datetime_cols:
            date_column = st.selectbox("Select Date Column", all_datetime_cols)
            time_period = st.selectbox("Time Period", ["Year", "Month", "Quarter", "Day of Week"])
            
            if date_column:
                # Convert to datetime if not already
                if date_column in potential_datetime_cols:
                    date_data = pd.to_datetime(df[date_column], errors='coerce')
                else:
                    date_data = df[date_column]
                
                # Extract time components
                if time_period == "Year":
                    time_component = date_data.dt.year
                elif time_period == "Month":
                    time_component = date_data.dt.month_name()
                elif time_period == "Quarter":
                    time_component = "Q" + date_data.dt.quarter.astype(str)
                else:  # Day of Week
                    time_component = date_data.dt.day_name()
                
                # Group by time component
                time_groups = time_component.value_counts().sort_index()
                
                st.markdown(f"#### Time-based Analysis: {time_period}")
                st.dataframe(time_groups, use_container_width=True)
                
                # Time series visualization
                fig = px.bar(
                    x=time_groups.index,
                    y=time_groups.values,
                    title=f"Distribution by {time_period}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date/datetime columns found")

# Navigation
st.markdown("---")
nav_cols = st.columns(4)

with nav_cols[0]:
    if st.button("📥 Back to Upload"):
        st.switch_page("pages/01_Upload.py")

with nav_cols[1]:
    if st.button("🧹 Clean Pipeline"):
        st.switch_page("pages/04_Clean_Pipeline.py")

with nav_cols[2]:
    if st.button("🔍 Auto Clean"):
        st.switch_page("pages/04_Clean_Pipeline.py")

with nav_cols[3]:
    if st.button("💾 Export Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"overview_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
