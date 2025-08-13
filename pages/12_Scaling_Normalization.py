import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Scaling Normalization", page_icon="📊", layout="wide")

st.title("📊 Scaling Normalization")
st.markdown("Advanced scaling techniques with distribution analysis and intelligent method selection")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.warning("⚠️ No numeric columns found for scaling.")
    st.stop()

# Scaling utility functions
def analyze_distribution(series):
    """Analyze the distribution of a numeric series"""
    clean_data = series.dropna()
    
    if len(clean_data) == 0:
        return {}
    
    try:
        skewness = clean_data.skew()
        kurtosis = clean_data.kurtosis()
        
        # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
        if len(clean_data) <= 5000:
            _, p_value = stats.shapiro(clean_data.sample(min(5000, len(clean_data))))
            normality_test = "Shapiro-Wilk"
        else:
            stat, crit_vals, sig_level = stats.anderson(clean_data.sample(5000), dist='norm')
            p_value = 0.05 if stat > crit_vals[2] else 0.1  # Approximate p-value
            normality_test = "Anderson-Darling"
        
        is_normal = p_value > 0.05
        
        return {
            'mean': clean_data.mean(),
            'std': clean_data.std(),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'range': clean_data.max() - clean_data.min(),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'is_normal': is_normal,
            'normality_p_value': p_value,
            'normality_test': normality_test,
            'has_outliers': len(clean_data[(np.abs(stats.zscore(clean_data)) > 3)]) > 0,
            'zero_values': (clean_data == 0).sum(),
            'negative_values': (clean_data < 0).sum()
        }
    except:
        return {'error': 'Distribution analysis failed'}

def recommend_scaler(series):
    """Recommend the best scaling method based on data characteristics"""
    analysis = analyze_distribution(series)
    
    if 'error' in analysis:
        return "StandardScaler", "Default choice - unable to analyze distribution"
    
    # Decision logic based on data characteristics
    if analysis['is_normal'] and not analysis['has_outliers']:
        return "StandardScaler", "Data is normally distributed without outliers"
    
    elif analysis['has_outliers'] or abs(analysis['skewness']) > 1:
        return "RobustScaler", "Data has outliers or is significantly skewed"
    
    elif analysis['min'] >= 0 and analysis['range'] > 0:
        return "MinMaxScaler", "Data is non-negative with reasonable range"
    
    elif abs(analysis['skewness']) > 2:
        return "PowerTransformer", "Data is highly skewed, needs transformation"
    
    else:
        return "StandardScaler", "Standard scaling is appropriate"

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Numeric Columns", len(numeric_cols))

with col2:
    # Calculate overall scale variance
    scales = []
    for col in numeric_cols:
        col_range = df[col].max() - df[col].min()
        if not pd.isna(col_range) and col_range > 0:
            scales.append(col_range)
    
    scale_variance = np.std(scales) / np.mean(scales) if scales else 0
    st.metric("Scale Variance", f"{scale_variance:.2f}")

with col3:
    # Count columns needing scaling
    needs_scaling = 0
    for col in numeric_cols:
        col_range = df[col].max() - df[col].min()
        if col_range > 100 or col_range < 0.01:
            needs_scaling += 1
    st.metric("Need Scaling", needs_scaling)

with col4:
    skewed_cols = sum(1 for col in numeric_cols if abs(df[col].skew()) > 1)
    st.metric("Skewed Columns", skewed_cols)

# Scaling tabs
scaling_tabs = st.tabs(["📊 Analysis", "⚖️ Individual Scaling", "🚀 Bulk Scaling", "🔬 Advanced Methods"])

with scaling_tabs[0]:
    st.markdown("### 📊 Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column analysis table
        analysis_data = []
        
        for col in numeric_cols:
            analysis = analyze_distribution(df[col])
            
            if 'error' not in analysis:
                recommended_scaler, reason = recommend_scaler(df[col])
                
                analysis_data.append({
                    'Column': col,
                    'Mean': f"{analysis['mean']:.2f}",
                    'Std': f"{analysis['std']:.2f}",
                    'Range': f"{analysis['range']:.2f}",
                    'Skewness': f"{analysis['skewness']:.2f}",
                    'Normal': "✅" if analysis['is_normal'] else "❌",
                    'Outliers': "⚠️" if analysis['has_outliers'] else "✅",
                    'Recommended Scaler': recommended_scaler
                })
            else:
                analysis_data.append({
                    'Column': col,
                    'Mean': 'Error',
                    'Std': 'Error',
                    'Range': 'Error',
                    'Skewness': 'Error',
                    'Normal': 'Error',
                    'Outliers': 'Error',
                    'Recommended Scaler': 'StandardScaler'
                })
        
        analysis_df = pd.DataFrame(analysis_data)
        
        # Convert to strings to avoid Arrow issues
        for col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].astype(str)
        
        st.dataframe(analysis_df, use_container_width=True)
    
    with col2:
        # Distribution visualization
        viz_col = st.selectbox("Select column for distribution visualization:", numeric_cols)
        
        col_data = df[viz_col].dropna()
        
        if len(col_data) > 0:
            # Histogram
            fig_hist = px.histogram(
                col_data, 
                nbins=30,
                title=f"Distribution: {viz_col}",
                labels={'value': viz_col, 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Box plot
            fig_box = px.box(y=col_data, title=f"Box Plot: {viz_col}")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Distribution statistics
            analysis = analyze_distribution(df[viz_col])
            if 'error' not in analysis:
                st.markdown("#### Distribution Stats")
                st.write(f"**Skewness:** {analysis['skewness']:.3f}")
                st.write(f"**Kurtosis:** {analysis['kurtosis']:.3f}")
                st.write(f"**Normal:** {'Yes' if analysis['is_normal'] else 'No'}")
                st.write(f"**Outliers:** {'Yes' if analysis['has_outliers'] else 'No'}")

with scaling_tabs[1]:
    st.markdown("### ⚖️ Individual Column Scaling")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Select Column and Method")
        
        scaling_column = st.selectbox("Select column to scale:", numeric_cols, key="scaling_col")
        
        scaling_method = st.selectbox(
            "Scaling method:",
            [
                "StandardScaler (Z-score)",
                "MinMaxScaler (0-1)", 
                "RobustScaler (Median-IQR)",
                "MaxAbsScaler (Max Absolute)",
                "PowerTransformer (Yeo-Johnson)",
                "QuantileTransformer (Uniform)",
                "Unit Vector Scaling"
            ],
            help="Choose the scaling method to apply"
        )
        
        # Method-specific parameters
        if scaling_method == "MinMaxScaler (0-1)":
            feature_range = st.selectbox("Feature range:", [(0, 1), (-1, 1), (0, 100)], format_func=str)
        
        elif scaling_method == "PowerTransformer (Yeo-Johnson)":
            standardize = st.checkbox("Standardize after transformation", value=True)
        
        elif scaling_method == "QuantileTransformer (Uniform)":
            n_quantiles = st.slider("Number of quantiles:", 100, 2000, 1000)
            output_distribution = st.selectbox("Output distribution:", ["uniform", "normal"])
        
        # Preview scaling
        if st.button("👁️ Preview Scaling", key="preview_scaling"):
            try:
                original_data = df[scaling_column].dropna().head(10)
                
                # Apply selected scaling method
                if scaling_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler(feature_range=feature_range)
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "RobustScaler (Median-IQR)":
                    scaler = RobustScaler()
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "MaxAbsScaler (Max Absolute)":
                    scaler = MaxAbsScaler()
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "PowerTransformer (Yeo-Johnson)":
                    scaler = PowerTransformer(method='yeo-johnson', standardize=standardize)
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "QuantileTransformer (Uniform)":
                    scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
                    scaled_data = scaler.fit_transform(original_data.values.reshape(-1, 1)).flatten()
                
                elif scaling_method == "Unit Vector Scaling":
                    scaler = Normalizer(norm='l2')
                    scaled_data = scaler.fit_transform(original_data.values.reshape(1, -1)).flatten()
                
                # Display comparison
                st.markdown("#### Before vs After Preview")
                
                preview_df = pd.DataFrame({
                    'Original': original_data.values,
                    'Scaled': scaled_data
                })
                
                for col in preview_df.columns:
                    preview_df[col] = preview_df[col].astype(str)
                
                st.dataframe(preview_df, use_container_width=True)
                
                # Statistics comparison
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("**Original Stats:**")
                    st.write(f"Mean: {original_data.mean():.3f}")
                    st.write(f"Std: {original_data.std():.3f}")
                    st.write(f"Range: {original_data.max() - original_data.min():.3f}")
                
                with col_b:
                    st.markdown("**Scaled Stats:**")
                    st.write(f"Mean: {np.mean(scaled_data):.3f}")
                    st.write(f"Std: {np.std(scaled_data):.3f}")
                    st.write(f"Range: {np.max(scaled_data) - np.min(scaled_data):.3f}")
                
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")
    
    with col2:
        st.markdown("#### Apply Scaling")
        
        if st.button("⚖️ Apply Scaling", type="primary", key="apply_individual_scaling"):
            try:
                # Get the full column data
                col_data = df[scaling_column].copy()
                
                # Handle missing values
                mask = col_data.notna()
                clean_data = col_data[mask].values.reshape(-1, 1)
                
                # Apply selected scaling method
                if scaling_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler(feature_range=feature_range)
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "RobustScaler (Median-IQR)":
                    scaler = RobustScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "MaxAbsScaler (Max Absolute)":
                    scaler = MaxAbsScaler()
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "PowerTransformer (Yeo-Johnson)":
                    scaler = PowerTransformer(method='yeo-johnson', standardize=standardize)
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "QuantileTransformer (Uniform)":
                    scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
                    scaled_data = scaler.fit_transform(clean_data)
                
                elif scaling_method == "Unit Vector Scaling":
                    # Unit vector scaling works differently - normalize across samples
                    scaler = Normalizer(norm='l2')
                    # For single column, we need to reshape appropriately
                    scaled_data = clean_data / np.linalg.norm(clean_data)
                
                # Create new column with scaled data
                new_col_name = f"{scaling_column}_scaled"
                df[new_col_name] = np.nan
                df.loc[mask, new_col_name] = scaled_data.flatten()
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Feature Scaling',
                    'details': f"Applied {scaling_method} to {scaling_column}, created {new_col_name}"
                })
                
                st.success(f"✅ Applied {scaling_method} to {scaling_column}! Created column: {new_col_name}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Scaling failed: {str(e)}")
        
        # Show current column analysis
        if scaling_column:
            analysis = analyze_distribution(df[scaling_column])
            
            if 'error' not in analysis:
                st.markdown("#### Current Column Analysis")
                st.write(f"**Range:** {analysis['range']:.2f}")
                st.write(f"**Skewness:** {analysis['skewness']:.2f}")
                st.write(f"**Has Outliers:** {'Yes' if analysis['has_outliers'] else 'No'}")
                st.write(f"**Normal Distribution:** {'Yes' if analysis['is_normal'] else 'No'}")
                
                recommended_scaler, reason = recommend_scaler(df[scaling_column])
                st.info(f"**Recommended:** {recommended_scaler}\n\n{reason}")

with scaling_tabs[2]:
    st.markdown("### 🚀 Bulk Scaling Operations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Bulk Scaling Strategy")
        
        bulk_strategy = st.selectbox(
            "Select bulk scaling strategy:",
            [
                "Smart Auto-Selection",
                "StandardScaler for All",
                "MinMaxScaler for All", 
                "RobustScaler for All",
                "Custom Selection"
            ],
            help="Choose how to scale multiple columns at once"
        )
        
        # Column selection
        if bulk_strategy == "Custom Selection":
            selected_columns = st.multiselect(
                "Select columns to scale:",
                numeric_cols,
                default=numeric_cols,
                help="Choose which columns to include in bulk scaling"
            )
        else:
            selected_columns = numeric_cols
        
        # Additional options
        preserve_original = st.checkbox(
            "Preserve original columns",
            value=True,
            help="Keep original columns and create new scaled versions"
        )
        
        if st.button("🚀 Apply Bulk Scaling", type="primary"):
            scaled_columns = []
            
            try:
                for col in selected_columns:
                    # Get column data
                    col_data = df[col].copy()
                    mask = col_data.notna()
                    clean_data = col_data[mask].values.reshape(-1, 1)
                    
                    if len(clean_data) == 0:
                        continue
                    
                    # Choose scaler based on strategy
                    if bulk_strategy == "Smart Auto-Selection":
                        recommended_scaler, _ = recommend_scaler(df[col])
                        
                        if "Standard" in recommended_scaler:
                            scaler = StandardScaler()
                        elif "Robust" in recommended_scaler:
                            scaler = RobustScaler()
                        elif "MinMax" in recommended_scaler:
                            scaler = MinMaxScaler()
                        elif "Power" in recommended_scaler:
                            scaler = PowerTransformer(method='yeo-johnson')
                        else:
                            scaler = StandardScaler()
                    
                    elif bulk_strategy == "StandardScaler for All":
                        scaler = StandardScaler()
                    elif bulk_strategy == "MinMaxScaler for All":
                        scaler = MinMaxScaler()
                    elif bulk_strategy == "RobustScaler for All":
                        scaler = RobustScaler()
                    elif bulk_strategy == "Custom Selection":
                        scaler = StandardScaler()  # Default for custom
                    
                    # Apply scaling
                    scaled_data = scaler.fit_transform(clean_data)
                    
                    # Create new column or replace
                    if preserve_original:
                        new_col_name = f"{col}_scaled"
                        df[new_col_name] = np.nan
                        df.loc[mask, new_col_name] = scaled_data.flatten()
                        scaled_columns.append(new_col_name)
                    else:
                        df.loc[mask, col] = scaled_data.flatten()
                        scaled_columns.append(col)
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Bulk Feature Scaling',
                    'details': f"Applied {bulk_strategy} to {len(selected_columns)} columns: {', '.join(selected_columns)}"
                })
                
                st.success(f"✅ Applied {bulk_strategy} to {len(selected_columns)} columns!")
                if preserve_original:
                    st.info(f"Created {len(scaled_columns)} new scaled columns")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Bulk scaling failed: {str(e)}")
    
    with col2:
        st.markdown("#### Scaling Impact Analysis")
        
        # Show before/after statistics
        scaled_cols = [col for col in df.columns if '_scaled' in col]
        
        if scaled_cols:
            st.markdown("#### Available Scaled Columns")
            
            for scaled_col in scaled_cols[:10]:  # Show first 10
                original_col = scaled_col.replace('_scaled', '')
                if original_col in df.columns:
                    
                    orig_range = df[original_col].max() - df[original_col].min()
                    scaled_range = df[scaled_col].max() - df[scaled_col].min()
                    
                    st.write(f"**{original_col}**")
                    st.write(f"  Original range: {orig_range:.2f}")
                    st.write(f"  Scaled range: {scaled_range:.2f}")
                    st.write("---")
        
        # Column comparison
        st.markdown("#### Before/After Comparison")
        
        if scaled_cols:
            compare_col = st.selectbox("Select scaled column for comparison:", scaled_cols, key="compare_scaled")
            
            original_col = compare_col.replace('_scaled', '')
            
            if original_col in df.columns:
                # Statistics comparison
                orig_stats = df[original_col].describe()
                scaled_stats = df[compare_col].describe()
                
                comparison_df = pd.DataFrame({
                    'Original': orig_stats,
                    'Scaled': scaled_stats
                })
                
                # Round for display and convert to strings
                comparison_df = comparison_df.round(3)
                for col in comparison_df.columns:
                    comparison_df[col] = comparison_df[col].astype(str)
                
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("No scaled columns found. Apply scaling first.")

with scaling_tabs[3]:
    st.markdown("### 🔬 Advanced Scaling Methods")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Distribution-Based Scaling")
        
        advanced_col = st.selectbox("Select column for advanced scaling:", numeric_cols, key="advanced_col")
        
        advanced_method = st.selectbox(
            "Advanced method:",
            [
                "Box-Cox Transformation",
                "Quantile-based Scaling",
                "Rank Transformation",
                "Log-based Scaling",
                "Winsorization + Scaling"
            ]
        )
        
        # Method-specific parameters
        if advanced_method == "Box-Cox Transformation":
            st.info("Box-Cox requires positive values. Negative values will be shifted.")
        
        elif advanced_method == "Quantile-based Scaling":
            quantile_range = st.slider("Quantile range:", 0.01, 0.5, 0.25, help="Range for quantile normalization")
        
        elif advanced_method == "Log-based Scaling":
            log_type = st.selectbox("Log transformation:", ["Natural log", "Log10", "Log2", "Log1p (log(1+x))"])
        
        elif advanced_method == "Winsorization + Scaling":
            winsorize_limits = st.slider("Winsorization limits:", 0.01, 0.2, 0.05, help="Fraction of data to winsorize")
        
        if st.button("🔬 Apply Advanced Method", key="advanced_scaling"):
            try:
                col_data = df[advanced_col].copy()
                
                if advanced_method == "Box-Cox Transformation":
                    # Shift data to make it positive if needed
                    min_val = col_data.min()
                    if min_val <= 0:
                        col_data = col_data - min_val + 1
                    
                    # Apply Box-Cox
                    transformed_data, lambda_param = stats.boxcox(col_data.dropna())
                    
                    # Update dataframe
                    mask = col_data.notna()
                    df[f"{advanced_col}_boxcox"] = np.nan
                    df.loc[mask, f"{advanced_col}_boxcox"] = transformed_data
                    
                    result_msg = f"Applied Box-Cox transformation (λ={lambda_param:.3f})"
                
                elif advanced_method == "Quantile-based Scaling":
                    from scipy.stats import trim_mean
                    
                    # Calculate quantile-based statistics
                    q_low = col_data.quantile(quantile_range)
                    q_high = col_data.quantile(1 - quantile_range)
                    q_median = col_data.median()
                    
                    # Scale using quantile range
                    scaled_data = (col_data - q_median) / (q_high - q_low)
                    df[f"{advanced_col}_quantile_scaled"] = scaled_data
                    
                    result_msg = f"Applied quantile-based scaling (range: {quantile_range:.2f})"
                
                elif advanced_method == "Rank Transformation":
                    # Convert to ranks and then normalize
                    ranked_data = col_data.rank() / len(col_data.dropna())
                    df[f"{advanced_col}_rank_normalized"] = ranked_data
                    
                    result_msg = "Applied rank transformation"
                
                elif advanced_method == "Log-based Scaling":
                    if log_type == "Natural log":
                        # Ensure positive values
                        positive_data = col_data - col_data.min() + 1 if col_data.min() <= 0 else col_data
                        transformed_data = np.log(positive_data)
                    elif log_type == "Log10":
                        positive_data = col_data - col_data.min() + 1 if col_data.min() <= 0 else col_data
                        transformed_data = np.log10(positive_data)
                    elif log_type == "Log2":
                        positive_data = col_data - col_data.min() + 1 if col_data.min() <= 0 else col_data
                        transformed_data = np.log2(positive_data)
                    else:  # Log1p
                        transformed_data = np.log1p(col_data)
                    
                    df[f"{advanced_col}_log_scaled"] = transformed_data
                    result_msg = f"Applied {log_type} transformation"
                
                elif advanced_method == "Winsorization + Scaling":
                    from scipy.stats import trim_mean
                    
                    # Winsorize data (clip extreme values)
                    q_low = col_data.quantile(winsorize_limits)
                    q_high = col_data.quantile(1 - winsorize_limits)
                    
                    winsorized_data = col_data.clip(lower=q_low, upper=q_high)
                    
                    # Then apply standard scaling
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(winsorized_data.values.reshape(-1, 1)).flatten()
                    
                    df[f"{advanced_col}_winsorized_scaled"] = scaled_data
                    result_msg = f"Applied winsorization ({winsorize_limits:.2%}) + scaling"
                
                # Update session state
                st.session_state.current_dataset = df
                
                st.success(f"✅ {result_msg}")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Advanced scaling failed: {str(e)}")
    
    with col2:
        st.markdown("#### Scaling Validation")
        
        # Show all scaled columns
        all_scaled_cols = [col for col in df.columns if any(suffix in col for suffix in ['_scaled', '_boxcox', '_quantile', '_rank', '_log', '_winsorized'])]
        
        if all_scaled_cols:
            st.markdown("#### Available Transformed Columns")
            
            for col in all_scaled_cols[:15]:  # Show first 15
                transformation_type = col.split('_')[-1]
                st.write(f"• **{col}** ({transformation_type} transformation)")
            
            # Validation metrics
            st.markdown("#### Validation Metrics")
            
            validate_col = st.selectbox("Select column to validate:", all_scaled_cols, key="validate_scaled_col")
            
            if st.button("📊 Validate Scaling"):
                try:
                    col_data = df[validate_col].dropna()
                    
                    validation_metrics = {
                        'Mean': f"{col_data.mean():.4f}",
                        'Standard Deviation': f"{col_data.std():.4f}",
                        'Skewness': f"{col_data.skew():.4f}",
                        'Kurtosis': f"{col_data.kurtosis():.4f}",
                        'Range': f"{col_data.max() - col_data.min():.4f}",
                        'Min Value': f"{col_data.min():.4f}",
                        'Max Value': f"{col_data.max():.4f}",
                        'Zero Values': f"{(col_data == 0).sum():,}",
                        'Negative Values': f"{(col_data < 0).sum():,}"
                    }
                    
                    for metric, value in validation_metrics.items():
                        st.write(f"**{metric}:** {value}")
                    
                    # Distribution plot
                    fig = px.histogram(col_data, nbins=30, title=f"Distribution: {validate_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Validation failed: {str(e)}")
        
        else:
            st.info("No scaled columns found. Apply scaling methods first.")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Scaled Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"scaled_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Feature Engineering"):
        st.switch_page("pages/13_Feature_Engineering.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Scaling Guide")
    
    st.markdown("#### Scaling Methods:")
    methods = [
        "**StandardScaler:** Mean=0, Std=1",
        "**MinMaxScaler:** Range 0-1",
        "**RobustScaler:** Uses median & IQR", 
        "**PowerTransformer:** Reduces skewness",
        "**QuantileTransformer:** Uniform/normal output"
    ]
    
    for method in methods:
        st.markdown(f"• {method}")
    
    st.markdown("---")
    st.markdown("#### When to Use:")
    
    use_cases = [
        "**Normal data, no outliers:** StandardScaler",
        "**Data with outliers:** RobustScaler",
        "**Need specific range:** MinMaxScaler",
        "**Highly skewed data:** PowerTransformer",
        "**ML algorithms sensitive to scale:** Any scaler"
    ]
    
    for case in use_cases:
        st.markdown(f"• {case}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Analyze distribution before scaling
    • Consider outlier impact
    • Preserve original columns for comparison
    • Validate scaling results
    • Choose appropriate method for ML algorithm
    """)