import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")

st.title("⚙️ Feature Engineering")
st.markdown("Intelligent feature creation, selection, and dimensionality reduction with ML-powered insights")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Features", len(df.columns))

with col2:
    st.metric("Numeric Features", len(numeric_cols))

with col3:
    st.metric("Categorical Features", len(categorical_cols))

with col4:
    engineered_cols = [col for col in df.columns if any(suffix in col for suffix in ['_poly_', '_interaction_', '_binned_', '_pca_', '_ratio_'])]
    st.metric("Engineered Features", len(engineered_cols))

# Feature engineering tabs
fe_tabs = st.tabs(["🔧 Feature Creation", "📊 Feature Selection", "🎯 Dimensionality Reduction", "🤖 Automated Engineering"])

with fe_tabs[0]:
    st.markdown("### 🔧 Feature Creation")
    
    creation_subtabs = st.tabs(["📈 Polynomial Features", "🔗 Interaction Features", "📊 Binning Features", "⚡ Mathematical Features"])
    
    with creation_subtabs[0]:
        st.markdown("#### 📈 Polynomial Feature Generation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                poly_columns = st.multiselect(
                    "Select columns for polynomial features:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    help="Choose numeric columns to generate polynomial features"
                )
                
                poly_degree = st.slider("Polynomial degree:", 2, 5, 2, help="Higher degrees create more complex features")
                
                include_bias = st.checkbox("Include bias term", value=False, help="Include intercept term")
                
                interaction_only = st.checkbox("Interaction features only", value=False, 
                                             help="Only create interaction terms, no pure polynomial terms")
                
                if st.button("🚀 Generate Polynomial Features", type="primary") and poly_columns:
                    try:
                        # Prepare data
                        poly_data = df[poly_columns].fillna(0)  # Fill NaN with 0 for polynomial features
                        
                        # Create polynomial features
                        poly = PolynomialFeatures(
                            degree=poly_degree, 
                            include_bias=include_bias, 
                            interaction_only=interaction_only
                        )
                        
                        poly_features = poly.fit_transform(poly_data)
                        
                        # Get feature names
                        feature_names = poly.get_feature_names_out(poly_columns)
                        
                        # Add to dataframe (skip first column if it's bias)
                        start_idx = 1 if include_bias and not interaction_only else len(poly_columns)
                        
                        for i, name in enumerate(feature_names[start_idx:], start=start_idx):
                            df[f"poly_{name}"] = poly_features[:, i]
                        
                        # Update session state
                        st.session_state.current_dataset = df
                        
                        created_features = len(feature_names) - start_idx
                        
                        # Log action
                        if 'processing_log' not in st.session_state:
                            st.session_state.processing_log = []
                        
                        st.session_state.processing_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Feature Engineering - Polynomial',
                            'details': f"Created {created_features} polynomial features (degree={poly_degree}) from {len(poly_columns)} columns"
                        })
                        
                        st.success(f"✅ Created {created_features} polynomial features!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Polynomial feature generation failed: {str(e)}")
            else:
                st.info("No numeric columns available for polynomial features.")
        
        with col2:
            st.markdown("#### Polynomial Features Preview")
            
            if poly_columns:
                st.markdown("**Expected Features:**")
                
                if poly_degree == 2:
                    st.write("• Original features")
                    st.write("• Squared terms (x²)")
                    if not interaction_only:
                        st.write("• Cross products (x₁×x₂)")
                
                elif poly_degree == 3:
                    st.write("• Linear terms (x)")
                    st.write("• Squared terms (x²)")
                    st.write("• Cubic terms (x³)")
                    st.write("• Interaction terms (x₁×x₂, x₁×x₂×x₃)")
                
                # Estimate number of features
                from math import comb
                n_features = len(poly_columns)
                
                if interaction_only:
                    total_features = sum(comb(n_features, i) for i in range(2, poly_degree + 1))
                else:
                    total_features = sum(comb(n_features + i - 1, i) for i in range(1, poly_degree + 1))
                
                st.write(f"**Estimated new features:** {total_features}")
                
                if total_features > 100:
                    st.warning("⚠️ Large number of features will be created. Consider reducing degree or number of columns.")
    
    with creation_subtabs[1]:
        st.markdown("#### 🔗 Custom Interaction Features")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if len(numeric_cols) >= 2:
                # Manual interaction selection
                st.markdown("##### Manual Interaction Selection")
                
                interaction_col1 = st.selectbox("First column:", numeric_cols, key="int_col1")
                interaction_col2 = st.selectbox("Second column:", [col for col in numeric_cols if col != interaction_col1], key="int_col2")
                
                interaction_type = st.selectbox(
                    "Interaction type:",
                    ["Multiply (x₁ × x₂)", "Add (x₁ + x₂)", "Subtract (x₁ - x₂)", "Divide (x₁ / x₂)", "Ratio (x₁ / (x₁ + x₂))"],
                    help="Choose how to combine the two features"
                )
                
                if st.button("➕ Create Interaction Feature"):
                    try:
                        col1_data = df[interaction_col1]
                        col2_data = df[interaction_col2]
                        
                        if interaction_type == "Multiply (x₁ × x₂)":
                            new_feature = col1_data * col2_data
                            feature_name = f"{interaction_col1}_x_{interaction_col2}"
                        
                        elif interaction_type == "Add (x₁ + x₂)":
                            new_feature = col1_data + col2_data
                            feature_name = f"{interaction_col1}_plus_{interaction_col2}"
                        
                        elif interaction_type == "Subtract (x₁ - x₂)":
                            new_feature = col1_data - col2_data
                            feature_name = f"{interaction_col1}_minus_{interaction_col2}"
                        
                        elif interaction_type == "Divide (x₁ / x₂)":
                            new_feature = col1_data / col2_data.replace(0, np.nan)  # Avoid division by zero
                            feature_name = f"{interaction_col1}_div_{interaction_col2}"
                        
                        elif interaction_type == "Ratio (x₁ / (x₁ + x₂))":
                            denominator = col1_data + col2_data
                            new_feature = col1_data / denominator.replace(0, np.nan)
                            feature_name = f"{interaction_col1}_ratio_{interaction_col2}"
                        
                        df[feature_name] = new_feature
                        st.session_state.current_dataset = df
                        
                        st.success(f"✅ Created interaction feature: {feature_name}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Interaction feature creation failed: {str(e)}")
                
                # Bulk interaction creation
                st.markdown("##### Bulk Interaction Creation")
                
                bulk_columns = st.multiselect(
                    "Select columns for all pairwise interactions:",
                    numeric_cols,
                    help="Create all possible pairwise multiplication interactions"
                )
                
                if st.button("🚀 Create All Pairwise Interactions") and len(bulk_columns) >= 2:
                    try:
                        created_features = []
                        
                        for i, col1 in enumerate(bulk_columns):
                            for col2 in bulk_columns[i+1:]:
                                feature_name = f"{col1}_x_{col2}"
                                df[feature_name] = df[col1] * df[col2]
                                created_features.append(feature_name)
                        
                        st.session_state.current_dataset = df
                        
                        # Log action
                        if 'processing_log' not in st.session_state:
                            st.session_state.processing_log = []
                        
                        st.session_state.processing_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Feature Engineering - Interactions',
                            'details': f"Created {len(created_features)} pairwise interaction features from {len(bulk_columns)} columns"
                        })
                        
                        st.success(f"✅ Created {len(created_features)} interaction features!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Bulk interaction creation failed: {str(e)}")
            
            else:
                st.info("Need at least 2 numeric columns for interaction features.")
        
        with col2:
            st.markdown("#### Interaction Preview")
            
            if len(numeric_cols) >= 2:
                # Show potential interactions
                st.markdown("**Available Interactions:**")
                
                interaction_count = len(numeric_cols) * (len(numeric_cols) - 1) // 2
                st.write(f"Possible pairwise interactions: {interaction_count}")
                
                if interaction_count <= 10:
                    st.markdown("**Specific Interactions:**")
                    for i, col1 in enumerate(numeric_cols):
                        for col2 in numeric_cols[i+1:]:
                            st.write(f"• {col1} × {col2}")
                else:
                    st.write("Too many interactions to list individually")
                
                # Sample calculation preview
                if interaction_col1 and interaction_col2:
                    st.markdown("#### Sample Calculation")
                    
                    sample_data = df[[interaction_col1, interaction_col2]].head(5)
                    
                    for idx, row in sample_data.iterrows():
                        val1, val2 = row[interaction_col1], row[interaction_col2]
                        if pd.notna(val1) and pd.notna(val2):
                            if interaction_type == "Multiply (x₁ × x₂)":
                                result = val1 * val2
                            elif interaction_type == "Add (x₁ + x₂)":
                                result = val1 + val2
                            elif interaction_type == "Subtract (x₁ - x₂)":
                                result = val1 - val2
                            elif interaction_type == "Divide (x₁ / x₂)" and val2 != 0:
                                result = val1 / val2
                            elif interaction_type == "Ratio (x₁ / (x₁ + x₂))" and (val1 + val2) != 0:
                                result = val1 / (val1 + val2)
                            else:
                                result = "undefined"
                            
                            st.write(f"Row {idx}: {val1:.2f} & {val2:.2f} → {result}")
                            
                            if idx >= 2:  # Show only first 3 examples
                                break
    
    with creation_subtabs[2]:
        st.markdown("#### 📊 Feature Binning")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                binning_column = st.selectbox("Select column to bin:", numeric_cols, key="bin_col")
                
                binning_method = st.selectbox(
                    "Binning method:",
                    ["Equal-width bins", "Equal-frequency bins", "Quantile bins", "Custom bins"],
                    help="Choose how to create bins"
                )
                
                if binning_method in ["Equal-width bins", "Equal-frequency bins", "Quantile bins"]:
                    n_bins = st.slider("Number of bins:", 2, 20, 5, help="Number of bins to create")
                
                elif binning_method == "Custom bins":
                    # Allow custom bin edges
                    col_min = df[binning_column].min()
                    col_max = df[binning_column].max()
                    
                    st.write(f"Column range: {col_min:.2f} to {col_max:.2f}")
                    
                    custom_bins_input = st.text_input(
                        "Custom bin edges (comma-separated):",
                        value=f"{col_min:.2f}, {(col_min+col_max)/2:.2f}, {col_max:.2f}",
                        help="Enter bin edges as comma-separated values"
                    )
                
                # Labels option
                create_labels = st.checkbox("Create meaningful labels", value=True, help="Create descriptive labels for bins")
                
                if st.button("📊 Create Binned Feature", type="primary"):
                    try:
                        col_data = df[binning_column]
                        
                        if binning_method == "Equal-width bins":
                            binned_data, bin_edges = pd.cut(col_data, bins=n_bins, retbins=True, include_lowest=True)
                        
                        elif binning_method == "Equal-frequency bins":
                            binned_data, bin_edges = pd.qcut(col_data, q=n_bins, retbins=True, duplicates='drop')
                        
                        elif binning_method == "Quantile bins":
                            quantiles = np.linspace(0, 1, n_bins + 1)
                            bin_edges = col_data.quantile(quantiles).unique()
                            binned_data = pd.cut(col_data, bins=bin_edges, include_lowest=True)
                        
                        elif binning_method == "Custom bins":
                            try:
                                bin_edges = [float(x.strip()) for x in custom_bins_input.split(',')]
                                binned_data = pd.cut(col_data, bins=bin_edges, include_lowest=True)
                            except:
                                st.error("Invalid bin edges format. Use comma-separated numbers.")
                                st.stop()
                        
                        # Create feature name
                        feature_name = f"{binning_column}_binned"
                        
                        if create_labels:
                            # Create meaningful labels
                            if binning_method in ["Equal-width bins", "Custom bins"]:
                                labels = [f"Bin_{i+1}_({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})" for i in range(len(bin_edges)-1)]
                            else:
                                labels = [f"Q{i+1}" for i in range(len(bin_edges)-1)]
                            
                            # Apply labels
                            binned_data = pd.cut(col_data, bins=bin_edges, labels=labels, include_lowest=True)
                        
                        df[feature_name] = binned_data
                        
                        # Also create binary indicators for each bin
                        if st.checkbox("Create binary indicators for each bin", value=False):
                            binned_dummies = pd.get_dummies(binned_data, prefix=f"{binning_column}_bin")
                            df = pd.concat([df, binned_dummies], axis=1)
                        
                        st.session_state.current_dataset = df
                        
                        st.success(f"✅ Created binned feature: {feature_name}")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Binning failed: {str(e)}")
            else:
                st.info("No numeric columns available for binning.")
        
        with col2:
            st.markdown("#### Binning Preview")
            
            if binning_column:
                col_data = df[binning_column].dropna()
                
                # Show distribution
                fig = px.histogram(col_data, nbins=30, title=f"Current Distribution: {binning_column}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                st.markdown("**Column Statistics:**")
                st.write(f"Min: {col_data.min():.2f}")
                st.write(f"Max: {col_data.max():.2f}")
                st.write(f"Mean: {col_data.mean():.2f}")
                st.write(f"Median: {col_data.median():.2f}")
                st.write(f"Std: {col_data.std():.2f}")
    
    with creation_subtabs[3]:
        st.markdown("#### ⚡ Mathematical Features")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                math_columns = st.multiselect(
                    "Select columns for mathematical features:",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    help="Choose columns to apply mathematical transformations"
                )
                
                math_operations = st.multiselect(
                    "Select operations:",
                    [
                        "Square root", "Square", "Cube", "Logarithm", 
                        "Exponential", "Reciprocal", "Absolute value",
                        "Sign (positive/negative)", "Power of 2"
                    ],
                    default=["Square", "Square root"],
                    help="Choose mathematical operations to apply"
                )
                
                if st.button("⚡ Create Mathematical Features", type="primary") and math_columns and math_operations:
                    try:
                        created_features = []
                        
                        for col in math_columns:
                            col_data = df[col]
                            
                            for operation in math_operations:
                                if operation == "Square root":
                                    # Handle negative values
                                    new_feature = np.sqrt(np.abs(col_data))
                                    feature_name = f"{col}_sqrt"
                                
                                elif operation == "Square":
                                    new_feature = col_data ** 2
                                    feature_name = f"{col}_squared"
                                
                                elif operation == "Cube":
                                    new_feature = col_data ** 3
                                    feature_name = f"{col}_cubed"
                                
                                elif operation == "Logarithm":
                                    # Use log1p to handle zeros and negatives
                                    new_feature = np.log1p(np.abs(col_data))
                                    feature_name = f"{col}_log"
                                
                                elif operation == "Exponential":
                                    # Clip to prevent overflow
                                    clipped_data = col_data.clip(-10, 10)
                                    new_feature = np.exp(clipped_data)
                                    feature_name = f"{col}_exp"
                                
                                elif operation == "Reciprocal":
                                    new_feature = 1 / col_data.replace(0, np.nan)
                                    feature_name = f"{col}_reciprocal"
                                
                                elif operation == "Absolute value":
                                    new_feature = np.abs(col_data)
                                    feature_name = f"{col}_abs"
                                
                                elif operation == "Sign (positive/negative)":
                                    new_feature = np.sign(col_data)
                                    feature_name = f"{col}_sign"
                                
                                elif operation == "Power of 2":
                                    new_feature = 2 ** col_data.clip(-10, 10)  # Prevent overflow
                                    feature_name = f"{col}_pow2"
                                
                                df[feature_name] = new_feature
                                created_features.append(feature_name)
                        
                        st.session_state.current_dataset = df
                        
                        # Log action
                        if 'processing_log' not in st.session_state:
                            st.session_state.processing_log = []
                        
                        st.session_state.processing_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Feature Engineering - Mathematical',
                            'details': f"Created {len(created_features)} mathematical features using {len(math_operations)} operations on {len(math_columns)} columns"
                        })
                        
                        st.success(f"✅ Created {len(created_features)} mathematical features!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Mathematical feature creation failed: {str(e)}")
            else:
                st.info("No numeric columns available for mathematical features.")
        
        with col2:
            st.markdown("#### Mathematical Operations Info")
            
            operations_info = {
                "Square root": "√x - Good for reducing right skewness",
                "Square": "x² - Amplifies larger values",
                "Cube": "x³ - Strong amplification, preserves sign",
                "Logarithm": "ln(|x|+1) - Reduces right skewness",
                "Exponential": "eˣ - Exponential growth pattern",
                "Reciprocal": "1/x - Inverse relationship",
                "Absolute value": "|x| - Remove negative values",
                "Sign": "sign(x) - Just positive/negative/zero",
                "Power of 2": "2ˣ - Exponential base 2"
            }
            
            for op, description in operations_info.items():
                st.markdown(f"**{op}:** {description}")

with fe_tabs[1]:
    st.markdown("### 📊 Feature Selection")
    
    selection_subtabs = st.tabs(["📈 Statistical Selection", "🤖 ML-Based Selection", "🧹 Correlation-Based"])
    
    with selection_subtabs[0]:
        st.markdown("#### 📈 Statistical Feature Selection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                # Target variable selection
                target_column = st.selectbox("Select target variable:", numeric_cols + categorical_cols, 
                                           help="Variable to predict (for supervised selection)")
                
                # Feature selection method
                selection_method = st.selectbox(
                    "Selection method:",
                    ["SelectKBest (f_classif)", "SelectKBest (mutual_info)", "Variance Threshold", "Correlation Threshold"],
                    help="Choose feature selection method"
                )
                
                if selection_method in ["SelectKBest (f_classif)", "SelectKBest (mutual_info)"]:
                    k_features = st.slider("Number of features to select:", 1, min(20, len(numeric_cols)), 10)
                
                elif selection_method == "Variance Threshold":
                    variance_threshold = st.slider("Minimum variance:", 0.0, 1.0, 0.01, 
                                                  help="Features with variance below this will be removed")
                
                elif selection_method == "Correlation Threshold":
                    corr_threshold = st.slider("Correlation threshold:", 0.5, 0.99, 0.95,
                                             help="Remove features with correlation above this")
                
                if st.button("🔍 Apply Feature Selection", type="primary"):
                    try:
                        # Prepare feature matrix
                        feature_cols = [col for col in numeric_cols if col != target_column]
                        X = df[feature_cols].fillna(0)
                        y = df[target_column]
                        
                        if selection_method == "SelectKBest (f_classif)":
                            selector = SelectKBest(score_func=f_classif, k=k_features)
                            X_selected = selector.fit_transform(X, y)
                            
                            # Get selected feature names
                            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                            
                        elif selection_method == "SelectKBest (mutual_info)":
                            # Determine if classification or regression
                            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 20:
                                selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
                            else:
                                from sklearn.feature_selection import mutual_info_regression
                                selector = SelectKBest(score_func=mutual_info_regression, k=k_features)
                            
                            X_selected = selector.fit_transform(X, y)
                            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                        
                        elif selection_method == "Variance Threshold":
                            from sklearn.feature_selection import VarianceThreshold
                            selector = VarianceThreshold(threshold=variance_threshold)
                            X_selected = selector.fit_transform(X)
                            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                        
                        elif selection_method == "Correlation Threshold":
                            # Calculate correlation matrix
                            corr_matrix = X.corr().abs()
                            
                            # Find highly correlated pairs
                            high_corr_pairs = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i+1, len(corr_matrix.columns)):
                                    if corr_matrix.iloc[i, j] > corr_threshold:
                                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
                            
                            # Remove one feature from each highly correlated pair
                            features_to_remove = set()
                            for feat1, feat2, corr_val in high_corr_pairs:
                                if feat1 not in features_to_remove:
                                    features_to_remove.add(feat2)
                            
                            selected_features = [col for col in feature_cols if col not in features_to_remove]
                        
                        # Store results
                        st.session_state.selected_features = selected_features
                        
                        st.success(f"✅ Selected {len(selected_features)} features out of {len(feature_cols)}")
                        
                        # Show selected features
                        st.markdown("**Selected Features:**")
                        for feat in selected_features:
                            st.write(f"• {feat}")
                        
                        if len(selected_features) < len(feature_cols):
                            removed_features = [col for col in feature_cols if col not in selected_features]
                            with st.expander(f"Removed Features ({len(removed_features)})"):
                                for feat in removed_features:
                                    st.write(f"• {feat}")
                        
                    except Exception as e:
                        st.error(f"❌ Feature selection failed: {str(e)}")
            else:
                st.info("No numeric columns available for feature selection.")
        
        with col2:
            st.markdown("#### Selection Results")
            
            if 'selected_features' in st.session_state:
                selected = st.session_state.selected_features
                
                st.markdown("**Selection Summary:**")
                st.write(f"Original features: {len(numeric_cols)}")
                st.write(f"Selected features: {len(selected)}")
                st.write(f"Reduction: {((len(numeric_cols) - len(selected)) / len(numeric_cols) * 100):.1f}%")
                
                # Create filtered dataset option
                if st.button("📄 Create Filtered Dataset"):
                    # Create new dataframe with selected features only
                    filtered_df = df[selected + [target_column]].copy()
                    
                    # Option to replace current dataset
                    replace_current = st.checkbox("Replace current dataset with filtered version")
                    
                    if replace_current and st.button("🔄 Replace Dataset"):
                        st.session_state.current_dataset = filtered_df
                        st.success("✅ Dataset replaced with selected features!")
                        st.rerun()
            
            else:
                st.info("Run feature selection to see results here.")
    
    with selection_subtabs[1]:
        st.markdown("#### 🤖 ML-Based Feature Selection")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                ml_target = st.selectbox("Target variable:", numeric_cols + categorical_cols, key="ml_target")
                
                ml_method = st.selectbox(
                    "ML selection method:",
                    ["Random Forest Importance", "Recursive Feature Elimination (RFE)", "LASSO Regularization"],
                    help="Choose ML-based selection method"
                )
                
                if ml_method == "Random Forest Importance":
                    n_estimators = st.slider("Number of trees:", 10, 200, 100)
                    importance_threshold = st.slider("Importance threshold:", 0.001, 0.1, 0.01)
                
                elif ml_method == "Recursive Feature Elimination (RFE)":
                    n_features_to_select = st.slider("Features to select:", 1, min(15, len(numeric_cols)), 10)
                
                if st.button("🤖 Apply ML Selection", type="primary"):
                    try:
                        # Prepare data
                        feature_cols = [col for col in numeric_cols if col != ml_target]
                        X = df[feature_cols].fillna(0)
                        y = df[ml_target]
                        
                        if ml_method == "Random Forest Importance":
                            # Determine if classification or regression
                            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                                rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                            else:
                                rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                            
                            rf.fit(X, y)
                            
                            # Get feature importances
                            importances = rf.feature_importances_
                            
                            # Select features above threshold
                            selected_indices = importances > importance_threshold
                            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_indices[i]]
                            
                            # Store importance scores
                            feature_importance = list(zip(feature_cols, importances))
                            feature_importance.sort(key=lambda x: x[1], reverse=True)
                            
                            st.session_state.feature_importance = feature_importance
                        
                        elif ml_method == "Recursive Feature Elimination (RFE)":
                            # Use random forest as base estimator
                            if y.dtype in ['object', 'category'] or y.nunique() < 20:
                                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
                            else:
                                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
                            
                            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
                            rfe.fit(X, y)
                            
                            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
                        
                        st.session_state.ml_selected_features = selected_features
                        
                        st.success(f"✅ ML selection completed! Selected {len(selected_features)} features.")
                        
                        # Show results
                        st.markdown("**Selected Features:**")
                        for feat in selected_features:
                            st.write(f"• {feat}")
                        
                    except Exception as e:
                        st.error(f"❌ ML feature selection failed: {str(e)}")
            else:
                st.info("No numeric columns available for ML-based selection.")
        
        with col2:
            st.markdown("#### Feature Importance")
            
            if 'feature_importance' in st.session_state:
                importance_data = st.session_state.feature_importance
                
                # Show top 10 most important features
                st.markdown("**Top 10 Most Important Features:**")
                
                top_features = importance_data[:10]
                feature_names = [item[0] for item in top_features]
                importance_scores = [item[1] for item in top_features]
                
                # Create bar chart
                fig = px.bar(
                    x=importance_scores,
                    y=feature_names,
                    orientation='h',
                    title="Feature Importance",
                    labels={'x': 'Importance Score', 'y': 'Features'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Show all importances in expandable section
                with st.expander("All Feature Importances"):
                    for feat, imp in importance_data:
                        st.write(f"• **{feat}:** {imp:.4f}")
            
            else:
                st.info("Run ML-based selection to see feature importance.")
    
    with selection_subtabs[2]:
        st.markdown("#### 🧹 Correlation-Based Feature Selection")
        
        if len(numeric_cols) > 1:
            # Correlation matrix visualization
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # High correlation pairs
            st.markdown("#### High Correlation Pairs")
            
            threshold = st.slider("Correlation threshold:", 0.5, 0.99, 0.8)
            
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > threshold:
                        high_corr_pairs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_matrix.iloc[i, j]:.3f}"
                        })
            
            if high_corr_pairs:
                pairs_df = pd.DataFrame(high_corr_pairs)
                
                # Convert to strings to avoid Arrow issues
                for col in pairs_df.columns:
                    pairs_df[col] = pairs_df[col].astype(str)
                
                st.dataframe(pairs_df, use_container_width=True)
                
                st.write(f"Found {len(high_corr_pairs)} highly correlated feature pairs.")
            else:
                st.info(f"No feature pairs with correlation > {threshold} found.")
        
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

with fe_tabs[2]:
    st.markdown("### 🎯 Dimensionality Reduction")
    
    dim_reduction_subtabs = st.tabs(["📊 PCA", "🔧 Feature Aggregation"])
    
    with dim_reduction_subtabs[0]:
        st.markdown("#### 📊 Principal Component Analysis (PCA)")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if len(numeric_cols) > 1:
                pca_columns = st.multiselect(
                    "Select columns for PCA:",
                    numeric_cols,
                    default=numeric_cols,
                    help="Choose numeric columns for PCA transformation"
                )
                
                pca_method = st.selectbox(
                    "PCA method:",
                    ["Specify number of components", "Specify variance explained", "Automatic (95% variance)"],
                    help="How to determine number of components"
                )
                
                if pca_method == "Specify number of components":
                    n_components = st.slider("Number of components:", 1, min(len(pca_columns), 10), 
                                            min(3, len(pca_columns)))
                
                elif pca_method == "Specify variance explained":
                    variance_explained = st.slider("Minimum variance to explain:", 0.5, 0.99, 0.95)
                
                if st.button("🎯 Apply PCA", type="primary") and pca_columns:
                    try:
                        # Prepare data
                        pca_data = df[pca_columns].fillna(0)
                        
                        # Standardize data for PCA
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        pca_data_scaled = scaler.fit_transform(pca_data)
                        
                        # Apply PCA
                        if pca_method == "Specify number of components":
                            pca = PCA(n_components=n_components)
                        elif pca_method == "Specify variance explained":
                            pca = PCA(n_components=variance_explained)
                        else:  # Automatic
                            pca = PCA(n_components=0.95)
                        
                        pca_result = pca.fit_transform(pca_data_scaled)
                        
                        # Add PCA components to dataframe
                        for i in range(pca_result.shape[1]):
                            df[f"pca_component_{i+1}"] = pca_result[:, i]
                        
                        # Store PCA information
                        st.session_state.pca_explained_variance = pca.explained_variance_ratio_
                        st.session_state.pca_components = pca.components_
                        st.session_state.pca_feature_names = pca_columns
                        
                        st.session_state.current_dataset = df
                        
                        # Log action
                        if 'processing_log' not in st.session_state:
                            st.session_state.processing_log = []
                        
                        st.session_state.processing_log.append({
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Dimensionality Reduction - PCA',
                            'details': f"Applied PCA to {len(pca_columns)} features, created {pca_result.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.1%} variance"
                        })
                        
                        st.success(f"✅ PCA completed! Created {pca_result.shape[1]} components explaining {pca.explained_variance_ratio_.sum():.1%} of variance.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ PCA failed: {str(e)}")
            else:
                st.info("Need at least 2 numeric columns for PCA.")
        
        with col2:
            st.markdown("#### PCA Results")
            
            if 'pca_explained_variance' in st.session_state:
                explained_var = st.session_state.pca_explained_variance
                
                # Variance explained chart
                fig = px.bar(
                    x=[f"PC{i+1}" for i in range(len(explained_var))],
                    y=explained_var,
                    title="Variance Explained by Each Component",
                    labels={'x': 'Principal Component', 'y': 'Variance Explained'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cumulative variance
                cumulative_var = np.cumsum(explained_var)
                
                st.markdown("**Variance Explained:**")
                for i, (individual, cumulative) in enumerate(zip(explained_var, cumulative_var)):
                    st.write(f"PC{i+1}: {individual:.3f} (Cumulative: {cumulative:.3f})")
                
                # Component interpretation
                if 'pca_components' in st.session_state:
                    st.markdown("#### Component Interpretation")
                    
                    components = st.session_state.pca_components
                    feature_names = st.session_state.pca_feature_names
                    
                    selected_pc = st.selectbox("Select component to interpret:", 
                                             [f"PC{i+1}" for i in range(len(components))])
                    
                    pc_idx = int(selected_pc[2:]) - 1  # Extract number from "PC1", "PC2", etc.
                    
                    # Show feature loadings
                    loadings = components[pc_idx]
                    
                    loading_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Loading': loadings,
                        'Abs_Loading': np.abs(loadings)
                    }).sort_values('Abs_Loading', ascending=False)
                    
                    # Convert to strings to avoid Arrow issues
                    for col in loading_df.columns:
                        loading_df[col] = loading_df[col].astype(str)
                    
                    st.dataframe(loading_df, use_container_width=True)
            
            else:
                st.info("Run PCA to see results here.")
    
    with dim_reduction_subtabs[1]:
        st.markdown("#### 🔧 Feature Aggregation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if numeric_cols:
                aggregation_columns = st.multiselect(
                    "Select columns to aggregate:",
                    numeric_cols,
                    help="Choose columns to create aggregate features"
                )
                
                aggregation_methods = st.multiselect(
                    "Aggregation methods:",
                    ["Mean", "Sum", "Min", "Max", "Standard Deviation", "Range"],
                    default=["Mean", "Sum"],
                    help="Choose how to aggregate the selected columns"
                )
                
                if st.button("🔧 Create Aggregate Features") and aggregation_columns and aggregation_methods:
                    try:
                        agg_data = df[aggregation_columns]
                        created_features = []
                        
                        for method in aggregation_methods:
                            if method == "Mean":
                                df[f"agg_mean_{len(aggregation_columns)}cols"] = agg_data.mean(axis=1)
                                created_features.append(f"agg_mean_{len(aggregation_columns)}cols")
                            
                            elif method == "Sum":
                                df[f"agg_sum_{len(aggregation_columns)}cols"] = agg_data.sum(axis=1)
                                created_features.append(f"agg_sum_{len(aggregation_columns)}cols")
                            
                            elif method == "Min":
                                df[f"agg_min_{len(aggregation_columns)}cols"] = agg_data.min(axis=1)
                                created_features.append(f"agg_min_{len(aggregation_columns)}cols")
                            
                            elif method == "Max":
                                df[f"agg_max_{len(aggregation_columns)}cols"] = agg_data.max(axis=1)
                                created_features.append(f"agg_max_{len(aggregation_columns)}cols")
                            
                            elif method == "Standard Deviation":
                                df[f"agg_std_{len(aggregation_columns)}cols"] = agg_data.std(axis=1)
                                created_features.append(f"agg_std_{len(aggregation_columns)}cols")
                            
                            elif method == "Range":
                                df[f"agg_range_{len(aggregation_columns)}cols"] = agg_data.max(axis=1) - agg_data.min(axis=1)
                                created_features.append(f"agg_range_{len(aggregation_columns)}cols")
                        
                        st.session_state.current_dataset = df
                        
                        st.success(f"✅ Created {len(created_features)} aggregate features!")
                        
                        for feature in created_features:
                            st.write(f"• {feature}")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Feature aggregation failed: {str(e)}")
            else:
                st.info("No numeric columns available for aggregation.")
        
        with col2:
            st.markdown("#### Aggregation Preview")
            
            if aggregation_columns:
                st.markdown("**Selected Columns Statistics:**")
                
                sample_data = df[aggregation_columns].head(5)
                
                for method in ["Mean", "Sum", "Min", "Max", "Std"]:
                    if method == "Mean":
                        values = sample_data.mean(axis=1)
                    elif method == "Sum":
                        values = sample_data.sum(axis=1)
                    elif method == "Min":
                        values = sample_data.min(axis=1)
                    elif method == "Max":
                        values = sample_data.max(axis=1)
                    elif method == "Std":
                        values = sample_data.std(axis=1)
                    
                    st.write(f"**{method}:** {values.iloc[0]:.2f}, {values.iloc[1]:.2f}, {values.iloc[2]:.2f}...")

with fe_tabs[3]:
    st.markdown("### 🤖 Automated Feature Engineering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🚀 Smart Auto Feature Engineering")
        
        st.markdown("""
        This will automatically create a comprehensive set of engineered features based on your data characteristics:
        """)
        
        auto_features = st.multiselect(
            "Select automated feature types:",
            [
                "Polynomial features (degree 2)",
                "Top interaction features",
                "Mathematical transformations",
                "Binning for skewed columns",
                "Aggregate statistics"
            ],
            default=["Top interaction features", "Mathematical transformations"],
            help="Choose which types of features to automatically generate"
        )
        
        max_new_features = st.slider("Maximum new features to create:", 10, 100, 50, 
                                    help="Limit the total number of new features")
        
        if st.button("🤖 Run Auto Feature Engineering", type="primary") and auto_features:
            try:
                initial_feature_count = len(df.columns)
                created_features = []
                
                # 1. Polynomial features
                if "Polynomial features (degree 2)" in auto_features and len(numeric_cols) >= 2:
                    # Select top correlated features with target if available
                    poly_cols = numeric_cols[:5]  # Limit to top 5 to avoid explosion
                    
                    poly_data = df[poly_cols].fillna(0)
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    poly_features = poly.fit_transform(poly_data)
                    
                    feature_names = poly.get_feature_names_out(poly_cols)
                    
                    # Add only interaction terms (skip original features)
                    for i, name in enumerate(feature_names[len(poly_cols):], start=len(poly_cols)):
                        if len(created_features) < max_new_features:
                            df[f"auto_poly_{name}"] = poly_features[:, i]
                            created_features.append(f"auto_poly_{name}")
                
                # 2. Top interaction features
                if "Top interaction features" in auto_features and len(numeric_cols) >= 2:
                    # Create top interactions based on correlation with potential targets
                    interaction_count = 0
                    for i, col1 in enumerate(numeric_cols[:8]):  # Limit to top 8 columns
                        for col2 in numeric_cols[i+1:8]:
                            if len(created_features) < max_new_features and interaction_count < 10:
                                feature_name = f"auto_int_{col1}_x_{col2}"
                                df[feature_name] = df[col1] * df[col2]
                                created_features.append(feature_name)
                                interaction_count += 1
                
                # 3. Mathematical transformations
                if "Mathematical transformations" in auto_features:
                    for col in numeric_cols[:5]:  # Limit to top 5 columns
                        if len(created_features) < max_new_features:
                            col_data = df[col]
                            
                            # Square root for positive skewed data
                            if col_data.skew() > 1:
                                df[f"auto_sqrt_{col}"] = np.sqrt(np.abs(col_data))
                                created_features.append(f"auto_sqrt_{col}")
                            
                            # Log for highly skewed data
                            if col_data.skew() > 2 and len(created_features) < max_new_features:
                                df[f"auto_log_{col}"] = np.log1p(np.abs(col_data))
                                created_features.append(f"auto_log_{col}")
                            
                            # Square for symmetric data
                            if abs(col_data.skew()) < 0.5 and len(created_features) < max_new_features:
                                df[f"auto_sq_{col}"] = col_data ** 2
                                created_features.append(f"auto_sq_{col}")
                
                # 4. Binning for skewed columns
                if "Binning for skewed columns" in auto_features:
                    for col in numeric_cols:
                        if len(created_features) < max_new_features and abs(df[col].skew()) > 1:
                            try:
                                binned_data, bin_edges = pd.qcut(df[col], q=5, retbins=True, duplicates='drop')
                                df[f"auto_bin_{col}"] = binned_data.cat.codes
                                created_features.append(f"auto_bin_{col}")
                            except:
                                continue
                
                # 5. Aggregate statistics
                if "Aggregate statistics" in auto_features and len(numeric_cols) >= 3:
                    # Create rolling statistics for groups of columns
                    col_groups = [numeric_cols[i:i+3] for i in range(0, len(numeric_cols), 3)]
                    
                    for group in col_groups[:3]:  # Limit to first 3 groups
                        if len(created_features) < max_new_features:
                            group_data = df[group]
                            
                            df[f"auto_agg_mean_{len(group)}"] = group_data.mean(axis=1)
                            created_features.append(f"auto_agg_mean_{len(group)}")
                            
                            if len(created_features) < max_new_features:
                                df[f"auto_agg_std_{len(group)}"] = group_data.std(axis=1)
                                created_features.append(f"auto_agg_std_{len(group)}")
                
                # Update session state
                st.session_state.current_dataset = df
                
                # Log action
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                st.session_state.processing_log.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': 'Automated Feature Engineering',
                    'details': f"Automatically created {len(created_features)} engineered features using {len(auto_features)} feature types"
                })
                
                st.success(f"🎉 Automated feature engineering complete!")
                st.write(f"✅ Created {len(created_features)} new features")
                st.write(f"📊 Dataset expanded from {initial_feature_count} to {len(df.columns)} columns")
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Automated feature engineering failed: {str(e)}")
    
    with col2:
        st.markdown("#### 📊 Engineering Summary")
        
        # Show all engineered features
        all_engineered = [col for col in df.columns if any(prefix in col for prefix in ['poly_', 'auto_', '_interaction_', '_scaled', '_binned', 'pca_', 'agg_'])]
        
        if all_engineered:
            st.markdown(f"#### All Engineered Features ({len(all_engineered)})")
            
            # Group by type
            feature_types = {
                'Polynomial': [col for col in all_engineered if 'poly_' in col],
                'Interactions': [col for col in all_engineered if any(x in col for x in ['_x_', 'interaction', 'auto_int'])],
                'Mathematical': [col for col in all_engineered if any(x in col for x in ['_sqrt', '_log', '_sq', 'auto_sqrt', 'auto_log', 'auto_sq'])],
                'Scaled': [col for col in all_engineered if '_scaled' in col],
                'Binned': [col for col in all_engineered if any(x in col for x in ['_binned', 'auto_bin'])],
                'PCA': [col for col in all_engineered if 'pca_' in col],
                'Aggregated': [col for col in all_engineered if 'agg_' in col or 'auto_agg' in col]
            }
            
            for feature_type, features in feature_types.items():
                if features:
                    st.markdown(f"**{feature_type} Features ({len(features)}):**")
                    for feat in features[:5]:  # Show first 5
                        st.write(f"• {feat}")
                    if len(features) > 5:
                        st.write(f"... and {len(features) - 5} more")
                    st.markdown("---")
        
        else:
            st.info("No engineered features found. Create some features first!")
        
        # Dataset summary
        st.markdown("#### Dataset Summary")
        st.write(f"**Total Features:** {len(df.columns):,}")
        st.write(f"**Original Features:** {len(numeric_cols) + len(categorical_cols):,}")
        st.write(f"**Engineered Features:** {len(all_engineered):,}")
        st.write(f"**Engineering Ratio:** {len(all_engineered) / len(df.columns) * 100:.1f}%")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Enhanced Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"feature_engineered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to History & Export"):
        st.switch_page("pages/14_History_Export.py")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Feature Engineering Guide")
    
    st.markdown("#### Key Techniques:")
    techniques = [
        "**Polynomial:** x², x³, interactions",
        "**Mathematical:** √x, log(x), 1/x",
        "**Binning:** Categorical from numeric",
        "**Interactions:** x₁ × x₂, ratios", 
        "**Aggregation:** Mean, sum, std across features",
        "**PCA:** Dimensionality reduction"
    ]
    
    for technique in techniques:
        st.markdown(f"• {technique}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Start with domain knowledge
    • Create features based on data distribution
    • Use feature selection to reduce noise
    • Validate feature importance
    • Monitor for overfitting with too many features
    """)
    
    if st.session_state.get('processing_log'):
        st.markdown("#### 📝 Recent Engineering")
        fe_actions = [log for log in st.session_state.processing_log if 'Feature' in log.get('action', '')][-3:]
        
        for action in fe_actions:
            st.caption(f"✅ {action.get('details', 'Feature engineering applied')}")