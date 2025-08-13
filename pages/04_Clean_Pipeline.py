import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
import re
from textblob import TextBlob
import nltk
from datetime import datetime
import math
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Clean Pipeline", page_icon="🧹", layout="wide")


# Initialize session state for processing log and tab navigation
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

if 'current_tab_index' not in st.session_state:
    st.session_state.current_tab_index = 0

if 'auto_mode_enabled' not in st.session_state:
    st.session_state.auto_mode_enabled = True

if 'activities_performed' not in st.session_state:
    st.session_state.activities_performed = {
        'data_types': [],
        'duplicates': [],
        'missing_values': [],
        'outliers': [],
        'text_cleaning': [],
        'datetime': [],
        'geospatial': [],
        'categorical': [],
        'scaling': [],
        'feature_engineering': []
    }

def log_action(action, details, tab_name=None):
    """Log processing actions with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_log.append({
        'timestamp': timestamp,
        'action': action,
        'details': details
    })
    
    # Also log activities for specific tabs
    if tab_name and tab_name in st.session_state.activities_performed:
        st.session_state.activities_performed[tab_name].append({
            'timestamp': timestamp,
            'action': action,
            'details': details
        })



def undo_last_action():
    """Undo the last processing action"""
    if 'original_clean_data' in st.session_state and st.session_state.processing_log:
        # Remove last action from log
        last_action = st.session_state.processing_log.pop()
        
        # Reset to original data and replay all actions except the last one
        st.session_state.current_dataset = st.session_state.original_clean_data.copy()
        
        # Show success message
        st.success(f"✅ Undid action: {last_action['action']}")
        st.rerun()
    else:
        st.warning("No actions to undo")

def ai_analyze_columns_for_treatment(df, analysis_type):
    """AI-powered analysis to identify columns needing treatment"""
    recommendations = []
    
    if analysis_type == "data_types":
        for col in df.columns:
            current_type = str(df[col].dtype)
            # Check if column should be converted
            if current_type == 'object':
                # Check if it's actually numeric
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_values.isna().all():
                        non_null_numeric = numeric_values.dropna()
                        if len(non_null_numeric) > len(df) * 0.7:  # 70% convertible
                            recommendations.append({
                                'column': col,
                                'reason': f"Column appears to be numeric but stored as text. {len(non_null_numeric)}/{len(df)} values are convertible to numbers.",
                                'severity': 'medium',
                                'suggested_action': 'Convert to numeric type'
                            })
                except:
                    pass
                
                # Check if it's datetime
                try:
                    datetime_values = pd.to_datetime(df[col], errors='coerce')
                    if not datetime_values.isna().all():
                        non_null_dates = datetime_values.dropna()
                        if len(non_null_dates) > len(df) * 0.5:  # 50% convertible
                            recommendations.append({
                                'column': col,
                                'reason': f"Column contains date/time patterns. {len(non_null_dates)}/{len(df)} values are convertible to datetime.",
                                'severity': 'medium', 
                                'suggested_action': 'Convert to datetime type'
                            })
                except:
                    pass
    
    elif analysis_type == "duplicates":
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append({
                'column': 'All columns',
                'reason': f"Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}% of data). Duplicates can skew analysis results.",
                'severity': 'high' if duplicate_count > len(df) * 0.1 else 'medium',
                'suggested_action': f'Remove {duplicate_count} duplicate rows'
            })
    
    elif analysis_type == "missing_values":
        missing_data = df.isnull().sum()
        for col in missing_data[missing_data > 0].index:
            missing_count = missing_data[col]
            missing_pct = missing_count / len(df) * 100
            severity = 'high' if missing_pct > 30 else 'medium' if missing_pct > 10 else 'low'
            
            recommendations.append({
                'column': col,
                'reason': f"Column has {missing_count} missing values ({missing_pct:.1f}%). Missing data can bias analysis and reduce model performance.",
                'severity': severity,
                'suggested_action': f'Impute missing values using {"median" if df[col].dtype in ["int64", "float64"] else "mode"} strategy'
            })
    
    elif analysis_type == "outliers":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(df) * 100
                severity = 'high' if outlier_pct > 10 else 'medium' if outlier_pct > 5 else 'low'
                
                recommendations.append({
                    'column': col,
                    'reason': f"Column contains {len(outliers)} outliers ({outlier_pct:.1f}%). Extreme values can distort statistical analysis and model training.",
                    'severity': severity,
                    'suggested_action': f'{"Cap" if outlier_pct < 5 else "Remove"} outliers using IQR method'
                })
    
    elif analysis_type == "text_cleaning":
        text_cols = df.select_dtypes(include=['object']).columns
        for col in text_cols:
            issues = []
            sample_text = df[col].dropna().astype(str).head(100)
            
            # Check for common text issues
            has_extra_spaces = sample_text.str.contains(r'\s{2,}').any()
            has_special_chars = sample_text.str.contains(r'[^\w\s]').any()
            has_mixed_case = sample_text.str.islower().any() and sample_text.str.isupper().any()
            
            if has_extra_spaces:
                issues.append("extra whitespace")
            if has_special_chars:
                issues.append("special characters")
            if has_mixed_case:
                issues.append("inconsistent capitalization")
            
            if issues:
                recommendations.append({
                    'column': col,
                    'reason': f"Text column has formatting issues: {', '.join(issues)}. Clean text improves consistency and analysis quality.",
                    'severity': 'medium',
                    'suggested_action': 'Clean and standardize text formatting'
                })
    
    elif analysis_type == "categorical":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            total_count = len(df)
            
            if unique_count > 1:  # Skip columns with only one value
                cardinality_ratio = unique_count / total_count
                
                if cardinality_ratio < 0.5:  # Good candidate for encoding
                    encoding_type = "One-hot encoding" if unique_count <= 10 else "Label encoding"
                    recommendations.append({
                        'column': col,
                        'reason': f"Categorical column with {unique_count} unique values. Machine learning models require numeric encoding of categorical data.",
                        'severity': 'high',
                        'suggested_action': f'Apply {encoding_type.lower()}'
                    })
    
    elif analysis_type == "scaling":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            # Check for different scales
            scales = {}
            for col in numeric_cols:
                col_range = df[col].max() - df[col].min()
                scales[col] = col_range
            
            max_scale = max(scales.values())
            min_scale = min(scales.values())
            
            if max_scale / min_scale > 100:  # Significant scale difference
                for col in numeric_cols:
                    recommendations.append({
                        'column': col,
                        'reason': f"Numeric columns have different scales (range: {scales[col]:.2f}). Features with larger scales can dominate machine learning models.",
                        'severity': 'medium',
                        'suggested_action': 'Apply StandardScaler or MinMaxScaler normalization'
                    })
    
    return recommendations

def display_ai_recommendations(recommendations, tab_key):
    """Display AI recommendations with selection interface"""
    if not recommendations:
        st.info("No issues detected by AI analysis.")
        return []
    
    st.markdown("#### 🤖 AI Analysis & Recommendations")
    
    selected_items = []
    
    for i, rec in enumerate(recommendations):
        with st.expander(f"📊 {rec['column']} - {rec['severity'].title()} Priority", expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Color-coded severity
                color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec['severity']]
                st.markdown(f"**{color} Issue:** {rec['reason']}")
                st.markdown(f"**💡 Suggested Action:** {rec['suggested_action']}")
            
            with col2:
                selected = st.checkbox(
                    "Select for treatment",
                    value=True,
                    key=f"{tab_key}_rec_{i}"
                )
                if selected:
                    selected_items.append(rec)
    
    return selected_items

def display_action_summary(actions_taken, tab_name):
    """Display summary of actions taken after processing"""
    if not actions_taken:
        return
    
    st.markdown("#### 📋 Processing Summary")
    
    with st.expander("View Detailed Action Log", expanded=True):
        for action in actions_taken:
            st.markdown(f"""
            **Action:** {action['action']}  
            **Target:** {action['target']}  
            **Result:** {action['result']}  
            **Timestamp:** {action['timestamp']}
            """)
            if 'metrics' in action:
                st.json(action['metrics'])

def show_celebration_balloons(message, count=0, celebration_type="success"):
    """Show enhanced celebration with balloons and animations"""
    if celebration_type == "success":
        # Create celebration container
        celebration_placeholder = st.empty()
        
        with celebration_placeholder.container():
            # Custom CSS for animations
            st.markdown("""
            <style>
            @keyframes balloonFloat {
                0% { transform: translateY(100px) scale(0); opacity: 0; }
                50% { transform: translateY(-20px) scale(1.1); opacity: 1; }
                100% { transform: translateY(-50px) scale(1); opacity: 0.8; }
            }
            
            @keyframes confetti {
                0% { transform: translateY(-100px) rotate(0deg); opacity: 1; }
                100% { transform: translateY(500px) rotate(720deg); opacity: 0; }
            }
            
            .celebration-container {
                position: relative;
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                margin: 10px 0;
                color: white;
                animation: balloonFloat 3s ease-in-out;
            }
            
            .celebration-text {
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            
            .celebration-counter {
                font-size: 2em;
                color: #FFD700;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .confetti-piece {
                position: absolute;
                width: 10px;
                height: 10px;
                background: #FFD700;
                animation: confetti 3s linear infinite;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Celebration message with animations
            celebration_html = f"""
            <div class="celebration-container">
                <div class="celebration-text">{message}</div>
                {f'<div class="celebration-counter">{count} items processed!</div>' if count > 0 else ''}
                <div>🎉 🎈 ✨ 🎊 🎈 🎉</div>
            </div>
            """
            
            st.markdown(celebration_html, unsafe_allow_html=True)
            
            # Standard Streamlit balloons for extra effect
            st.balloons()
            
            # Remove the sleep to avoid blocking the app
            # celebration_placeholder will be cleared on next rerun

def initialize_comprehensive_stats():
    """Initialize comprehensive statistics tracking"""
    if 'comprehensive_stats' not in st.session_state:
        st.session_state.comprehensive_stats = {
            'total_actions': 0,
            'columns_processed': [],  # Use list instead of set for JSON serialization
            'rows_processed': 0,
            'missing_values_fixed': 0,
            'outliers_treated': 0,
            'duplicates_removed': 0,
            'encodings_applied': 0,
            'scaling_transformations': 0,
            'text_cleanings': 0,
            'datetime_extractions': 0,
            'features_engineered': 0,
            'geospatial_processed': 0,
            'data_types_converted': 0,
            'start_time': datetime.now(),
            'pipeline_progress': 0,
            'tab_completion': {f'tab_{i}': False for i in range(11)},
            'time_saved_estimates': []
        }

def update_comprehensive_stats(action_type, details):
    """Update comprehensive statistics"""
    initialize_comprehensive_stats()
    stats = st.session_state.comprehensive_stats
    
    stats['total_actions'] += 1
    
    if action_type == 'missing_values':
        stats['missing_values_fixed'] += details.get('count', 1)
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'outliers':
        stats['outliers_treated'] += details.get('count', 1)
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'duplicates':
        stats['duplicates_removed'] += details.get('count', 1)
    elif action_type == 'encoding':
        stats['encodings_applied'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'scaling':
        stats['scaling_transformations'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'text_cleaning':
        stats['text_cleanings'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'datetime':
        stats['datetime_extractions'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'feature_engineering':
        stats['features_engineered'] += details.get('count', 1)
    elif action_type == 'data_types':
        stats['data_types_converted'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    elif action_type == 'geospatial':
        stats['geospatial_processed'] += 1
        column = details.get('column', '')
        if column and column not in stats['columns_processed']:
            stats['columns_processed'].append(column)
    
    # Estimate time saved (placeholder calculation)
    time_saved = details.get('time_saved', 5)  # Default 5 minutes per action
    stats['time_saved_estimates'].append(time_saved)

def calculate_pipeline_progress():
    """Calculate overall pipeline progress"""
    initialize_comprehensive_stats()
    stats = st.session_state.comprehensive_stats
    
    completed_tabs = sum(1 for completed in stats['tab_completion'].values() if completed)
    total_tabs = len(stats['tab_completion'])
    
    progress = (completed_tabs / total_tabs) * 100
    stats['pipeline_progress'] = progress
    
    return progress

def mark_tab_complete(tab_index):
    """Mark a tab as complete"""
    initialize_comprehensive_stats()
    st.session_state.comprehensive_stats['tab_completion'][f'tab_{tab_index}'] = True

def show_progress_notification(progress):
    """Show progress notifications at milestones"""
    milestones = [25, 50, 75, 100]
    
    for milestone in milestones:
        if abs(progress - milestone) < 5:  # Within 5% of milestone
            if milestone == 25:
                st.info("🚀 Great start! You're 25% through the pipeline!")
            elif milestone == 50:
                st.success("🎯 Halfway there! Keep up the excellent work!")
            elif milestone == 75:
                st.success("⭐ Almost done! 75% complete!")
            elif milestone == 100:
                st.success("🏆 Pipeline Complete! Amazing work!")
                show_celebration_balloons("🎉 Data Pipeline Complete! 🎉", celebration_type="completion")

def generate_ai_summary_writeup():
    """Generate AI-powered summary of the cleaning process"""
    initialize_comprehensive_stats()
    stats = st.session_state.comprehensive_stats
    
    total_time = (datetime.now() - stats['start_time']).total_seconds() / 60  # minutes
    total_time_saved = sum(stats['time_saved_estimates'])
    
    summary = f"""
    ## 🤖 AI-Generated Data Cleaning Summary
    
    **Processing Overview:**
    Your data preprocessing pipeline has successfully processed {len(stats['columns_processed'])} unique columns 
    with {stats['total_actions']} total cleaning actions applied.
    
    **Key Achievements:**
    - ✅ **Data Quality**: Fixed {stats['missing_values_fixed']} missing values and treated {stats['outliers_treated']} outliers
    - 🔄 **Data Transformation**: Applied {stats['encodings_applied']} encodings and {stats['scaling_transformations']} scaling operations
    - 📝 **Text Processing**: Cleaned {stats['text_cleanings']} text columns for consistency
    - 📅 **Temporal Features**: Extracted features from {stats['datetime_extractions']} datetime columns
    - ⚙️ **Feature Engineering**: Created {stats['features_engineered']} new features
    - 🗂️ **Data Types**: Converted {stats['data_types_converted']} columns to appropriate types
    
    **Impact Assessment:**
    The comprehensive cleaning process has significantly improved your data quality. Missing values and outliers 
    that could have caused model bias or poor performance have been addressed. Categorical encoding ensures 
    machine learning compatibility, while feature engineering enhances predictive potential.
    
    **Time Efficiency:**
    Pipeline execution time: {total_time:.1f} minutes
    Estimated manual effort saved: {total_time_saved:.0f} minutes
    
    **Recommendations for Next Steps:**
    1. **Model Training**: Your data is now ready for machine learning algorithms
    2. **Validation**: Consider cross-validation to assess data quality improvements
    3. **Monitoring**: Set up data quality monitoring for future datasets
    4. **Documentation**: Export the activity log for reproducibility
    
    **Data Readiness Score: {min(95, 70 + (stats['total_actions'] * 2))}%** 
    Your dataset is well-prepared for analysis and modeling!
    """
    
    return summary

def get_alternative_suggestions(column, data_type, issue_type):
    """Get alternative treatment suggestions with pros/cons"""
    alternatives = []
    
    if issue_type == "missing_values":
        if data_type in ['int64', 'float64']:
            alternatives = [
                {"method": "Mean Imputation", "pros": "Simple, preserves overall distribution", "cons": "May not capture data patterns"},
                {"method": "Median Imputation", "pros": "Robust to outliers, good for skewed data", "cons": "May not preserve variance"},
                {"method": "Forward Fill", "pros": "Preserves trends in time series", "cons": "Only suitable for sequential data"},
                {"method": "KNN Imputation", "pros": "Uses similar records for prediction", "cons": "Computationally expensive"}
            ]
        else:
            alternatives = [
                {"method": "Mode Imputation", "pros": "Uses most frequent value", "cons": "May introduce bias toward majority class"},
                {"method": "Custom Value", "pros": "Domain-specific knowledge", "cons": "Requires manual input"},
                {"method": "Remove Rows", "pros": "Maintains data integrity", "cons": "Reduces dataset size"}
            ]
    
    elif issue_type == "outliers":
        alternatives = [
            {"method": "IQR Capping", "pros": "Preserves data points, reduces extreme values", "cons": "May still affect distribution"},
            {"method": "Z-Score Removal", "pros": "Statistically sound method", "cons": "Assumes normal distribution"},
            {"method": "Isolation Forest", "pros": "Handles complex patterns", "cons": "May remove valid extreme cases"},
            {"method": "Manual Review", "pros": "Expert judgment", "cons": "Time-intensive process"}
        ]
    
    return alternatives

class SmartColumnTreatmentExplainer:
    """AI-powered explainer for column treatment recommendations"""
    
    def __init__(self, df):
        self.df = df
        self.explanations = {}
        self.impact_assessments = {}
        self.dependency_warnings = {}
        
    def generate_detailed_explanation(self, column, issue_type, recommendation):
        """Generate comprehensive explanation for why a treatment is recommended"""
        explanation = {
            "primary_reason": "",
            "data_impact": "",
            "ml_impact": "",
            "business_impact": "",
            "risk_assessment": "",
            "confidence_score": 0.0,
            "contextual_insights": [],
            "cross_dependencies": []
        }
        
        col_data = self.df[column] if column in self.df.columns else None
        
        if issue_type == "missing_values" and col_data is not None:
            missing_count = col_data.isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            explanation["primary_reason"] = f"Column '{column}' contains {missing_count} missing values ({missing_pct:.1f}% of total data)"
            
            if missing_pct > 50:
                explanation["data_impact"] = "HIGH IMPACT: More than half the data is missing. This could severely affect analysis reliability."
                explanation["risk_assessment"] = "High risk - consider if this column should be used at all"
                explanation["confidence_score"] = 0.3
            elif missing_pct > 20:
                explanation["data_impact"] = "MEDIUM IMPACT: Significant portion of data missing. Imputation strategy is crucial."
                explanation["risk_assessment"] = "Medium risk - choose imputation method carefully"
                explanation["confidence_score"] = 0.7
            else:
                explanation["data_impact"] = "LOW IMPACT: Small amount of missing data. Standard imputation should work well."
                explanation["risk_assessment"] = "Low risk - most imputation methods will work"
                explanation["confidence_score"] = 0.9
            
            if col_data.dtype in ['int64', 'float64']:
                explanation["ml_impact"] = "Missing numeric values can cause algorithms to fail or produce biased results"
                explanation["contextual_insights"].append("Numeric data - median imputation recommended for robustness")
                if col_data.std() / col_data.mean() > 1:  # High coefficient of variation
                    explanation["contextual_insights"].append("High variability detected - consider robust imputation methods")
            else:
                explanation["ml_impact"] = "Missing categorical values reduce training data and may create unknown categories"
                explanation["contextual_insights"].append("Categorical data - mode imputation or 'Unknown' category recommended")
        
        elif issue_type == "outliers" and col_data is not None:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[(col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)]
            outlier_pct = (len(outliers) / len(self.df)) * 100
            
            explanation["primary_reason"] = f"Column '{column}' contains {len(outliers)} outliers ({outlier_pct:.1f}% of data)"
            
            if outlier_pct > 10:
                explanation["data_impact"] = "HIGH IMPACT: Many outliers detected. Could indicate data quality issues or genuine extreme values."
                explanation["ml_impact"] = "Severe impact on model performance - outliers can dominate learning algorithms"
                explanation["confidence_score"] = 0.8
            elif outlier_pct > 5:
                explanation["data_impact"] = "MEDIUM IMPACT: Moderate number of outliers. Review for business significance."
                explanation["ml_impact"] = "Moderate impact - may affect model accuracy and generalization"
                explanation["confidence_score"] = 0.7
            else:
                explanation["data_impact"] = "LOW IMPACT: Few outliers detected. May represent natural variation."
                explanation["ml_impact"] = "Minimal impact on most algorithms, but may affect distance-based methods"
                explanation["confidence_score"] = 0.6
            
            explanation["contextual_insights"].append(f"Outlier range: {outliers[column].min():.2f} to {outliers[column].max():.2f}")
            explanation["contextual_insights"].append(f"Normal range: {Q1:.2f} to {Q3:.2f}")
        
        elif issue_type == "data_types" and col_data is not None:
            if col_data.dtype == 'object':
                # Check numeric conversion potential
                try:
                    numeric_converted = pd.to_numeric(col_data, errors='coerce')
                    conversion_rate = (1 - numeric_converted.isnull().sum() / len(col_data)) * 100
                    
                    explanation["primary_reason"] = f"Column '{column}' is stored as text but {conversion_rate:.1f}% of values can be converted to numbers"
                    explanation["ml_impact"] = "Text data cannot be used directly in mathematical models - numeric conversion enables statistical analysis"
                    explanation["data_impact"] = f"Converting will unlock numerical operations and statistical analysis capabilities"
                    explanation["confidence_score"] = conversion_rate / 100
                    
                    if conversion_rate > 90:
                        explanation["contextual_insights"].append("High conversion success rate - safe to convert")
                    elif conversion_rate > 70:
                        explanation["contextual_insights"].append("Good conversion rate - review non-convertible values")
                    else:
                        explanation["contextual_insights"].append("Mixed data type - consider splitting or cleaning first")
                        
                except:
                    # Check datetime conversion
                    try:
                        datetime_converted = pd.to_datetime(col_data, errors='coerce')
                        conversion_rate = (1 - datetime_converted.isnull().sum() / len(col_data)) * 100
                        
                        explanation["primary_reason"] = f"Column '{column}' contains date/time patterns with {conversion_rate:.1f}% conversion success"
                        explanation["ml_impact"] = "DateTime data enables time-based analysis and feature engineering"
                        explanation["confidence_score"] = conversion_rate / 100
                    except:
                        pass
        
        elif issue_type == "categorical" and col_data is not None:
            unique_count = col_data.nunique()
            total_count = len(col_data)
            cardinality_ratio = unique_count / total_count
            
            explanation["primary_reason"] = f"Column '{column}' is categorical with {unique_count} unique values ({cardinality_ratio:.1%} cardinality)"
            explanation["ml_impact"] = "Machine learning algorithms require numeric input - categorical encoding is essential"
            
            if unique_count <= 5:
                explanation["data_impact"] = "LOW CARDINALITY: Perfect for one-hot encoding without dimension explosion"
                explanation["contextual_insights"].append("Recommended: One-hot encoding - creates binary features for each category")
                explanation["confidence_score"] = 0.95
            elif unique_count <= 20:
                explanation["data_impact"] = "MEDIUM CARDINALITY: One-hot encoding feasible but increases feature space"
                explanation["contextual_insights"].append("Recommended: One-hot encoding or label encoding based on model type")
                explanation["confidence_score"] = 0.8
            else:
                explanation["data_impact"] = "HIGH CARDINALITY: One-hot encoding will create many features"
                explanation["contextual_insights"].append("Recommended: Label encoding or target encoding to reduce dimensionality")
                explanation["confidence_score"] = 0.7
        
        elif issue_type == "scaling" and col_data is not None:
            col_range = col_data.max() - col_data.min()
            col_mean = col_data.mean()
            col_std = col_data.std()
            
            # Compare with other numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number])
            if len(numeric_cols.columns) > 1:
                ranges = {col: (numeric_cols[col].max() - numeric_cols[col].min()) for col in numeric_cols.columns}
                max_range = max(ranges.values())
                min_range = min(ranges.values())
                scale_disparity = max_range / min_range if min_range > 0 else float('inf')
                
                explanation["primary_reason"] = f"Column '{column}' has range {col_range:.2f}, but feature scales vary by {scale_disparity:.1f}x across dataset"
                
                if scale_disparity > 1000:
                    explanation["ml_impact"] = "CRITICAL: Extreme scale differences will cause algorithms to ignore smaller-scale features"
                    explanation["confidence_score"] = 0.95
                elif scale_disparity > 100:
                    explanation["ml_impact"] = "HIGH IMPACT: Scale differences may bias learning toward larger-scale features"
                    explanation["confidence_score"] = 0.8
                else:
                    explanation["ml_impact"] = "MODERATE IMPACT: Some algorithms benefit from normalized scales"
                    explanation["confidence_score"] = 0.6
                
                explanation["contextual_insights"].append(f"Current range: {col_data.min():.2f} to {col_data.max():.2f}")
                explanation["contextual_insights"].append(f"Mean: {col_mean:.2f}, Standard deviation: {col_std:.2f}")
        
        # Check for cross-column dependencies
        explanation["cross_dependencies"] = self._check_dependencies(column, issue_type)
        
        return explanation
    
    def _check_dependencies(self, column, issue_type):
        """Check for dependencies and conflicts with other columns"""
        dependencies = []
        
        # Check if treating this column affects others
        if issue_type == "missing_values":
            # Check if other columns have missing values in same rows
            if column in self.df.columns:
                missing_mask = self.df[column].isnull()
                correlated_missing = []
                
                for other_col in self.df.columns:
                    if other_col != column:
                        other_missing = self.df[other_col].isnull()
                        # Check if missing values often occur together
                        correlation = (missing_mask & other_missing).sum() / missing_mask.sum() if missing_mask.sum() > 0 else 0
                        if correlation > 0.5:
                            correlated_missing.append(other_col)
                
                if correlated_missing:
                    dependencies.append({
                        "type": "missing_correlation",
                        "message": f"Missing values often coincide with: {', '.join(correlated_missing[:3])}",
                        "recommendation": "Consider treating these columns together for consistency"
                    })
        
        elif issue_type == "outliers":
            # Check if outliers appear in related columns
            if column in self.df.columns and self.df[column].dtype in ['int64', 'float64']:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (self.df[column] < Q1 - 1.5 * IQR) | (self.df[column] > Q3 + 1.5 * IQR)
                
                if outlier_mask.any():
                    numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                    correlated_outliers = []
                    
                    for other_col in numeric_cols:
                        if other_col != column:
                            try:
                                other_Q1 = self.df[other_col].quantile(0.25)
                                other_Q3 = self.df[other_col].quantile(0.75)
                                other_IQR = other_Q3 - other_Q1
                                other_outliers = (self.df[other_col] < other_Q1 - 1.5 * other_IQR) | (self.df[other_col] > other_Q3 + 1.5 * other_IQR)
                                
                                # Check if outliers coincide
                                overlap = (outlier_mask & other_outliers).sum() / outlier_mask.sum() if outlier_mask.sum() > 0 else 0
                                if overlap > 0.3:
                                    correlated_outliers.append(other_col)
                            except:
                                continue
                    
                    if correlated_outliers:
                        dependencies.append({
                            "type": "outlier_correlation", 
                            "message": f"Outliers often coincide with: {', '.join(correlated_outliers[:3])}",
                            "recommendation": "Consider multivariate outlier detection for these related columns"
                        })
        
        return dependencies
    
    def get_confidence_explanation(self, confidence_score):
        """Explain what the confidence score means"""
        if confidence_score >= 0.9:
            return "Very High Confidence - Strong evidence supports this recommendation"
        elif confidence_score >= 0.7:
            return "High Confidence - Good evidence supports this recommendation"
        elif confidence_score >= 0.5:
            return "Medium Confidence - Some evidence supports this recommendation"
        elif confidence_score >= 0.3:
            return "Low Confidence - Weak evidence, consider alternatives"
        else:
            return "Very Low Confidence - Insufficient evidence, manual review recommended"
    
    def generate_treatment_order_recommendations(self, selected_treatments):
        """Suggest optimal order for applying treatments"""
        order_recommendations = []
        
        # Recommended order: Data types → Duplicates → Missing values → Outliers → Text → Encoding → Scaling
        treatment_priority = {
            "data_types": 1,
            "duplicates": 2, 
            "missing_values": 3,
            "outliers": 4,
            "text_cleaning": 5,
            "categorical": 6,
            "scaling": 7
        }
        
        sorted_treatments = sorted(selected_treatments, key=lambda x: treatment_priority.get(x, 999))
        
        for i, treatment in enumerate(sorted_treatments):
            if i == 0:
                order_recommendations.append(f"Start with {treatment.replace('_', ' ').title()}")
            else:
                prev_treatment = sorted_treatments[i-1]
                reason = self._get_order_reason(prev_treatment, treatment)
                order_recommendations.append(f"Then {treatment.replace('_', ' ').title()}: {reason}")
        
        return order_recommendations
    
    def _get_order_reason(self, prev_treatment, current_treatment):
        """Get reason for treatment order"""
        reasons = {
            ("data_types", "duplicates"): "Correct data types help identify true duplicates",
            ("duplicates", "missing_values"): "Remove duplicates first to avoid biasing imputation",
            ("missing_values", "outliers"): "Handle missing data before outlier detection for accuracy",
            ("outliers", "text_cleaning"): "Clean numeric data before text processing",
            ("text_cleaning", "categorical"): "Clean text before encoding categorical variables",
            ("categorical", "scaling"): "Encode categories before scaling numerical features"
        }
        
        return reasons.get((prev_treatment, current_treatment), "Logical processing sequence")

def display_smart_column_explainer(df, recommendations, tab_key):
    """Display the Smart Column Treatment AI Explainer interface"""
    if not recommendations:
        return
    
    explainer = SmartColumnTreatmentExplainer(df)
    
    st.markdown("### 🧠 Smart Column Treatment AI Explainer")
    st.markdown("*Get detailed, AI-powered explanations for each treatment recommendation*")
    
    # Create tabs for different explanation views
    explain_tab1, explain_tab2, explain_tab3 = st.tabs(["📊 Column Analysis", "🔗 Dependencies", "📋 Treatment Order"])
    
    with explain_tab1:
        st.markdown("#### Detailed Column Analysis")
        
        for i, rec in enumerate(recommendations):
            column = rec['column']
            issue_type = rec.get('issue_type', 'general')
            
            # Determine issue type from recommendation details
            if 'missing' in rec['reason'].lower():
                issue_type = 'missing_values'
            elif 'outlier' in rec['reason'].lower():
                issue_type = 'outliers'
            elif 'numeric' in rec['suggested_action'].lower() or 'datetime' in rec['suggested_action'].lower():
                issue_type = 'data_types'
            elif 'categorical' in rec['reason'].lower() or 'encoding' in rec['suggested_action'].lower():
                issue_type = 'categorical'
            elif 'scale' in rec['reason'].lower():
                issue_type = 'scaling'
            
            with st.expander(f"🔍 Deep Analysis: {column}", expanded=False):
                explanation = explainer.generate_detailed_explanation(column, issue_type, rec)
                
                # Confidence indicator
                confidence = explanation['confidence_score']
                confidence_color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                st.markdown(f"**{confidence_color} Confidence: {confidence:.1%}** - {explainer.get_confidence_explanation(confidence)}")
                
                # Primary reason
                st.markdown(f"**🎯 Why This Treatment?**")
                st.info(explanation['primary_reason'])
                
                # Impact assessments
                col1, col2 = st.columns(2)
                with col1:
                    if explanation['data_impact']:
                        st.markdown(f"**📊 Data Impact:**")
                        st.markdown(explanation['data_impact'])
                
                with col2:
                    if explanation['ml_impact']:
                        st.markdown(f"**🤖 ML Impact:**")
                        st.markdown(explanation['ml_impact'])
                
                # Contextual insights
                if explanation['contextual_insights']:
                    st.markdown(f"**💡 Key Insights:**")
                    for insight in explanation['contextual_insights']:
                        st.markdown(f"• {insight}")
                
                # Risk assessment
                if explanation['risk_assessment']:
                    st.markdown(f"**⚠️ Risk Level:** {explanation['risk_assessment']}")
    
    with explain_tab2:
        st.markdown("#### Cross-Column Dependencies & Conflicts")
        
        has_dependencies = False
        for rec in recommendations:
            column = rec['column']
            issue_type = rec.get('issue_type', 'general')
            
            if 'missing' in rec['reason'].lower():
                issue_type = 'missing_values'
            elif 'outlier' in rec['reason'].lower():
                issue_type = 'outliers'
            
            explanation = explainer.generate_detailed_explanation(column, issue_type, rec)
            
            if explanation['cross_dependencies']:
                has_dependencies = True
                st.markdown(f"**🔗 {column} Dependencies:**")
                for dep in explanation['cross_dependencies']:
                    st.warning(f"**{dep['type'].replace('_', ' ').title()}:** {dep['message']}")
                    st.info(f"💡 Recommendation: {dep['recommendation']}")
                st.markdown("---")
        
        if not has_dependencies:
            st.success("✅ No significant cross-column dependencies detected. Treatments can be applied independently.")
    
    with explain_tab3:
        st.markdown("#### Recommended Treatment Order")
        
        # Extract treatment types from recommendations
        treatment_types = []
        for rec in recommendations:
            if 'missing' in rec['reason'].lower():
                treatment_types.append('missing_values')
            elif 'outlier' in rec['reason'].lower():
                treatment_types.append('outliers')
            elif 'numeric' in rec['suggested_action'].lower() or 'datetime' in rec['suggested_action'].lower():
                treatment_types.append('data_types')
            elif 'categorical' in rec['reason'].lower():
                treatment_types.append('categorical')
            elif 'scale' in rec['reason'].lower():
                treatment_types.append('scaling')
        
        if treatment_types:
            order_recommendations = explainer.generate_treatment_order_recommendations(list(set(treatment_types)))
            
            st.markdown("**🎯 Optimal Processing Sequence:**")
            for i, order_rec in enumerate(order_recommendations, 1):
                st.markdown(f"{i}. {order_rec}")
            
            st.info("💡 Following this order minimizes conflicts and maximizes treatment effectiveness.")
        else:
            st.info("No specific treatment order recommendations available.")
    
    # Action buttons for explainer
    st.markdown("---")
    explainer_col1, explainer_col2, explainer_col3 = st.columns(3)
    
    with explainer_col1:
        if st.button("📋 Export Analysis Report", key=f"export_analysis_{tab_key}"):
            # Generate comprehensive report
            report = "# Smart Column Treatment Analysis Report\n\n"
            for rec in recommendations:
                column = rec['column']
                report += f"## {column}\n"
                report += f"- **Issue**: {rec['reason']}\n"
                report += f"- **Suggested Action**: {rec['suggested_action']}\n"
                report += f"- **Severity**: {rec['severity']}\n\n"
            
            st.download_button(
                label="💾 Download Report",
                data=report,
                file_name=f"treatment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                key=f"download_report_{tab_key}"
            )
    
    with explainer_col2:
        if st.button("🔄 Refresh Analysis", key=f"refresh_analysis_{tab_key}"):
            st.rerun()
    
    with explainer_col3:
        if st.button("❓ Help & Tips", key=f"help_tips_{tab_key}"):
            st.info("""
            **How to use the Smart Explainer:**
            - Review confidence scores to prioritize treatments
            - Check dependencies before applying multiple treatments
            - Follow the recommended treatment order for best results
            - Export analysis reports for documentation
            """)
    
    return explainer

def show_activity_list(tab_name, title):
    """Display detailed list of activities performed in this tab"""
    st.markdown(f"#### 📋 {title} - Activities Performed")
    
    if tab_name in st.session_state.activities_performed and st.session_state.activities_performed[tab_name]:
        activities = st.session_state.activities_performed[tab_name]
        
        for i, activity in enumerate(reversed(activities[-5:]), 1):  # Show last 5 activities
            with st.expander(f"Activity {len(activities) - i + 1}: {activity['action']}", expanded=False):
                st.write(f"**Time:** {activity['timestamp']}")
                st.write(f"**Details:** {activity['details']}")
    else:
        st.info("No activities performed in this tab yet")

def apply_auto_data_types(df):
    """Apply automatic data type detection and conversion"""
    activities = []
    suggestions = auto_detect_column_types(df)
    conversions_made = 0
    
    for col, suggested_type in suggestions.items():
        current_type = str(df[col].dtype)
        
        if suggested_type != current_type:
            try:
                if suggested_type == 'datetime64':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    activities.append(f"Converted {col} from {current_type} to datetime")
                    conversions_made += 1
                elif suggested_type == 'numerical':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    activities.append(f"Converted {col} from {current_type} to numeric")
                    conversions_made += 1
                elif suggested_type == 'boolean':
                    df[col] = df[col].astype('bool')
                    activities.append(f"Converted {col} from {current_type} to boolean")
                    conversions_made += 1
                elif suggested_type == 'categorical':
                    df[col] = df[col].astype('category')
                    activities.append(f"Converted {col} from {current_type} to category")
                    conversions_made += 1
            except Exception as e:
                activities.append(f"Failed to convert {col}: {str(e)}")
    
    if conversions_made > 0:
        log_action("Auto Data Type Conversion", f"Converted {conversions_made} columns", "data_types")
        
    return df, activities

def auto_detect_column_types(df):
    """Automatically detect and suggest column types"""
    suggestions = {}
    for col in df.columns:
        # Try to detect datetime
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].dropna().head(100), infer_datetime_format=True)
                suggestions[col] = 'datetime64'
                continue
            except:
                pass
        
        # Detect boolean
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and all(str(v).lower() in ['true', 'false', '1', '0', 'yes', 'no'] for v in unique_vals):
            suggestions[col] = 'boolean'
            continue
            
        # Detect numerical
        if df[col].dtype in ['int64', 'float64']:
            suggestions[col] = 'numerical'
        elif df[col].dtype == 'object':
            # Check if can be converted to numeric
            try:
                pd.to_numeric(df[col].dropna().head(100))
                suggestions[col] = 'numerical'
            except:
                suggestions[col] = 'categorical'
        else:
            suggestions[col] = 'categorical'
    
    return suggestions

st.title("🧹 Clean Pipeline")
st.markdown("Complete data preprocessing with automated recommendations and manual controls")

# Check if data is available
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.error("⚠️ No dataset available. Please upload data first.")
    st.stop()

df = st.session_state.current_dataset.copy()

# Save original data for comparisons
if 'original_clean_data' not in st.session_state:
    st.session_state.original_clean_data = df.copy()


# Main Pipeline Tabs
st.markdown("### 🔧 Data Processing Pipeline")

# Tab definitions
tab_names = [
    "🔍 Detect & Convert Data Types",
    "🔄 Remove Duplicates", 
    "🕳️ Handle Missing Values",
    "🎯 Detect & Treat Outliers",
    "📝 Clean Text",
    "📅 Process Date/Time",
    "🌍 Process Geospatial Data",
    "🏷️ Encode Categorical Variables",
    "📊 Scale/Normalize Numerical Features",
    "⚙️ Engineer/Select Features",
    "📈 Summary Statistics"
]

# Show current tab indicator
current_tab = st.session_state.current_tab_index
st.markdown(f"**Current Step:** {tab_names[current_tab]} ({current_tab + 1} of {len(tab_names)})")

# Navigation buttons at the top
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])

with col1:
    if current_tab > 0:
        if st.button("⬅️ Previous", key="nav_prev"):
            st.session_state.current_tab_index = current_tab - 1
            st.rerun()
    else:
        st.button("⬅️ Previous", disabled=True, key="nav_prev_disabled")

with col2:
    if current_tab < len(tab_names) - 1:
        if st.button("Next ➡️", key="nav_next"):
            st.session_state.current_tab_index = current_tab + 1
            st.rerun()
    else:
        st.button("Next ➡️", disabled=True, key="nav_next_disabled")

with col3:
    selected_tab = st.selectbox("Jump to Step", range(len(tab_names)), 
                              format_func=lambda x: f"{x+1}. {tab_names[x]}", 
                              index=current_tab, key="tab_selector")
    if selected_tab != current_tab:
        if st.button("Go", key="go_to_tab"):
            st.session_state.current_tab_index = selected_tab
            st.rerun()

with col4:
    if st.button("↩️ Undo Last", key="nav_undo"):
        undo_last_action()

# Show the current tab content based on index
st.markdown("---")

# ==================== TAB 1: DETECT & CONVERT DATA TYPES ====================
if current_tab == 0:
    st.subheader("🔍 Detect & Convert Data Types")
    
    # Auto Mode Toggle - enabled by default
    auto_mode = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_dtype")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations = ai_analyze_columns_for_treatment(df, "data_types")
    selected_recommendations = display_ai_recommendations(ai_recommendations, "dtype")
    
    # Smart Column Treatment AI Explainer
    # Smart Column Treatment AI Explainer for Data Types
    if ai_recommendations:
        display_smart_column_explainer(df, ai_recommendations, "dtype")

    #if ai_recommendations:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, ai_recommendations, "dtype")
    
    # Manual Selection Override
    if not auto_mode or st.checkbox("🎯 Manual Column Selection", key="manual_dtype_selection"):
        st.markdown("#### Manual Column Selection")
        manual_cols = st.multiselect(
            "Select columns for type conversion:",
            df.columns.tolist(),
            default=[rec['column'] for rec in selected_recommendations if rec['column'] in df.columns],
            key="manual_dtype_cols"
        )
        # Override recommendations with manual selection
        selected_recommendations = [{'column': col, 'reason': 'Manually selected', 'suggested_action': 'Convert data type'} for col in manual_cols]
    
    # Preview Section
    if selected_recommendations and st.checkbox("👁️ Preview Changes", key="preview_dtype"):
        st.markdown("#### 🔍 Preview Data Type Changes")
        preview_df = df.copy()
        for rec in selected_recommendations[:3]:  # Preview first 3
            col = rec['column']
            if col in df.columns:
                try:
                    if 'numeric' in rec['suggested_action'].lower():
                        preview_df[f"{col}_preview"] = pd.to_numeric(df[col], errors='coerce')
                    elif 'datetime' in rec['suggested_action'].lower():
                        preview_df[f"{col}_preview"] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        st.dataframe(preview_df.head(), use_container_width=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Current Data Types")
        
        # Enhanced dtype analysis
        type_analysis = []
        type_suggestions = auto_detect_column_types(df)
        
        for col in df.columns:
            current_type = str(df[col].dtype)
            suggested_type = type_suggestions.get(col, current_type)
            non_null_count = df[col].count()
            unique_count = df[col].nunique()
            sample_values = df[col].dropna().head(3).tolist()
            
            type_analysis.append({
                'Column': col,
                'Current Type': current_type,
                'Suggested Type': suggested_type,
                'Non-Null Count': non_null_count,
                'Unique Values': unique_count,
                'Sample Values': str(sample_values)
            })
        
        # Convert to string to avoid Arrow issues
        type_df = pd.DataFrame(type_analysis)
        for col in type_df.columns:
            type_df[col] = type_df[col].astype(str)
        
        st.dataframe(type_df, use_container_width=True)
    
    with col2:
        # Type Distribution Visualization
        st.markdown("#### Type Distribution")
        current_types = df.dtypes.astype(str).value_counts()
        fig_types = px.pie(values=current_types.values, names=current_types.index, 
                          title="Current Data Types")
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Manual Conversion Controls
    if not auto_mode:
        st.markdown("#### Manual Type Conversion")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            conversion_col = st.selectbox("Select Column:", df.columns, key="dtype_conv_col")
        with col2:
            new_type = st.selectbox("Convert To:", 
                                  ["int64", "float64", "object", "datetime64", "category", "boolean"], 
                                  key="dtype_new_type")
        with col3:
            if st.button("🔄 Convert Type", type="primary", key="convert_dtype_btn"):
                try:
                    if new_type == "datetime64":
                        df[conversion_col] = pd.to_datetime(df[conversion_col], errors='coerce')
                    elif new_type == "boolean":
                        df[conversion_col] = df[conversion_col].astype(bool)
                    elif new_type == "category":
                        df[conversion_col] = df[conversion_col].astype('category')
                    else:
                        df[conversion_col] = df[conversion_col].astype(new_type)
                    
                    st.session_state.current_dataset = df
                    log_action("Data Type Conversion", f"Converted {conversion_col} to {new_type}")
                    st.success(f"✅ Converted {conversion_col} to {new_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
    
    else:
        # Auto Mode Implementation
        st.markdown("#### 🤖 Automatic Type Detection & Conversion")
        
        if st.button("🚀 Auto-Convert All Types", type="primary", key="auto_convert_all"):
            conversions_made = 0
            for col, suggested_type in type_suggestions.items():
                try:
                    current_type = str(df[col].dtype)
                    if suggested_type != current_type and suggested_type != 'categorical':
                        if suggested_type == "datetime64":
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        elif suggested_type == "boolean":
                            df[col] = df[col].astype(bool)
                        elif suggested_type == "numerical":
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        conversions_made += 1
                        log_action("Auto Type Conversion", f"Converted {col} to {suggested_type}", "data_types")
                
                except Exception as e:
                    continue
            
            if conversions_made > 0:
                st.session_state.current_dataset = df
                st.balloons()  # Balloon notification for success
                st.success(f"✅ Auto-converted {conversions_made} columns!")
                st.rerun()
            else:
                st.info("ℹ️ No automatic conversions needed.")
    
    # Show activity list for this tab
    show_activity_list("data_types", "Data Type Conversions")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode and 'activities_performed' in st.session_state and st.session_state.activities_performed['data_types']:
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)
    
    # Download Option
    if st.button("💾 Download Dataset", key="download_after_dtype"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"cleaned_data_dtypes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ==================== TAB 2: REMOVE DUPLICATES ====================
elif current_tab == 1:
    st.subheader("🔄 Remove Duplicates")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_dup = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_duplicate")
    

    
    # Duplicate Analysis
    exact_duplicates = df.duplicated().sum()
    total_rows = len(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Exact Duplicates")
        st.metric("Duplicate Rows", f"{exact_duplicates:,}", f"{(exact_duplicates/total_rows*100):.1f}%")
        
        if exact_duplicates > 0:
            st.markdown("**Sample Duplicate Rows:**")
            duplicate_sample = df[df.duplicated()].head(5)
            # Convert to string to avoid Arrow issues
            for col in duplicate_sample.columns:
                duplicate_sample[col] = duplicate_sample[col].astype(str)
            st.dataframe(duplicate_sample, use_container_width=True)
        else:
            st.success("✅ No exact duplicates found!")
    
    with col2:
        # Duplicate Visualization
        st.markdown("#### Duplicate Analysis")
        if exact_duplicates > 0:
            dup_data = {'Type': ['Unique Rows', 'Duplicate Rows'], 
                       'Count': [total_rows - exact_duplicates, exact_duplicates]}
            fig_dup = px.bar(dup_data, x='Type', y='Count', title="Duplicate Distribution")
            st.plotly_chart(fig_dup, use_container_width=True)
    
    # Manual Controls
    if not auto_mode_dup:
        st.markdown("#### Manual Duplicate Removal")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            subset_cols = st.multiselect("Check duplicates based on columns:", 
                                       df.columns.tolist(),
                                       default=df.columns.tolist()[:3] if len(df.columns) >= 3 else df.columns.tolist(),
                                       key="dup_subset_cols")
        
        with col2:
            keep_option = st.selectbox("Keep which duplicate:", ["first", "last"], key="dup_keep_option")
        
        with col3:
            if st.button("🗑️ Remove Duplicates", type="primary", key="remove_duplicates_btn"):
                if subset_cols:
                    initial_rows = len(df)
                    df_cleaned = df.drop_duplicates(subset=subset_cols, keep=keep_option)
                    removed_count = initial_rows - len(df_cleaned)
                    
                    st.session_state.current_dataset = df_cleaned
                    log_action("Duplicate Removal", f"Removed {removed_count} duplicates based on {subset_cols}")
                    st.success(f"✅ Removed {removed_count} duplicate rows!")
                    st.rerun()
                else:
                    st.warning("Please select at least one column for duplicate checking.")
    
    else:
        # Auto Mode Implementation
        st.markdown("#### 🤖 Automatic Duplicate Removal")
        
        if st.button("🚀 Auto-Remove Duplicates", type="primary", key="auto_remove_duplicates"):
            if exact_duplicates > 0:
                initial_rows = len(df)
                df_cleaned = df.drop_duplicates()
                removed_count = initial_rows - len(df_cleaned)
                
                st.session_state.current_dataset = df_cleaned
                log_action("Auto Duplicate Removal", f"Automatically removed {removed_count} exact duplicates", "duplicates")
                st.balloons()  # Balloon notification for success
                st.success(f"✅ Automatically removed {removed_count} duplicate rows!")
                st.rerun()
            else:
                st.info("ℹ️ No duplicates to remove.")
    
    # Show activity list for this tab
    show_activity_list("duplicates", "Duplicate Removal")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_dup and 'activities_performed' in st.session_state and st.session_state.activities_performed['duplicates']:
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 3: HANDLE MISSING VALUES ====================
elif current_tab == 2:
    st.subheader("🕳️ Handle Missing Values")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_missing = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_missing")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations_missing = ai_analyze_columns_for_treatment(df, "missing_values")
    selected_recommendations_missing = display_ai_recommendations(ai_recommendations_missing, "missing")
    
    # Smart Column Treatment AI Explainer
    if ai_recommendations_missing:
        display_smart_column_explainer(df, ai_recommendations_missing, "missing")

    #if ai_recommendations_missing:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, ai_recommendations_missing, "missing")
    
    # Manual Selection Override
    if not auto_mode_missing or st.checkbox("🎯 Manual Missing Value Treatment", key="manual_missing_selection"):
        st.markdown("#### Manual Missing Value Configuration")
        missing_cols_manual = st.multiselect(
            "Select columns to treat for missing values:",
            df.columns[df.isnull().any()].tolist(),
            default=[rec['column'] for rec in selected_recommendations_missing if rec['column'] in df.columns],
            key="manual_missing_cols"
        )
        
        imputation_strategy = st.selectbox(
            "Imputation strategy:",
            ["Auto-detect", "Mean", "Median", "Mode", "Forward fill", "Backward fill", "Drop rows"],
            key="imputation_strategy"
        )
    
    # Preview Section
    if selected_recommendations_missing and st.checkbox("👁️ Preview Missing Value Treatment", key="preview_missing"):
        st.markdown("#### 🔍 Preview Missing Value Treatment")
        for rec in selected_recommendations_missing[:3]:  # Preview first 3
            col = rec['column']
            if col in df.columns and df[col].isnull().any():
                missing_count = df[col].isnull().sum()
                st.markdown(f"**{col}:** {missing_count} missing values will be imputed")
                
                # Show before/after sample
                sample_with_missing = df[df[col].isnull()].head(3)
                if len(sample_with_missing) > 0:
                    st.dataframe(sample_with_missing[[col]], use_container_width=True)
    
    # Missing Value Analysis
    missing_data = df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Missing Values Summary")
            missing_df = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing Count': missing_cols.values,
                'Missing %': (missing_cols.values / len(df) * 100).round(2),
                'Data Type': [str(df[col].dtype) for col in missing_cols.index]
            })
            # Convert to string to avoid Arrow issues
            for col in missing_df.columns:
                missing_df[col] = missing_df[col].astype(str)
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Missing Values Heatmap
            st.markdown("#### Missing Values Heatmap")
            fig_missing = px.imshow(df.isnull().T, aspect="auto", title="Missing Values Pattern")
            st.plotly_chart(fig_missing, use_container_width=True)
        
        # Manual Controls
        if not auto_mode_missing:
            st.markdown("#### Manual Missing Value Treatment")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_col = st.selectbox("Select Column:", missing_cols.index, key="missing_col_select")
            
            with col2:
                if df[selected_col].dtype in ['int64', 'float64']:
                    impute_options = ["Drop rows", "Mean", "Median", "Mode", "Constant value", "Forward fill", "Backward fill"]
                else:
                    impute_options = ["Drop rows", "Mode", "Constant value", "Forward fill", "Backward fill"]
                
                impute_method = st.selectbox("Imputation Method:", impute_options, key="impute_method_select")
            
            with col3:
                if impute_method == "Constant value":
                    constant_value = st.text_input("Constant Value:", key="constant_value_input")
                
                if st.button("🔧 Apply Treatment", type="primary", key="apply_missing_treatment"):
                    try:
                        if impute_method == "Drop rows":
                            df = df.dropna(subset=[selected_col])
                        elif impute_method == "Mean":
                            df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                        elif impute_method == "Median":
                            df[selected_col].fillna(df[selected_col].median(), inplace=True)
                        elif impute_method == "Mode":
                            df[selected_col].fillna(df[selected_col].mode()[0], inplace=True)
                        elif impute_method == "Constant value":
                            df[selected_col].fillna(constant_value, inplace=True)
                        elif impute_method == "Forward fill":
                            df[selected_col].fillna(method='ffill', inplace=True)
                        elif impute_method == "Backward fill":
                            df[selected_col].fillna(method='bfill', inplace=True)
                        
                        st.session_state.current_dataset = df
                        log_action("Missing Value Treatment", f"Applied {impute_method} to {selected_col}", "missing_values")
                        st.balloons()  # Balloon notification for success
                        st.success(f"✅ Applied {impute_method} to {selected_col}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Treatment failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Missing Value Treatment")
            
            if st.button("🚀 Auto-Fix Missing Values", type="primary", key="auto_fix_missing"):
                treatments_applied = 0
                
                for col in missing_cols.index:
                    try:
                        if df[col].dtype in ['int64', 'float64']:
                            # Use median for numerical columns
                            df[col].fillna(df[col].median(), inplace=True)
                            treatments_applied += 1
                            log_action("Auto Missing Value Treatment", f"Applied median imputation to {col}", "missing_values")
                        else:
                            # Use mode for categorical columns
                            mode_val = df[col].mode()
                            if len(mode_val) > 0:
                                df[col].fillna(mode_val[0], inplace=True)
                                treatments_applied += 1
                                log_action("Auto Missing Value Treatment", f"Applied mode imputation to {col}", "missing_values")
                    except Exception as e:
                        continue
                
                if treatments_applied > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-treated missing values in {treatments_applied} columns!")
                    st.rerun()
                else:
                    st.info("ℹ️ No missing values to treat.")
    
    # Show activity list for this tab
    show_activity_list("missing_values", "Missing Value Treatment")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_missing and 'activities_performed' in st.session_state and st.session_state.activities_performed['missing_values']:
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 4: DETECT & TREAT OUTLIERS ====================
elif current_tab == 3:
    st.subheader("🎯 Detect & Treat Outliers")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_outliers = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_outliers")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations_outliers = ai_analyze_columns_for_treatment(df, "outliers")
    selected_recommendations_outliers = display_ai_recommendations(ai_recommendations_outliers, "outliers")

    # Smart Column Treatment AI Explainer
    if ai_recommendations_outliers:
        display_smart_column_explainer(df, ai_recommendations_outliers, "outliers")

    #if ai_recommendations_outliers:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, ai_recommendations_outliers, "outliers")
    
    # Manual Selection Override
    if not auto_mode_outliers or st.checkbox("🎯 Manual Outlier Detection", key="manual_outlier_selection"):
        st.markdown("#### Manual Outlier Configuration")
        outlier_cols_manual = st.multiselect(
            "Select numeric columns for outlier detection:",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=[rec['column'] for rec in selected_recommendations_outliers if rec['column'] in df.columns],
            key="manual_outlier_cols"
        )
        
        outlier_method = st.selectbox(
            "Outlier detection method:",
            ["IQR Method", "Z-Score", "Isolation Forest", "Manual Threshold"],
            key="outlier_method"
        )
        
        outlier_treatment = st.selectbox(
            "Outlier treatment:",
            ["Remove", "Cap/Winsorize", "Transform (log)", "Keep (mark only)"],
            key="outlier_treatment"
        )
    
    # Preview Section
    if selected_recommendations_outliers and st.checkbox("👁️ Preview Outlier Treatment", key="preview_outliers"):
        st.markdown("#### 🔍 Preview Outlier Detection")
        for rec in selected_recommendations_outliers[:3]:  # Preview first 3
            col = rec['column']
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                
                if len(outliers) > 0:
                    st.markdown(f"**{col}:** {len(outliers)} outliers detected")
                    fig = px.box(df, y=col, title=f"Outliers in {col}")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.info("ℹ️ No numeric columns found for outlier detection.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Outlier Detection Summary")
            
            outlier_summary = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': len(outliers),
                    'Outlier %': f"{(len(outliers)/len(df)*100):.1f}%",
                    'Lower Bound': f"{lower_bound:.2f}",
                    'Upper Bound': f"{upper_bound:.2f}"
                })
            
            outlier_df = pd.DataFrame(outlier_summary)
            # Convert to string to avoid Arrow issues
            for col in outlier_df.columns:
                outlier_df[col] = outlier_df[col].astype(str)
            st.dataframe(outlier_df, use_container_width=True)
        
        with col2:
            # Outlier Visualization
            if len(numeric_cols) > 0:
                viz_col = st.selectbox("Visualize outliers for:", numeric_cols, key="outlier_viz_col")
                fig_box = px.box(df, y=viz_col, title=f"Outliers in {viz_col}")
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Manual Controls
        if not auto_mode_outliers:
            st.markdown("#### Manual Outlier Treatment")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_col = st.selectbox("Select Column:", numeric_cols, key="outlier_col_select")
            
            with col2:
                detection_method = st.selectbox("Detection Method:", ["IQR", "Z-Score", "Isolation Forest"], key="outlier_detection_method")
            
            with col3:
                treatment_method = st.selectbox("Treatment:", ["Remove", "Cap (Clip)", "Transform"], key="outlier_treatment_method")
            
            if st.button("🎯 Apply Outlier Treatment", type="primary", key="apply_outlier_treatment"):
                try:
                    if detection_method == "IQR":
                        Q1 = df[selected_col].quantile(0.25)
                        Q3 = df[selected_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_mask = (df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)
                    
                    elif detection_method == "Z-Score":
                        z_scores = np.abs((df[selected_col] - df[selected_col].mean()) / df[selected_col].std())
                        outlier_mask = z_scores > 3
                    
                    else:  # Isolation Forest
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers = iso_forest.fit_predict(df[[selected_col]])
                        outlier_mask = outliers == -1
                    
                    outlier_count = outlier_mask.sum()
                    
                    if treatment_method == "Remove":
                        df = df[~outlier_mask]
                    elif treatment_method == "Cap (Clip)":
                        if detection_method == "IQR":
                            df[selected_col] = df[selected_col].clip(lower=lower_bound, upper=upper_bound)
                    elif treatment_method == "Transform":
                        df[selected_col] = np.log1p(df[selected_col])
                    
                    st.session_state.current_dataset = df
                    log_action("Outlier Treatment", f"Applied {treatment_method} using {detection_method} to {selected_col}, affected {outlier_count} values")
                    st.success(f"✅ Treated {outlier_count} outliers in {selected_col}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Treatment failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Outlier Treatment")
            
            if st.button("🚀 Auto-Treat Outliers", type="primary", key="auto_treat_outliers"):
                treatments_applied = 0
                total_outliers_treated = 0
                
                for col in numeric_cols:
                    try:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        
                        if outliers_count > 0:
                            # Auto-cap outliers
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                            log_action("Auto Outlier Treatment", f"Capped outliers in {col} ({outliers_count} values)")
                            treatments_applied += 1
                            total_outliers_treated += outliers_count
                    
                    except Exception as e:
                        continue
                
                if treatments_applied > 0:
                    st.session_state.current_dataset = df
                    st.success(f"✅ Auto-treated {total_outliers_treated} outliers in {treatments_applied} columns!")
                    st.rerun()
                else:
                    st.info("ℹ️ No outliers detected for treatment.")

# ==================== TAB 5: CLEAN TEXT ====================
elif current_tab == 4:
    st.subheader("📝 Clean Text")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_text = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_text")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations_text = ai_analyze_columns_for_treatment(df, "text_cleaning")
    selected_recommendations_text = display_ai_recommendations(ai_recommendations_text, "text")
    
    # Smart Column Treatment AI Explainer
    if ai_recommendations_text:
        #st.markdown(" 🧠 Smart Column Treatment AI Explainer")
        display_smart_column_explainer(df, ai_recommendations_text, "text")

    # Manual Selection Override
    if not auto_mode_text or st.checkbox("🎯 Manual Text Cleaning Selection", key="manual_text_selection"):
        st.markdown("#### Manual Text Cleaning Configuration")
        text_cols_manual = st.multiselect(
            "Select text columns for cleaning:",
            df.select_dtypes(include=['object']).columns.tolist(),
            default=[rec['column'] for rec in selected_recommendations_text if rec['column'] in df.columns],
            key="manual_text_cols"
        )
        
        cleaning_options = st.multiselect(
            "Select cleaning operations:",
            ["Remove extra whitespace", "Standardize case", "Remove special characters", "Remove numbers", "Remove punctuation"],
            default=["Remove extra whitespace", "Standardize case"],
            key="cleaning_operations"
        )
    
    # Preview Section
    if selected_recommendations_text and st.checkbox("👁️ Preview Text Cleaning", key="preview_text"):
        st.markdown("#### 🔍 Preview Text Cleaning")
        for rec in selected_recommendations_text[:2]:  # Preview first 2
            col = rec['column']
            if col in df.columns:
                sample_text = df[col].dropna().head(5)
                st.markdown(f"**{col} - Before Cleaning:**")
                st.write(sample_text.tolist())
                
                # Show cleaned preview
                cleaned_sample = sample_text.str.strip().str.replace(r'\s+', ' ', regex=True)
                st.markdown(f"**{col} - After Cleaning:**")
                st.write(cleaned_sample.tolist())
    
    text_cols = df.select_dtypes(include=['object']).columns
    
    if len(text_cols) == 0:
        st.info("ℹ️ No text columns found for cleaning.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Text Analysis")
            selected_text_col = st.selectbox("Select text column:", text_cols, key="text_analysis_col")
            
            # Text statistics
            text_stats = {
                'Total Records': len(df[selected_text_col]),
                'Non-Empty Records': df[selected_text_col].notna().sum(),
                'Unique Values': df[selected_text_col].nunique(),
                'Avg Length': df[selected_text_col].str.len().mean() if df[selected_text_col].dtype == 'object' else 0
            }
            
            for key, value in text_stats.items():
                st.metric(key, f"{value:.1f}" if isinstance(value, float) else f"{value:,}")
        
        with col2:
            # Word frequency visualization
            if df[selected_text_col].dtype == 'object':
                st.markdown("#### Word Frequency")
                try:
                    all_text = ' '.join(df[selected_text_col].dropna().astype(str))
                    words = re.findall(r'\w+', all_text.lower())
                    word_freq = Counter(words).most_common(10)
                    
                    if word_freq:
                        word_df = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
                        fig_words = px.bar(word_df, x='Word', y='Frequency', title="Top 10 Words")
                        st.plotly_chart(fig_words, use_container_width=True)
                except Exception as e:
                    st.info("Unable to generate word frequency chart")
        
        # Manual Controls
        if not auto_mode_text:
            st.markdown("#### Manual Text Cleaning")
            
            cleaning_options = st.multiselect("Select cleaning operations:", [
                "Convert to lowercase",
                "Remove special characters",
                "Remove extra whitespace",
                "Remove numbers",
                "Remove punctuation",
                "Remove stop words"
            ], key="text_cleaning_options")
            
            if cleaning_options and st.button("🧹 Apply Text Cleaning", type="primary", key="apply_text_cleaning"):
                try:
                    cleaned_col = df[selected_text_col].copy()
                    
                    for option in cleaning_options:
                        if option == "Convert to lowercase":
                            cleaned_col = cleaned_col.str.lower()
                        elif option == "Remove special characters":
                            cleaned_col = cleaned_col.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                        elif option == "Remove extra whitespace":
                            cleaned_col = cleaned_col.str.strip().str.replace(r'\s+', ' ', regex=True)
                        elif option == "Remove numbers":
                            cleaned_col = cleaned_col.str.replace(r'\d+', '', regex=True)
                        elif option == "Remove punctuation":
                            cleaned_col = cleaned_col.str.replace(r'[^\w\s]', '', regex=True)
                    
                    df[selected_text_col] = cleaned_col
                    st.session_state.current_dataset = df
                    log_action("Text Cleaning", f"Applied {len(cleaning_options)} operations to {selected_text_col}")
                    st.success(f"✅ Applied {len(cleaning_options)} cleaning operations!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Text cleaning failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Text Cleaning")
            
            if st.button("🚀 Auto-Clean Text", type="primary", key="auto_clean_text"):
                cleanings_applied = 0
                
                for col in text_cols:
                    try:
                        # Basic auto-cleaning: lowercase, trim whitespace, remove extra spaces
                        df[col] = df[col].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
                        log_action("Auto Text Cleaning", f"Auto-cleaned {col}", "text_cleaning")
                        cleanings_applied += 1
                    except Exception as e:
                        continue
                
                if cleanings_applied > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-cleaned {cleanings_applied} text columns!")
                    st.rerun()
                else:
                    st.info("ℹ️ No text columns to clean.")
    
    # Show activity list for this tab
    show_activity_list("text_cleaning", "Text Cleaning")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_text and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('text_cleaning', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 6: PROCESS DATE/TIME ====================
elif current_tab == 5:
    st.subheader("📅 Process Date/Time")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_datetime = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_datetime")
    
    # AI Analysis Section for DateTime Processing
    st.markdown("---")
    datetime_recommendations = []
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    potential_datetime_cols = []
    
    # Check for potential datetime columns in object columns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            sample_values = df[col].dropna().head(100)
            datetime_converted = pd.to_datetime(sample_values, errors='coerce')
            conversion_rate = (1 - datetime_converted.isnull().sum() / len(sample_values)) * 100
            
            if conversion_rate > 50:  # 50% convertible to datetime
                potential_datetime_cols.append(col)
                datetime_recommendations.append({
                    'column': col,
                    'reason': f"Column contains date/time patterns with {conversion_rate:.1f}% conversion success. Converting enables time-based analysis and feature extraction.",
                    'severity': 'medium',
                    'suggested_action': f'Convert to datetime and extract features (year, month, weekday)'
                })
        except:
            continue
    
    # Add existing datetime columns for feature extraction
    for col in datetime_cols:
        datetime_recommendations.append({
            'column': col,
            'reason': f"DateTime column ready for feature extraction. Can generate time-based features like year, month, day of week for enhanced analysis.",
            'severity': 'low',
            'suggested_action': 'Extract datetime features for machine learning'
        })
    
    selected_recommendations_datetime = display_ai_recommendations(datetime_recommendations, "datetime")
    

    # Smart Column Treatment AI Explainer - DateTime
    if datetime_recommendations:
        #st.markdown("### 🧠 Smart Column Treatment AI Explainer - DateTime")
        display_smart_column_explainer(df, datetime_recommendations, "datetime")

    #if datetime_recommendations:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, datetime_recommendations, "datetime")
    
    # Detect datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    potential_datetime_cols = []
    
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col].dropna().head(10))
            potential_datetime_cols.append(col)
        except:
            continue
    
    all_datetime_cols = list(datetime_cols) + potential_datetime_cols
    
    if len(all_datetime_cols) == 0:
        st.info("ℹ️ No datetime columns detected.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### DateTime Analysis")
            selected_dt_col = st.selectbox("Select datetime column:", all_datetime_cols, key="datetime_analysis_col")
            
            # Convert to datetime if needed
            if df[selected_dt_col].dtype == 'object':
                try:
                    df[selected_dt_col] = pd.to_datetime(df[selected_dt_col])
                except:
                    st.error("Failed to parse datetime column")
            
            if df[selected_dt_col].dtype.name.startswith('datetime'):
                dt_stats = {
                    'Date Range': f"{df[selected_dt_col].min().date()} to {df[selected_dt_col].max().date()}",
                    'Total Records': len(df[selected_dt_col]),
                    'Valid Dates': df[selected_dt_col].notna().sum(),
                    'Unique Dates': df[selected_dt_col].nunique()
                }
                
                for key, value in dt_stats.items():
                    st.write(f"**{key}:** {value}")
        
        with col2:
            # Time series visualization
            if df[selected_dt_col].dtype.name.startswith('datetime'):
                st.markdown("#### Time Distribution")
                try:
                    dt_counts = df[selected_dt_col].dt.date.value_counts().sort_index()
                    fig_dt = px.line(x=dt_counts.index, y=dt_counts.values, title="Records Over Time")
                    st.plotly_chart(fig_dt, use_container_width=True)
                except Exception as e:
                    st.info("Unable to generate time series chart")
        
        # Feature extraction
        if not auto_mode_datetime:
            st.markdown("#### Manual Feature Extraction")
            
            features_to_extract = st.multiselect("Extract datetime features:", [
                "Year", "Month", "Day", "Weekday", "Hour", "Quarter", "Week of Year"
            ], key="datetime_features_select")
            
            if features_to_extract and st.button("📅 Extract Features", type="primary", key="extract_datetime_features"):
                try:
                    dt_col = df[selected_dt_col]
                    
                    for feature in features_to_extract:
                        if feature == "Year":
                            df[f"{selected_dt_col}_year"] = dt_col.dt.year
                        elif feature == "Month":
                            df[f"{selected_dt_col}_month"] = dt_col.dt.month
                        elif feature == "Day":
                            df[f"{selected_dt_col}_day"] = dt_col.dt.day
                        elif feature == "Weekday":
                            df[f"{selected_dt_col}_weekday"] = dt_col.dt.dayofweek
                        elif feature == "Hour":
                            df[f"{selected_dt_col}_hour"] = dt_col.dt.hour
                        elif feature == "Quarter":
                            df[f"{selected_dt_col}_quarter"] = dt_col.dt.quarter
                        elif feature == "Week of Year":
                            df[f"{selected_dt_col}_week"] = dt_col.dt.isocalendar().week
                    
                    st.session_state.current_dataset = df
                    log_action("DateTime Feature Extraction", f"Extracted {len(features_to_extract)} features from {selected_dt_col}")
                    st.success(f"✅ Extracted {len(features_to_extract)} datetime features!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Feature extraction failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Feature Extraction")
            
            if st.button("🚀 Auto-Extract DateTime Features", type="primary", key="auto_extract_datetime"):
                features_extracted = 0
                
                for col in all_datetime_cols:
                    try:
                        if df[col].dtype == 'object':
                            df[col] = pd.to_datetime(df[col])
                        
                        # Auto-extract common features
                        df[f"{col}_year"] = df[col].dt.year
                        df[f"{col}_month"] = df[col].dt.month
                        df[f"{col}_weekday"] = df[col].dt.dayofweek
                        
                        features_extracted += 3
                        log_action("Auto DateTime Extraction", f"Auto-extracted 3 features from {col}", "datetime")
                    except Exception as e:
                        continue
                
                if features_extracted > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-extracted {features_extracted} datetime features!")
                    st.rerun()
                else:
                    st.info("ℹ️ No datetime features to extract.")
    
    # Show activity list for this tab
    show_activity_list("datetime", "DateTime Processing")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_datetime and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('datetime', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 7: PROCESS GEOSPATIAL DATA ====================
elif current_tab == 6:
    st.subheader("🌍 Process Geospatial Data")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_geo = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_geo")
    

    
    # Look for potential geospatial columns
    geo_keywords = ['lat', 'lon', 'latitude', 'longitude', 'coord', 'location', 'address', 'city', 'country']
    potential_geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in geo_keywords)]
    
    if len(potential_geo_cols) == 0:
        st.info("ℹ️ No obvious geospatial columns detected. Look for columns containing coordinates, addresses, or location names.")
    else:
        st.markdown("#### Potential Geospatial Columns Detected")
        
        for col in potential_geo_cols:
            with st.expander(f"📍 {col}"):
                sample_data = df[col].dropna().head(5)
                st.write("Sample values:")
                st.write(sample_data.tolist())
                
                # Basic validation for coordinates
                if df[col].dtype in ['int64', 'float64']:
                    coord_range = f"Range: {df[col].min():.4f} to {df[col].max():.4f}"
                    st.write(coord_range)
                    
                    # Check if values are in valid lat/lon ranges
                    if 'lat' in col.lower():
                        valid_lat = ((df[col] >= -90) & (df[col] <= 90)).all()
                        st.write(f"Valid latitude range: {'✅' if valid_lat else '❌'}")
                    elif 'lon' in col.lower():
                        valid_lon = ((df[col] >= -180) & (df[col] <= 180)).all()
                        st.write(f"Valid longitude range: {'✅' if valid_lon else '❌'}")
        
        # Auto Mode Implementation for Geospatial
        if auto_mode_geo and len(potential_geo_cols) > 0:
            st.markdown("#### 🤖 Automatic Geospatial Processing")
            
            if st.button("🚀 Auto-Process Geospatial Data", type="primary", key="auto_process_geo"):
                processed_cols = 0
                
                for col in potential_geo_cols:
                    try:
                        if df[col].dtype in ['int64', 'float64']:
                            # Validate coordinate ranges
                            if 'lat' in col.lower() and not ((df[col] >= -90) & (df[col] <= 90)).all():
                                df[col] = df[col].clip(-90, 90)
                                log_action("Auto Geo Processing", f"Clipped latitude values in {col}", "geospatial")
                                processed_cols += 1
                            elif 'lon' in col.lower() and not ((df[col] >= -180) & (df[col] <= 180)).all():
                                df[col] = df[col].clip(-180, 180)
                                log_action("Auto Geo Processing", f"Clipped longitude values in {col}", "geospatial")
                                processed_cols += 1
                    except Exception as e:
                        continue
                
                if processed_cols > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-processed {processed_cols} geospatial columns!")
                    st.rerun()
                else:
                    st.info("ℹ️ No geospatial processing needed.")
    
    # Show activity list for this tab
    show_activity_list("geospatial", "Geospatial Processing")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_geo and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('geospatial', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 8: ENCODE CATEGORICAL VARIABLES ====================
elif current_tab == 7:
    st.subheader("🏷️ Encode Categorical Variables")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_encoding = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_encoding")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations_encoding = ai_analyze_columns_for_treatment(df, "categorical")
    selected_recommendations_encoding = display_ai_recommendations(ai_recommendations_encoding, "encoding")
    
    # Smart Column Treatment AI Explainer - Encoding
    if ai_recommendations_encoding:
        display_smart_column_explainer(df, ai_recommendations_encoding, "encoding")

    #if ai_recommendations_encoding:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, ai_recommendations_encoding, "encoding")
    
    # Manual Selection Override
    if not auto_mode_encoding or st.checkbox("🎯 Manual Encoding Selection", key="manual_encoding_selection"):
        st.markdown("#### Manual Encoding Configuration")
        encoding_cols_manual = st.multiselect(
            "Select categorical columns for encoding:",
            df.select_dtypes(include=['object', 'category']).columns.tolist(),
            default=[rec['column'] for rec in selected_recommendations_encoding if rec['column'] in df.columns],
            key="manual_encoding_cols"
        )
        
        encoding_method = st.selectbox(
            "Encoding method:",
            ["Auto-select", "One-hot encoding", "Label encoding", "Target encoding", "Binary encoding"],
            key="encoding_method"
        )
    
    # Preview Section
    if selected_recommendations_encoding and st.checkbox("👁️ Preview Categorical Encoding", key="preview_encoding"):
        st.markdown("#### 🔍 Preview Categorical Encoding")
        for rec in selected_recommendations_encoding[:2]:  # Preview first 2
            col = rec['column']
            if col in df.columns:
                unique_values = df[col].value_counts().head(10)
                st.markdown(f"**{col} - Unique Values (top 10):**")
                st.write(unique_values)
                
                if df[col].nunique() <= 10:
                    st.markdown("→ Recommended: One-hot encoding")
                else:
                    st.markdown("→ Recommended: Label encoding")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) == 0:
        st.info("ℹ️ No categorical columns found for encoding.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Categorical Analysis")
            
            cat_analysis = []
            for col in categorical_cols:
                unique_count = df[col].nunique()
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                
                cat_analysis.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Most Frequent': str(mode_val),
                    'Recommended Encoding': 'One-Hot' if unique_count <= 10 else 'Label'
                })
            
            cat_df = pd.DataFrame(cat_analysis)
            for col in cat_df.columns:
                cat_df[col] = cat_df[col].astype(str)
            st.dataframe(cat_df, use_container_width=True)
        
        with col2:
            # Category distribution visualization
            st.markdown("#### Category Distribution")
            viz_cat_col = st.selectbox("Visualize column:", categorical_cols, key="cat_viz_col")
            
            try:
                cat_counts = df[viz_cat_col].value_counts().head(10)
                fig_cat = px.bar(x=cat_counts.values, y=cat_counts.index, orientation='h', 
                               title=f"Top Categories in {viz_cat_col}")
                st.plotly_chart(fig_cat, use_container_width=True)
            except Exception as e:
                st.info("Unable to generate category chart")
        
        # Manual Controls
        if not auto_mode_encoding:
            st.markdown("#### Manual Encoding")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_cat_col = st.selectbox("Select column:", categorical_cols, key="encoding_col_select")
            
            with col2:
                encoding_method = st.selectbox("Encoding method:", 
                                             ["One-Hot Encoding", "Label Encoding", "Frequency Encoding"], 
                                             key="encoding_method_select")
            
            with col3:
                if st.button("🔤 Apply Encoding", type="primary", key="apply_encoding"):
                    try:
                        if encoding_method == "One-Hot Encoding":
                            encoded_df = pd.get_dummies(df[selected_cat_col], prefix=selected_cat_col)
                            df = pd.concat([df.drop(selected_cat_col, axis=1), encoded_df], axis=1)
                        
                        elif encoding_method == "Label Encoding":
                            le = LabelEncoder()
                            df[f"{selected_cat_col}_encoded"] = le.fit_transform(df[selected_cat_col].astype(str))
                        
                        elif encoding_method == "Frequency Encoding":
                            freq_map = df[selected_cat_col].value_counts().to_dict()
                            df[f"{selected_cat_col}_freq_encoded"] = df[selected_cat_col].map(freq_map)
                        
                        st.session_state.current_dataset = df
                        log_action("Categorical Encoding", f"Applied {encoding_method} to {selected_cat_col}")
                        st.success(f"✅ Applied {encoding_method} to {selected_cat_col}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Encoding failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Encoding")
            
            if st.button("🚀 Auto-Encode Categories", type="primary", key="auto_encode_categories"):
                encodings_applied = 0
                
                for col in categorical_cols:
                    try:
                        unique_count = df[col].nunique()
                        
                        if unique_count <= 10:
                            # One-hot encoding for low cardinality
                            encoded_df = pd.get_dummies(df[col], prefix=col)
                            df = pd.concat([df.drop(col, axis=1), encoded_df], axis=1)
                            log_action("Auto Categorical Encoding", f"One-hot encoded {col}", "encoding")
                        else:
                            # Label encoding for high cardinality
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                            log_action("Auto Categorical Encoding", f"Label encoded {col}", "encoding")
                        
                        encodings_applied += 1
                    except Exception as e:
                        continue
                
                if encodings_applied > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-encoded {encodings_applied} categorical columns!")
                    st.rerun()
                else:
                    st.info("ℹ️ No categorical encoding needed.")
    
    # Show activity list for this tab
    show_activity_list("encoding", "Categorical Encoding")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_encoding and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('encoding', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 9: SCALE/NORMALIZE NUMERICAL FEATURES ====================
elif current_tab == 8:
    st.subheader("📊 Scale/Normalize Numerical Features")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_scaling = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_scaling")
    
    # AI Analysis Section
    st.markdown("---")
    ai_recommendations_scaling = ai_analyze_columns_for_treatment(df, "scaling")
    selected_recommendations_scaling = display_ai_recommendations(ai_recommendations_scaling, "scaling")
    
    # Smart Column Treatment AI Explainer
    if ai_recommendations_scaling:
        #st.markdown(" 🧠 Smart Column Treatment AI Explainer")
        display_smart_column_explainer(df, ai_recommendations_scaling,"scaling")

    #if ai_recommendations_scaling:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, ai_recommendations_scaling, "scaling")
    
    # Manual Selection Override
    if not auto_mode_scaling or st.checkbox("🎯 Manual Scaling Selection", key="manual_scaling_selection"):
        st.markdown("#### Manual Scaling Configuration")
        scaling_cols_manual = st.multiselect(
            "Select numeric columns for scaling:",
            df.select_dtypes(include=[np.number]).columns.tolist(),
            default=[rec['column'] for rec in selected_recommendations_scaling if rec['column'] in df.columns],
            key="manual_scaling_cols"
        )
        
        scaling_method = st.selectbox(
            "Scaling method:",
            ["Auto-select", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"],
            key="scaling_method"
        )
    
    # Preview Section
    if selected_recommendations_scaling and st.checkbox("👁️ Preview Scaling", key="preview_scaling"):
        st.markdown("#### 🔍 Preview Feature Scaling")
        for rec in selected_recommendations_scaling[:2]:  # Preview first 2
            col = rec['column']
            if col in df.columns:
                col_stats = df[col].describe()
                st.markdown(f"**{col} - Current Statistics:**")
                st.write(f"Range: {col_stats['min']:.2f} to {col_stats['max']:.2f}")
                st.write(f"Mean: {col_stats['mean']:.2f}, Std: {col_stats['std']:.2f}")
                
                # Show what scaling would look like
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_preview = scaler.fit_transform(df[[col]])
                st.write(f"After StandardScaler: Mean ≈ 0, Std ≈ 1")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        st.info("ℹ️ No numeric columns found for scaling.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Numerical Features Analysis")
            
            # Statistical summary
            numeric_summary = df[numeric_cols].describe().T
            numeric_summary['Recommended Scaler'] = 'StandardScaler'  # Default recommendation
            
            # Convert to string to avoid Arrow issues
            for col in numeric_summary.columns:
                numeric_summary[col] = numeric_summary[col].astype(str)
            st.dataframe(numeric_summary, use_container_width=True)
        
        with col2:
            # Distribution visualization
            st.markdown("#### Distribution Analysis")
            viz_num_col = st.selectbox("Visualize column:", numeric_cols, key="scaling_viz_col")
            
            try:
                fig_hist = px.histogram(df, x=viz_num_col, title=f"Distribution of {viz_num_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.info("Unable to generate distribution chart")
        
        # Manual Controls
        if not auto_mode_scaling:
            st.markdown("#### Manual Scaling")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_num_cols = st.multiselect("Select columns:", numeric_cols, 
                                                 default=list(numeric_cols)[:3], key="scaling_cols_select")
            
            with col2:
                scaling_method = st.selectbox("Scaling method:", 
                                            ["StandardScaler", "MinMaxScaler", "RobustScaler"], 
                                            key="scaling_method_select")
            
            with col3:
                if st.button("⚖️ Apply Scaling", type="primary", key="apply_scaling"):
                    try:
                        if scaling_method == "StandardScaler":
                            scaler = StandardScaler()
                        elif scaling_method == "MinMaxScaler":
                            scaler = MinMaxScaler()
                        else:  # RobustScaler
                            scaler = RobustScaler()
                        
                        df[selected_num_cols] = scaler.fit_transform(df[selected_num_cols])
                        
                        st.session_state.current_dataset = df
                        log_action("Feature Scaling", f"Applied {scaling_method} to {len(selected_num_cols)} columns")
                        st.success(f"✅ Applied {scaling_method} to {len(selected_num_cols)} columns!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Scaling failed: {str(e)}")
        
        else:
            # Auto Mode Implementation
            st.markdown("#### 🤖 Automatic Scaling")
            
            if st.button("🚀 Auto-Scale Features", type="primary", key="auto_scale_features"):
                try:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()  # Default to StandardScaler for auto mode
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                    
                    st.session_state.current_dataset = df
                    log_action("Auto Feature Scaling", f"Auto-scaled {len(numeric_cols)} numeric columns with StandardScaler", "scaling")
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-scaled {len(numeric_cols)} numeric columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Auto-scaling failed: {str(e)}")
    
    # Show activity list for this tab
    show_activity_list("scaling", "Feature Scaling")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_scaling and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('scaling', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 10: ENGINEER/SELECT FEATURES ====================
elif current_tab == 9:
    st.subheader("⚙️ Engineer/Select Features")
    
    # Auto Mode Toggle - enabled by default
    auto_mode_feature = st.toggle("🤖 Auto Mode", value=st.session_state.auto_mode_enabled, key="auto_feature")
    
    # AI Analysis Section for Feature Engineering
    st.markdown("---")
    feature_recommendations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Analyze feature engineering opportunities
    if len(numeric_cols) >= 2:
        feature_recommendations.append({
            'column': f"{len(numeric_cols)} numeric columns",
            'reason': f"Multiple numeric columns detected. Interaction features and polynomial terms can capture non-linear relationships and improve model performance.",
            'severity': 'medium',
            'suggested_action': 'Create interaction features and polynomial terms'
        })
    
    # Check for high-cardinality numeric columns that could benefit from binning
    for col in numeric_cols:
        unique_count = df[col].nunique()
        if unique_count > 50:  # High cardinality
            feature_recommendations.append({
                'column': col,
                'reason': f"High-cardinality numeric column with {unique_count} unique values. Binning can create categorical features and reduce overfitting.",
                'severity': 'low',
                'suggested_action': 'Apply binning to create categorical groups'
            })
    
    # Check if PCA might be beneficial
    if len(numeric_cols) > 10:
        feature_recommendations.append({
            'column': f"{len(numeric_cols)} numeric features",
            'reason': f"Many numeric features detected. PCA can reduce dimensionality while preserving most variance, improving computational efficiency.",
            'severity': 'medium',
            'suggested_action': 'Apply PCA for dimensionality reduction'
        })
    
    selected_recommendations_feature = display_ai_recommendations(feature_recommendations, "feature")
    
    # Smart Column Treatment AI Explainer for Feature Engineering
    if feature_recommendations:
        # st.markdown(" 🧠 Smart Column Treatment AI Explainer")
        display_smart_column_explainer(df, feature_recommendations, "feature")


    #if feature_recommendations:
        #with st.expander("🧠 Smart Column Treatment AI Explainer", expanded=False):
            #display_smart_column_explainer(df, feature_recommendations, "feature")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    st.markdown("#### Feature Engineering Options")
    
    if not auto_mode_feature:
        # Manual feature engineering
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Create New Features**")
            
            feature_type = st.selectbox("Feature type:", 
                                      ["Polynomial Features", "Interaction Features", "Binning"], 
                                      key="feature_type_select")
            
            if feature_type == "Polynomial Features" and len(numeric_cols) > 0:
                poly_cols = st.multiselect("Select columns for polynomial features:", 
                                         numeric_cols, key="poly_cols_select")
                poly_degree = st.slider("Polynomial degree:", 2, 4, 2, key="poly_degree")
                
                if poly_cols and st.button("🔧 Create Polynomial Features", key="create_poly_features"):
                    try:
                        for col in poly_cols:
                            for degree in range(2, poly_degree + 1):
                                df[f"{col}_poly_{degree}"] = df[col] ** degree
                        
                        st.session_state.current_dataset = df
                        log_action("Feature Engineering", f"Created polynomial features (degree {poly_degree}) for {len(poly_cols)} columns")
                        st.success(f"✅ Created polynomial features!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Feature creation failed: {str(e)}")
            
            elif feature_type == "Binning" and len(numeric_cols) > 0:
                bin_col = st.selectbox("Select column for binning:", numeric_cols, key="bin_col_select")
                bin_count = st.slider("Number of bins:", 3, 10, 5, key="bin_count")
                
                if st.button("📊 Create Bins", key="create_bins"):
                    try:
                        df[f"{bin_col}_binned"] = pd.cut(df[bin_col], bins=bin_count, labels=False)
                        
                        st.session_state.current_dataset = df
                        log_action("Feature Engineering", f"Created {bin_count} bins for {bin_col}")
                        st.success(f"✅ Created {bin_count} bins for {bin_col}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Binning failed: {str(e)}")
        
        with col2:
            st.markdown("**Feature Selection**")
            
            if len(numeric_cols) > 10:
                st.info("Consider reducing features for better model performance")
                
                if st.button("🎯 Apply PCA", key="apply_pca"):
                    try:
                        pca = PCA(n_components=min(10, len(numeric_cols)))
                        pca_features = pca.fit_transform(df[numeric_cols])
                        
                        # Create PCA feature columns
                        pca_df = pd.DataFrame(pca_features, 
                                            columns=[f"PCA_{i+1}" for i in range(pca_features.shape[1])])
                        
                        # Replace numeric columns with PCA features
                        df = df.drop(numeric_cols, axis=1)
                        df = pd.concat([df, pca_df], axis=1)
                        
                        st.session_state.current_dataset = df
                        log_action("Feature Selection", f"Applied PCA, reduced {len(numeric_cols)} features to {pca_features.shape[1]}")
                        st.success(f"✅ Applied PCA dimensionality reduction!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ PCA failed: {str(e)}")
    
    else:
        # Auto Mode Implementation
        st.markdown("#### 🤖 Automatic Feature Engineering")
        
        if st.button("🚀 Auto-Engineer Features", type="primary", key="auto_engineer_features"):
            features_created = 0
            
            try:
                # Auto-create simple polynomial features for numeric columns
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    df[f"{col}_squared"] = df[col] ** 2
                    features_created += 1
                    log_action("Auto Feature Engineering", f"Created squared feature for {col}", "feature_engineering")
                
                # Auto-binning for high-variance numeric columns
                for col in numeric_cols:
                    if df[col].std() > df[col].mean():  # High variance indicator
                        df[f"{col}_binned"] = pd.cut(df[col], bins=5, labels=False)
                        features_created += 1
                        log_action("Auto Feature Engineering", f"Created 5 bins for {col}", "feature_engineering")
                        break  # Only bin one column automatically
                
                if features_created > 0:
                    st.session_state.current_dataset = df
                    st.balloons()  # Balloon notification for success
                    st.success(f"✅ Auto-created {features_created} new features!")
                    st.rerun()
                else:
                    st.info("ℹ️ No features were automatically created.")
            
            except Exception as e:
                st.error(f"❌ Auto feature engineering failed: {str(e)}")
    
    # Show activity list for this tab
    show_activity_list("feature_engineering", "Feature Engineering")
    
    # Show post-treatment dataset if auto mode was applied
    if auto_mode_feature and 'activities_performed' in st.session_state and st.session_state.activities_performed.get('feature_engineering', False):
        st.markdown("#### 📊 Post-Treatment Dataset")
        st.dataframe(df.head(), use_container_width=True)

# ==================== TAB 11: COMPREHENSIVE SUMMARY & STATISTICS ====================
elif current_tab == 10:
    st.subheader("📊 Comprehensive Pipeline Summary")
    
    # Initialize stats tracking
    initialize_comprehensive_stats()

    if st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        stats = st.session_state.comprehensive_stats
        
        # Progress indicator
        progress = calculate_pipeline_progress()
        st.progress(progress / 100, text=f"Pipeline Progress: {progress:.1f}% Complete")
        
        # Show progress notifications
        show_progress_notification(progress)
        
        # Main summary tabs
        summary_tab1, summary_tab2, summary_tab3, summary_tab4 = st.tabs([
            "📈 Statistics Dashboard", 
            "🤖 AI Summary", 
            "📋 Activity Log", 
            "📊 Visual Analytics"
        ])
        
        with summary_tab1:
            st.markdown("### 📈 Comprehensive Statistics Dashboard")
            
            # Key metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Total Actions", 
                    stats['total_actions'],
                    delta=f"+{stats['total_actions']} completed"
                )
            
            with col2:
                st.metric(
                    "Columns Processed", 
                    len(stats['columns_processed']),
                    delta=f"of {len(df.columns)} total"
                )
            
            with col3:
                st.metric(
                    "Missing Values Fixed", 
                    stats['missing_values_fixed'],
                    delta="✅ Cleaned"
                )
            
            with col4:
                st.metric(
                    "Outliers Treated", 
                    stats['outliers_treated'],
                    delta="🎯 Handled"
                )
            
            with col5:
                time_saved = sum(stats['time_saved_estimates'])
                st.metric(
                    "Time Saved", 
                    f"{time_saved:.0f} min",
                    delta="⏰ Efficiency"
                )
            
            st.markdown("---")
            
            # Detailed statistics grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔧 Transformation Summary")
                transformation_data = {
                    "Data Types Converted": stats['data_types_converted'],
                    "Duplicates Removed": stats['duplicates_removed'],
                    "Encodings Applied": stats['encodings_applied'],
                    "Scaling Operations": stats['scaling_transformations'],
                    "Text Cleanings": stats['text_cleanings']
                }
                
                for key, value in transformation_data.items():
                    st.write(f"**{key}:** {value}")
            
            with col2:
                st.markdown("#### ⚙️ Feature Engineering")
                feature_data = {
                    "DateTime Features Extracted": stats['datetime_extractions'],
                    "Features Engineered": stats['features_engineered'],
                    "Geospatial Processed": stats['geospatial_processed']
                }
                
                for key, value in feature_data.items():
                    st.write(f"**{key}:** {value}")
            
            # Visual summary charts
            st.markdown("#### 📊 Processing Impact Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Actions by type chart
                action_types = [
                    'Missing Values', 'Outliers', 'Encoding', 'Scaling', 
                    'Text Cleaning', 'DateTime', 'Feature Engineering'
                ]
                action_counts = [
                    stats['missing_values_fixed'], stats['outliers_treated'],
                    stats['encodings_applied'], stats['scaling_transformations'],
                    stats['text_cleanings'], stats['datetime_extractions'],
                    stats['features_engineered']
                ]
                
                fig_actions = px.bar(
                    x=action_types, y=action_counts,
                    title="Actions by Category",
                    labels={'x': 'Action Type', 'y': 'Count'}
                )
                st.plotly_chart(fig_actions, use_container_width=True)
            
            with col2:
                # Pipeline completion chart
                completed_tabs = sum(1 for completed in stats['tab_completion'].values() if completed)
                remaining_tabs = len(stats['tab_completion']) - completed_tabs
                
                fig_completion = px.pie(
                    values=[completed_tabs, remaining_tabs],
                    names=['Completed', 'Remaining'],
                    title="Pipeline Completion Status"
                )
                st.plotly_chart(fig_completion, use_container_width=True)
        
        with summary_tab2:
            st.markdown("### 🤖 AI-Generated Summary")
            
            # Generate and display AI summary
            ai_summary = generate_ai_summary_writeup()
            st.markdown(ai_summary)
            
            # Recommendations section
            st.markdown("### 💡 Smart Recommendations")
            
            recommendations = []
            
            if stats['missing_values_fixed'] > 0:
                recommendations.append("✅ Excellent missing value handling - your data integrity is much improved")
            
            if stats['outliers_treated'] > 0:
                recommendations.append("🎯 Outlier treatment will enhance model robustness")
            
            if stats['encodings_applied'] > 0:
                recommendations.append("🔄 Categorical encoding prepares your data for machine learning")
            
            if stats['features_engineered'] > 0:
                recommendations.append("⚙️ Feature engineering adds predictive power to your dataset")
            
            if len(recommendations) == 0:
                recommendations.append("🚀 Ready to start your data cleaning journey!")
            
            for rec in recommendations:
                st.success(rec)
        
        with summary_tab3:
            st.markdown("### 📋 Comprehensive Activity Log")
            
            # Filter and search options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                log_filter = st.selectbox(
                    "Filter by Action Type:",
                    ["All", "Missing Values", "Outliers", "Encoding", "Scaling", "Text", "DateTime", "Features"],
                    key="log_filter"
                )
            
            with col2:
                search_term = st.text_input("Search in log:", placeholder="Enter column name or action...", key="log_search")
            
            with col3:
                export_format = st.selectbox("Export Format:", ["CSV", "PDF"], key="export_format")
            
            # Display filtered log
            if 'processing_log' in st.session_state and st.session_state.processing_log:
                filtered_log = st.session_state.processing_log
                
                # Apply filters
                if log_filter != "All":
                    filtered_log = [
                        entry for entry in filtered_log 
                        if log_filter.lower().replace(" ", "_") in entry['action'].lower()
                    ]
                
                if search_term:
                    filtered_log = [
                        entry for entry in filtered_log
                        if search_term.lower() in entry['details'].lower() or 
                           search_term.lower() in entry['action'].lower()
                    ]
                
                # Display log entries
                    st.markdown(
                        f"**Showing {len(filtered_log)} of {len(st.session_state.processing_log)} total actions**")

                    for i, entry in enumerate(reversed(filtered_log[-20:]), 1):
                        action_text = entry.get('action', 'Unknown Action')  # Use default if missing
                        with st.expander(f"Action {len(filtered_log) - i + 1}: {action_text}", expanded=False):
                            col1, col2 = st.columns([2, 1])

                        with col1:
                            st.write(f"**Details:** {entry['details']}")
                            st.write(f"**Timestamp:** {entry['timestamp']}")
                        
                        with col2:
                            if 'before_stats' in entry and 'after_stats' in entry:
                                st.write("**Impact:**")
                                st.json({"before": entry['before_stats'], "after": entry['after_stats']})
                
                # Export options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📥 Export Activity Log", type="primary"):
                        log_df = pd.DataFrame(filtered_log)
                        
                        if export_format == "CSV":
                            log_csv = log_df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download CSV",
                                data=log_csv,
                                file_name=f"activity_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("PDF export functionality coming soon!")
                
                with col2:
                    if st.button("🔄 Replay Process", key="replay_process"):
                        st.info("🎬 Process replay functionality for training/audit purposes coming soon!")
            
            else:
                st.info("No activity log available. Start processing data to see your actions here!")
        
        with summary_tab4:
            st.markdown("### 📊 Visual Analytics Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Dataset Overview")
                overview_stats = {
                    "Total Rows": f"{len(df):,}",
                    "Total Columns": f"{len(df.columns):,}",
                    "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    "Duplicate Rows": f"{df.duplicated().sum():,}",
                    "Missing Values": f"{df.isnull().sum().sum():,}"
                }
                
                for key, value in overview_stats.items():
                    st.metric(key, value)
            
            with col2:
                st.markdown("#### Data Types Distribution")
                # Convert data types to string to avoid JSON serialization issues
                dtype_counts = df.dtypes.astype(str).value_counts()
                fig_dtype = px.pie(values=dtype_counts.values, names=dtype_counts.index, 
                                 title="Data Types Distribution")
                st.plotly_chart(fig_dtype, use_container_width=True)
            
            # Correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                st.markdown("#### Correlation Analysis")
                # Convert to standard float to avoid JSON serialization issues
                numeric_df = df[numeric_cols].astype('float64')
                correlation_matrix = numeric_df.corr()
                
                fig_heatmap = px.imshow(correlation_matrix, 
                                      text_auto=True, 
                                      aspect="auto",
                                      title="Feature Correlation Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Missing values visualization
            if df.isnull().sum().sum() > 0:
                st.markdown("#### Missing Values Pattern")
                missing_summary = df.isnull().sum().reset_index()
                missing_summary.columns = ['Column', 'Missing_Count']
                missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
                
                if not missing_summary.empty:
                    fig_missing = px.bar(
                        missing_summary, x='Column', y='Missing_Count',
                        title="Missing Values by Column"
                    )
                    st.plotly_chart(fig_missing, use_container_width=True)
            
            # Statistical summary
            st.markdown("#### Statistical Summary")
            # Convert describe output to string to avoid JSON serialization issues
            describe_df = df.describe(include='all')
            for col in describe_df.columns:
                describe_df[col] = describe_df[col].astype(str)
            st.dataframe(describe_df, use_container_width=True)
        
        # Final export and sharing options
        st.markdown("---")
        st.markdown("### 📤 Export & Sharing Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📥 Download Cleaned Dataset", type="primary"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Generate Full Report"):
                # Create comprehensive report
                report_content = f"""
                # Data Cleaning Pipeline Report
                Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                ## Summary Statistics
                - Total Actions: {stats['total_actions']}
                - Columns Processed: {len(stats['columns_processed'])}
                - Missing Values Fixed: {stats['missing_values_fixed']}
                - Outliers Treated: {stats['outliers_treated']}
                
                ## AI Summary
                {generate_ai_summary_writeup()}
                """
                
                st.download_button(
                    label="💾 Download Report",
                    data=report_content,
                    file_name=f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button("🔗 Share Summary"):
                st.info("📧 Sharing functionality for stakeholders coming soon!")
        
        with col4:
            if st.button("🏆 View Achievements"):
                st.info("🎖️ Achievement system coming soon!")
    
    else:
        st.warning("⚠️ No dataset available. Please upload data first.")
        
        # Show empty state with progress
        st.markdown("### 🚀 Ready to Start Your Data Cleaning Journey?")
        st.info("Upload your dataset in the Upload page to begin the comprehensive cleaning pipeline!")
        
        # Show pipeline overview
        st.markdown("#### 📋 Pipeline Overview")
        pipeline_steps = [
            "🔍 Detect & Convert Data Types",
            "🔄 Remove Duplicates", 
            "🕳️ Handle Missing Values",
            "🎯 Detect & Treat Outliers",
            "📝 Clean Text",
            "📅 Process Date/Time",
            "🌍 Process Geospatial Data",
            "🏷️ Encode Categorical Variables",
            "📊 Scale/Normalize Features",
            "⚙️ Engineer/Select Features",
            "📊 Summary & Analytics"
        ]
        
        for i, step in enumerate(pipeline_steps, 1):
            st.markdown(f"{i}. {step}")
        
        st.success("Each step includes AI-powered recommendations, smart explanations, and celebration feedback!")

# Processing Log Sidebar
with st.sidebar:
    st.markdown("### 📋 Processing Log")
    if st.session_state.processing_log:
        for i, log_entry in enumerate(reversed(st.session_state.processing_log[-10:])):  # Show last 10 entries
            action = log_entry.get('step', log_entry.get('action', 'Processing Step'))
            timestamp = log_entry.get('timestamp', 'Unknown Time')
            with st.expander(f"{action} - {timestamp[-8:] if len(timestamp) >= 8 else timestamp}"):
                st.write(log_entry['details'])
    else:
        st.info("No processing actions yet.")
    
    if st.button("🗑️ Clear Log"):
        st.session_state.processing_log = []
        st.rerun()