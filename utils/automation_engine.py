import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
import streamlit as st

class AutomationEngine:
    def __init__(self):
        self.recommendations = []
        self.automation_history = []
    
    def analyze_and_recommend(self, df):
        """Analyze data and generate automated recommendations"""
        if df is None or df.empty:
            return []
        
        recommendations = []
        
        # Missing values analysis
        missing_recommendations = self._analyze_missing_values(df)
        recommendations.extend(missing_recommendations)
        
        # Outlier analysis
        outlier_recommendations = self._analyze_outliers(df)
        recommendations.extend(outlier_recommendations)
        
        # Data type analysis
        dtype_recommendations = self._analyze_data_types(df)
        recommendations.extend(dtype_recommendations)
        
        # Duplicate analysis
        duplicate_recommendations = self._analyze_duplicates(df)
        recommendations.extend(duplicate_recommendations)
        
        # Feature scaling analysis
        scaling_recommendations = self._analyze_scaling_needs(df)
        recommendations.extend(scaling_recommendations)
        
        # Categorical encoding analysis
        encoding_recommendations = self._analyze_encoding_needs(df)
        recommendations.extend(encoding_recommendations)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_missing_values(self, df):
        """Analyze missing values and recommend imputation strategies"""
        recommendations = []
        missing_cols = df.columns[df.isnull().any()]
        
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if missing_pct > 80:
                recommendations.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': 'high',
                    'issue': f'Column has {missing_pct:.1f}% missing values',
                    'recommendation': 'Consider dropping this column',
                    'action': 'drop_column',
                    'auto_fixable': True,
                    'estimated_time_saved': '10 minutes'
                })
            elif missing_pct > 50:
                recommendations.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': 'high',
                    'issue': f'Column has {missing_pct:.1f}% missing values',
                    'recommendation': 'Use advanced imputation (KNN or model-based)',
                    'action': 'knn_impute',
                    'auto_fixable': True,
                    'estimated_time_saved': '30 minutes'
                })
            else:
                if df[col].dtype in ['int64', 'float64']:
                    if abs(df[col].skew()) > 1:
                        strategy = 'median'
                    else:
                        strategy = 'mean'
                else:
                    strategy = 'most_frequent'
                
                recommendations.append({
                    'type': 'missing_values',
                    'column': col,
                    'severity': 'medium',
                    'issue': f'Column has {missing_pct:.1f}% missing values',
                    'recommendation': f'Impute using {strategy}',
                    'action': f'impute_{strategy}',
                    'auto_fixable': True,
                    'estimated_time_saved': '15 minutes'
                })
        
        return recommendations
    
    def _analyze_outliers(self, df):
        """Analyze outliers and recommend treatment"""
        recommendations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            # IQR method for outlier detection
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR))
            outlier_pct = (outliers.sum() / len(col_data)) * 100
            
            if outlier_pct > 10:
                recommendations.append({
                    'type': 'outliers',
                    'column': col,
                    'severity': 'high',
                    'issue': f'Column has {outlier_pct:.1f}% outliers',
                    'recommendation': 'Apply robust scaling or winsorization',
                    'action': 'winsorize',
                    'auto_fixable': True,
                    'estimated_time_saved': '20 minutes'
                })
            elif outlier_pct > 5:
                recommendations.append({
                    'type': 'outliers',
                    'column': col,
                    'severity': 'medium',
                    'issue': f'Column has {outlier_pct:.1f}% outliers',
                    'recommendation': 'Consider outlier treatment',
                    'action': 'flag_outliers',
                    'auto_fixable': True,
                    'estimated_time_saved': '15 minutes'
                })
        
        return recommendations
    
    def _analyze_data_types(self, df):
        """Analyze data types and recommend conversions"""
        recommendations = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric values are stored as strings
                sample_values = df[col].dropna().head(100)
                numeric_count = 0
                
                for val in sample_values:
                    if isinstance(val, str):
                        # Remove common numeric formatting
                        clean_val = str(val).replace(',', '').replace('$', '').replace('%', '').strip()
                        try:
                            float(clean_val)
                            numeric_count += 1
                        except:
                            pass
                
                if numeric_count > len(sample_values) * 0.8:
                    recommendations.append({
                        'type': 'data_types',
                        'column': col,
                        'severity': 'medium',
                        'issue': 'Numeric data stored as text',
                        'recommendation': 'Convert to numeric type',
                        'action': 'convert_to_numeric',
                        'auto_fixable': True,
                        'estimated_time_saved': '10 minutes'
                    })
        
        return recommendations
    
    def _analyze_duplicates(self, df):
        """Analyze duplicate records"""
        recommendations = []
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        if duplicate_count > 0:
            severity = 'high' if duplicate_pct > 10 else 'medium'
            recommendations.append({
                'type': 'duplicates',
                'column': 'all',
                'severity': severity,
                'issue': f'{duplicate_count} duplicate records ({duplicate_pct:.1f}%)',
                'recommendation': 'Remove duplicate records',
                'action': 'remove_duplicates',
                'auto_fixable': True,
                'estimated_time_saved': '5 minutes'
            })
        
        return recommendations
    
    def _analyze_scaling_needs(self, df):
        """Analyze if features need scaling"""
        recommendations = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) <= 1:
            return recommendations
        
        # Check scale differences
        scales = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                scales[col] = col_data.std()
        
        if scales:
            max_scale = max(scales.values())
            min_scale = min(scales.values())
            
            if max_scale / min_scale > 10:  # Significant scale difference
                recommendations.append({
                    'type': 'scaling',
                    'column': 'numeric_features',
                    'severity': 'medium',
                    'issue': 'Features have different scales',
                    'recommendation': 'Apply feature scaling',
                    'action': 'auto_scale',
                    'auto_fixable': True,
                    'estimated_time_saved': '25 minutes'
                })
        
        return recommendations
    
    def _analyze_encoding_needs(self, df):
        """Analyze categorical encoding needs"""
        recommendations = []
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_count = df[col].nunique()
            
            if unique_count > 1:
                if unique_count <= 10:
                    method = 'one-hot encoding'
                    action = 'onehot_encode'
                else:
                    method = 'label encoding'
                    action = 'label_encode'
                
                recommendations.append({
                    'type': 'encoding',
                    'column': col,
                    'severity': 'low',
                    'issue': f'Categorical column needs encoding ({unique_count} categories)',
                    'recommendation': f'Apply {method}',
                    'action': action,
                    'auto_fixable': True,
                    'estimated_time_saved': '15 minutes'
                })
        
        return recommendations
    
    def apply_recommendation(self, df, recommendation, data_processor):
        """Apply a specific recommendation"""
        action = recommendation['action']
        column = recommendation['column']
        
        try:
            if action == 'drop_column':
                df = df.drop(columns=[column])
                success_msg = f"Dropped column '{column}'"
            
            elif action.startswith('impute_'):
                strategy = action.split('_')[1]
                data_processor.impute_missing_values(strategy=strategy, columns=[column])
                df = data_processor.processed_data
                success_msg = f"Imputed missing values in '{column}' using {strategy}"
            
            elif action == 'knn_impute':
                data_processor.impute_missing_values(strategy='knn', columns=[column])
                df = data_processor.processed_data
                success_msg = f"Applied KNN imputation to '{column}'"
            
            elif action == 'remove_duplicates':
                original_len = len(df)
                df = df.drop_duplicates()
                removed = original_len - len(df)
                success_msg = f"Removed {removed} duplicate records"
            
            elif action == 'convert_to_numeric':
                df[column] = pd.to_numeric(df[column].astype(str).str.replace(r'[,$%]', '', regex=True), errors='coerce')
                success_msg = f"Converted '{column}' to numeric type"
            
            elif action == 'auto_scale':
                data_processor.scale_features(method='auto')
                df = data_processor.processed_data
                success_msg = "Applied automatic feature scaling"
            
            elif action in ['onehot_encode', 'label_encode']:
                method = 'onehot' if action == 'onehot_encode' else 'label'
                data_processor.encode_categorical(method=method, columns=[column])
                df = data_processor.processed_data
                success_msg = f"Applied {method} encoding to '{column}'"
            
            else:
                return df, False, "Unknown action"
            
            # Log the automation step
            self.automation_history.append({
                'timestamp': pd.Timestamp.now(),
                'recommendation': recommendation,
                'success': True,
                'message': success_msg
            })
            
            return df, True, success_msg
        
        except Exception as e:
            error_msg = f"Failed to apply {action}: {str(e)}"
            self.automation_history.append({
                'timestamp': pd.Timestamp.now(),
                'recommendation': recommendation,
                'success': False,
                'message': error_msg
            })
            return df, False, error_msg
    
    def apply_all_recommendations(self, df, recommendations, data_processor):
        """Apply all auto-fixable recommendations"""
        results = []
        current_df = df.copy()
        
        # Sort by severity to handle critical issues first
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_recommendations = sorted(recommendations, 
                                      key=lambda x: severity_order.get(x['severity'], 3))
        
        for rec in sorted_recommendations:
            if rec['auto_fixable']:
                current_df, success, message = self.apply_recommendation(current_df, rec, data_processor)
                results.append({
                    'recommendation': rec,
                    'success': success,
                    'message': message
                })
        
        return current_df, results
    
    def calculate_time_saved(self, recommendations):
        """Calculate estimated time saved through automation"""
        total_minutes = 0
        
        for rec in recommendations:
            if rec['auto_fixable']:
                time_str = rec.get('estimated_time_saved', '0 minutes')
                minutes = int(time_str.split()[0]) if time_str.split()[0].isdigit() else 0
                total_minutes += minutes
        
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    
    def get_automation_summary(self, recommendations):
        """Get summary of automation capabilities"""
        total_issues = len(recommendations)
        auto_fixable = len([r for r in recommendations if r['auto_fixable']])
        high_severity = len([r for r in recommendations if r['severity'] == 'high'])
        
        return {
            'total_issues': total_issues,
            'auto_fixable': auto_fixable,
            'high_severity': high_severity,
            'automation_coverage': (auto_fixable / total_issues * 100) if total_issues > 0 else 0,
            'estimated_time_saved': self.calculate_time_saved(recommendations)
        }
    
    def generate_data_insights(self, df):
        """Generate AI-powered data quality insights"""
        insights = []
        
        # Missing values insights
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 20:
            insights.append({
                'severity': 'high',
                'message': f"Dataset has {missing_pct:.1f}% missing values - significant data quality issues detected"
            })
        elif missing_pct > 5:
            insights.append({
                'severity': 'medium',
                'message': f"Dataset has {missing_pct:.1f}% missing values - moderate cleaning required"
            })
        
        # Duplicate insights
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        if duplicate_pct > 10:
            insights.append({
                'severity': 'high',
                'message': f"High duplicate rate: {duplicate_pct:.1f}% of records are duplicates"
            })
        
        # Data type insights
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            # Check if it might be numeric
            sample_vals = df[col].dropna().head(50)
            numeric_count = 0
            for val in sample_vals:
                try:
                    float(str(val).replace(',', '').replace('$', '').strip())
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(sample_vals) * 0.8:
                insights.append({
                    'severity': 'medium',
                    'message': f"Column '{col}' appears to contain numeric data stored as text"
                })
        
        # Cardinality insights
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                insights.append({
                    'severity': 'low',
                    'message': f"Column '{col}' has very high cardinality ({unique_ratio:.1%}) - might be an identifier"
                })
        
        return insights
    
    def detect_all_issues(self, df):
        """Detect all data quality issues for the overview page"""
        issues = []
        issue_id = 1
        
        # Missing values issues
        missing_cols = df.columns[df.isnull().any()]
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            severity = 'high' if missing_pct > 50 else 'medium' if missing_pct > 10 else 'low'
            
            issues.append({
                'id': issue_id,
                'category': 'Missing Values',
                'title': f'Missing values in {col}',
                'description': f'Column has {missing_pct:.1f}% missing values',
                'recommendation': 'Apply appropriate imputation method',
                'severity': severity
            })
            issue_id += 1
        
        # Duplicate issues
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append({
                'id': issue_id,
                'category': 'Duplicates',
                'title': 'Duplicate records detected',
                'description': f'Found {duplicate_count} duplicate rows',
                'recommendation': 'Remove or merge duplicate records',
                'severity': 'high' if duplicate_count > len(df) * 0.1 else 'medium'
            })
            issue_id += 1
        
        # Data type issues
        for col in df.select_dtypes(include=['object']).columns:
            # Check for potential numeric columns
            sample_vals = df[col].dropna().head(50)
            numeric_count = 0
            for val in sample_vals:
                try:
                    float(str(val).replace(',', '').replace('$', '').strip())
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(sample_vals) * 0.8:
                issues.append({
                    'id': issue_id,
                    'category': 'Data Types',
                    'title': f'Incorrect data type in {col}',
                    'description': 'Numeric data stored as text',
                    'recommendation': 'Convert to appropriate numeric type',
                    'severity': 'medium'
                })
                issue_id += 1
        
        return issues
    
    def detect_fuzzy_duplicates(self, df):
        """Simplified fuzzy duplicate detection"""
        # This is a simplified implementation
        # In a real scenario, you'd use libraries like fuzzywuzzy
        text_columns = df.select_dtypes(include=['object']).columns
        
        fuzzy_count = 0
        for col in text_columns[:2]:  # Limit to first 2 text columns for performance
            values = df[col].dropna().astype(str)
            # Simple approach: check for similar strings (same length, similar characters)
            for i, val1 in enumerate(values):
                for j, val2 in enumerate(values[i+1:], i+1):
                    if len(val1) == len(val2) and val1.lower() != val2.lower():
                        # Simple similarity check
                        similarity = sum(c1 == c2 for c1, c2 in zip(val1.lower(), val2.lower())) / len(val1)
                        if similarity > 0.8:
                            fuzzy_count += 1
                            break
        
        return min(fuzzy_count, 50)  # Cap at 50 for display purposes
    
    def recommend_imputation_methods(self, df, columns):
        """Recommend specific imputation methods for columns"""
        recommendations = {}
        
        for col in columns:
            col_data = df[col].dropna()
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            
            if df[col].dtype in ['int64', 'float64']:
                # Numeric column
                skewness = abs(stats.skew(col_data)) if len(col_data) > 0 else 0
                
                if missing_pct > 50:
                    method = 'KNN Imputation'
                    reason = 'High missing percentage requires advanced method'
                    confidence = 8
                elif skewness > 1:
                    method = 'Median Imputation'
                    reason = 'Data is skewed, median is more robust'
                    confidence = 9
                else:
                    method = 'Mean Imputation'
                    reason = 'Data is normally distributed'
                    confidence = 8
            else:
                # Categorical column
                if missing_pct > 30:
                    method = 'Create "Unknown" category'
                    reason = 'High missing percentage, separate category is safer'
                    confidence = 7
                else:
                    method = 'Mode Imputation'
                    reason = 'Use most frequent category'
                    confidence = 8
            
            recommendations[col] = {
                'method': method,
                'reason': reason,
                'confidence': confidence,
                'characteristics': f'{missing_pct:.1f}% missing'
            }
        
        return recommendations
    
    def recommend_scaling_methods(self, df, columns):
        """Recommend scaling methods for numeric columns"""
        recommendations = {}
        
        for col in columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue
            
            skewness = abs(stats.skew(col_data))
            
            # Check for outliers using IQR
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < Q1 - 1.5*IQR) | (col_data > Q3 + 1.5*IQR)).sum()
            has_outliers = outliers > len(col_data) * 0.05
            
            if skewness > 2:
                method = 'PowerTransformer'
                reason = 'Highly skewed data needs transformation'
                characteristics = f'Skewness: {skewness:.2f}'
                confidence = 9
            elif has_outliers and skewness > 1:
                method = 'RobustScaler'
                reason = 'Outliers present with moderate skew'
                characteristics = f'{outliers} outliers, Skewness: {skewness:.2f}'
                confidence = 8
            elif has_outliers:
                method = 'RobustScaler'
                reason = 'Outliers detected, robust scaling recommended'
                characteristics = f'{outliers} outliers detected'
                confidence = 9
            elif skewness > 1:
                method = 'QuantileTransformer'
                reason = 'Moderately skewed data'
                characteristics = f'Skewness: {skewness:.2f}'
                confidence = 7
            else:
                method = 'StandardScaler'
                reason = 'Well-behaved normal distribution'
                characteristics = 'Low skew, no outliers'
                confidence = 9
            
            recommendations[col] = {
                'method': method,
                'reason': reason,
                'characteristics': characteristics,
                'confidence': confidence
            }
        
        return recommendations
