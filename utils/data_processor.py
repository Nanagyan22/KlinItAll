import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Core data processing utilities for KlinItAll
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.processing_history = []
    
    def add_to_history(self, operation, details):
        """Add operation to processing history"""
        self.processing_history.append({
            'operation': operation,
            'details': details,
            'timestamp': pd.Timestamp.now()
        })
    
    def detect_data_types(self, df):
        """Automatically detect and suggest data types"""
        suggestions = {}
        
        for col in df.columns:
            # Check for numeric conversion possibilities
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    suggestions[col] = 'numeric'
                except:
                    # Check for datetime
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        suggestions[col] = 'datetime'
                    except:
                        # Check if it's categorical
                        unique_ratio = df[col].nunique() / len(df)
                        if unique_ratio < 0.1:
                            suggestions[col] = 'categorical'
                        else:
                            suggestions[col] = 'text'
            else:
                suggestions[col] = 'keep_current'
        
        return suggestions
    
    def handle_missing_values(self, df, strategy='auto', columns=None):
        """Handle missing values with various strategies"""
        if columns is None:
            columns = df.columns
        
        result_df = df.copy()
        
        for col in columns:
            if df[col].isnull().sum() == 0:
                continue
                
            if strategy == 'auto':
                # Automatic strategy selection
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric columns - use median for robustness
                    strategy_used = 'median'
                    result_df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Categorical columns - use mode
                    strategy_used = 'mode'
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        result_df[col].fillna(mode_value[0], inplace=True)
                    else:
                        result_df[col].fillna('Unknown', inplace=True)
            else:
                # Use specified strategy
                if strategy == 'mean' and df[col].dtype in ['int64', 'float64']:
                    result_df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median' and df[col].dtype in ['int64', 'float64']:
                    result_df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        result_df[col].fillna(mode_value[0], inplace=True)
                elif strategy == 'drop':
                    result_df.dropna(subset=[col], inplace=True)
            
            self.add_to_history('missing_values', f"Applied {strategy} to {col}")
        
        return result_df
    
    def detect_outliers(self, df, method='iqr', columns=None):
        """Detect outliers using various methods"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > 3].index
            
            elif method == 'isolation_forest':
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(df[[col]])
                outliers[col] = df.index[outlier_labels == -1]
        
        return outliers
    
    def encode_categorical(self, df, method='auto', columns=None):
        """Encode categorical variables"""
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns
        
        result_df = df.copy()
        
        for col in columns:
            if method == 'auto':
                # Choose encoding based on cardinality
                cardinality = df[col].nunique()
                if cardinality <= 10:
                    # Low cardinality - use one-hot encoding
                    dummies = pd.get_dummies(df[col], prefix=col)
                    result_df = pd.concat([result_df.drop(col, axis=1), dummies], axis=1)
                    self.add_to_history('encoding', f"One-hot encoded {col}")
                else:
                    # High cardinality - use label encoding
                    le = LabelEncoder()
                    result_df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                    self.add_to_history('encoding', f"Label encoded {col}")
            
            elif method == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                result_df = pd.concat([result_df.drop(col, axis=1), dummies], axis=1)
            
            elif method == 'label':
                le = LabelEncoder()
                result_df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        return result_df
    
    def scale_features(self, df, method='standard', columns=None):
        """Scale numerical features"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        result_df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
            result_df[columns] = scaler.fit_transform(df[columns])
            self.scalers['standard'] = scaler
        
        self.add_to_history('scaling', f"Applied {method} scaling to {len(columns)} columns")
        return result_df
    
    def create_features(self, df):
        """Automated feature engineering"""
        result_df = df.copy()
        
        # Create interaction features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Create interaction feature
                    result_df[f"{col1}_{col2}_interaction"] = df[col1] * df[col2]
        
        # Create binned features
        for col in numeric_cols:
            result_df[f"{col}_binned"] = pd.cut(df[col], bins=5, labels=False)
        
        self.add_to_history('feature_engineering', f"Created interaction and binned features")
        return result_df
    
    def detect_duplicates(self, df, subset=None):
        """Detect duplicate rows"""
        if subset is None:
            duplicates = df.duplicated()
        else:
            duplicates = df.duplicated(subset=subset)
        
        return df[duplicates]
    
    def validate_data_quality(self, df):
        """Comprehensive data quality assessment"""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicates': df.duplicated().sum(),
            'data_types': dict(df.dtypes),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }
        
        return quality_report
    
    def get_processing_history(self):
        """Return processing history"""
        return self.processing_history
    
    def reset_history(self):
        """Reset processing history"""
        self.processing_history = []
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
