import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

def create_missing_heatmap(df):
    """Create a heatmap showing missing values pattern"""
    missing_data = df.isnull()
    
    # Create heatmap
    fig = px.imshow(
        missing_data.T,
        title="Missing Values Heatmap",
        labels=dict(x="Row Index", y="Columns", color="Missing"),
        color_continuous_scale=["white", "red"],
        aspect="auto"
    )
    
    fig.update_layout(
        height=max(400, len(df.columns) * 20),
        xaxis_title="Row Index",
        yaxis_title="Columns"
    )
    
    return fig

def create_correlation_matrix(df, method='pearson'):
    """Create correlation matrix visualization"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return None
    
    corr_matrix = numeric_df.corr(method=method)
    
    fig = px.imshow(
        corr_matrix,
        title=f"Correlation Matrix ({method.title()})",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )
    
    # Add text annotations
    fig.update_traces(
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10}
    )
    
    return fig

def create_distribution_plots(df, columns=None):
    """Create distribution plots for numeric columns"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if len(columns) == 0:
        return None
    
    # Create subplots
    n_cols = min(3, len(columns))
    n_rows = (len(columns) - 1) // n_cols + 1
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=columns,
        vertical_spacing=0.1
    )
    
    for i, col in enumerate(columns):
        row = i // n_cols + 1
        col_pos = i % n_cols + 1
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                nbinsx=30,
                showlegend=False
            ),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        title="Distribution of Numeric Variables",
        height=300 * n_rows,
        showlegend=False
    )
    
    return fig

def create_outlier_boxplots(df, columns=None):
    """Create box plots to visualize outliers"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if len(columns) == 0:
        return None
    
    fig = go.Figure()
    
    for col in columns:
        fig.add_trace(go.Box(
            y=df[col],
            name=col,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Box Plots - Outlier Detection",
        yaxis_title="Values",
        height=500
    )
    
    return fig

def create_data_preview(df, sample_size=10):
    """Create a preview of the dataset"""
    preview_data = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': dict(df.dtypes),
        'sample': df.head(sample_size),
        'missing_count': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum()
    }
    
    return preview_data

def create_upload_summary(df):
    """Create summary visualization for uploaded data"""
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicates': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    return summary

def create_before_after_comparison(before_data, after_data, column_name, operation):
    """Create before/after comparison visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Before {operation}", f"After {operation}"],
        shared_yaxes=True
    )
    
    # Before histogram
    fig.add_trace(
        go.Histogram(
            x=before_data,
            name="Before",
            nbinsx=30,
            opacity=0.7,
            marker_color="blue"
        ),
        row=1, col=1
    )
    
    # After histogram
    fig.add_trace(
        go.Histogram(
            x=after_data,
            name="After",
            nbinsx=30,
            opacity=0.7,
            marker_color="green"
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=f"Before/After Comparison: {column_name}",
        height=400,
        showlegend=False
    )
    
    return fig

def create_categorical_distribution(df, column, max_categories=20):
    """Create distribution plot for categorical data"""
    value_counts = df[column].value_counts().head(max_categories)
    
    fig = px.bar(
        x=value_counts.values,
        y=value_counts.index,
        orientation='h',
        title=f"Distribution of {column}",
        labels={'x': 'Count', 'y': column}
    )
    
    fig.update_layout(height=max(400, len(value_counts) * 25))
    
    return fig

def create_time_series_plot(df, date_column, value_column):
    """Create time series plot"""
    fig = px.line(
        df,
        x=date_column,
        y=value_column,
        title=f"Time Series: {value_column} over {date_column}"
    )
    
    fig.update_layout(
        height=400,
        xaxis_title=date_column,
        yaxis_title=value_column
    )
    
    return fig

def create_feature_importance_plot(importance_scores, feature_names):
    """Create feature importance visualization"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title="Feature Importance"
    )
    
    fig.update_layout(height=max(400, len(feature_names) * 25))
    
    return fig

def create_data_quality_dashboard(df):
    """Create comprehensive data quality dashboard"""
    # Calculate quality metrics
    missing_by_column = df.isnull().sum()
    missing_pct_by_column = (missing_by_column / len(df)) * 100
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Missing Values by Column",
            "Data Types Distribution",
            "Column Cardinality",
            "Memory Usage by Column"
        ],
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Missing values plot
    fig.add_trace(
        go.Bar(
            x=missing_by_column.index,
            y=missing_by_column.values,
            name="Missing Count",
            marker_color="red"
        ),
        row=1, col=1
    )
    
    # Data types pie chart
    dtype_counts = df.dtypes.value_counts()
    fig.add_trace(
        go.Pie(
            labels=dtype_counts.index.astype(str),
            values=dtype_counts.values,
            name="Data Types"
        ),
        row=1, col=2
    )
    
    # Column cardinality
    cardinality = [df[col].nunique() for col in df.columns]
    fig.add_trace(
        go.Bar(
            x=df.columns,
            y=cardinality,
            name="Unique Values",
            marker_color="blue"
        ),
        row=2, col=1
    )
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True)
    fig.add_trace(
        go.Bar(
            x=memory_usage.index,
            y=memory_usage.values / 1024,  # Convert to KB
            name="Memory (KB)",
            marker_color="green"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Data Quality Dashboard",
        height=800,
        showlegend=False
    )
    
    return fig
