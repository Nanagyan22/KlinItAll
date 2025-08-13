import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile
import io
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="History Export", page_icon="📚", layout="wide")

st.title("📚 History Export")
st.markdown("Complete audit trail, processing logs, and comprehensive export capabilities")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Initialize processing log if it doesn't exist
if 'processing_log' not in st.session_state:
    st.session_state.processing_log = []

# History and Export tabs
history_tabs = st.tabs(["📝 Processing History", "📊 Dataset Summary", "💾 Export Options", "📋 Reports"])

with history_tabs[0]:
    st.markdown("### 📝 Complete Processing History")
    
    if st.session_state.processing_log:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Detailed Processing Log")
            
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                action_filter = st.multiselect(
                    "Filter by action type:",
                    list(set(log.get('action', 'Unknown') for log in st.session_state.processing_log)),
                    help="Filter log entries by processing action type"
                )
            
            with filter_col2:
                date_filter = st.date_input("Filter by date:", value=None, help="Show only entries from specific date")
            
            # Apply filters
            filtered_log = st.session_state.processing_log
            
            if action_filter:
                filtered_log = [log for log in filtered_log if log.get('action') in action_filter]
            
            if date_filter:
                date_str = date_filter.strftime('%Y-%m-%d')
                filtered_log = [log for log in filtered_log if log.get('timestamp', '').startswith(date_str)]
            
            # Display log entries
            if filtered_log:
                for i, log_entry in enumerate(reversed(filtered_log)):
                    timestamp = log_entry.get('timestamp', 'Unknown time')
                    action = log_entry.get('action', 'Unknown action')
                    details = log_entry.get('details', 'No details available')
                    
                    with st.expander(f"📅 {timestamp} - {action}", expanded=(i < 3)):
                        st.markdown(f"**Action:** {action}")
                        st.markdown(f"**Time:** {timestamp}")
                        st.markdown(f"**Details:** {details}")
                        
                        # Add metadata if available
                        if 'metadata' in log_entry:
                            st.markdown("**Additional Info:**")
                            for key, value in log_entry['metadata'].items():
                                st.write(f"• {key}: {value}")
            else:
                st.info("No log entries match the current filters.")
        
        with col2:
            st.markdown("#### Processing Statistics")
            
            # Action type distribution
            action_counts = {}
            for log in st.session_state.processing_log:
                action = log.get('action', 'Unknown')
                action_counts[action] = action_counts.get(action, 0) + 1
            
            if action_counts:
                fig = px.pie(
                    values=list(action_counts.values()),
                    names=list(action_counts.keys()),
                    title="Processing Actions Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Timeline visualization
            if len(st.session_state.processing_log) > 1:
                st.markdown("#### Processing Timeline")
                
                # Extract dates for timeline
                dates = []
                actions = []
                
                for log in st.session_state.processing_log:
                    try:
                        timestamp = datetime.strptime(log.get('timestamp', ''), '%Y-%m-%d %H:%M:%S')
                        dates.append(timestamp)
                        actions.append(log.get('action', 'Unknown'))
                    except:
                        continue
                
                if dates:
                    timeline_df = pd.DataFrame({
                        'Date': dates,
                        'Action': actions,
                        'Count': [1] * len(dates)
                    })
                    
                    # Group by hour for better visualization
                    timeline_df['Hour'] = timeline_df['Date'].dt.floor('H')
                    hourly_counts = timeline_df.groupby('Hour').size()
                    
                    fig = px.line(
                        x=hourly_counts.index,
                        y=hourly_counts.values,
                        title="Processing Activity Over Time",
                        labels={'x': 'Time', 'y': 'Number of Actions'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Processing summary metrics
            st.markdown("#### Summary Metrics")
            
            total_actions = len(st.session_state.processing_log)
            st.metric("Total Actions", f"{total_actions:,}")
            
            if st.session_state.processing_log:
                # Calculate processing duration
                try:
                    first_action = datetime.strptime(st.session_state.processing_log[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
                    last_action = datetime.strptime(st.session_state.processing_log[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
                    duration = last_action - first_action
                    
                    st.metric("Processing Duration", f"{duration.total_seconds() / 60:.1f} min")
                except:
                    st.metric("Processing Duration", "Unknown")
                
                # Most frequent action
                if action_counts:
                    most_frequent_action = max(action_counts, key=action_counts.get)
                    st.metric("Most Frequent Action", most_frequent_action)
        
        # Export log options
        st.markdown("---")
        st.markdown("#### Export Processing Log")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Log as JSON"):
                log_json = json.dumps(st.session_state.processing_log, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=log_json,
                    file_name=f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Download Log as CSV"):
                log_df = pd.DataFrame(st.session_state.processing_log)
                csv = log_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("🗑️ Clear History"):
                if st.checkbox("I understand this will delete all processing history"):
                    st.session_state.processing_log = []
                    st.success("✅ Processing history cleared!")
                    st.rerun()
    
    else:
        st.info("No processing history available. Start by performing some data processing operations.")

with history_tabs[1]:
    st.markdown("### 📊 Comprehensive Dataset Summary")
    
    # Current dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    with col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Data type breakdown
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Data Type Distribution")
        
        type_counts = df.dtypes.astype(str).value_counts()
        
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Column Data Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed type breakdown
        st.markdown("#### Detailed Column Analysis")
        
        column_analysis = []
        for col in df.columns:
            col_data = df[col]
            
            analysis = {
                'Column': col,
                'Type': str(col_data.dtype),
                'Non-Null': f"{col_data.count():,}",
                'Null': f"{col_data.isnull().sum():,}",
                'Unique': f"{col_data.nunique():,}",
                'Memory (KB)': f"{col_data.memory_usage(deep=True) / 1024:.1f}"
            }
            
            column_analysis.append(analysis)
        
        analysis_df = pd.DataFrame(column_analysis)
        
        # Convert to strings to avoid Arrow issues
        for col in analysis_df.columns:
            analysis_df[col] = analysis_df[col].astype(str)
        
        st.dataframe(analysis_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Missing Value Analysis")
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
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
            st.success("✅ No missing values in the dataset!")
        
        # Feature engineering summary
        st.markdown("#### Feature Engineering Summary")
        
        # Detect engineered features
        engineered_features = {
            'Scaled': [col for col in df.columns if '_scaled' in col],
            'Polynomial': [col for col in df.columns if 'poly_' in col],
            'Interactions': [col for col in df.columns if '_x_' in col or 'interaction' in col.lower()],
            'Binned': [col for col in df.columns if '_binned' in col or '_bin_' in col],
            'Encoded': [col for col in df.columns if '_encoded' in col],
            'PCA': [col for col in df.columns if 'pca_' in col],
            'Mathematical': [col for col in df.columns if any(suffix in col for suffix in ['_sqrt', '_log', '_sq', '_exp'])],
            'Aggregated': [col for col in df.columns if 'agg_' in col]
        }
        
        engineering_summary = []
        for category, features in engineered_features.items():
            if features:
                engineering_summary.append({
                    'Category': category,
                    'Count': len(features),
                    'Percentage': f"{len(features) / len(df.columns) * 100:.1f}%"
                })
        
        if engineering_summary:
            eng_df = pd.DataFrame(engineering_summary)
            
            # Convert to strings
            for col in eng_df.columns:
                eng_df[col] = eng_df[col].astype(str)
            
            st.dataframe(eng_df, use_container_width=True)
        else:
            st.info("No engineered features detected in the dataset.")
    
    # Data quality metrics
    st.markdown("---")
    st.markdown("#### Data Quality Metrics")
    
    quality_col1, quality_col2, quality_col3 = st.columns(3)
    
    with quality_col1:
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / len(df)) * 100
        
        st.markdown("**Duplicate Analysis**")
        st.write(f"Duplicate Rows: {duplicate_rows:,} ({duplicate_pct:.1f}%)")
        
        # Unique row percentage
        unique_pct = (len(df.drop_duplicates()) / len(df)) * 100
        st.write(f"Unique Rows: {unique_pct:.1f}%")
    
    with quality_col2:
        # Numeric data quality
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        st.markdown("**Numeric Data Quality**")
        
        if len(numeric_cols) > 0:
            # Calculate outliers using IQR method
            total_outliers = 0
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                total_outliers += outliers
            
            outlier_pct = (total_outliers / (len(df) * len(numeric_cols))) * 100
            st.write(f"Outliers: {total_outliers:,} ({outlier_pct:.1f}%)")
            
            # Skewness analysis
            avg_skewness = df[numeric_cols].skew().abs().mean()
            st.write(f"Avg Absolute Skewness: {avg_skewness:.2f}")
        else:
            st.write("No numeric columns for analysis")
    
    with quality_col3:
        # Text data quality
        text_cols = df.select_dtypes(include=['object']).columns
        
        st.markdown("**Text Data Quality**")
        
        if len(text_cols) > 0:
            # Calculate text statistics
            empty_strings = 0
            total_text_cells = 0
            
            for col in text_cols:
                empty_strings += (df[col] == '').sum()
                total_text_cells += df[col].count()
            
            if total_text_cells > 0:
                empty_pct = (empty_strings / total_text_cells) * 100
                st.write(f"Empty Strings: {empty_strings:,} ({empty_pct:.1f}%)")
            
            # Average text length
            avg_lengths = []
            for col in text_cols:
                avg_len = df[col].astype(str).str.len().mean()
                if not pd.isna(avg_len):
                    avg_lengths.append(avg_len)
            
            if avg_lengths:
                st.write(f"Avg Text Length: {np.mean(avg_lengths):.1f} chars")
        else:
            st.write("No text columns for analysis")

with history_tabs[2]:
    st.markdown("### 💾 Comprehensive Export Options")
    
    export_col1, export_col2 = st.columns([1, 1])
    
    with export_col1:
        st.markdown('<div class="export-section">', unsafe_allow_html=True)
        st.markdown("#### 📄 Standard Data Formats")
        
        # CSV Export
        st.markdown("**CSV Export**")
        include_index = st.checkbox("Include row index", value=False, key="csv_index")
        
        csv_data = df.to_csv(index=include_index)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name=f"processed_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Excel Export
        st.markdown("**Excel Export**")
        
        if st.button("📊 Prepare Excel Export"):
            # Create Excel file with multiple sheets
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Processed_Data', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Memory Usage (MB)', 'Processing Actions'],
                    'Value': [
                        len(df),
                        len(df.columns),
                        df.isnull().sum().sum(),
                        df.memory_usage(deep=True).sum() / (1024**2),
                        len(st.session_state.processing_log)
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                # Column info sheet
                column_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Unique Values': df.nunique(),
                    'Missing Values': df.isnull().sum()
                })
                column_info.to_excel(writer, sheet_name='Column_Info', index=False)
            
            excel_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Excel File",
                data=excel_buffer.getvalue(),
                file_name=f"complete_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # JSON Export
        st.markdown("**JSON Export**")
        json_orient = st.selectbox("JSON orientation:", ["records", "index", "values", "table"], index=0)
        
        json_data = df.to_json(orient=json_orient, indent=2)
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with export_col2:
        st.markdown('<div class="export-section">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Advanced Export Options")
        
        # Parquet Export
        st.markdown("**High-Performance Formats**")
        
        if st.button("⚡ Export as Parquet"):
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            
            st.download_button(
                label="⚡ Download Parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        # Feather Export
        if st.button("🪶 Export as Feather"):
            feather_buffer = io.BytesIO()
            df.to_feather(feather_buffer)
            
            st.download_button(
                label="🪶 Download Feather",
                data=feather_buffer.getvalue(),
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.feather",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        # Subset Export
        st.markdown("**Subset Export**")
        
        # Column selection
        selected_columns = st.multiselect(
            "Select columns to export:",
            df.columns.tolist(),
            help="Choose specific columns for export"
        )
        
        # Row filtering
        if len(df) > 1000:
            export_rows = st.selectbox(
                "Number of rows to export:",
                ["All rows", "First 1000", "Last 1000", "Random 1000", "Custom range"],
                help="Choose how many rows to include"
            )
            
            if export_rows == "Custom range":
                start_row = st.number_input("Start row:", 0, len(df)-1, 0)
                end_row = st.number_input("End row:", start_row+1, len(df), min(start_row+1000, len(df)))
        else:
            export_rows = "All rows"
        
        if st.button("📤 Export Subset") and selected_columns:
            # Prepare subset
            if export_rows == "All rows":
                subset_df = df[selected_columns]
            elif export_rows == "First 1000":
                subset_df = df[selected_columns].head(1000)
            elif export_rows == "Last 1000":
                subset_df = df[selected_columns].tail(1000)
            elif export_rows == "Random 1000":
                subset_df = df[selected_columns].sample(n=min(1000, len(df)))
            elif export_rows == "Custom range":
                subset_df = df[selected_columns].iloc[start_row:end_row]
            
            csv_subset = subset_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Subset CSV",
                data=csv_subset,
                file_name=f"subset_{len(selected_columns)}cols_{len(subset_df)}rows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Complete Export Package
    st.markdown("---")
    st.markdown("#### 📦 Complete Export Package")
    
    st.info("Create a comprehensive ZIP package containing all data, reports, and processing history.")
    
    if st.button("📦 Create Complete Package", type="primary"):
        # Create ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add main dataset
            zip_file.writestr("processed_dataset.csv", df.to_csv(index=False))
            
            # Add processing log
            if st.session_state.processing_log:
                log_json = json.dumps(st.session_state.processing_log, indent=2)
                zip_file.writestr("processing_log.json", log_json)
                
                log_df = pd.DataFrame(st.session_state.processing_log)
                zip_file.writestr("processing_log.csv", log_df.to_csv(index=False))
            
            # Add dataset summary
            summary_info = f"""
Dataset Processing Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET OVERVIEW:
- Total Rows: {len(df):,}
- Total Columns: {len(df.columns):,}
- Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB
- Data Completeness: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

DATA TYPES:
{df.dtypes.value_counts().to_string()}

PROCESSING HISTORY:
- Total Actions: {len(st.session_state.processing_log)}
- Actions Performed:
"""
            
            # Add action summary
            if st.session_state.processing_log:
                action_counts = {}
                for log in st.session_state.processing_log:
                    action = log.get('action', 'Unknown')
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                for action, count in action_counts.items():
                    summary_info += f"  - {action}: {count}\n"
            
            zip_file.writestr("dataset_summary.txt", summary_info)
            
            # Add column information
            column_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Unique Values': df.nunique(),
                'Missing Values': df.isnull().sum()
            })
            zip_file.writestr("column_information.csv", column_info.to_csv(index=False))
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📦 Download Complete Package",
            data=zip_buffer.getvalue(),
            file_name=f"KlinItAll_Complete_Package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True
        )
        
        st.success("✅ Complete package prepared successfully!")

with history_tabs[3]:
    st.markdown("### 📋 Automated Reports")
    
    report_col1, report_col2 = st.columns([1, 1])
    
    with report_col1:
        st.markdown("#### 📊 Data Quality Report")
        
        if st.button("📊 Generate Data Quality Report", type="primary"):
            # Generate comprehensive data quality report
            report_content = f"""
# Data Quality Assessment Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report provides a comprehensive assessment of data quality for the processed dataset.

## Dataset Overview
- **Total Records:** {len(df):,}
- **Total Features:** {len(df.columns):,}
- **Memory Usage:** {df.memory_usage(deep=True).sum() / (1024**2):.1f} MB
- **Overall Completeness:** {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

## Data Quality Metrics

### 1. Completeness Analysis
"""
            
            # Missing value analysis
            missing_data = df.isnull().sum()
            missing_cols = missing_data[missing_data > 0]
            
            if len(missing_cols) > 0:
                report_content += f"\n**Columns with Missing Values:** {len(missing_cols)}\n\n"
                for col, missing_count in missing_cols.head(10).items():
                    pct = (missing_count / len(df)) * 100
                    report_content += f"- {col}: {missing_count:,} ({pct:.1f}%)\n"
            else:
                report_content += "\n✅ **No missing values detected**\n"
            
            # Uniqueness analysis
            report_content += f"\n### 2. Uniqueness Analysis\n"
            report_content += f"- **Duplicate Records:** {df.duplicated().sum():,} ({df.duplicated().sum()/len(df)*100:.1f}%)\n"
            report_content += f"- **Unique Records:** {len(df.drop_duplicates()):,} ({len(df.drop_duplicates())/len(df)*100:.1f}%)\n"
            
            # Data type analysis
            report_content += f"\n### 3. Data Type Distribution\n"
            type_counts = df.dtypes.value_counts()
            for dtype, count in type_counts.items():
                report_content += f"- {dtype}: {count} columns\n"
            
            # Quality issues and recommendations
            report_content += f"\n### 4. Quality Issues & Recommendations\n"
            
            issues_found = []
            
            # Check for high missing value columns
            high_missing = missing_data[missing_data > len(df) * 0.3]
            if len(high_missing) > 0:
                issues_found.append(f"🔴 **High Missing Values:** {len(high_missing)} columns have >30% missing values")
            
            # Check for duplicate rows
            if df.duplicated().sum() > 0:
                issues_found.append(f"🟡 **Duplicate Rows:** {df.duplicated().sum():,} duplicate rows detected")
            
            # Check for single-value columns
            single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if single_value_cols:
                issues_found.append(f"🟠 **Single Value Columns:** {len(single_value_cols)} columns have only one unique value")
            
            if issues_found:
                for issue in issues_found:
                    report_content += f"\n{issue}\n"
            else:
                report_content += "\n✅ **No major quality issues detected**\n"
            
            # Processing history summary
            if st.session_state.processing_log:
                report_content += f"\n### 5. Processing History Summary\n"
                report_content += f"- **Total Processing Actions:** {len(st.session_state.processing_log)}\n"
                
                action_counts = {}
                for log in st.session_state.processing_log:
                    action = log.get('action', 'Unknown')
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                report_content += "\n**Actions Performed:**\n"
                for action, count in sorted(action_counts.items()):
                    report_content += f"- {action}: {count}\n"
            
            report_content += f"\n---\n*Report generated by KlinItAll Data Preprocessing Platform*"
            
            # Download report
            st.download_button(
                label="📄 Download Quality Report",
                data=report_content,
                file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with report_col2:
        st.markdown("#### 🔧 Processing Summary Report")
        
        if st.button("🔧 Generate Processing Summary", type="primary"):
            # Generate processing summary report
            processing_report = f"""
# Data Processing Summary Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Overview
This report summarizes all data preprocessing operations performed on the dataset.

## Dataset Transformation

### Before Processing
"""
            
            # Original dataset info (if available)
            if 'original_dataset' in st.session_state:
                orig_df = st.session_state.original_dataset
                processing_report += f"- **Original Rows:** {len(orig_df):,}\n"
                processing_report += f"- **Original Columns:** {len(orig_df.columns):,}\n"
                processing_report += f"- **Original Missing Values:** {orig_df.isnull().sum().sum():,}\n"
            
            processing_report += f"""
### After Processing
- **Current Rows:** {len(df):,}
- **Current Columns:** {len(df.columns):,}
- **Current Missing Values:** {df.isnull().sum().sum():,}
- **Data Completeness:** {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

## Processing Actions Performed
"""
            
            if st.session_state.processing_log:
                processing_report += f"**Total Actions:** {len(st.session_state.processing_log)}\n\n"
                
                # Group actions by type
                action_details = {}
                for log in st.session_state.processing_log:
                    action = log.get('action', 'Unknown')
                    details = log.get('details', '')
                    timestamp = log.get('timestamp', '')
                    
                    if action not in action_details:
                        action_details[action] = []
                    
                    action_details[action].append({
                        'timestamp': timestamp,
                        'details': details
                    })
                
                # Write detailed actions
                for action_type, actions in action_details.items():
                    processing_report += f"\n### {action_type}\n"
                    processing_report += f"**Count:** {len(actions)}\n\n"
                    
                    for i, action in enumerate(actions, 1):
                        processing_report += f"{i}. **{action['timestamp']}**\n"
                        processing_report += f"   {action['details']}\n\n"
            
            else:
                processing_report += "No processing actions recorded.\n"
            
            # Feature engineering summary
            engineered_features = [col for col in df.columns if any(suffix in col for suffix in ['_scaled', '_encoded', '_binned', 'poly_', 'pca_', 'agg_'])]
            
            if engineered_features:
                processing_report += f"\n## Feature Engineering Summary\n"
                processing_report += f"**Engineered Features Created:** {len(engineered_features)}\n\n"
                
                # Group by type
                feature_types = {
                    'Scaled Features': [col for col in engineered_features if '_scaled' in col],
                    'Encoded Features': [col for col in engineered_features if '_encoded' in col],
                    'Binned Features': [col for col in engineered_features if '_binned' in col],
                    'Polynomial Features': [col for col in engineered_features if 'poly_' in col],
                    'PCA Components': [col for col in engineered_features if 'pca_' in col],
                    'Aggregate Features': [col for col in engineered_features if 'agg_' in col]
                }
                
                for feature_type, features in feature_types.items():
                    if features:
                        processing_report += f"\n**{feature_type}:** {len(features)}\n"
                        for feat in features[:5]:  # Show first 5
                            processing_report += f"- {feat}\n"
                        if len(features) > 5:
                            processing_report += f"... and {len(features) - 5} more\n"
            
            processing_report += f"\n---\n*Processing Summary generated by KlinItAll*"
            
            # Download processing report
            st.download_button(
                label="📄 Download Processing Summary",
                data=processing_report,
                file_name=f"processing_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col2:
    if st.button("🔧 Back to Clean Pipeline"):
        st.switch_page("pages/04_Clean_Pipeline.py")

with col3:
    if st.button("⚙️ Batch Processing"):
        st.switch_page("pages/15_Batch_Processing.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📚 History & Export Guide")
    
    st.markdown("#### Available Exports:")
    exports = [
        "**CSV:** Standard comma-separated",
        "**Excel:** Multi-sheet workbook",
        "**JSON:** Structured data format",
        "**Parquet:** High-performance format",
        "**Complete Package:** ZIP with everything"
    ]
    
    for export in exports:
        st.markdown(f"• {export}")
    
    st.markdown("---")
    st.markdown("#### Reports Available:")
    
    reports = [
        "**Quality Report:** Data completeness & issues",
        "**Processing Summary:** All transformations",
        "**Column Analysis:** Detailed field information",
        "**Complete Package:** Everything combined"
    ]
    
    for report in reports:
        st.markdown(f"• {report}")
    
    st.markdown("---")
    if st.session_state.processing_log:
        st.markdown("#### Quick Stats")
        st.metric("Total Actions", len(st.session_state.processing_log))
        
        if st.session_state.processing_log:
            latest_action = st.session_state.processing_log[-1]
            st.caption(f"Last: {latest_action.get('action', 'Unknown')}")