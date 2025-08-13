import streamlit as st
import pandas as pd
import numpy as np
import concurrent.futures
import threading
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Batch Processing", page_icon="⚙️", layout="wide")

st.title("⚙️ Batch Processing")
st.markdown("Process multiple datasets simultaneously with parallel execution and automated workflows")

# Initialize batch processing session state
if 'batch_jobs' not in st.session_state:
    st.session_state.batch_jobs = []

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = {}

def run_batch_processing(datasets, operations, job_id):
    """Run batch processing operations on multiple datasets"""
    results = {}
    
    for dataset_name, dataset in datasets.items():
        try:
            processed_df = dataset.copy()
            operation_log = []
            
            # Apply each operation
            for operation in operations:
                if operation['type'] == 'remove_duplicates':
                    initial_rows = len(processed_df)
                    processed_df = processed_df.drop_duplicates()
                    removed = initial_rows - len(processed_df)
                    operation_log.append(f"Removed {removed} duplicate rows")
                
                elif operation['type'] == 'handle_missing':
                    strategy = operation.get('strategy', 'drop')
                    if strategy == 'drop':
                        initial_rows = len(processed_df)
                        processed_df = processed_df.dropna()
                        removed = initial_rows - len(processed_df)
                        operation_log.append(f"Dropped {removed} rows with missing values")
                    
                    elif strategy == 'fill_numeric':
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            filled_count = processed_df[col].isnull().sum()
                            processed_df[col].fillna(processed_df[col].median(), inplace=True)
                            operation_log.append(f"Filled {filled_count} missing values in {col} with median")
                
                elif operation['type'] == 'standardize_columns':
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
                    operation_log.append(f"Standardized {len(numeric_cols)} numeric columns")
                
                elif operation['type'] == 'encode_categorical':
                    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns
                    encoded_count = 0
                    for col in categorical_cols:
                        if processed_df[col].nunique() <= 10:  # One-hot encode low cardinality
                            encoded_df = pd.get_dummies(processed_df[col], prefix=col)
                            processed_df = pd.concat([processed_df.drop(col, axis=1), encoded_df], axis=1)
                            encoded_count += len(encoded_df.columns)
                        else:  # Label encode high cardinality
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col].astype(str))
                            encoded_count += 1
                    operation_log.append(f"Encoded {len(categorical_cols)} categorical columns, created {encoded_count} new features")
                
                # Add more operations as needed...
            
            results[dataset_name] = {
                'processed_data': processed_df,
                'operation_log': operation_log,
                'status': 'completed',
                'original_shape': dataset.shape,
                'final_shape': processed_df.shape,
                'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            results[dataset_name] = {
                'processed_data': None,
                'operation_log': [],
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return results

# Batch Processing tabs
batch_tabs = st.tabs(["📁 Dataset Management", "🔧 Operation Setup", "⚡ Parallel Processing", "📊 Results Monitor"])

with batch_tabs[0]:
    st.markdown("### 📁 Multiple Dataset Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Upload Multiple Datasets")
        
        # Multi-file upload
        uploaded_files = st.file_uploader(
            "Upload multiple CSV files for batch processing:",
            type=['csv'],
            accept_multiple_files=True,
            help="Select multiple CSV files to process simultaneously"
        )
        
        # Dataset storage
        if uploaded_files:
            if 'batch_datasets' not in st.session_state:
                st.session_state.batch_datasets = {}
            
            for uploaded_file in uploaded_files:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.batch_datasets[uploaded_file.name] = df
                    st.success(f"✅ Loaded {uploaded_file.name}: {len(df)} rows × {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"❌ Failed to load {uploaded_file.name}: {str(e)}")
        
        # Current dataset addition
        if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
            if st.button("➕ Add Current Dataset to Batch"):
                if 'batch_datasets' not in st.session_state:
                    st.session_state.batch_datasets = {}
                
                dataset_name = f"current_dataset_{datetime.now().strftime('%H%M%S')}"
                st.session_state.batch_datasets[dataset_name] = st.session_state.current_dataset.copy()
                st.success(f"✅ Added current dataset as {dataset_name}")
                st.rerun()
        
        # Manual dataset creation
        st.markdown("#### Create Sample Dataset")
        if st.button("🎲 Generate Sample Dataset"):
            sample_data = {
                'ID': range(1, 101),
                'Value1': np.random.randn(100),
                'Value2': np.random.randn(100),
                'Category': np.random.choice(['A', 'B', 'C'], 100),
                'Flag': np.random.choice([True, False], 100)
            }
            
            # Add some missing values
            sample_df = pd.DataFrame(sample_data)
            sample_df.loc[np.random.choice(sample_df.index, 10), 'Value1'] = np.nan
            
            if 'batch_datasets' not in st.session_state:
                st.session_state.batch_datasets = {}
            
            dataset_name = f"sample_dataset_{datetime.now().strftime('%H%M%S')}"
            st.session_state.batch_datasets[dataset_name] = sample_df
            st.success(f"✅ Created sample dataset: {dataset_name}")
            st.rerun()
    
    with col2:
        st.markdown("#### Loaded Datasets")
        
        if 'batch_datasets' in st.session_state and st.session_state.batch_datasets:
            total_datasets = len(st.session_state.batch_datasets)
            total_rows = sum(len(df) for df in st.session_state.batch_datasets.values())
            total_columns = sum(len(df.columns) for df in st.session_state.batch_datasets.values())
            
            st.metric("Total Datasets", total_datasets)
            st.metric("Total Rows", f"{total_rows:,}")
            st.metric("Avg Columns", f"{total_columns // total_datasets:.0f}")
            
            st.markdown("#### Dataset List")
            
            for dataset_name, dataset in st.session_state.batch_datasets.items():
                with st.expander(f"📄 {dataset_name}"):
                    st.write(f"**Shape:** {dataset.shape[0]} rows × {dataset.shape[1]} columns")
                    st.write(f"**Memory:** {dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
                    st.write(f"**Missing:** {dataset.isnull().sum().sum():,} values")
                    
                    # Quick preview
                    st.dataframe(dataset.head(3), use_container_width=True)
                    
                    # Remove dataset option
                    if st.button(f"🗑️ Remove {dataset_name}", key=f"remove_{dataset_name}"):
                        del st.session_state.batch_datasets[dataset_name]
                        st.rerun()
            
            # Clear all datasets
            if st.button("🗑️ Clear All Datasets"):
                if st.checkbox("Confirm clear all datasets"):
                    st.session_state.batch_datasets = {}
                    st.success("✅ All datasets cleared!")
                    st.rerun()
        
        else:
            st.info("No datasets loaded. Upload files above to get started.")

with batch_tabs[1]:
    st.markdown("### 🔧 Batch Operation Setup")
    
    if 'batch_datasets' not in st.session_state or not st.session_state.batch_datasets:
        st.warning("⚠️ Please load datasets first in the Dataset Management tab.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Select Operations")
            
            # Available operations
            available_operations = {
                'remove_duplicates': 'Remove Duplicate Rows',
                'handle_missing': 'Handle Missing Values',
                'standardize_columns': 'Standardize Numeric Columns',
                'encode_categorical': 'Encode Categorical Variables',
                'detect_outliers': 'Detect and Handle Outliers',
                'text_cleaning': 'Basic Text Cleaning'
            }
            
            selected_operations = st.multiselect(
                "Choose operations to apply to all datasets:",
                list(available_operations.keys()),
                format_func=lambda x: available_operations[x],
                help="These operations will be applied to all datasets in the batch"
            )
            
            # Operation-specific parameters
            operation_config = {}
            
            if 'handle_missing' in selected_operations:
                st.markdown("##### Missing Value Strategy")
                missing_strategy = st.selectbox(
                    "Missing value handling:",
                    ['drop', 'fill_numeric', 'fill_categorical'],
                    format_func=lambda x: {
                        'drop': 'Drop rows with missing values',
                        'fill_numeric': 'Fill numeric with median',
                        'fill_categorical': 'Fill categorical with mode'
                    }[x]
                )
                operation_config['handle_missing'] = {'strategy': missing_strategy}
            
            if 'detect_outliers' in selected_operations:
                st.markdown("##### Outlier Detection")
                outlier_method = st.selectbox("Outlier detection method:", ['iqr', 'zscore'])
                outlier_action = st.selectbox("Outlier action:", ['remove', 'cap', 'mark'])
                operation_config['detect_outliers'] = {
                    'method': outlier_method,
                    'action': outlier_action
                }
            
            # Processing options
            st.markdown("#### Processing Options")
            
            parallel_processing = st.checkbox(
                "Enable parallel processing",
                value=True,
                help="Process datasets simultaneously using multiple threads"
            )
            
            if parallel_processing:
                max_workers = st.slider(
                    "Maximum parallel workers:",
                    1, min(8, len(st.session_state.batch_datasets)), 
                    min(4, len(st.session_state.batch_datasets)),
                    help="Number of datasets to process simultaneously"
                )
            else:
                max_workers = 1
        
        with col2:
            st.markdown("#### Operation Preview")
            
            if selected_operations:
                st.markdown("**Operations to be applied:**")
                
                for i, operation in enumerate(selected_operations, 1):
                    op_name = available_operations[operation]
                    st.write(f"{i}. {op_name}")
                    
                    # Show operation details
                    if operation in operation_config:
                        config = operation_config[operation]
                        for key, value in config.items():
                            st.write(f"   - {key}: {value}")
                
                # Estimate processing time
                total_datasets = len(st.session_state.batch_datasets)
                estimated_time_per_dataset = len(selected_operations) * 2  # 2 seconds per operation (rough estimate)
                
                if parallel_processing:
                    estimated_total_time = (total_datasets / max_workers) * estimated_time_per_dataset
                else:
                    estimated_total_time = total_datasets * estimated_time_per_dataset
                
                st.info(f"**Estimated processing time:** {estimated_total_time:.0f} seconds")
                
                # Dataset impact preview
                st.markdown("#### Impact Preview")
                
                sample_dataset = next(iter(st.session_state.batch_datasets.values()))
                
                st.write("**Sample dataset impact:**")
                st.write(f"- Original shape: {sample_dataset.shape}")
                
                # Estimate changes
                estimated_changes = []
                
                if 'remove_duplicates' in selected_operations:
                    dup_count = sample_dataset.duplicated().sum()
                    estimated_changes.append(f"Remove ~{dup_count} duplicate rows")
                
                if 'handle_missing' in selected_operations:
                    missing_count = sample_dataset.isnull().sum().sum()
                    estimated_changes.append(f"Handle {missing_count} missing values")
                
                if 'standardize_columns' in selected_operations:
                    numeric_count = len(sample_dataset.select_dtypes(include=[np.number]).columns)
                    estimated_changes.append(f"Standardize {numeric_count} numeric columns")
                
                if 'encode_categorical' in selected_operations:
                    cat_count = len(sample_dataset.select_dtypes(include=['object', 'category']).columns)
                    estimated_changes.append(f"Encode {cat_count} categorical columns")
                
                for change in estimated_changes:
                    st.write(f"- {change}")
            
            else:
                st.info("Select operations to see preview")

with batch_tabs[2]:
    st.markdown("### ⚡ Parallel Processing Execution")
    
    if 'batch_datasets' not in st.session_state or not st.session_state.batch_datasets:
        st.warning("⚠️ Please load datasets and configure operations first.")
    
    elif 'selected_operations' not in locals() or not selected_operations:
        st.warning("⚠️ Please select operations to perform in the Operation Setup tab.")
    
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="progress-section">', unsafe_allow_html=True)
            st.markdown("#### Batch Processing Control")
            
            # Job configuration
            job_name = st.text_input(
                "Job name:",
                value=f"batch_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Name for this batch processing job"
            )
            
            # Dataset selection
            dataset_names = list(st.session_state.batch_datasets.keys())
            selected_datasets = st.multiselect(
                "Select datasets to process:",
                dataset_names,
                default=dataset_names,
                help="Choose which datasets to include in this batch job"
            )
            
            # Start processing
            if st.button("🚀 Start Batch Processing", type="primary") and selected_datasets:
                
                # Prepare datasets for processing
                datasets_to_process = {
                    name: st.session_state.batch_datasets[name] 
                    for name in selected_datasets
                }
                
                # Prepare operations list
                operations_list = []
                for op in selected_operations:
                    op_config = {'type': op}
                    if op in operation_config:
                        op_config.update(operation_config[op])
                    operations_list.append(op_config)
                
                # Create job entry
                job_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                
                job_info = {
                    'job_id': job_id,
                    'job_name': job_name,
                    'datasets': list(selected_datasets),
                    'operations': operations_list,
                    'status': 'running',
                    'start_time': datetime.now(),
                    'parallel': parallel_processing,
                    'max_workers': max_workers if parallel_processing else 1
                }
                
                st.session_state.batch_jobs.append(job_info)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Start processing
                if parallel_processing and max_workers > 1:
                    # Parallel processing
                    status_text.text(f"Starting parallel processing with {max_workers} workers...")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit individual dataset processing jobs
                        future_to_dataset = {}
                        
                        for dataset_name in selected_datasets:
                            single_dataset = {dataset_name: datasets_to_process[dataset_name]}
                            future = executor.submit(run_batch_processing, single_dataset, operations_list, job_id)
                            future_to_dataset[future] = dataset_name
                        
                        # Collect results
                        completed = 0
                        batch_results = {}
                        
                        for future in concurrent.futures.as_completed(future_to_dataset):
                            dataset_name = future_to_dataset[future]
                            try:
                                result = future.result()
                                batch_results.update(result)
                                completed += 1
                                
                                progress = completed / len(selected_datasets)
                                progress_bar.progress(progress)
                                status_text.text(f"Completed {completed}/{len(selected_datasets)} datasets...")
                                
                            except Exception as e:
                                batch_results[dataset_name] = {
                                    'status': 'failed',
                                    'error': str(e)
                                }
                                completed += 1
                
                else:
                    # Sequential processing
                    status_text.text("Starting sequential processing...")
                    batch_results = run_batch_processing(datasets_to_process, operations_list, job_id)
                    progress_bar.progress(1.0)
                
                # Store results
                st.session_state.batch_results[job_id] = batch_results
                
                # Update job status
                for job in st.session_state.batch_jobs:
                    if job['job_id'] == job_id:
                        job['status'] = 'completed'
                        job['end_time'] = datetime.now()
                        job['results'] = batch_results
                        break
                
                status_text.text("✅ Batch processing completed!")
                st.success(f"🎉 Batch job '{job_name}' completed successfully!")
                st.balloons()
                
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Active Jobs Monitor")
            
            if st.session_state.batch_jobs:
                for job in reversed(st.session_state.batch_jobs[-5:]):  # Show last 5 jobs
                    status_color = {
                        'running': '🟡',
                        'completed': '✅',
                        'failed': '❌'
                    }.get(job['status'], '⚪')
                    
                    with st.expander(f"{status_color} {job['job_name']} ({job['status']})"):
                        st.write(f"**Job ID:** {job['job_id']}")
                        st.write(f"**Status:** {job['status']}")
                        st.write(f"**Datasets:** {len(job['datasets'])}")
                        st.write(f"**Operations:** {len(job['operations'])}")
                        st.write(f"**Start Time:** {job['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        if 'end_time' in job:
                            duration = job['end_time'] - job['start_time']
                            st.write(f"**Duration:** {duration.total_seconds():.1f} seconds")
                        
                        if job['status'] == 'completed' and 'results' in job:
                            successful = sum(1 for r in job['results'].values() if r['status'] == 'completed')
                            st.write(f"**Success Rate:** {successful}/{len(job['results'])}")
            else:
                st.info("No batch jobs have been run yet.")

with batch_tabs[3]:
    st.markdown("### 📊 Results Monitor & Export")
    
    if not st.session_state.batch_results:
        st.info("No batch processing results available yet. Run some batch jobs first.")
    
    else:
        # Job selection
        job_ids = list(st.session_state.batch_results.keys())
        selected_job = st.selectbox("Select job to view results:", job_ids)
        
        if selected_job:
            results = st.session_state.batch_results[selected_job]
            
            # Results overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_datasets = len(results)
                st.metric("Total Datasets", total_datasets)
            
            with col2:
                successful = sum(1 for r in results.values() if r['status'] == 'completed')
                st.metric("Successful", successful)
            
            with col3:
                failed = sum(1 for r in results.values() if r['status'] == 'failed')
                st.metric("Failed", failed)
            
            with col4:
                success_rate = (successful / total_datasets * 100) if total_datasets > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Detailed results
            st.markdown("#### Detailed Results")
            
            for dataset_name, result in results.items():
                status_icon = "✅" if result['status'] == 'completed' else "❌"
                
                with st.expander(f"{status_icon} {dataset_name} - {result['status']}"):
                    if result['status'] == 'completed':
                        st.write(f"**Original Shape:** {result['original_shape']}")
                        st.write(f"**Final Shape:** {result['final_shape']}")
                        st.write(f"**Completion Time:** {result['completion_time']}")
                        
                        st.markdown("**Operations Applied:**")
                        for op in result['operation_log']:
                            st.write(f"• {op}")
                        
                        # Data preview
                        if result['processed_data'] is not None:
                            st.markdown("**Processed Data Preview:**")
                            preview_df = result['processed_data'].head(5)
                            # Convert to strings to avoid Arrow issues
                            for col in preview_df.columns:
                                preview_df[col] = preview_df[col].astype(str)
                            st.dataframe(preview_df, use_container_width=True)
                        
                        # Export individual result
                        if st.button(f"💾 Export {dataset_name}", key=f"export_{dataset_name}"):
                            csv_data = result['processed_data'].to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {dataset_name}",
                                data=csv_data,
                                file_name=f"processed_{dataset_name}",
                                mime="text/csv",
                                key=f"download_{dataset_name}"
                            )
                    
                    else:  # Failed
                        st.error(f"**Error:** {result.get('error', 'Unknown error')}")
            
            # Bulk export
            st.markdown("---")
            st.markdown("#### Bulk Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📦 Export All Successful Results", type="primary"):
                    # Create ZIP file with all successful results
                    import zipfile
                    import io
                    
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add each successful dataset
                        for dataset_name, result in results.items():
                            if result['status'] == 'completed' and result['processed_data'] is not None:
                                csv_data = result['processed_data'].to_csv(index=False)
                                zip_file.writestr(f"processed_{dataset_name}", csv_data)
                        
                        # Add processing summary
                        summary = f"""
Batch Processing Results Summary
Job ID: {selected_job}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
- Total Datasets: {len(results)}
- Successful: {successful}
- Failed: {failed}
- Success Rate: {success_rate:.1f}%

DETAILED RESULTS:
"""
                        
                        for dataset_name, result in results.items():
                            summary += f"\n{dataset_name}:\n"
                            summary += f"  Status: {result['status']}\n"
                            if result['status'] == 'completed':
                                summary += f"  Original Shape: {result['original_shape']}\n"
                                summary += f"  Final Shape: {result['final_shape']}\n"
                                summary += f"  Operations: {len(result['operation_log'])}\n"
                            else:
                                summary += f"  Error: {result.get('error', 'Unknown')}\n"
                        
                        zip_file.writestr("batch_processing_summary.txt", summary)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download All Results",
                        data=zip_buffer.getvalue(),
                        file_name=f"batch_results_{selected_job}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            
            with col2:
                # Generate batch report
                if st.button("📊 Generate Batch Report"):
                    report_content = f"""
# Batch Processing Report
**Job ID:** {selected_job}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Total Datasets Processed:** {len(results)}
- **Successful Processes:** {successful}
- **Failed Processes:** {failed}
- **Overall Success Rate:** {success_rate:.1f}%

## Processing Details

### Successful Datasets
"""
                    
                    for dataset_name, result in results.items():
                        if result['status'] == 'completed':
                            report_content += f"""
#### {dataset_name}
- **Original Size:** {result['original_shape'][0]} rows × {result['original_shape'][1]} columns
- **Final Size:** {result['final_shape'][0]} rows × {result['final_shape'][1]} columns
- **Operations Applied:** {len(result['operation_log'])}
- **Completion Time:** {result['completion_time']}

**Operation Details:**
"""
                            for op in result['operation_log']:
                                report_content += f"- {op}\n"
                    
                    if failed > 0:
                        report_content += "\n### Failed Datasets\n"
                        for dataset_name, result in results.items():
                            if result['status'] == 'failed':
                                report_content += f"""
#### {dataset_name}
- **Error:** {result.get('error', 'Unknown error')}
- **Failure Time:** {result.get('completion_time', 'Unknown')}
"""
                    
                    report_content += "\n---\n*Report generated by KlinItAll Batch Processing*"
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=report_content,
                        file_name=f"batch_report_{selected_job}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col2:
    if st.button("📚 History & Export"):
        st.switch_page("pages/14_History_Export.py")

with col3:
    if st.button("🏠 Back to Home"):
        st.switch_page("app.py")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Batch Processing Guide")
    
    st.markdown("#### Key Features:")
    features = [
        "**Multi-Dataset:** Process multiple files",
        "**Parallel:** Simultaneous execution",
        "**Automated:** Consistent operations",
        "**Monitoring:** Real-time progress",
        "**Export:** Bulk results download"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.markdown("---")
    st.markdown("#### Available Operations:")
    
    operations = [
        "Remove duplicates",
        "Handle missing values", 
        "Standardize columns",
        "Encode categorical variables",
        "Detect outliers",
        "Basic text cleaning"
    ]
    
    for op in operations:
        st.markdown(f"• {op}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Tips:**
    • Test operations on single dataset first
    • Use parallel processing for large batches
    • Monitor memory usage with many datasets
    • Export results immediately after completion
    • Keep processing logs for audit trail
    """)
    
    # Quick stats
    if st.session_state.batch_jobs:
        st.markdown("---")
        st.markdown("#### Quick Stats")
        
        total_jobs = len(st.session_state.batch_jobs)
        completed_jobs = sum(1 for job in st.session_state.batch_jobs if job['status'] == 'completed')
        
        st.metric("Total Jobs", total_jobs)
        st.metric("Completed", completed_jobs)
        
        if st.session_state.batch_results:
            total_datasets_processed = sum(len(results) for results in st.session_state.batch_results.values())
            st.metric("Datasets Processed", total_datasets_processed)