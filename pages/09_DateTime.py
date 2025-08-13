import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import calendar
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="DateTime", page_icon="📅", layout="wide")

st.title("📅 DateTime")
st.markdown("Advanced datetime analysis, conversion, and feature extraction capabilities")

if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Detect datetime columns and potential datetime columns
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
potential_datetime_cols = []

# Check object columns for potential datetime data
for col in df.select_dtypes(include=['object']).columns:
    try:
        sample_data = df[col].dropna().head(20)
        if len(sample_data) > 0:
            parsed_dates = pd.to_datetime(sample_data, errors='coerce', infer_datetime_format=True)
            success_rate = parsed_dates.notna().sum() / len(sample_data)
            if success_rate > 0.7:  # 70% of samples successfully parsed
                potential_datetime_cols.append(col)
    except:
        continue

all_datetime_cols = datetime_cols + potential_datetime_cols

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("DateTime Columns", len(datetime_cols))

with col2:
    st.metric("Potential DateTime", len(potential_datetime_cols))

with col3:
    st.metric("Total Candidates", len(all_datetime_cols))

with col4:
    if all_datetime_cols:
        sample_col = all_datetime_cols[0]
        if sample_col in datetime_cols:
            date_range = df[sample_col].max() - df[sample_col].min()
            st.metric("Sample Range", f"{date_range.days} days")
        else:
            st.metric("Processing Status", "Ready")
    else:
        st.metric("Processing Status", "No Data")

if len(all_datetime_cols) == 0:
    st.warning("⚠️ No datetime columns detected in the dataset.")
    
    # Allow manual datetime conversion
    st.markdown("### 🔧 Manual DateTime Conversion")
    
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        manual_col = st.selectbox("Try to convert this column to datetime:", object_cols)
        
        # Show sample values
        sample_vals = df[manual_col].dropna().head(10).astype(str).tolist()
        st.write("**Sample values:**")
        for val in sample_vals[:5]:
            st.write(f"• {val}")
        
        date_format = st.text_input("Specify date format (optional):", 
                                   placeholder="e.g., %Y-%m-%d, %m/%d/%Y")
        
        if st.button("🔄 Convert to DateTime"):
            try:
                if date_format:
                    df[manual_col] = pd.to_datetime(df[manual_col], format=date_format, errors='coerce')
                else:
                    df[manual_col] = pd.to_datetime(df[manual_col], errors='coerce', infer_datetime_format=True)
                
                st.session_state.current_dataset = df
                st.success(f"✅ Converted {manual_col} to datetime!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Conversion failed: {str(e)}")
    else:
        st.info("No text columns available for conversion.")
    st.stop()

# DateTime processing tabs
datetime_tabs = st.tabs(["📊 Analysis", "🔄 Conversion", "⚙️ Feature Engineering", "📈 Time Series"])

with datetime_tabs[0]:
    st.markdown("### 📊 DateTime Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Column analysis
        analysis_col = st.selectbox("Select datetime column for analysis:", all_datetime_cols)
        
        # Convert to datetime if not already
        if analysis_col not in datetime_cols:
            try:
                df[analysis_col] = pd.to_datetime(df[analysis_col], errors='coerce', infer_datetime_format=True)
                st.info(f"Converted {analysis_col} to datetime for analysis")
            except:
                st.error(f"Failed to convert {analysis_col} to datetime")
                st.stop()
        
        dt_data = df[analysis_col].dropna()
        
        if len(dt_data) == 0:
            st.warning("No valid datetime values found.")
        else:
            # DateTime statistics
            dt_stats = {
                'Metric': [
                    'Total Records',
                    'Valid Dates', 
                    'Invalid Dates',
                    'Date Range (Days)',
                    'Earliest Date',
                    'Latest Date',
                    'Most Common Year',
                    'Most Common Month'
                ],
                'Value': [
                    f"{len(df[analysis_col]):,}",
                    f"{len(dt_data):,}",
                    f"{df[analysis_col].isna().sum():,}",
                    f"{(dt_data.max() - dt_data.min()).days:,}",
                    str(dt_data.min().date()),
                    str(dt_data.max().date()),
                    str(dt_data.dt.year.mode().iloc[0]) if len(dt_data.dt.year.mode()) > 0 else "N/A",
                    calendar.month_name[dt_data.dt.month.mode().iloc[0]] if len(dt_data.dt.month.mode()) > 0 else "N/A"
                ]
            }
            
            stats_df = pd.DataFrame(dt_stats)
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Time distribution visualizations
        if len(dt_data) > 0:
            # Year distribution
            year_counts = dt_data.dt.year.value_counts().sort_index()
            if len(year_counts) > 1:
                fig_year = px.bar(x=year_counts.index, y=year_counts.values, 
                                title=f"Records by Year: {analysis_col}")
                fig_year.update_layout(height=300)
                st.plotly_chart(fig_year, use_container_width=True)
            
            # Month distribution
            month_counts = dt_data.dt.month.value_counts().sort_index()
            month_names = [calendar.month_abbr[i] for i in month_counts.index]
            
            fig_month = px.bar(x=month_names, y=month_counts.values,
                             title=f"Records by Month: {analysis_col}")
            fig_month.update_layout(height=300)
            st.plotly_chart(fig_month, use_container_width=True)

with datetime_tabs[1]:
    st.markdown("### 🔄 DateTime Conversion & Validation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Convert Text to DateTime")
        
        conversion_candidates = [col for col in potential_datetime_cols if col not in datetime_cols]
        
        if conversion_candidates:
            convert_col = st.selectbox("Select column to convert:", conversion_candidates)
            
            # Show sample values for format detection
            sample_values = df[convert_col].dropna().head(10).astype(str).tolist()
            st.markdown("**Sample values:**")
            for i, val in enumerate(sample_values[:5], 1):
                st.write(f"{i}. {val}")
            
            # Conversion options
            conversion_method = st.selectbox(
                "Conversion method:",
                ["Auto-detect format", "Specify custom format", "Try multiple formats"],
                help="Choose how to parse the datetime strings"
            )
            
            if conversion_method == "Specify custom format":
                custom_format = st.text_input(
                    "DateTime format:",
                    placeholder="e.g., %Y-%m-%d %H:%M:%S, %m/%d/%Y",
                    help="Use Python strftime codes"
                )
                
                # Format help
                with st.expander("📋 Format Codes Reference"):
                    st.markdown("""
                    **Common Format Codes:**
                    - `%Y` - 4-digit year (2023)
                    - `%y` - 2-digit year (23)
                    - `%m` - Month as number (01-12)
                    - `%d` - Day of month (01-31)
                    - `%H` - Hour 24-hour (00-23)
                    - `%I` - Hour 12-hour (01-12)
                    - `%M` - Minute (00-59)
                    - `%S` - Second (00-59)
                    - `%p` - AM/PM
                    """)
            
            if st.button("🔄 Convert Column", type="primary"):
                try:
                    original_type = str(df[convert_col].dtype)
                    
                    if conversion_method == "Auto-detect format":
                        df[convert_col] = pd.to_datetime(df[convert_col], errors='coerce', infer_datetime_format=True)
                    
                    elif conversion_method == "Specify custom format":
                        if custom_format:
                            df[convert_col] = pd.to_datetime(df[convert_col], format=custom_format, errors='coerce')
                        else:
                            st.error("Please specify a custom format")
                            st.stop()
                    
                    elif conversion_method == "Try multiple formats":
                        # Try common formats
                        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']
                        converted = False
                        
                        for fmt in formats:
                            try:
                                df[convert_col] = pd.to_datetime(df[convert_col], format=fmt, errors='coerce')
                                if df[convert_col].notna().sum() > 0:
                                    converted = True
                                    st.info(f"Successfully used format: {fmt}")
                                    break
                            except:
                                continue
                        
                        if not converted:
                            df[convert_col] = pd.to_datetime(df[convert_col], errors='coerce', infer_datetime_format=True)
                    
                    # Update session state
                    st.session_state.current_dataset = df
                    
                    # Success metrics
                    valid_dates = df[convert_col].notna().sum()
                    total_rows = len(df)
                    success_rate = (valid_dates / total_rows) * 100
                    
                    # Log action
                    if 'processing_log' not in st.session_state:
                        st.session_state.processing_log = []
                    
                    st.session_state.processing_log.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'DateTime Conversion',
                        'details': f"Converted {convert_col} from {original_type} to datetime64 ({success_rate:.1f}% success rate)"
                    })
                    
                    st.success(f"✅ Converted {convert_col}! Success rate: {success_rate:.1f}% ({valid_dates:,}/{total_rows:,})")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
        else:
            st.info("All potential datetime columns have been converted.")
    
    with col2:
        st.markdown("#### DateTime Validation")
        
        if datetime_cols:
            validate_col = st.selectbox("Column to validate:", datetime_cols, key="validate_col")
            
            dt_col_data = df[validate_col]
            
            # Validation metrics
            total_count = len(dt_col_data)
            null_count = dt_col_data.isna().sum()
            valid_count = total_count - null_count
            
            st.metric("Total Records", f"{total_count:,}")
            st.metric("Valid Dates", f"{valid_count:,}")
            st.metric("Invalid/Missing", f"{null_count:,}")
            
            if null_count > 0:
                st.warning(f"⚠️ {null_count:,} invalid or missing datetime values detected")
                
                # Options to handle invalid dates
                if st.button("🔧 Handle Invalid Dates"):
                    handling_method = st.selectbox(
                        "How to handle invalid dates:",
                        ["Drop rows with invalid dates", "Fill with median date", "Fill with most recent date", "Fill with earliest date"]
                    )
                    
                    if st.button("Apply Fix", key="apply_fix"):
                        if handling_method == "Drop rows with invalid dates":
                            df = df.dropna(subset=[validate_col])
                            st.success(f"✅ Dropped {null_count} rows with invalid dates")
                        
                        elif handling_method == "Fill with median date":
                            median_date = dt_col_data.dropna().median()
                            df[validate_col].fillna(median_date, inplace=True)
                            st.success(f"✅ Filled invalid dates with median: {median_date.date()}")
                        
                        elif handling_method == "Fill with most recent date":
                            recent_date = dt_col_data.dropna().max()
                            df[validate_col].fillna(recent_date, inplace=True)
                            st.success(f"✅ Filled invalid dates with most recent: {recent_date.date()}")
                        
                        elif handling_method == "Fill with earliest date":
                            earliest_date = dt_col_data.dropna().min()
                            df[validate_col].fillna(earliest_date, inplace=True)
                            st.success(f"✅ Filled invalid dates with earliest: {earliest_date.date()}")
                        
                        st.session_state.current_dataset = df
                        st.rerun()
            else:
                st.success("✅ All datetime values are valid!")

with datetime_tabs[2]:
    st.markdown("### ⚙️ DateTime Feature Engineering")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Extract DateTime Components")
        
        if datetime_cols:
            feature_col = st.selectbox("Select datetime column:", datetime_cols, key="feature_col")
            
            # Feature extraction options
            features_to_extract = st.multiselect(
                "Select features to extract:",
                [
                    "Year", "Month", "Day", "Weekday", "Week of Year",
                    "Quarter", "Hour", "Minute", "Day of Year",
                    "Is Weekend", "Is Month End", "Is Month Start",
                    "Season", "Time Period (Morning/Afternoon/Evening/Night)"
                ],
                default=["Year", "Month", "Weekday"],
                help="Choose datetime components to extract as new columns"
            )
            
            if st.button("⚙️ Extract Features", type="primary") and features_to_extract:
                try:
                    dt_data = df[feature_col]
                    features_created = []
                    
                    for feature in features_to_extract:
                        if feature == "Year":
                            df[f"{feature_col}_year"] = dt_data.dt.year
                            features_created.append(f"{feature_col}_year")
                        
                        elif feature == "Month":
                            df[f"{feature_col}_month"] = dt_data.dt.month
                            features_created.append(f"{feature_col}_month")
                        
                        elif feature == "Day":
                            df[f"{feature_col}_day"] = dt_data.dt.day
                            features_created.append(f"{feature_col}_day")
                        
                        elif feature == "Weekday":
                            df[f"{feature_col}_weekday"] = dt_data.dt.dayofweek
                            features_created.append(f"{feature_col}_weekday")
                        
                        elif feature == "Week of Year":
                            df[f"{feature_col}_week"] = dt_data.dt.isocalendar().week
                            features_created.append(f"{feature_col}_week")
                        
                        elif feature == "Quarter":
                            df[f"{feature_col}_quarter"] = dt_data.dt.quarter
                            features_created.append(f"{feature_col}_quarter")
                        
                        elif feature == "Hour":
                            df[f"{feature_col}_hour"] = dt_data.dt.hour
                            features_created.append(f"{feature_col}_hour")
                        
                        elif feature == "Minute":
                            df[f"{feature_col}_minute"] = dt_data.dt.minute
                            features_created.append(f"{feature_col}_minute")
                        
                        elif feature == "Day of Year":
                            df[f"{feature_col}_dayofyear"] = dt_data.dt.dayofyear
                            features_created.append(f"{feature_col}_dayofyear")
                        
                        elif feature == "Is Weekend":
                            df[f"{feature_col}_is_weekend"] = (dt_data.dt.dayofweek >= 5).astype(int)
                            features_created.append(f"{feature_col}_is_weekend")
                        
                        elif feature == "Is Month End":
                            df[f"{feature_col}_is_month_end"] = dt_data.dt.is_month_end.astype(int)
                            features_created.append(f"{feature_col}_is_month_end")
                        
                        elif feature == "Is Month Start":
                            df[f"{feature_col}_is_month_start"] = dt_data.dt.is_month_start.astype(int)
                            features_created.append(f"{feature_col}_is_month_start")
                        
                        elif feature == "Season":
                            seasons = dt_data.dt.month.map({
                                12: 'Winter', 1: 'Winter', 2: 'Winter',
                                3: 'Spring', 4: 'Spring', 5: 'Spring',
                                6: 'Summer', 7: 'Summer', 8: 'Summer',
                                9: 'Fall', 10: 'Fall', 11: 'Fall'
                            })
                            df[f"{feature_col}_season"] = seasons
                            features_created.append(f"{feature_col}_season")
                        
                        elif feature == "Time Period (Morning/Afternoon/Evening/Night)":
                            time_periods = dt_data.dt.hour.map(lambda x: 
                                'Night' if x < 6 else
                                'Morning' if x < 12 else
                                'Afternoon' if x < 18 else
                                'Evening'
                            )
                            df[f"{feature_col}_time_period"] = time_periods
                            features_created.append(f"{feature_col}_time_period")
                    
                    # Update session state
                    st.session_state.current_dataset = df
                    
                    # Log action
                    if 'processing_log' not in st.session_state:
                        st.session_state.processing_log = []
                    
                    st.session_state.processing_log.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'action': 'DateTime Feature Engineering',
                        'details': f"Extracted {len(features_created)} features from {feature_col}: {', '.join(features_created)}"
                    })
                    
                    st.success(f"✅ Created {len(features_created)} new datetime features!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Feature extraction failed: {str(e)}")
        else:
            st.info("No datetime columns available for feature engineering.")
    
    with col2:
        st.markdown("#### Advanced Time Features")
        
        if datetime_cols:
            adv_col = st.selectbox("DateTime column:", datetime_cols, key="adv_col")
            
            # Advanced feature options
            advanced_features = st.multiselect(
                "Advanced features:",
                [
                    "Days since earliest date",
                    "Days until latest date", 
                    "Business days from start",
                    "Age in years (from date)",
                    "Time since midnight (seconds)",
                    "Cyclical features (sin/cos)"
                ],
                help="Create advanced time-based features"
            )
            
            if st.button("🚀 Create Advanced Features") and advanced_features:
                try:
                    dt_data = df[adv_col].dropna()
                    
                    for feature in advanced_features:
                        if feature == "Days since earliest date":
                            earliest = dt_data.min()
                            df[f"{adv_col}_days_since_start"] = (df[adv_col] - earliest).dt.days
                        
                        elif feature == "Days until latest date":
                            latest = dt_data.max()
                            df[f"{adv_col}_days_until_end"] = (latest - df[adv_col]).dt.days
                        
                        elif feature == "Business days from start":
                            earliest = dt_data.min()
                            df[f"{adv_col}_bdays_since_start"] = df[adv_col].apply(
                                lambda x: pd.bdate_range(earliest, x).size if pd.notna(x) else np.nan
                            )
                        
                        elif feature == "Age in years (from date)":
                            reference_date = datetime.now()
                            df[f"{adv_col}_age_years"] = (reference_date - df[adv_col]).dt.days / 365.25
                        
                        elif feature == "Time since midnight (seconds)":
                            df[f"{adv_col}_seconds_since_midnight"] = (
                                df[adv_col].dt.hour * 3600 + 
                                df[adv_col].dt.minute * 60 + 
                                df[adv_col].dt.second
                            )
                        
                        elif feature == "Cyclical features (sin/cos)":
                            # Create cyclical features for month
                            df[f"{adv_col}_month_sin"] = np.sin(2 * np.pi * df[adv_col].dt.month / 12)
                            df[f"{adv_col}_month_cos"] = np.cos(2 * np.pi * df[adv_col].dt.month / 12)
                            
                            # Create cyclical features for day of week
                            df[f"{adv_col}_weekday_sin"] = np.sin(2 * np.pi * df[adv_col].dt.dayofweek / 7)
                            df[f"{adv_col}_weekday_cos"] = np.cos(2 * np.pi * df[adv_col].dt.dayofweek / 7)
                    
                    st.session_state.current_dataset = df
                    st.success(f"✅ Created {len(advanced_features)} advanced time features!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Advanced feature creation failed: {str(e)}")

with datetime_tabs[3]:
    st.markdown("### 📈 Time Series Analysis")
    
    if datetime_cols:
        ts_col = st.selectbox("Select datetime column for time series:", datetime_cols, key="ts_col")
        
        # Get numeric columns for time series analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            value_col = st.selectbox("Select value column:", numeric_cols, key="value_col")
            
            # Time series aggregation
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Time Series Aggregation")
                
                aggregation_period = st.selectbox(
                    "Aggregation period:",
                    ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                    help="How to group the time series data"
                )
                
                aggregation_method = st.selectbox(
                    "Aggregation method:",
                    ["sum", "mean", "median", "count", "min", "max"],
                    help="How to aggregate values within each period"
                )
                
                if st.button("📊 Create Time Series", type="primary"):
                    try:
                        # Set datetime column as index
                        ts_df = df[[ts_col, value_col]].copy()
                        ts_df = ts_df.dropna()
                        ts_df.set_index(ts_col, inplace=True)
                        
                        # Aggregate based on period
                        if aggregation_period == "Daily":
                            ts_aggregated = ts_df.resample('D')[value_col].agg(aggregation_method)
                        elif aggregation_period == "Weekly":
                            ts_aggregated = ts_df.resample('W')[value_col].agg(aggregation_method)
                        elif aggregation_period == "Monthly":
                            ts_aggregated = ts_df.resample('M')[value_col].agg(aggregation_method)
                        elif aggregation_period == "Quarterly":
                            ts_aggregated = ts_df.resample('Q')[value_col].agg(aggregation_method)
                        else:  # Yearly
                            ts_aggregated = ts_df.resample('Y')[value_col].agg(aggregation_method)
                        
                        # Plot time series
                        fig = px.line(
                            x=ts_aggregated.index, 
                            y=ts_aggregated.values,
                            title=f"{aggregation_period} {aggregation_method.title()} of {value_col}",
                            labels={'x': ts_col, 'y': f"{aggregation_method.title()} {value_col}"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Time series statistics
                        st.markdown("#### Time Series Statistics")
                        ts_stats = {
                            'Periods': len(ts_aggregated),
                            'Start Date': str(ts_aggregated.index.min().date()),
                            'End Date': str(ts_aggregated.index.max().date()),
                            'Mean Value': f"{ts_aggregated.mean():.2f}",
                            'Std Deviation': f"{ts_aggregated.std():.2f}",
                            'Min Value': f"{ts_aggregated.min():.2f}",
                            'Max Value': f"{ts_aggregated.max():.2f}"
                        }
                        
                        for key, value in ts_stats.items():
                            st.write(f"**{key}:** {value}")
                        
                    except Exception as e:
                        st.error(f"❌ Time series creation failed: {str(e)}")
            
            with col2:
                st.markdown("#### Trend Analysis")
                
                # Simple trend analysis
                if st.button("📈 Analyze Trends"):
                    try:
                        ts_df = df[[ts_col, value_col]].copy().dropna()
                        ts_df = ts_df.sort_values(ts_col)
                        
                        # Calculate rolling averages
                        window_size = min(30, len(ts_df) // 4)  # Adaptive window size
                        ts_df['rolling_mean'] = ts_df[value_col].rolling(window=window_size).mean()
                        
                        # Plot with trend
                        fig = go.Figure()
                        
                        # Original data
                        fig.add_trace(go.Scatter(
                            x=ts_df[ts_col],
                            y=ts_df[value_col],
                            mode='markers',
                            name='Data Points',
                            opacity=0.6
                        ))
                        
                        # Rolling average
                        fig.add_trace(go.Scatter(
                            x=ts_df[ts_col],
                            y=ts_df['rolling_mean'],
                            mode='lines',
                            name=f'Rolling Mean ({window_size} periods)',
                            line=dict(color='red', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f"Trend Analysis: {value_col} over {ts_col}",
                            xaxis_title=ts_col,
                            yaxis_title=value_col
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Basic trend statistics
                        first_half = ts_df[value_col].iloc[:len(ts_df)//2].mean()
                        second_half = ts_df[value_col].iloc[len(ts_df)//2:].mean()
                        trend_direction = "Upward" if second_half > first_half else "Downward"
                        trend_magnitude = abs(second_half - first_half)
                        
                        st.write(f"**Trend Direction:** {trend_direction}")
                        st.write(f"**Trend Magnitude:** {trend_magnitude:.2f}")
                        st.write(f"**First Half Average:** {first_half:.2f}")
                        st.write(f"**Second Half Average:** {second_half:.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ Trend analysis failed: {str(e)}")
        
        else:
            st.info("No numeric columns available for time series analysis.")
    
    else:
        st.info("No datetime columns available for time series analysis.")

# Export and Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Download Enhanced Dataset"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"datetime_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔍 View Data Overview"):
        st.switch_page("pages/02_Data_Overview.py")

with col3:
    if st.button("➡️ Continue to Geospatial"):
        st.switch_page("pages/10_Geospatial.py")

# Sidebar
with st.sidebar:
    st.markdown("### 📅 DateTime Processing Guide")
    
    st.markdown("#### Key Features:")
    features = [
        "**Conversion:** Text to datetime parsing",
        "**Validation:** Data quality checking",
        "**Engineering:** Component extraction",
        "**Analysis:** Time series insights"
    ]
    
    for feature in features:
        st.markdown(f"• {feature}")
    
    st.markdown("---")
    st.markdown("#### Common DateTime Features:")
    
    common_features = [
        "Year, Month, Day components",
        "Weekday and weekend flags",
        "Seasonal classifications",
        "Business vs weekend indicators",
        "Time periods (morning/afternoon)",
        "Cyclical encodings for ML"
    ]
    
    for cf in common_features:
        st.markdown(f"• {cf}")
    
    st.markdown("---")
    st.markdown("#### 💡 Best Practices")
    
    st.info("""
    **Guidelines:**
    • Validate datetime ranges make sense
    • Handle timezone considerations
    • Create relevant business features
    • Use cyclical encoding for ML models
    • Consider seasonality patterns
    """)