import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import zipfile
import sqlite3
from datetime import datetime
import sqlalchemy as sa
import pymongo
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
import dropbox
import pyarrow
import h5py
import xmltodict
from fuzzywuzzy import fuzz, process
from fuzzywuzzy.utils import full_process
import mysql.connector
import psycopg2
import math
from difflib import SequenceMatcher
from utils.guided_tour import guided_tour
from utils.milestone_rewards import milestone_rewards



st.set_page_config(page_title="Upload Data", page_icon="📥", layout="wide")


# Initialize session state
if 'upload_log' not in st.session_state:
    st.session_state.upload_log = []

if 'merge_datasets' not in st.session_state:
    st.session_state.merge_datasets = []

if 'automation_stats' not in st.session_state:
    st.session_state.automation_stats = {
        'numeric_cols_for_scaling': 0,
        'categorical_cols_for_encoding': 0,
        'manual_time_est': 0,
        'auto_time_est': 0
    }

# Show contextual hints based on tour state
if st.session_state.tour_active:
    if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
        guided_tour.show_character_hint("upload", custom_message="Great! You're on the Upload page. Let's get your data uploaded so we can start cleaning it together!", hint_type="info")
    else:
        guided_tour.show_character_hint("upload", custom_message="Perfect! I see you have data uploaded. Now you can either preview it below or jump straight to cleaning. I'm here to help guide you!", hint_type="success")


st.title("📥 Upload Data")
st.markdown("Upload your data files for preprocessing and analysis")

# Show tour controls and hints
guided_tour.show_tour_controls()

def log_dataset_upload(source, source_type, row_count, column_count, size_info=""):
    """Enhanced logging for dataset uploads"""
    log_entry = {
        'source': source,
        'source_type': source_type,
        'rows': row_count,
        'columns': column_count,
        'size_info': size_info,
        'upload_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'Success'
    }
    st.session_state.upload_log.append(log_entry)
    update_automation_stats()

def update_automation_stats():
    """Update automation statistics based on current dataset"""
    if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        
        # Count numeric columns for scaling
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        
        # Count categorical columns for encoding
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Estimate processing times
        manual_time = (numeric_cols * 8 + categorical_cols * 12) / 60  # hours
        auto_time = (numeric_cols * 0.5 + categorical_cols * 1) / 60  # hours
        
        st.session_state.automation_stats = {
            'numeric_cols_for_scaling': numeric_cols,
            'categorical_cols_for_encoding': categorical_cols,
            'manual_time_est': manual_time,
            'auto_time_est': auto_time
        }

def suggest_join_columns(df1, df2):
    """Suggest best columns for joining based on column names and data similarity"""
    suggestions = []
    
    for col1 in df1.columns:
        for col2 in df2.columns:
            # Check name similarity
            name_similarity = SequenceMatcher(None, col1.lower(), col2.lower()).ratio()
            
            if name_similarity > 0.7:  # High name similarity
                suggestions.append({
                    'df1_col': col1,
                    'df2_col': col2,
                    'similarity': name_similarity,
                    'reason': 'High name similarity'
                })
            
            # Check for common join patterns
            if any(pattern in col1.lower() for pattern in ['id', 'key', 'code']) and \
               any(pattern in col2.lower() for pattern in ['id', 'key', 'code']):
                suggestions.append({
                    'df1_col': col1,
                    'df2_col': col2,
                    'similarity': 0.9,
                    'reason': 'Common key pattern'
                })
    
    return sorted(suggestions, key=lambda x: x['similarity'], reverse=True)

def perform_fuzzy_join(df1, df2, col1, col2, threshold=80):
    """Perform fuzzy join based on string similarity"""
    matches = []
    
    for idx1, val1 in df1[col1].items():
        if pd.isna(val1):
            continue
            
        best_match = process.extractOne(str(val1), df2[col2].astype(str), score_cutoff=threshold)
        
        if best_match:
            match_idx = df2[df2[col2].astype(str) == best_match[0]].index[0]
            matches.append({
                'df1_idx': idx1,
                'df2_idx': match_idx,
                'similarity': best_match[1]
            })
    
    return matches

def perform_range_join(df1, df2, col1, col2, range_col, tolerance=1):
    """Join based on value ranges"""
    matches = []
    
    for idx1, val1 in df1[col1].items():
        if pd.isna(val1):
            continue
            
        # Find matches within range
        range_matches = df2[
            (df2[col2] >= val1 - tolerance) & 
            (df2[col2] <= val1 + tolerance)
        ]
        
        for idx2 in range_matches.index:
            matches.append({
                'df1_idx': idx1,
                'df2_idx': idx2,
                'distance': abs(val1 - df2.loc[idx2, col2])
            })
    
    return matches

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def perform_spatial_join(df1, df2, lat1_col, lon1_col, lat2_col, lon2_col, max_distance=1):
    """Join based on spatial proximity"""
    matches = []
    
    for idx1, row1 in df1.iterrows():
        if pd.isna(row1[lat1_col]) or pd.isna(row1[lon1_col]):
            continue
            
        for idx2, row2 in df2.iterrows():
            if pd.isna(row2[lat2_col]) or pd.isna(row2[lon2_col]):
                continue
                
            distance = calculate_distance(
                row1[lat1_col], row1[lon1_col],
                row2[lat2_col], row2[lon2_col]
            )
            
            if distance <= max_distance:
                matches.append({
                    'df1_idx': idx1,
                    'df2_idx': idx2,
                    'distance': distance
                })
    
    return matches

# ==================== DATABASE CONNECTIONS ====================
def connect_to_database(db_type, connection_params):
    """Enhanced database connection with multiple database types"""
    try:
        if db_type == "PostgreSQL":
            conn = psycopg2.connect(
                host=connection_params['host'],
                port=connection_params['port'],
                database=connection_params['database'],
                user=connection_params['username'],
                password=connection_params['password']
            )
            return pd.read_sql_query(connection_params['query'], conn)
        
        elif db_type == "MySQL":
            conn = mysql.connector.connect(
                host=connection_params['host'],
                port=connection_params['port'],
                database=connection_params['database'],
                user=connection_params['username'],
                password=connection_params['password']
            )
            return pd.read_sql_query(connection_params['query'], conn)
        
        elif db_type == "SQLite":
            conn = sqlite3.connect(connection_params['database_path'])
            return pd.read_sql_query(connection_params['query'], conn)
        
        elif db_type == "MongoDB":
            client = pymongo.MongoClient(
                host=connection_params['host'],
                port=connection_params['port'],
                username=connection_params.get('username'),
                password=connection_params.get('password')
            )
            db = client[connection_params['database']]
            collection = db[connection_params['collection']]
            data = list(collection.find({}))
            return pd.DataFrame(data)
        
        elif db_type == "SQL Server":
            connection_string = f"mssql+pyodbc://{connection_params['username']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['database']}?driver=ODBC+Driver+17+for+SQL+Server"
            engine = sa.create_engine(connection_string)
            return pd.read_sql_query(connection_params['query'], engine)
        
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# ==================== CLOUD STORAGE FUNCTIONS ====================
def load_from_aws_s3(access_key, secret_key, bucket_name, file_key):
    """Load data from AWS S3"""
    try:
        s3_client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        
        if file_key.endswith('.csv'):
            return pd.read_csv(obj['Body'])
        elif file_key.endswith('.json'):
            return pd.read_json(obj['Body'])
        elif file_key.endswith('.parquet'):
            return pd.read_parquet(obj['Body'])
        
    except Exception as e:
        st.error(f"S3 connection failed: {str(e)}")
        return None

def load_from_gcp_storage(service_account_json, bucket_name, blob_name):
    """Load data from Google Cloud Storage"""
    try:
        client = storage.Client.from_service_account_info(json.loads(service_account_json))
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        data = blob.download_as_bytes()
        
        if blob_name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(data))
        elif blob_name.endswith('.json'):
            return pd.read_json(io.BytesIO(data))
        elif blob_name.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(data))
        
    except Exception as e:
        st.error(f"GCP Storage connection failed: {str(e)}")
        return None

def load_from_azure_blob(connection_string, container_name, blob_name):
    """Load data from Azure Blob Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        data = blob_client.download_blob().readall()
        
        if blob_name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(data))
        elif blob_name.endswith('.json'):
            return pd.read_json(io.BytesIO(data))
        elif blob_name.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(data))
        
    except Exception as e:
        st.error(f"Azure Blob connection failed: {str(e)}")
        return None

def load_from_dropbox(access_token, file_path):
    """Load data from Dropbox"""
    try:
        dbx = dropbox.Dropbox(access_token)
        metadata, response = dbx.files_download(file_path)
        
        data = response.content
        
        if file_path.endswith('.csv'):
            return pd.read_csv(io.BytesIO(data))
        elif file_path.endswith('.json'):
            return pd.read_json(io.BytesIO(data))
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(io.BytesIO(data))
        
    except Exception as e:
        st.error(f"Dropbox connection failed: {str(e)}")
        return None

# ==================== SPECIAL FORMAT PROCESSORS ====================
def process_special_formats(uploaded_file):
    """Process special file formats"""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'parquet':
            return pd.read_parquet(uploaded_file)
        
        elif file_extension == 'feather':
            return pd.read_feather(uploaded_file)
        
        elif file_extension == 'h5' or file_extension == 'hdf5':
            # For HDF5, we need to specify which dataset to read
            st.info("HDF5 file detected. Attempting to read the first dataset...")
            with h5py.File(uploaded_file, 'r') as f:
                # Get the first dataset key
                first_key = list(f.keys())[0]
                data = f[first_key][:]
                return pd.DataFrame(data)
        
        elif file_extension == 'xml':
            content = uploaded_file.read().decode('utf-8')
            xml_dict = xmltodict.parse(content)
            return pd.json_normalize(xml_dict)
        
        elif file_extension == 'zip':
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                st.write("Files in ZIP:", file_list)
                
                # Try to read the first CSV file in the ZIP
                for file_name in file_list:
                    if file_name.endswith('.csv'):
                        with zip_ref.open(file_name) as csv_file:
                            return pd.read_csv(csv_file)
                
                st.warning("No CSV files found in ZIP archive")
                return None
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return None
            
    except Exception as e:
        st.error(f"Error processing {file_extension} file: {str(e)}")
        return None

# ==================== MAIN UPLOAD INTERFACE ====================

# Create tabs for different upload methods
upload_tabs = st.tabs([
    "📂 File Upload", 
    "🗄️ Database Connection", 
    "☁️ Cloud Storage", 
    "📋 Special Formats",
    "🔗 API & Web Data"
])

# ==================== TAB 1: FILE UPLOAD ====================
with upload_tabs[0]:
    st.markdown("### 📂 Standard File Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>💡 Supported Formats:</strong> CSV, Excel (.xlsx, .xls), JSON, TSV, TXT
            <br><strong>🚀 Auto-Detection:</strong> Data types, encodings, delimiters, headers
            <br><strong>📊 Instant Preview:</strong> Sample data and basic statistics
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'xlsx', 'xls', 'json', 'tsv', 'txt'],
            help="Upload your dataset for automatic processing and profiling"
        )
        
        if uploaded_file is not None:
            try:
                # Determine file type and process
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv' or file_extension == 'tsv':
                    separator = '\t' if file_extension == 'tsv' else ','
                    df = pd.read_csv(uploaded_file, sep=separator)
                elif file_extension in ['xlsx', 'xls']:
                    # Handle multiple sheets
                    excel_file = pd.ExcelFile(uploaded_file)
                    if len(excel_file.sheet_names) > 1:
                        sheet_name = st.selectbox("Select sheet:", excel_file.sheet_names)
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    else:
                        df = pd.read_excel(uploaded_file)
                elif file_extension == 'json':
                    content = uploaded_file.read()
                    json_data = json.loads(content)
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    else:
                        df = pd.json_normalize(json_data)
                
                # Store in session state
                st.session_state.current_dataset = df
                st.session_state.original_dataset = df.copy()
                
                # Track milestone activity
                milestone_rewards.track_user_activity('dataset_uploaded', {'size': len(df), 'filename': uploaded_file.name})
                
                # Log the upload
                log_dataset_upload(
                    uploaded_file.name, 
                    file_extension.upper(), 
                    len(df), 
                    len(df.columns),
                    f"{uploaded_file.size / 1024:.1f} KB"
                )
                
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
                st.write(f"📊 Dataset: {len(df)} rows × {len(df.columns)} columns")
                
                # Trigger tour progression if active
                if st.session_state.tour_active:
                    guided_tour.show_celebration("first_upload")
                    guided_tour.check_trigger_conditions("upload", {"action": "file_uploaded"})
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    with col2:
        # Upload statistics
        if st.session_state.upload_log:
            st.markdown("### 📈 Upload History")
            recent_uploads = st.session_state.upload_log[-5:]  # Show last 5 uploads
            
            for upload in reversed(recent_uploads):
                with st.expander(f"📄 {upload['source']} - {upload['upload_date'][-8:]}"):
                    st.write(f"**Type:** {upload['source_type']}")
                    st.write(f"**Size:** {upload['rows']} rows × {upload['columns']} cols")
                    st.write(f"**Info:** {upload['size_info']}")

# ==================== TAB 2: DATABASE CONNECTION ====================
with upload_tabs[1]:
    st.markdown("### 🗄️ Database Integration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Connection Settings")
        
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite", "MongoDB", "SQL Server", "Oracle"],
            help="Select your database type for connection"
        )
        
        if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
            host = st.text_input("Host", value="localhost", help="Database server hostname or IP")
            port = st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306, help="Database port number")
            database = st.text_input("Database Name", help="Name of the database to connect to")
            username = st.text_input("Username", help="Database username")
            password = st.text_input("Password", type="password", help="Database password")
            
            query = st.text_area(
                "SQL Query",
                value="SELECT * FROM your_table LIMIT 1000;",
                help="SQL query to fetch data from the database"
            )
            
        elif db_type == "SQLite":
            database_path = st.text_input("Database File Path", help="Path to SQLite database file")
            query = st.text_area(
                "SQL Query",
                value="SELECT * FROM your_table LIMIT 1000;",
                help="SQL query to fetch data"
            )
            
        elif db_type == "MongoDB":
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=27017)
            database = st.text_input("Database Name")
            collection = st.text_input("Collection Name")
            username = st.text_input("Username (optional)")
            password = st.text_input("Password (optional)", type="password")
    
    with col2:
        st.markdown("#### 🔗 Test & Connect")
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Supported Databases:</strong>
            <ul>
                <li>PostgreSQL - Advanced relational database</li>
                <li>MySQL - Popular open-source database</li>
                <li>SQLite - Lightweight file-based database</li>
                <li>MongoDB - Document-oriented NoSQL database</li>
                <li>SQL Server - Microsoft database system</li>
                <li>Oracle - Enterprise database solution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔌 Connect & Load Data", type="primary"):
            with st.spinner("Connecting to database..."):
                connection_params = {}
                
                if db_type in ["PostgreSQL", "MySQL", "SQL Server", "Oracle"]:
                    connection_params = {
                        'host': host,
                        'port': port,
                        'database': database,
                        'username': username,
                        'password': password,
                        'query': query
                    }
                elif db_type == "SQLite":
                    connection_params = {
                        'database_path': database_path,
                        'query': query
                    }
                elif db_type == "MongoDB":
                    connection_params = {
                        'host': host,
                        'port': port,
                        'database': database,
                        'collection': collection,
                        'username': username if username else None,
                        'password': password if password else None
                    }
                
                df = connect_to_database(db_type, connection_params)
                
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    
                    log_dataset_upload(
                        f"{db_type} - {database}",
                        "Database",
                        len(df),
                        len(df.columns),
                        f"Query: {query[:50]}..."
                    )
                    
                    st.success(f"✅ Connected to {db_type} successfully!")
                    st.write(f"📊 Loaded: {len(df)} rows × {len(df.columns)} columns")

# ==================== TAB 3: CLOUD STORAGE ====================
with upload_tabs[2]:
    st.markdown("### ☁️ Cloud Storage Integration")
    
    cloud_provider = st.selectbox(
        "Cloud Provider",
        ["AWS S3", "Google Cloud Storage", "Azure Blob Storage", "Dropbox"],
        help="Select your cloud storage provider"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if cloud_provider == "AWS S3":
            st.markdown("#### 🟠 AWS S3 Configuration")
            aws_access_key = st.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
            s3_bucket = st.text_input("S3 Bucket Name")
            s3_file_key = st.text_input("File Key (path/filename.csv)")
            
            if st.button("📥 Load from S3", type="primary"):
                df = load_from_aws_s3(aws_access_key, aws_secret_key, s3_bucket, s3_file_key)
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    log_dataset_upload(f"S3: {s3_bucket}/{s3_file_key}", "AWS S3", len(df), len(df.columns))
                    st.success(f"✅ Loaded from S3: {len(df)} rows × {len(df.columns)} columns")
        
        elif cloud_provider == "Google Cloud Storage":
            st.markdown("#### 🔵 Google Cloud Storage Configuration")
            gcp_service_account = st.text_input("Service Account JSON", type="password")
            #gcp_service_account = st.text_area("Service Account JSON", type="password")
            gcp_bucket = st.text_input("GCS Bucket Name")
            gcp_blob = st.text_input("Blob Name (filename.csv)")
            
            if st.button("📥 Load from GCS", type="primary"):
                df = load_from_gcp_storage(gcp_service_account, gcp_bucket, gcp_blob)
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    log_dataset_upload(f"GCS: {gcp_bucket}/{gcp_blob}", "Google Cloud", len(df), len(df.columns))
                    st.success(f"✅ Loaded from GCS: {len(df)} rows × {len(df.columns)} columns")
        
        elif cloud_provider == "Azure Blob Storage":
            st.markdown("#### 🔷 Azure Blob Storage Configuration")
            azure_connection_string = st.text_input("Connection String", type="password")
            azure_container = st.text_input("Container Name")
            azure_blob = st.text_input("Blob Name (filename.csv)")
            
            if st.button("📥 Load from Azure", type="primary"):
                df = load_from_azure_blob(azure_connection_string, azure_container, azure_blob)
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    log_dataset_upload(f"Azure: {azure_container}/{azure_blob}", "Azure Blob", len(df), len(df.columns))
                    st.success(f"✅ Loaded from Azure: {len(df)} rows × {len(df.columns)} columns")
        
        elif cloud_provider == "Dropbox":
            st.markdown("#### 📦 Dropbox Configuration")
            dropbox_token = st.text_input("Access Token", type="password")
            dropbox_path = st.text_input("File Path (/path/to/file.csv)")
            
            if st.button("📥 Load from Dropbox", type="primary"):
                df = load_from_dropbox(dropbox_token, dropbox_path)
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    log_dataset_upload(f"Dropbox: {dropbox_path}", "Dropbox", len(df), len(df.columns))
                    st.success(f"✅ Loaded from Dropbox: {len(df)} rows × {len(df.columns)} columns")
    
    with col2:
        st.markdown("#### 🔑 Authentication Help")
        
        if cloud_provider == "AWS S3":
            st.markdown("""
            <div class="info-box">
                <strong>AWS S3 Setup:</strong><br>
                1. Go to AWS IAM Console<br>
                2. Create new Access Key<br>
                3. Ensure S3 read permissions<br>
                4. Copy Access Key ID and Secret
            </div>
            """, unsafe_allow_html=True)
        
        elif cloud_provider == "Google Cloud Storage":
            st.markdown("""
            <div class="info-box">
                <strong>GCS Setup:</strong><br>
                1. Go to GCP Console<br>
                2. Create Service Account<br>
                3. Download JSON key file<br>
                4. Paste JSON content above
            </div>
            """, unsafe_allow_html=True)
        
        elif cloud_provider == "Azure Blob Storage":
            st.markdown("""
            <div class="info-box">
                <strong>Azure Setup:</strong><br>
                1. Go to Azure Portal<br>
                2. Navigate to Storage Account<br>
                3. Copy Connection String<br>
                4. Paste in field above
            </div>
            """, unsafe_allow_html=True)
        
        elif cloud_provider == "Dropbox":
            st.markdown("""
            <div class="info-box">
                <strong>Dropbox Setup:</strong><br>
                1. Go to Dropbox App Console<br>
                2. Create new App<br>
                3. Generate Access Token<br>
                4. Use full file paths (/folder/file.csv)
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 4: SPECIAL FORMATS ====================
with upload_tabs[3]:
    st.markdown("### 📋 Special File Formats")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <strong>🚀 Advanced Format Support:</strong>
            <ul>
                <li><strong>Parquet:</strong> High-performance columnar format</li>
                <li><strong>Feather:</strong> Fast binary format for Python/R</li>
                <li><strong>HDF5:</strong> Hierarchical data format</li>
                <li><strong>XML:</strong> Extensible markup language files</li>
                <li><strong>ZIP:</strong> Compressed archive files</li>
                <li><strong>Avro:</strong> Data serialization format</li>
                <li><strong>ORC:</strong> Optimized row columnar format</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        special_file = st.file_uploader(
            "Upload Special Format File",
            type=['parquet', 'feather', 'h5', 'hdf5', 'xml', 'zip', 'avro', 'orc'],
            help="Upload advanced format files for processing"
        )
        
        if special_file is not None:
            with st.spinner("Processing special format file..."):
                df = process_special_formats(special_file)
                
                if df is not None:
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    
                    file_extension = special_file.name.split('.')[-1].upper()
                    log_dataset_upload(
                        special_file.name,
                        file_extension,
                        len(df),
                        len(df.columns),
                        f"{special_file.size / 1024:.1f} KB"
                    )
                    
                    st.success(f"✅ Successfully processed {file_extension} file")
                    st.write(f"📊 Dataset: {len(df)} rows × {len(df.columns)} columns")
    
    with col2:
        st.markdown("#### 📊 Format Advantages")
        
        format_info = {
            "Parquet": "🚀 Fast columnar storage, excellent compression",
            "Feather": "⚡ Ultra-fast binary format for data exchange",
            "HDF5": "🗂️ Hierarchical data, supports complex structures",
            "XML": "📋 Structured markup, web-friendly format",
            "ZIP": "📦 Compressed archives, multiple files support"
        }
        
        for format_name, description in format_info.items():
            st.markdown(f"**{format_name}:** {description}")

# ==================== TAB 5: API & WEB DATA ====================
with upload_tabs[4]:
    st.markdown("### 🔗 API & Web Data Integration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🌐 REST API Data")
        
        api_url = st.text_input("API URL", help="Enter the REST API endpoint URL")
        
        # API Authentication
        auth_type = st.selectbox("Authentication", ["None", "API Key", "Bearer Token", "Basic Auth"])
        
        if auth_type == "API Key":
            api_key_name = st.text_input("API Key Parameter Name", value="api_key")
            api_key_value = st.text_input("API Key Value", type="password")
        elif auth_type == "Bearer Token":
            bearer_token = st.text_input("Bearer Token", type="password")
        elif auth_type == "Basic Auth":
            basic_username = st.text_input("Username")
            basic_password = st.text_input("Password", type="password")
        
        # HTTP Method
        http_method = st.selectbox("HTTP Method", ["GET", "POST"])
        
        if http_method == "POST":
            post_data = st.text_area("POST Data (JSON)", help="JSON data to send with POST request")
        
        if st.button("🌐 Fetch API Data", type="primary"):
            try:
                import requests
                
                headers = {}
                params = {}
                
                if auth_type == "API Key":
                    params[api_key_name] = api_key_value
                elif auth_type == "Bearer Token":
                    headers["Authorization"] = f"Bearer {bearer_token}"
                
                if http_method == "GET":
                    response = requests.get(api_url, headers=headers, params=params)
                else:
                    data = json.loads(post_data) if post_data else {}
                    response = requests.post(api_url, headers=headers, params=params, json=data)
                
                if response.status_code == 200:
                    json_data = response.json()
                    
                    if isinstance(json_data, list):
                        df = pd.DataFrame(json_data)
                    else:
                        df = pd.json_normalize(json_data)
                    
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    
                    log_dataset_upload(
                        api_url,
                        "REST API",
                        len(df),
                        len(df.columns),
                        f"HTTP {response.status_code}"
                    )
                    
                    st.success(f"✅ API data loaded: {len(df)} rows × {len(df.columns)} columns")
                else:
                    st.error(f"❌ API request failed: HTTP {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ API request error: {str(e)}")
    
    with col2:
        st.markdown("#### 📄 Web Scraping")
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Web Scraping Notice:</strong><br>
            Always ensure you comply with website terms of service and robots.txt.
            Some websites may require permission for data extraction.
        </div>
        """, unsafe_allow_html=True)
        
        web_url = st.text_input("Website URL", help="URL containing table data to scrape")
        
        if st.button("🕷️ Scrape Web Data"):
            try:
                import requests
                from bs4 import BeautifulSoup
                
                response = requests.get(web_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find tables in the webpage
                tables = pd.read_html(web_url)
                
                if tables:
                    df = tables[0]  # Use the first table found
                    
                    st.session_state.current_dataset = df
                    st.session_state.original_dataset = df.copy()
                    
                    log_dataset_upload(
                        web_url,
                        "Web Scraping",
                        len(df),
                        len(df.columns),
                        f"Table from webpage"
                    )
                    
                    st.success(f"✅ Web data scraped: {len(df)} rows × {len(df.columns)} columns")
                else:
                    st.warning("⚠️ No tables found on the webpage")
                    
            except Exception as e:
                st.error(f"❌ Web scraping error: {str(e)}")

# ==================== DATA PREVIEW SECTION ====================
if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
    st.markdown("---")
    st.markdown("### 📊 Dataset Preview & Auto-Analysis")
    
    df = st.session_state.current_dataset
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Data preview tabs
    preview_tabs = st.tabs(["🔍 Data Preview", "📈 Quick Stats", "🔧 Data Quality", "⚠️ Issues Detected"])
    
    with preview_tabs[0]:
        st.markdown("#### First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("#### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        # Convert to string to avoid Arrow issues
        for col in col_info.columns:
            col_info[col] = col_info[col].astype(str)
        st.dataframe(col_info, use_container_width=True)
    
    with preview_tabs[1]:
        st.markdown("#### Statistical Summary")
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                stats_df = df[numeric_cols].describe()
                # Convert to string to avoid Arrow issues
                for col in stats_df.columns:
                    stats_df[col] = stats_df[col].astype(str)
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
        except Exception as e:
            st.warning(f"Unable to generate statistics: {str(e)}")
    
    with preview_tabs[2]:
        st.markdown("#### Data Quality Assessment")
        
        # Quality metrics
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with col2:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicate_rows:,}")
        with col3:
            mixed_types = sum(1 for col in df.columns if df[col].dtype == 'object')
            st.metric("Text Columns", f"{mixed_types}")
    
    with preview_tabs[3]:
        st.markdown("#### Potential Issues")
        
        issues = []
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"🔴 Missing values in {len(missing_cols)} columns: {', '.join(missing_cols[:5])}")
        
        # Check for duplicates
        if df.duplicated().any():
            issues.append(f"🟡 {df.duplicated().sum()} duplicate rows found")
        
        # Check for potential date columns
        potential_dates = []
        for col in df.select_dtypes(include=['object']).columns:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                potential_dates.append(col)
            except:
                pass
        
        if potential_dates:
            issues.append(f"🔵 Potential date columns detected: {', '.join(potential_dates)}")
        
        # Check for high cardinality categorical columns
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.7:
                high_card_cols.append(col)
        
        if high_card_cols:
            issues.append(f"🟠 High cardinality columns: {', '.join(high_card_cols)}")
        
        if issues:
            for issue in issues:
                st.markdown(f"- {issue}")
        else:
            st.success("✅ No major data quality issues detected!")
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔍 View Data Overview", type="primary"):
            st.switch_page("pages/02_Data_Overview.py")
    
    with col2:
        if st.button("🤖 Auto Clean Pipeline", type="primary"):
            if st.session_state.tour_active:
                guided_tour.check_trigger_conditions("upload", {"action": "choose_pipeline"})
                guided_tour.show_character_hint("", custom_message="Great choice! Auto mode is perfect for beginners. I'll handle most of the cleaning decisions for you!", hint_type="tip")
            st.switch_page("pages/04_Clean_Pipeline.py")
    
    with col3:
        if st.button("🧹 Manual Clean Pipeline", type="primary"):
            if st.session_state.tour_active:
                guided_tour.check_trigger_conditions("upload", {"action": "choose_pipeline"})
            st.switch_page("pages/04_Clean_Pipeline.py")

# ==================== ADVANCED MERGING & JOINING OPERATIONS ====================
st.markdown("---")
st.markdown("### 🔗 Advanced Grouping, Merging & Joining Operations")

# Main operations tabs (removed Dataset Information and Upload Logging)
merge_tabs = st.tabs([
    "🔄 Automated Merging", 
    "🎯 Advanced Joining", 
    "📊 Grouping Operations"
])

# ====================TAB 1: AUTOMATED MERGING ====================
with merge_tabs[0]:
    st.markdown(" 🔄 Merge/Join Datasets Automatically (AI-Selected Keys)")
    st.markdown("""
    <div style="background: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 10px 0;">
        <strong>💡 Smart Automation:</strong> The app will automatically suggest the best keys and join types 
        and process dataset merges without manual intervention.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📥 Upload Additional Dataset to Merge")
        
        merge_uploaded_file = st.file_uploader(
            "Choose a dataset to merge with current data",
            type=['csv', 'xlsx', 'xls', 'json'],
            help="Max file size: 200 MB. Supported formats: CSV, XLSX, XLS, JSON",
            key="merge_upload"
        )
        
        if merge_uploaded_file is not None:
            # Validate file size (200 MB limit)
            file_size = len(merge_uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            
            if file_size > 200:
                st.error(f"❌ File size ({file_size:.1f} MB) exceeds 200 MB limit")
            else:
                    try:
                        # Process the uploaded file
                        file_extension = merge_uploaded_file.name.split('.')[-1].lower()
                        
                        if file_extension == 'csv':
                            merge_df = pd.read_csv(merge_uploaded_file)
                        elif file_extension in ['xlsx', 'xls']:
                            merge_df = pd.read_excel(merge_uploaded_file)
                        elif file_extension == 'json':
                            content = merge_uploaded_file.read()
                            json_data = json.loads(content)
                            if isinstance(json_data, list):
                                merge_df = pd.DataFrame(json_data)
                            else:
                                merge_df = pd.json_normalize(json_data)
                        
                        st.success(f"✅ File uploaded successfully: {len(merge_df)} rows × {len(merge_df.columns)} columns")
                        
                        # Store merge dataset
                        st.session_state.merge_datasets.append({
                            'name': merge_uploaded_file.name,
                            'data': merge_df,
                            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        # Auto-suggest merge options if main dataset exists
                        if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
                            main_df = st.session_state.current_dataset
                            
                            st.markdown("#### 🤖 AI-Powered Merge Suggestions")
                            
                            # Get column suggestions
                            suggestions = suggest_join_columns(main_df, merge_df)
                            
                            if suggestions:
                                best_suggestion = suggestions[0]
                                
                                st.markdown(f"""
                                <div style="background: #f0f8f0; padding: 15px; border-left: 4px solid #4CAF50; margin: 10px 0;">
                                    <strong>🎯 Best Match Found:</strong><br>
                                    Main Dataset: <code>{best_suggestion['df1_col']}</code> ↔ 
                                    Merge Dataset: <code>{best_suggestion['df2_col']}</code><br>
                                    <strong>Reason:</strong> {best_suggestion['reason']} 
                                    ({best_suggestion['similarity']:.1%} similarity)
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Auto merge options
                                col_a, col_b, col_c = st.columns(3)
                                
                                with col_a:
                                    join_type = st.selectbox(
                                        "Join Type",
                                        ["inner", "left", "right", "outer"],
                                        index=1,  # Default to left join
                                        help="AI recommends 'left' join to preserve main dataset"
                                    )
                                
                                with col_b:
                                    handle_conflicts = st.selectbox(
                                        "Handle Conflicts",
                                        ["Add suffix (_left, _right)", "Drop duplicates", "Keep left", "Keep right"],
                                        help="How to handle conflicting column names"
                                    )
                                
                                with col_c:
                                    if st.button("🚀 Auto-Merge Datasets", type="primary"):
                                        try:
                                            # Perform the merge
                                            if handle_conflicts == "Add suffix (_left, _right)":
                                                merged_df = pd.merge(
                                                    main_df, merge_df,
                                                    left_on=best_suggestion['df1_col'],
                                                    right_on=best_suggestion['df2_col'],
                                                    how=join_type,
                                                    suffixes=('_left', '_right')
                                                )
                                            else:
                                                merged_df = pd.merge(
                                                    main_df, merge_df,
                                                    left_on=best_suggestion['df1_col'],
                                                    right_on=best_suggestion['df2_col'],
                                                    how=join_type
                                                )
                                            
                                            # Update session state
                                            st.session_state.current_dataset = merged_df
                                            
                                            log_dataset_upload(
                                                f"Merged: {merge_uploaded_file.name}",
                                                "Auto-Merge",
                                                len(merged_df),
                                                len(merged_df.columns),
                                                f"{join_type} join on {best_suggestion['df1_col']}"
                                            )
                                            
                                            st.balloons()
                                            st.success(f"✅ Datasets merged successfully! New shape: {len(merged_df)} rows × {len(merged_df.columns)} columns")
                                            st.rerun()
                                            
                                        except Exception as e:
                                            st.error(f"❌ Merge failed: {str(e)}")
                            else:
                                st.warning("⚠️ No obvious join columns detected. Try manual configuration below.")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing file: {str(e)}")
        
    with col2:
        st.markdown("#### 🧭 Navigation (Automated Workflow)")
        
        if st.button("⬅️ Previous Step", use_container_width=True):
            st.info("Currently at Upload stage")
        
        if st.button("➡️ Next Step", use_container_width=True):
            if 'current_dataset' in st.session_state:
                st.switch_page("pages/02_Data_Overview.py")
            else:
                st.warning("Please upload a dataset first")
            
            st.markdown("---")
            st.markdown("#### 📊 Merge Statistics")
            
            if st.session_state.merge_datasets:
                st.metric("Datasets Ready", len(st.session_state.merge_datasets))
                for ds in st.session_state.merge_datasets[-3:]:  # Show last 3
                    st.text(f"📁 {ds['name'][:15]}...")
            else:
                st.info("No datasets uploaded for merging")
    
# ====================TAB 2: ADVANCED JOINING ====================
with merge_tabs[1]:
    st.markdown(" 🎯 Advanced Joining Operations")
    
    if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None and st.session_state.merge_datasets:
        main_df = st.session_state.current_dataset
        
        join_col1, join_col2 = st.columns([1, 1])
        
        with join_col1:
            st.markdown("#### 🔍 Fuzzy Joins")
            
            selected_merge_dataset = st.selectbox(
                "Select dataset to join",
                [ds['name'] for ds in st.session_state.merge_datasets],
                key="fuzzy_join_select"
            )
            
            if selected_merge_dataset:
                merge_df = next(ds['data'] for ds in st.session_state.merge_datasets if ds['name'] == selected_merge_dataset)
                    
                col_a, col_b = st.columns(2)
                with col_a:
                    main_col = st.selectbox("Main dataset column", main_df.columns, key="fuzzy_main_col")
                with col_b:
                    merge_col = st.selectbox("Merge dataset column", merge_df.columns, key="fuzzy_merge_col")
                    
                similarity_threshold = st.slider("Similarity threshold (%)", 50, 100, 80)
                
                if st.button("🔗 Perform Fuzzy Join"):
                    try:
                        matches = perform_fuzzy_join(main_df, merge_df, main_col, merge_col, similarity_threshold)
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} fuzzy matches")
                            
                            # Create a preview of matches
                            match_preview = []
                            for match in matches[:10]:  # Show first 10 matches
                                match_preview.append({
                                    'Main Value': main_df.loc[match['df1_idx'], main_col],
                                    'Matched Value': merge_df.loc[match['df2_idx'], merge_col],
                                    'Similarity': f"{match['similarity']:.1f}%"
                                })
                            
                            st.dataframe(pd.DataFrame(match_preview), use_container_width=True)
                        else:
                            st.warning("⚠️ No fuzzy matches found with current threshold")
                            
                    except Exception as e:
                        st.error(f"❌ Fuzzy join failed: {str(e)}")
            
        with join_col2:
            st.markdown("#### 📏 Range-Based Joins")
            
            numeric_cols_main = main_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols_merge = merge_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols_main and numeric_cols_merge:
                col_a, col_b = st.columns(2)
                with col_a:
                    range_main_col = st.selectbox("Main numeric column", numeric_cols_main, key="range_main_col")
                with col_b:
                    range_merge_col = st.selectbox("Merge numeric column", numeric_cols_merge, key="range_merge_col")
                    
                tolerance = st.number_input("Range tolerance", min_value=0.1, value=1.0, step=0.1)
                
                if st.button("🎯 Perform Range Join"):
                    try:
                        matches = perform_range_join(main_df, merge_df, range_main_col, range_merge_col, range_merge_col, tolerance)
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} range-based matches")
                            
                            # Create a preview of matches
                            match_preview = []
                            for match in matches[:10]:
                                match_preview.append({
                                    'Main Value': main_df.loc[match['df1_idx'], range_main_col],
                                    'Matched Value': merge_df.loc[match['df2_idx'], range_merge_col],
                                    'Distance': f"{match['distance']:.2f}"
                                })
                            
                            st.dataframe(pd.DataFrame(match_preview), use_container_width=True)
                        else:
                            st.warning("⚠️ No range matches found within tolerance")
                            
                    except Exception as e:
                        st.error(f"❌ Range join failed: {str(e)}")
            else:
                st.info("ℹ️ No numeric columns available for range joining")
            
            st.markdown("#### 🌍 Spatial Joins")
            
            # Look for potential coordinate columns
            coord_patterns = ['lat', 'lon', 'latitude', 'longitude']
            main_coord_cols = [col for col in main_df.columns if any(pattern in col.lower() for pattern in coord_patterns)]
            merge_coord_cols = [col for col in merge_df.columns if any(pattern in col.lower() for pattern in coord_patterns)]
            
            if main_coord_cols and merge_coord_cols:
                col_a, col_b = st.columns(2)
                with col_a:
                    main_lat = st.selectbox("Main latitude", main_coord_cols, key="main_lat")
                    main_lon = st.selectbox("Main longitude", main_coord_cols, key="main_lon")
                with col_b:
                    merge_lat = st.selectbox("Merge latitude", merge_coord_cols, key="merge_lat")
                    merge_lon = st.selectbox("Merge longitude", merge_coord_cols, key="merge_lon")
                
                max_distance = st.number_input("Max distance (km)", min_value=0.1, value=1.0, step=0.1)
                
                if st.button("🗺️ Perform Spatial Join"):
                    try:
                        matches = perform_spatial_join(main_df, merge_df, main_lat, main_lon, merge_lat, merge_lon, max_distance)
                        
                        if matches:
                            st.success(f"✅ Found {len(matches)} spatial matches")
                            
                            # Create a preview of matches
                            match_preview = []
                            for match in matches[:10]:
                                match_preview.append({
                                    'Main Coordinates': f"({main_df.loc[match['df1_idx'], main_lat]:.4f}, {main_df.loc[match['df1_idx'], main_lon]:.4f})",
                                    'Matched Coordinates': f"({merge_df.loc[match['df2_idx'], merge_lat]:.4f}, {merge_df.loc[match['df2_idx'], merge_lon]:.4f})",
                                    'Distance (km)': f"{match['distance']:.2f}"
                                })
                            
                            st.dataframe(pd.DataFrame(match_preview), use_container_width=True)
                        else:
                            st.warning("⚠️ No spatial matches found within distance")
                            
                    except Exception as e:
                        st.error(f"❌ Spatial join failed: {str(e)}")
            else:
                st.info("ℹ️ No coordinate columns detected for spatial joining")
    else:
        st.info("ℹ️ Upload datasets to enable advanced joining operations")
    
# ==================== TAB 3: GROUPING OPERATIONS ====================
with merge_tabs[2]:
    st.markdown("📊 Grouping Operations")
    
    if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        
        col1, col2 = st.columns([2, 1])
        
        with col1:

            # Suggest grouping columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_cols and numeric_cols:
                group_col = st.selectbox("Group by column", categorical_cols)
                
                agg_operations = st.multiselect(
                    "Aggregation operations",
                    ["count", "sum", "mean", "median", "std", "min", "max"],
                    default=["count", "mean"]
                )
                
                target_cols = st.multiselect("Target columns", numeric_cols, default=numeric_cols[:3])
                
                if st.button("📊 Perform Grouping"):
                    try:
                        grouped = df.groupby(group_col)[target_cols].agg(agg_operations)
                        
                        st.success(f"✅ Grouped by {group_col}")
                        st.dataframe(grouped, use_container_width=True)
                        
                        # Option to save grouped results
                        if st.button("💾 Save Grouped Results"):
                            st.session_state.current_dataset = grouped.reset_index()
                            st.success("✅ Grouped results saved as current dataset")
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Grouping failed: {str(e)}")
            else:
                st.info("ℹ️ Need both categorical and numeric columns for grouping")
        
        with col2:
            st.markdown("#### 📈 Grouping Statistics")
            
            if categorical_cols:
                for col in categorical_cols[:3]:  # Show stats for first 3 categorical columns
                    unique_count = df[col].nunique()
                    st.metric(f"{col} Groups", unique_count)
                    
                    if unique_count <= 10:  # Show value counts for low cardinality
                        value_counts = df[col].value_counts().head(5)
                        st.write(f"**Top {col} values:**")
                        for val, count in value_counts.items():
                            st.write(f"• {val}: {count}")
    else:
        st.info("ℹ️ Upload a dataset to enable grouping operations")

# ==================== DATASET INFORMATION SECTION====================
st.markdown("---")
st.markdown("### 📋 Dataset Information")

# Calculate aggregated information across all uploaded datasets
if st.session_state.upload_log:
    total_datasets = len(st.session_state.upload_log)
    total_rows = sum(upload['rows'] for upload in st.session_state.upload_log)
    total_columns = sum(upload['columns'] for upload in st.session_state.upload_log)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Datasets Uploaded", f"{total_datasets:,}")
    with col2:
        st.metric("Total Rows Processed", f"{total_rows:,}")
    with col3:
        st.metric("Total Columns Processed", f"{total_columns:,}")
    with col4:
        # Calculate total processing time saved across all datasets
        total_manual_time = 0
        total_auto_time = 0
        
        for upload in st.session_state.upload_log:
            # Estimate based on rows and columns
            manual_time = (upload['columns'] * 8) / 60  # hours
            auto_time = (upload['columns'] * 0.5) / 60  # hours
            total_manual_time += manual_time
            total_auto_time += auto_time
        
        time_saved = (total_manual_time - total_auto_time) * 60  # Convert to minutes
        st.metric("Total Time Saved", f"{time_saved:.0f} min")
    
    # Current dataset details
    if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Current Dataset Details")
            
            dataset_info = {
                'Active Dataset': 'Current Working Dataset',
                'File Type': 'DataFrame',
                'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
                'Row Count': f"{len(df):,}",
                'Column Count': f"{len(df.columns):,}",
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            for key, value in dataset_info.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            st.markdown("#### 🤖 Smart Automation Summary Panel")
            
            stats = st.session_state.automation_stats
            
            # Detected issues across all datasets
            st.markdown("**🔍 Detected Issues & Recommendations:**")
            st.metric("Numeric columns for scaling", f"{stats['numeric_cols_for_scaling']}")
            st.metric("Categorical columns for encoding", f"{stats['categorical_cols_for_encoding']}")
            
            # Time savings for current dataset
            st.markdown("**⏱️ Current Dataset Time Savings:**")
            manual_time = stats['manual_time_est'] * 60  # Convert to minutes
            auto_time = stats['auto_time_est'] * 60
            time_saved_current = manual_time - auto_time
            
            if manual_time > 0:
                savings_percent = (time_saved_current / manual_time) * 100
                
                st.metric("Manual Processing Time", f"{manual_time:.0f} min")
                st.metric("Automated Processing Time", f"{auto_time:.0f} min")
                st.metric("Time Saved", f"{time_saved_current:.0f} min ({savings_percent:.0f}%)")
            else:
                st.info("Current dataset time estimates")
else:
    st.info("ℹ️ Upload datasets to see aggregated information and processing statistics")

# ==================== DATASET UPLOAD LOGGING====================
st.markdown("---")
st.markdown("### 📝 Dataset Upload Logging")

if st.session_state.upload_log:
    # Create searchable, sortable upload log
    log_df = pd.DataFrame(st.session_state.upload_log)
    
    # Search functionality
    col1, col2 = st.columns([2, 1])
    with col1:
        search_term = st.text_input("🔍 Search uploads", placeholder="Search by filename, type, or date...")
    with col2:
        sort_by = st.selectbox("Sort by", ["upload_date", "source", "rows", "columns", "source_type"])
    
    # Filter based on search
    if search_term:
        mask = log_df.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)
        filtered_df = log_df[mask]
    else:
        filtered_df = log_df
    
    # Sort
    if sort_by == "upload_date":
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    else:
        filtered_df = filtered_df.sort_values(sort_by)
    
    # Display log table
    st.markdown(f"#### 📊 Upload History ({len(filtered_df)} entries)")
    
    # Convert all columns to strings for display
    display_df = filtered_df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str)
    
    st.dataframe(
        display_df[['source', 'source_type', 'rows', 'columns', 'size_info', 'upload_date', 'status']],
        use_container_width=True
    )
    
    # Export log
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Upload Log"):
            csv = log_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Log as CSV",
                data=csv,
                file_name=f"upload_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🗑️ Clear Upload Log"):
            st.session_state.upload_log = []
            st.success("✅ Upload log cleared")
            st.rerun()
else:
    st.info("📝 No uploads logged yet. Upload a dataset to start tracking.")

# ====================SIDEBAR: UPLOAD MANAGEMENT====================
with st.sidebar:
    st.markdown("### 📂 Upload Management")
    
    if st.session_state.upload_log:
        st.markdown("#### Recent Uploads")
        for i, upload in enumerate(reversed(st.session_state.upload_log[-5:])):
            with st.expander(f"{upload['source'][:20]}... - {upload['upload_date'][-8:]}"):
                st.write(f"**Source:** {upload['source']}")
                st.write(f"**Type:** {upload['source_type']}")
                st.write(f"**Rows:** {upload['rows']:,}")
                st.write(f"**Columns:** {upload['columns']:,}")
                st.write(f"**Info:** {upload['size_info']}")
                st.write(f"**Status:** {upload['status']}")
    
    else:
        st.info("No uploads yet")
    
    if st.button("🗑️ Clear Upload History"):
        st.session_state.upload_log = []
        st.rerun()
    
    # Quick actions
    st.markdown("---")
    st.markdown("#### ⚡ Quick Actions")
    
    if st.button("📊 Sample Dataset"):
        # Load a sample dataset for demonstration
        sample_data = {
            'ID': range(1, 101),
            'Name': [f'User_{i}' for i in range(1, 101)],
            'Age': np.random.randint(18, 65, 100),
            'Salary': np.random.randint(30000, 150000, 100),
            'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing'], 100),
            'Join_Date': pd.date_range('2020-01-01', periods=100, freq='D')
        }
        
        # Add some missing values
        sample_df = pd.DataFrame(sample_data)
        sample_df.loc[np.random.choice(sample_df.index, 10), 'Salary'] = np.nan
        sample_df.loc[np.random.choice(sample_df.index, 5), 'Department'] = np.nan
        
        st.session_state.current_dataset = sample_df
        st.session_state.original_dataset = sample_df.copy()
        
        log_dataset_upload("Sample Dataset", "Generated", 100, 6, "Demo data")
        st.success("✅ Sample dataset loaded!")
        st.rerun()
    
    if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
        if st.button("💾 Download Current Dataset"):
            csv = st.session_state.current_dataset.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
