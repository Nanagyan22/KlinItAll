import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.automation_engine import AutomationEngine
from utils.guided_tour import guided_tour
from utils.milestone_rewards import milestone_rewards
import base64
import os

# Configure page
st.set_page_config(
    page_title="KlinItAll - Smart Data Preprocessing",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'automation_engine' not in st.session_state:
    st.session_state.automation_engine = AutomationEngine()
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'preprocessing_history' not in st.session_state:
    st.session_state.preprocessing_history = []

# Initialize guided tour session state
if 'tour_active' not in st.session_state:
    st.session_state.tour_active = False
if 'tour_step' not in st.session_state:
    st.session_state.tour_step = 0
if 'tour_completed_steps' not in st.session_state:
    st.session_state.tour_completed_steps = set()
if 'show_hints' not in st.session_state:
    st.session_state.show_hints = True

# Custom CSS for branding
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #1f77b4, #17becf);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}
.feature-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.cta-button {
    background: #1f77b4;
    color: white;
    padding: 0.75rem 2rem;
    border-radius: 25px;
    text-decoration: none;
    font-weight: bold;
    display: inline-block;
    margin: 1rem 0;
}
.automation-highlight {
    background: #e8f4f8;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #17becf;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize milestone rewards system
    milestone_rewards.initialize_user_progress()
    
    # Show tour controls in sidebar
    guided_tour.show_tour_controls()
    
    # Show progress widget in sidebar
    milestone_rewards.show_progress_widget()
    
    # Header Section with branding
    st.markdown("""
    <div class="main-header">
        <h1>🧹 KlinItAll</h1>
        <h3>Smart Data Preprocessing System</h3>
        <p>Version 1.0</p>
        <p><em>Intelligent automation • Manual overrides • Complete reproducibility</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Introduction Section
    st.markdown("## Welcome to KlinItAll")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            **KlinItAll** is an intelligent, fully automated data preprocessing system designed to streamline 
            and simplify the cleaning, profiling, and transformation of structured datasets.
            It automates the tedious data preprocessing tasks that typically consume 50%-80% of a data analyst/scientist's time. 
            This intelligent system guides you through a comprehensive cleaning workflow with smart recommendations 
            and one-click solutions. With full automation,you can focus on insights, modeling, and impactful analysis, 
            while KlinItAll handles the repetitive tasks of data cleaning.

            ### Purpose of the App
            KlinItAll automates the entire data preprocessing pipeline—from ingestion to cleaning, profiling, 
            feature engineering, and export. With minimal user input, you get clean, analysis-ready data in no time.
            """)

        # CTA Button
        if st.button("🚀 Get Started", type="primary", use_container_width=True):
            st.switch_page("pages/01_Upload.py")

    with col2:
        st.markdown("""
            <div class="automation-highlight">
            <h4>⚡ Full Automation Features</h4>
            <ul>
            <li>🔍 Smart anomaly detection</li>
            <li>🛠️ One-click fixes</li>
            <li>📊 Automated profiling</li>
            <li>🎯 Intelligent recommendations</li>
            <li>🤖  Data Story Narrator</li>
            <li>⏱️ Time-saving analytics</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

    # System Overview
    st.markdown("---")
    st.markdown("## System Overview")

    # Key Features
    features = [
        {
            "title": "🤖 Full Automation",
            "description": "KlinItAll automates the entire process of data cleaning, from anomaly detection and missing value imputation to feature engineering and data scaling."
        },
        {
            "title": "📥 Data Ingestion",
            "description": "Seamless upload of datasets from multiple formats: CSV, Excel, JSON, SQL, APIs, and Cloud connectors."
        },
        {
            "title": "🧠 Smart Cleaning",
            "description": "Fully automated anomaly detection and suggested fixes with one-click application."
        },
        {
            "title": "⚙️ Feature Engineering",
            "description": "Automated binning, dimensionality reduction, custom feature creation, and encoding based on data characteristics."
        },
        {
            "title": "📚 History & Export",
            "description": "Save preprocessing steps for reproducibility and shareable reports."
        },
        {
            "title": "📈 Scalability",
            "description": "Handle large datasets with full automation, batch processing, and job scheduling."
        }
    ]

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="feature-card">
                <h4>{feature['title']}</h4>
                <p>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Show welcome tour hint for new users
    if st.session_state.tour_active and st.session_state.tour_step == 0:
        guided_tour.show_character_hint("", custom_message="Welcome to KlinItAll! I'm Klini, your friendly data cleaning assistant. Let's start by uploading some data. Click the 'Upload' page in the sidebar to begin!", hint_type="info")

    st.markdown("---")
    # Quick Start Workflow Section
    st.markdown("## 🚀 Quick Start Workflow")
    
    workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)
    
    with workflow_col1:
        if st.button("📥 Start Upload", type="primary", use_container_width=True):
            st.switch_page("pages/01_Upload.py")
        st.markdown("**1. Upload Data**\nCSV, Excel, JSON, SQL, APIs, Cloud")
    
    with workflow_col2:
        if st.button("📊 Profile Data", use_container_width=True):
            if st.session_state.current_dataset is not None:
                st.switch_page("pages/02_Data_Overview.py")
            else:
                st.warning("Upload data first!")
        st.markdown("**2. Profile & Analyze**\nAuto-detection, insights, quality assessment")
    
    with workflow_col3:
        if st.button("🧹 Clean Pipeline", use_container_width=True):
            if st.session_state.current_dataset is not None:
                st.switch_page("pages/04_Clean_Pipeline.py")
            else:
                st.warning("Upload data first!")
        st.markdown("**3. Clean & Process**\nAuto-suggestions, one-click fixes, manual overrides")
    
    with workflow_col4:
        if st.button("💾 Export Results", use_container_width=True):
            if st.session_state.current_dataset is not None:
                st.switch_page("pages/14_History_Export.py")
            else:
                st.warning("Upload data first!")
        st.markdown("**4. Export & Share**\nClean data, pipelines, reproducible code")

    st.markdown("---")

    # Key Features with icons
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            ### 📊 Smart Analysis
            - Automatic data profiling
            - Missing value detection
            - Outlier identification
            - Data type suggestions
            """)

    with col2:
        st.markdown("""
            ### 🤖 AI Recommendations  
            - One-click apply solutions
            - Intelligent preprocessing
            - Automated workflows
            - Manual override options
            """)

    with col3:
        st.markdown("""
            ### 🚀 Export Ready
            - Multiple format support
            - Generated Python code
            - Reproducible results
            - Progress tracking
            """)

    st.markdown("---")
    # Sample Datasets Section
    st.markdown("## 📋 Sample Datasets & Templates")
    
    sample_col1, sample_col2 = st.columns(2)
    
    with sample_col1:
        st.markdown("### 🎯 Industry Templates")
        
        if st.button("💰 Finance Dataset", use_container_width=True):
            # Create sample finance dataset
            np.random.seed(42)
            n_samples = 1000
            
            finance_data = {
                'customer_id': range(1, n_samples + 1),
                'account_balance': np.random.normal(5000, 2000, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'loan_amount': np.random.exponential(10000, n_samples),
                'account_type': np.random.choice(['Checking', 'Savings', 'Credit', 'Investment'], n_samples),
                'transaction_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
                'risk_category': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'annual_income': np.random.normal(60000, 20000, n_samples)
            }
            
            # Add some missing values and outliers
            finance_df = pd.DataFrame(finance_data)
            finance_df.loc[np.random.choice(finance_df.index, 50), 'credit_score'] = np.nan
            finance_df.loc[np.random.choice(finance_df.index, 30), 'annual_income'] = np.nan
            
            st.session_state.current_dataset = finance_df
            st.success("✅ Finance dataset loaded!")
            st.rerun()
        
        if st.button("🛒 Retail Dataset", use_container_width=True):
            # Create sample retail dataset
            np.random.seed(123)
            n_samples = 800
            
            retail_data = {
                'product_id': range(1, n_samples + 1),
                'product_name': [f'Product_{i}' for i in range(1, n_samples + 1)],
                'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], n_samples),
                'price': np.random.exponential(50, n_samples),
                'quantity_sold': np.random.poisson(20, n_samples),
                'customer_rating': np.random.uniform(1, 5, n_samples),
                'supplier_location': np.random.choice(['USA', 'China', 'Germany', 'Japan', 'India'], n_samples),
                'launch_date': pd.date_range('2022-01-01', periods=n_samples, freq='2D'),
                'is_bestseller': np.random.choice([True, False], n_samples),
                'description': [f'Description for product {i}' for i in range(1, n_samples + 1)]
            }
            
            retail_df = pd.DataFrame(retail_data)
            # Add missing values
            retail_df.loc[np.random.choice(retail_df.index, 40), 'customer_rating'] = np.nan
            retail_df.loc[np.random.choice(retail_df.index, 25), 'supplier_location'] = np.nan
            
            st.session_state.current_dataset = retail_df
            st.success("✅ Retail dataset loaded!")
            st.rerun()
    
    with sample_col2:
        st.markdown("### 📊 Specialized Data Types")
        
        if st.button("🌐 Geospatial Dataset", use_container_width=True):
            # Create sample geospatial dataset
            np.random.seed(456)
            n_samples = 500
            
            geo_data = {
                'location_id': range(1, n_samples + 1),
                'latitude': np.random.uniform(25.0, 49.0, n_samples),  # US latitudes
                'longitude': np.random.uniform(-125.0, -66.0, n_samples),  # US longitudes
                'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
                'state': np.random.choice(['NY', 'CA', 'IL', 'TX', 'AZ'], n_samples),
                'population': np.random.exponential(100000, n_samples),
                'temperature': np.random.normal(70, 15, n_samples),
                'elevation': np.random.uniform(0, 5000, n_samples),
                'zip_code': [f'{np.random.randint(10000, 99999):05d}' for _ in range(n_samples)],
                'address': [f'{np.random.randint(1, 9999)} Main St' for _ in range(n_samples)]
            }
            
            geo_df = pd.DataFrame(geo_data)
            # Add missing values
            geo_df.loc[np.random.choice(geo_df.index, 30), 'elevation'] = np.nan
            geo_df.loc[np.random.choice(geo_df.index, 20), 'zip_code'] = np.nan
            
            st.session_state.current_dataset = geo_df
            st.success("✅ Geospatial dataset loaded!")
            st.rerun()
        
        if st.button("📝 Text Dataset", use_container_width=True):
            # Create sample text dataset
            np.random.seed(789)
            n_samples = 300
            
            text_data = {
                'document_id': range(1, n_samples + 1),
                'title': [f'Document Title {i}' for i in range(1, n_samples + 1)],
                'content': [f'This is sample content for document {i}. ' * np.random.randint(5, 20) for i in range(1, n_samples + 1)],
                'author': np.random.choice(['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'], n_samples),
                'category': np.random.choice(['News', 'Blog', 'Research', 'Review', 'Tutorial'], n_samples),
                'word_count': np.random.randint(100, 2000, n_samples),
                'language': np.random.choice(['English', 'Spanish', 'French', 'German'], n_samples),
                'publication_date': pd.date_range('2023-01-01', periods=n_samples, freq='3D'),
                'tags': [f'tag{i},tag{i+1},tag{i+2}' for i in range(1, n_samples + 1)],
                'sentiment_score': np.random.uniform(-1, 1, n_samples)
            }
            
            text_df = pd.DataFrame(text_data)
            # Add missing values
            text_df.loc[np.random.choice(text_df.index, 25), 'author'] = np.nan
            text_df.loc[np.random.choice(text_df.index, 15), 'sentiment_score'] = np.nan
            
            st.session_state.current_dataset = text_df
            st.success("✅ Text dataset loaded!")
            st.rerun()

    st.markdown("---")
    # Smart Automation Summary
    st.markdown("## 🧠 Smart Automation Overview")

    if st.session_state.current_dataset is not None:
        df = st.session_state.current_dataset

        # Calculate automation metrics
        automation_col1, automation_col2, automation_col3, automation_col4 = st.columns(4)

        with automation_col1:
            anomaly_count = 0
            # Count missing values
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                anomaly_count += 1
            # Count duplicates
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                anomaly_count += 1
            # Count potential outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                if len(outliers) > 0:
                    anomaly_count += 1
                    break

            st.metric("Detected Anomalies", anomaly_count)

        with automation_col2:
            # Estimate time saved (mock calculation)
            manual_time_hours = len(df.columns) * 0.5  # 30 min per column manually
            auto_time_hours = len(df.columns) * 0.05  # 3 min per column with automation
            time_saved = manual_time_hours - auto_time_hours
            st.metric("Est. Time Saved", f"{time_saved:.1f}h")

        with automation_col3:
            processing_steps = len(st.session_state.get('processing_log', []))
            st.metric("Processing Steps", processing_steps)

        with automation_col4:
            if st.button("🔧 Fix It All", type="primary"):
                st.switch_page("pages/04_Clean_Pipeline.py")

        # Automation summary
        st.markdown("""
            <div class="automation-highlight">
            <h4>🎯 Intelligent Automation Features</h4>
            <ul>
            <li><strong>Auto-Detection:</strong> Smart identification of data types, anomalies, and quality issues</li>
            <li><strong>AI Suggestions:</strong> Severity-ranked recommendations with one-click fixes</li>
            <li><strong>Fix It All:</strong> Complete automation with review mode and undo capabilities</li>
            <li><strong>Manual Overrides:</strong> Full control over every processing step</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📊 Upload a dataset to see smart automation insights and recommendations!")

        st.markdown("---")
        # Personal Section
        st.markdown("## 👨🏽‍💻 Why I Built This")

        # Load image
        image_path = "images/Francis.jpeg"  #

        # Check if the image exists
        if os.path.exists(image_path):
            # Function to convert image to base64
            def get_base64_image(image_path):
                with open(image_path, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode()
                return f"data:image/jpeg;base64,{encoded}"

            # Convert image to base64
            image_base64 = get_base64_image(image_path)

            # Display image and text in a circular format
            st.markdown(f"""
                <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; margin-bottom:20px;">
                    <p style="font-size:1.2em; font-style:italic; color:#333;">
                    <img src="{image_base64}" style="width: 150px; height: 150px; border-radius: 50%; object-fit: cover; margin-bottom: 10px;">
                    <p style="text-align: center;">
                        “Every data project begins with excitement, until you meet the messy, chaotic reality of raw data.
                            On my path to becoming a data scientist, I discovered that nearly 60%-80% of my time disappeared into cleaning, formatting, and fixing datasets.
                            It was slow. It was repetitive. And it stole precious hours from the real work finding insights and building models.
                            That’s why I created KlinItAll: a smart, user-friendly data preprocessing companion that takes care of the tedious tasks for you. 
                            With KlinItAll, you spend less time scrubbing data and more time delivering impact...”
                    </p>
                    <p style="text-align:right; font-size:0.9em; color:#666;">
                        — Francis Afful Gyan, M.Sc.<br>
                        University of Ghana Business School
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Image not found: {image_path}")

            # --- 1. Intro Section ---
            st.markdown("""
                <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; margin-bottom:20px;">
                    <p style="font-size:1.2em; font-style:italic; color:#333;">
                        “As I progressed on my journey to becoming a data scientist, I discovered that nearly 80% of my time was consumed by cleaning and preparing raw data—an often repetitive and time-intensive process. KlinItAll was born out of that challenge: a smart, user-friendly tool designed to automate and simplify data preprocessing, so you can focus more on insights, modeling, and impact.”
                    </p>
                    <p style="text-align:right; font-size:0.9em; color:#666;">
                        — Francis Afful Gyan, M.Sc.<br>
                        University of Ghana Business School
                    </p>
                </div>
                """, unsafe_allow_html=True)


    # Footer with navigation
    st.markdown("---")

    # Navigation menu
    st.markdown("### 🧭 Navigation Menu")

    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)

    with nav_col1:
        st.markdown("**📥 Data Ingestion**")
        if st.button("Upload", key="nav_upload"):
            st.switch_page("pages/01_Upload.py")
        if st.button("Data Overview", key="nav_overview"):
            st.switch_page("pages/02_Data_Overview.py")

    with nav_col2:
        st.markdown("**🧹 Processing**")
        if st.button("Auto Clean", key="nav_auto"):
            st.switch_page("pages/04_Clean_Pipeline.py")
        if st.button("Clean Pipeline", key="nav_clean"):
            st.switch_page("pages/04_Clean_Pipeline.py")

    with nav_col3:
        st.markdown("**⚙️ Advanced**")
        if st.button("Feature Engineering", key="nav_feature"):
            st.switch_page("pages/13_Feature_Engineering.py")
        if st.button("Data Story Narrator", key="nav_story"):
            st.switch_page("pages/10_Data_Story_Narrator.py")

    with nav_col4:
        st.markdown("**📊 Export & History**")
        if st.button("History & Export", key="nav_history"):
            st.switch_page("pages/14_History_Export.py")
        if st.button("Settings & Help", key="nav_settings"):
            st.switch_page("pages/16_Settings.py")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>© 2025 KlinItAll - Smart Data Preprocessing Solution</p>
        <p>
            Contact: <a href="mailto:francisaffulgyan@gmail.com">francisaffulgyan@gmail.com</a> | 
            <a href="https://www.linkedin.com/in/francis-afful-gyan-2b27a5153//" target="_blank">LinkedIn</a> | 
            <a href="https://github.com/Nanagyan22" target="_blank">GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
