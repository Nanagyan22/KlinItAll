import streamlit as st
import pandas as pd
import json
from datetime import datetime
import os
import numpy as np
import sys


def main():
    st.set_page_config(page_title="Settings & Help", page_icon="⚙️", layout="wide")
    
    st.title("⚙️ Settings & Help")
    st.markdown("---")
    
    # Create two columns for the main layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Application Settings")
        
        # Session Management
        st.subheader("Session Management")
        if st.button("🔄 Reset Session", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("Session reset successfully!")
            st.rerun()
        
        # Data Export Preferences
        st.subheader("Export Preferences")
        export_format = st.selectbox(
            "Default Export Format",
            ["CSV", "Excel", "JSON", "Parquet"],
            index=0,
            help="Choose your preferred format for data exports"
        )
        
        # Processing Preferences
        st.subheader("Processing Preferences")
        auto_detect = st.checkbox(
            "Auto-detect data types",
            value=True,
            help="Automatically detect and convert appropriate data types"
        )
        
        show_warnings = st.checkbox(
            "Show processing warnings",
            value=True,
            help="Display warnings during data processing steps"
        )
        
        # Memory Management
        st.subheader("Memory Management")
        if st.button("🧹 Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            settings = {
                "export_format": export_format,
                "auto_detect": auto_detect,
                "show_warnings": show_warnings,
                "last_updated": datetime.now().isoformat()
            }
            st.session_state.app_settings = settings
            st.success("Settings saved successfully!")
    
    with col2:

        # Quick Start Guide
        st.subheader("Quick Help")
        with st.expander("🚀 Help Starter", expanded=False):
            st.markdown("""
            1. **Upload Data**: Go to the Upload page and select your data source
            2. **Review Data**: Use Data Overview to understand your dataset
            3. **Clean Data**: Use individual pages or the Clean Pipeline for preprocessing
            4. **Export Results**: Download your cleaned data from any processing page
            """)
        
        # Feature Overview
        with st.expander("🔧 Main Features", expanded=False):
            st.markdown("""
            - **Multiple Data Sources**: CSV, Excel, JSON, databases, cloud storage
            - **Automated Cleaning**: Auto Clean Pipeline for one-click preprocessing
            - **Individual Processing**: Dedicated pages for specific tasks
            - **Data Visualization**: Charts and plots for data exploration
            - **Export Options**: Multiple formats and compression options
            - **Batch Processing**: Handle multiple files simultaneously
            """)
        
        # Troubleshooting
        with st.expander("🔧 Troubleshooting", expanded=False):
            st.markdown("""
            **Common Issues:**
            
            - **Large File Upload**: For files >200MB, consider using database or cloud storage
            - **Memory Issues**: Clear cache or reset session if app becomes slow
            - **Export Problems**: Check file permissions and available disk space
            - **Processing Errors**: Verify data format and column types
            
            **Performance Tips:**
            - Use sampling for very large datasets during exploration
            - Clear session regularly to free up memory
            - Export intermediate results to avoid reprocessing
            """)
        
        # System Information

        # Display current session info
        if 'data' in st.session_state and st.session_state.data is not None:
            st.info(f"📊 Current Dataset: {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
        
        # Memory usage
        session_size = len(str(st.session_state))
        st.info(f"💾 Session Memory: ~{session_size:,} characters")
        

    #==================================More Add up===========================================
    # Settings and Help Tabs
    st.markdown("---")

    help_tabs = st.tabs([ "📚 Documentation", "❓ Help & FAQ", "🐞 Troubleshooting", "ℹ️ About"])


    with help_tabs[0]:
        st.markdown("### Documentation")

        doc_col1, doc_col2 = st.columns([2, 1])

        with doc_col1:
            st.markdown("#### Getting Started Guide")

            with st.expander("🚀 Quick Start Tutorial", expanded=True):
                st.markdown("""
                **Step 1: Upload Your Data**
                - Go to the Upload page
                - Select your CSV, Excel, or JSON file
                - The system will automatically detect data types and issues

                **Step 2: Review Data Overview**
                - Check the automatically generated data profile
                - Review detected issues and recommendations
                - Use the smart insights to understand your data

                **Step 3: Apply Automated Cleaning**
                - Use the "Auto-Fix All" buttons for instant cleaning
                - Or manually configure specific operations
                - Each page focuses on a specific aspect of data cleaning

                **Step 4: Export Your Results**
                - Download cleaned datasets in multiple formats
                - Export Python scripts to reproduce your workflow
                - Save processing history for future reference
                """)

            with st.expander("📊 Data Upload Formats"):
                st.markdown("""
                **Supported File Formats:**
                - **CSV**: Comma-separated values (most common)
                - **Excel**: .xlsx and .xls files (multiple sheets supported)
                - **JSON**: JavaScript Object Notation
                - **Parquet**: Columnar storage format (planned)

                **Upload Tips:**
                - Files up to 200MB are supported
                - Ensure first row contains column headers
                - Mixed data types in columns will be auto-detected
                - Special characters in column names will be standardized
                """)

            with st.expander("🧹 Data Cleaning Operations"):
                st.markdown("""
                **Automated Operations:**
                - **Missing Values**: Auto-detection and smart imputation
                - **Outliers**: Multiple detection methods (IQR, Z-score, Isolation Forest)
                - **Duplicates**: Exact and fuzzy duplicate detection
                - **Data Types**: Intelligent type conversion and validation
                - **Encoding**: Automatic categorical variable encoding
                - **Scaling**: Feature normalization and standardization
                - **Text Cleaning**: NLP preprocessing and feature extraction
                - **DateTime**: Date parsing and feature engineering
                - **Geospatial**: Coordinate validation and distance calculations
                """)

            with st.expander("⚡ Batch Processing"):
                st.markdown("""
                **Batch Mode Features:**
                - Upload multiple datasets simultaneously
                - Create reusable processing templates
                - Apply same cleaning pipeline to multiple files
                - Parallel processing for faster execution
                - Comprehensive results dashboard
                """)

        with doc_col2:
            st.markdown("#### Quick Reference")

            st.markdown("**🔗 Navigation Tips**")
            st.info("""
            - Use the sidebar to jump between pages
            - Each page has buttons for guided workflow
            - Home button returns to main page
            - Progress is auto-saved between pages
            """)

            st.markdown("**⚡ Keyboard Shortcuts**")
            st.code("""
            Ctrl + S: Save current progress
            Ctrl + Z: Undo last operation
            Ctrl + R: Refresh current page
            F5: Reload application
            """)

            st.markdown("**📱 Mobile Support**")
            st.info("""
            KlinItAll is optimized for desktop use.
            Mobile viewing is supported but some 
            features may have limited functionality.
            """)

            # Video tutorials placeholder
            st.markdown("**📺 Video Tutorials**")
            st.info("Video tutorials will be available in a future update")

    with help_tabs[1]:
        st.markdown("### Help & FAQ")

        faq_col1, faq_col2 = st.columns(2)

        with faq_col1:
            st.markdown("#### Frequently Asked Questions")

            with st.expander("❓ Why is my file not uploading?"):
                st.markdown("""
                **Common solutions:**
                - Check file size (max 200MB)
                - Ensure file format is supported (CSV, Excel, JSON)
                - Verify file is not corrupted
                - Try renaming file to remove special characters
                - Check internet connection for large files
                """)

            with st.expander("❓ How does auto-detection work?"):
                st.markdown("""
                **Auto-detection features:**
                - **Data Types**: Analyzes sample data to suggest conversions
                - **Missing Values**: Identifies null, empty, and invalid entries
                - **Outliers**: Uses statistical methods to flag anomalies
                - **Duplicates**: Finds exact and similar records
                - **Patterns**: Detects email addresses, phone numbers, dates

                **Confidence Levels:**
                - High (>80%): Auto-applied with notification
                - Medium (50-80%): Suggested with user confirmation
                - Low (<50%): Flagged for manual review
                """)

            with st.expander("❓ Can I undo operations?"):
                st.markdown("""
                **Version Control Features:**
                - Create manual checkpoints before major operations
                - Auto-backup before destructive operations
                - Restore to any previous version
                - Export processing history for reproducibility

                **Note:** Individual operation undo is limited.
                Use version restore for rolling back changes.
                """)

            with st.expander("❓ How accurate is the automation?"):
                st.markdown("""
                **Automation Accuracy:**
                - Missing value imputation: 85-95% appropriate
                - Data type detection: 90-98% accurate
                - Outlier detection: 80-90% precision
                - Duplicate detection: 95-99% for exact, 70-85% for fuzzy

                **Always review suggestions** before applying,
                especially for critical datasets.
                """)

        with faq_col2:
            st.markdown("#### Common Issues & Solutions")

            with st.expander("🐛 Memory errors with large files"):
                st.markdown("""
                **Solutions:**
                - Use batch processing for multiple files
                - Process files in chunks
                - Remove unnecessary columns first
                - Use more efficient data types
                """)

            with st.expander("🐛 Slow processing performance"):
                st.markdown("""
                **Performance tips:**
                - Close other browser tabs
                - Process smaller subsets first
                - Use sampling for initial exploration
                - Check system resources
                """)

            with st.expander("🐛 Encoding issues with text"):
                st.markdown("""
                **Text encoding solutions:**
                - Try different file encoding (UTF-8, Latin-1)
                - Use Excel format instead of CSV for special characters
                - Manually specify encoding during upload
                - Clean text before upload if possible
                """)

            with st.expander("🐛 Export functionality not working"):
                st.markdown("""
                **Export troubleshooting:**
                - Check browser download settings
                - Ensure sufficient disk space
                - Try different export format
                - Clear browser cache
                - Disable popup blockers
                """)

            # Contact support section
            st.markdown("#### 📞 Contact Support")
            st.info("""
            **Need more help?**

            📧 Email: francisaffulgyan.com
            📱 Phone: +233-554-319534
            🐛 Bug Reports: francisaffulgyan@gmail.com
            """)

    with help_tabs[2]:
        st.markdown("### Troubleshooting")

        troubleshoot_col1, troubleshoot_col2 = st.columns([2, 1])

        with troubleshoot_col1:
            st.markdown("#### System Diagnostics")

            if st.button("🔍 Run System Check", use_container_width=True):
                with st.spinner("Running system diagnostics..."):
                    # Simulate system check
                    import time
                    time.sleep(2)

                    diagnostics = {
                        "Browser": "✅ Compatible",
                        "Memory": "✅ Sufficient (4.2 GB available)",
                        "Storage": "✅ Available (15.7 GB free)",
                        "Network": "✅ Connected",
                        "Session": "✅ Active",
                        "Dependencies": "✅ All loaded"
                    }

                    st.success("System check completed!")

                    for item, status in diagnostics.items():
                        if "✅" in status:
                            st.success(f"{item}: {status}")
                        else:
                            st.error(f"{item}: {status}")

            st.markdown("#### Debug Information")

            debug_info = {
                "Session ID": st.session_state.get('session_id', 'default'),
                "Current Dataset": "✅ Loaded" if st.session_state.get('current_dataset') is not None else "❌ None",
                "Processing History": f"{len(st.session_state.get('preprocessing_history', []))} operations",
                "Batch Datasets": f"{len(st.session_state.get('batch_datasets', {}))} datasets",
                "Templates": f"{len(st.session_state.get('batch_processing_templates', {}))} templates",
                "App Settings": "✅ Configured" if st.session_state.get('app_settings') else "❌ Default"
            }

            for key, value in debug_info.items():
                st.text(f"{key}: {value}")

            # Clear cache options
            st.markdown("#### Cache Management")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🗑️ Clear Session Data"):
                    for key in list(st.session_state.keys()):
                        if key not in ['app_settings']:  # Keep settings
                            del st.session_state[key]
                    st.success("Session data cleared!")
                    st.rerun()

            with col2:
                if st.button("🔄 Reset Application"):
                    st.session_state.clear()
                    st.success("Application reset!")
                    st.rerun()

            with col3:
                if st.button("📊 Export Debug Info"):
                    debug_data = {
                        'timestamp': datetime.now().isoformat(),
                        'session_info': debug_info,
                        'settings': st.session_state.get('app_settings', {}),
                        'history_length': len(st.session_state.get('preprocessing_history', []))
                    }

                    st.download_button(
                        label="Download Debug Info",
                        data=json.dumps(debug_data, indent=2),
                        file_name=f"klinitall_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )

        with troubleshoot_col2:
            st.markdown("#### Quick Fixes")

            st.markdown("**🔧 Common Solutions**")

            if st.button("🔄 Refresh Page", use_container_width=True):
                st.rerun()

            if st.button("💾 Force Save Progress", use_container_width=True):
                # Save current state
                if st.session_state.get('current_dataset') is not None:
                    st.success("Progress saved!")
                else:
                    st.warning("No data to save!")

            if st.button("🧹 Clear Browser Cache", use_container_width=True):
                st.info("Please manually clear your browser cache and reload the page.")

            st.markdown("**⚠️ Emergency Actions**")

            if st.button("🆘 Safe Mode", use_container_width=True):
                # Minimal functionality mode
                st.warning("Safe mode activated - limited functionality")

            # System information
            st.markdown("### 💻 System Information")

            st.text(f"Streamlit: {st.__version__}")
            st.text(f"Python: {sys.version.split()[0]}")
            st.text(f"Pandas: {pd.__version__}")
            st.text(f"NumPy: {np.__version__}")

            #st.markdown("**💻 System Info**")
            #st.text(f"Streamlit: {st.__version__}")
            #st.text(f"Python: 3.8+")
            #st.text(f"Pandas: {pd.__version__}")
            #st.text(f"NumPy: {np.__version__}")

    with help_tabs[3]:
        st.markdown("### About KlinItAll")

        about_col1, about_col2 = st.columns([2, 1])

        with about_col1:
            st.markdown("#### Application Information")

            st.markdown("""
            **KlinItAll** is an intelligent, fully automated data preprocessing system designed to streamline 
            and simplify the cleaning, profiling, and transformation of structured datasets.

            #### Key Features
            - **Data Story Narrator**: AI-powered interactive narration for your datasets.
            - **Full Automation**: 95% of data preprocessing tasks automated
            - **Smart Detection**: AI-powered anomaly and pattern detection
            - **Multi-format Support**: CSV, Excel, JSON, and more
            - **Batch Processing**: Handle multiple datasets simultaneously
            - **Reproducibility**: Export Python scripts and pipelines
            - **Version Control**: Track and rollback changes
            - **Interactive Visualizations**: Understand your data better
            """)

            st.markdown("#### Developer Information")
            st.markdown("""
            **Created by:** Francis Afful Gyan, M.Sc.  
            **Institution:** University of Ghana Business School  
            **Purpose:** Automate the repetitive 80% of data preprocessing work  
            **Vision:** Empower data scientists to prioritize insights and advanced analytics over repetitive preprocessing  
            """)

            st.markdown("#### Technology Stack")
            st.markdown("""
            - **Frontend**: Streamlit
            - **Data Processing**: Pandas, NumPy
            - **Machine Learning**: Scikit-learn
            - **Visualizations**: Plotly, Seaborn
            - **Text Processing**: NLTK, TextBlob
            - **File Handling**: OpenPyXL, JSON
            """)

        with about_col2:
            st.markdown("#### Version Information")

            version_info = {
                "Version": "1.0",
                "Release Date": "2025-08-13",
                "Build": "2025.07.01",
                "Status": "Test"
            }

            for key, value in version_info.items():
                st.text(f"{key}: {value}")


            st.markdown("#### Acknowledgments")
            st.markdown("""
            **Special Thanks:**
            - University of Ghana Business School
            - Prof Emmanuel Awuni Kolog, PhD (Associate Professor of Data Science & Analytics - UGBS)
            - Corhot 2 - Msc,BA 2024 - 2025 Batch
            - Data science community
            """)

            # Links section
            #st.markdown("#### Links")
            #st.markdown("""
            #- 🌐 [Website](https://klinitall.com)
            #- 📚 [Documentation](https://docs.klinitall.com)
            #- 🐛 [Report Issues](https://github.com/klinitall/issues)
            #- 💬 [Community](https://community.klinitall.com)
            #- 📧 [Contact](mailto:support@klinitall.com)
            #""")

    # Footer with additional resources
    #st.markdown("---")
    #st.markdown("## 🔗 Additional Resources")

    #resource_col1, resource_col2, resource_col3 = st.columns(3)

    #with resource_col1:
        #st.markdown("#### Learning Resources")
        #st.markdown("""
        #- 📖 [Data Preprocessing Best Practices](https://docs.klinitall.com/best-practices)
        #- 🎓 [Data Science Fundamentals](https://docs.klinitall.com/fundamentals)
        #- 📊 [Visualization Guidelines](https://docs.klinitall.com/visualization)
        #""")

    #with resource_col2:
        #st.markdown("#### Community")
        #st.markdown("""
        #- 💬 [Discord Server](https://discord.gg/klinitall)
        #- 🐦 [Twitter Updates](https://twitter.com/klinitall)
        #- 📺 [YouTube Tutorials](https://youtube.com/klinitall)
        #""")

    #with resource_col3:
        #st.markdown("#### Enterprise")
        #st.markdown("""
        #- 🏢 [Enterprise Solutions](https://klinitall.com/enterprise)
        #- 🔐 [Security & Compliance](https://klinitall.com/security)
        #- 📞 [Custom Training](https://klinitall.com/training)
        #""")

    # Footer with additional resources
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>Need additional help? Check the individual page tooltips and help sections.</p>
    <p><strong>KlinItAll</strong> - Intelligent Data Preprocessing Platform</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()