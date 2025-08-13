import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import random

# Page config
st.set_page_config(page_title="Data Story Narrator - KlinItAll", layout="wide")

# Header
st.title("📖 Interactive AI-Powered Data Story Narrator")
st.markdown("*Transform your data into compelling narratives with AI-driven storytelling*")

# Check if data exists
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.warning("⚠️ No dataset found. Please upload data first.")
    if st.button("📥 Go to Upload Page"):
        st.switch_page("pages/01_Upload.py")
    st.stop()

df = st.session_state.current_dataset.copy()

# Initialize story state
if 'story_state' not in st.session_state:
    st.session_state.story_state = {
        'current_chapter': 0,
        'story_style': 'analytical',
        'selected_columns': [],
        'narrative_flow': [],
        'insights_discovered': []
    }

# Story configuration sidebar
with st.sidebar:
    st.markdown("### 📚 Story Configuration")
    
    # Story style selection
    story_style = st.selectbox(
        "Narrative Style",
        ["Analytical Detective", "Business Executive", "Scientific Explorer", "Creative Storyteller", "Data Journalist"],
        key="story_style_select"
    )
    
    # Update story style
    if story_style != st.session_state.story_state['story_style']:
        st.session_state.story_state['story_style'] = story_style.lower().replace(' ', '_')
    
    # Column selection for story focus
    st.markdown("#### 🎯 Story Focus")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    focus_columns = st.multiselect(
        "Select key columns for the story",
        df.columns.tolist(),
        default=st.session_state.story_state['selected_columns'][:3] if st.session_state.story_state['selected_columns'] else df.columns.tolist()[:3]
    )
    
    if focus_columns != st.session_state.story_state['selected_columns']:
        st.session_state.story_state['selected_columns'] = focus_columns
    
    # Story length preference
    story_length = st.slider("Story Depth", min_value=3, max_value=8, value=5, help="Number of narrative chapters")
    
    st.markdown("---")
    
    # Story progress
    st.markdown("### 📈 Story Progress")
    progress = st.session_state.story_state['current_chapter'] / story_length
    st.progress(progress)
    st.write(f"Chapter {st.session_state.story_state['current_chapter']} of {story_length}")

# Main story interface
col1, col2 = st.columns([2, 1])

with col1:
    # Story chapter tabs
    if st.session_state.story_state['current_chapter'] == 0:
        # Introduction chapter
        st.markdown("## 🌟 Chapter 1: The Data Discovery")
        
        with st.container():
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;">
                <h3>📊 Our Data Story Begins...</h3>
                <p style="font-size: 1.1em; line-height: 1.6;">
                Welcome to the fascinating world of your dataset! Today, we embark on a journey through 
                <strong>{len(df):,} rows</strong> and <strong>{len(df.columns)} columns</strong> of data, 
                where each number tells a story and every pattern reveals a secret.
                </p>
                <p style="font-size: 1em; opacity: 0.9;">
                Our narrative style: <strong>{story_style}</strong><br>
                Dataset memory footprint: <strong>{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB</strong><br>
                Data completeness: <strong>{((df.size - df.isnull().sum().sum()) / df.size) * 100:.1f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive data preview with story context
        st.markdown("#### 🔍 First Glimpse: What Lies Within")
        
        preview_col1, preview_col2 = st.columns([3, 1])
        with preview_col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with preview_col2:
            # Quick insights box
            st.markdown("""
            <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #007bff;">
                <h5>🔮 Quick Insights</h5>
            """, unsafe_allow_html=True)
            
            # Generate quick insights
            insights = []
            if len(numeric_cols) > 0:
                max_col = df[numeric_cols].max().idxmax()
                insights.append(f"📈 Highest values found in '{max_col}'")
            
            if len(categorical_cols) > 0:
                diverse_col = df[categorical_cols].nunique().idxmax()
                insights.append(f"🎨 Most diverse: '{diverse_col}' with {df[diverse_col].nunique()} categories")
            
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                insights.append(f"⚠️ Missing data detected in {len(missing_cols)} columns")
            else:
                insights.append("✅ No missing data detected!")
            
            for insight in insights:
                st.markdown(f"• {insight}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("🚀 Begin the Data Story", type="primary", use_container_width=True):
            st.session_state.story_state['current_chapter'] = 1
            st.rerun()
    
    elif st.session_state.story_state['current_chapter'] == 1:
        # Chapter 2: Character Introduction (Columns)
        st.markdown("## 👥 Chapter 2: Meet the Characters")
        
        st.markdown("""
        Every great story has memorable characters. In our data narrative, each column is a character 
        with its own personality, quirks, and role in the bigger picture.
        """)
        
        # Character profiles
        character_tabs = st.tabs(["🔢 Numeric Heroes", "📝 Categorical Cast", "📊 Character Stats"])
        
        with character_tabs[0]:
            if numeric_cols:
                st.markdown("### The Numeric Heroes")
                for i, col in enumerate(numeric_cols[:6]):  # Show top 6
                    col_data = df[col]
                    
                    # Character personality based on data distribution
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    skewness = col_data.skew()
                    
                    personality = "The Steady Reliable" if abs(skewness) < 0.5 else "The Dramatic Outlier"
                    personality = "The Mysterious Variable" if col_data.isnull().any() else personality
                    
                    with st.expander(f"🎭 {col} - {personality}"):
                        char_col1, char_col2 = st.columns([2, 1])
                        
                        with char_col1:
                            # Character description
                            st.markdown(f"""
                            **Character Profile:**
                            - **Range:** {col_data.min():.2f} to {col_data.max():.2f}
                            - **Average:** {mean_val:.2f}
                            - **Variability:** {std_val:.2f}
                            - **Personality:** {personality}
                            """)
                            
                            if abs(skewness) > 1:
                                st.warning(f"This character has a dramatic tendency (skewness: {skewness:.2f})")
                            elif col_data.isnull().any():
                                st.info(f"This character has some mysterious gaps ({col_data.isnull().sum()} missing values)")
                        
                        with char_col2:
                            # Mini histogram
                            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                            fig.update_layout(height=200, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric characters found in this dataset.")
        
        with character_tabs[1]:
            if categorical_cols:
                st.markdown("### The Categorical Cast")
                for col in categorical_cols[:6]:  # Show top 6
                    col_data = df[col]
                    unique_count = col_data.nunique()
                    
                    # Character role based on cardinality
                    if unique_count <= 5:
                        role = "The Lead Actor"
                    elif unique_count <= 20:
                        role = "The Supporting Cast"
                    else:
                        role = "The Ensemble Player"
                    
                    with st.expander(f"🎪 {col} - {role}"):
                        char_col1, char_col2 = st.columns([2, 1])
                        
                        with char_col1:
                            st.markdown(f"""
                            **Character Profile:**
                            - **Unique Roles:** {unique_count}
                            - **Most Common:** {col_data.mode().iloc[0] if len(col_data.mode()) > 0 else 'Unknown'}
                            - **Stage Presence:** {role}
                            """)
                            
                            # Show top categories
                            top_cats = col_data.value_counts().head(3)
                            st.markdown("**Top Performances:**")
                            for cat, count in top_cats.items():
                                st.write(f"• {cat}: {count} appearances ({count/len(df)*100:.1f}%)")
                        
                        with char_col2:
                            # Mini pie chart
                            if unique_count <= 10:
                                fig = px.pie(values=col_data.value_counts().values, 
                                           names=col_data.value_counts().index,
                                           title=f"Roles in {col}")
                                fig.update_layout(height=200, showlegend=False)
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical characters found in this dataset.")
        
        with character_tabs[2]:
            # Overall character statistics
            st.markdown("### 📈 Character Performance Metrics")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Total Characters", len(df.columns))
            with metrics_col2:
                st.metric("Numeric Heroes", len(numeric_cols))
            with metrics_col3:
                st.metric("Categorical Cast", len(categorical_cols))
            with metrics_col4:
                st.metric("Complete Profiles", len(df.columns) - len(df.columns[df.isnull().any()]))
        
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬅️ Back to Discovery", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 0
                st.rerun()
        with nav_col2:
            if st.button("➡️ Continue to Plot", type="primary", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 2
                st.rerun()
    
    elif st.session_state.story_state['current_chapter'] == 2:
        # Chapter 3: The Plot Thickens (Relationships)
        st.markdown("## 🕸️ Chapter 3: The Plot Thickens")
        
        st.markdown("""
        Now that we know our characters, let's explore how they interact with each other. 
        In data stories, relationships between variables often reveal the most surprising plot twists.
        """)
        
        if len(numeric_cols) >= 2:
            # Correlation heatmap story
            st.markdown("### 🎭 The Relationship Drama")
            
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest relationships
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Significant correlation
                        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            
            if corr_pairs:
                st.markdown("**🔍 Discovered Relationships:**")
                for var1, var2, corr_val in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                    relationship_type = "💕 Strong Alliance" if corr_val > 0.7 else "🤝 Friendship" if corr_val > 0.3 else "⚔️ Opposition" if corr_val < -0.7 else "🤔 Complex Relationship"
                    st.markdown(f"• **{var1}** and **{var2}**: {relationship_type} (correlation: {corr_val:.2f})")
            
            # Correlation heatmap
            fig = px.imshow(corr_matrix, 
                          title="The Relationship Map",
                          color_continuous_scale="RdBu_r",
                          aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive relationship explorer
        if len(focus_columns) >= 2:
            st.markdown("### 🔍 Interactive Relationship Explorer")
            
            explore_col1, explore_col2 = st.columns(2)
            with explore_col1:
                x_var = st.selectbox("Choose X character", focus_columns, key="x_var")
            with explore_col2:
                y_var = st.selectbox("Choose Y character", [col for col in focus_columns if col != x_var], key="y_var")
            
            if x_var and y_var:
                # Create scatter plot with story narrative
                fig = px.scatter(df, x=x_var, y=y_var, 
                               title=f"The {x_var} vs {y_var} Chronicles",
                               hover_data={col: True for col in focus_columns[:3]})
                
                # Add trendline if both are numeric
                if x_var in numeric_cols and y_var in numeric_cols:
                    fig = px.scatter(df, x=x_var, y=y_var, 
                                   trendline="ols",
                                   title=f"The {x_var} vs {y_var} Chronicles")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Narrative interpretation
                if x_var in numeric_cols and y_var in numeric_cols:
                    correlation = df[x_var].corr(df[y_var])
                    if abs(correlation) > 0.7:
                        st.success(f"📖 **Plot Twist!** {x_var} and {y_var} have a strong {'positive' if correlation > 0 else 'negative'} relationship!")
                    elif abs(correlation) > 0.3:
                        st.info(f"🤔 **Interesting...** {x_var} and {y_var} show a moderate connection worth investigating.")
                    else:
                        st.warning(f"🎭 **Mystery!** {x_var} and {y_var} seem to dance to their own rhythm - no clear pattern emerges.")
        
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬅️ Back to Characters", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 1
                st.rerun()
        with nav_col2:
            if st.button("➡️ Uncover Secrets", type="primary", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 3
                st.rerun()
    
    elif st.session_state.story_state['current_chapter'] == 3:
        # Chapter 4: Hidden Secrets (Outliers and Patterns)
        st.markdown("## 🕵️ Chapter 4: Hidden Secrets Revealed")
        
        st.markdown("""
        Every dataset holds secrets - outliers that break the rules, patterns that emerge from chaos, 
        and anomalies that tell their own fascinating stories.
        """)
        
        secret_tabs = st.tabs(["🎯 The Outliers", "📊 The Patterns", "🔍 The Anomalies"])
        
        with secret_tabs[0]:
            st.markdown("### 🎯 The Outlier Chronicles")
            
            if numeric_cols:
                outlier_stories = []
                
                for col in numeric_cols[:4]:  # Check first 4 numeric columns
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    
                    if len(outliers) > 0:
                        outlier_stories.append({
                            'column': col,
                            'count': len(outliers),
                            'percentage': len(outliers) / len(df) * 100,
                            'extreme_value': outliers[col].iloc[0] if len(outliers) > 0 else None
                        })
                
                if outlier_stories:
                    for story in outlier_stories:
                        with st.expander(f"🎭 The Tale of {story['column']} Outliers"):
                            st.markdown(f"""
                            **The Outlier Story:**
                            - **Rebel Count:** {story['count']} data points ({story['percentage']:.1f}% of total)
                            - **Most Extreme Rebel:** {story['extreme_value']:.2f}
                            - **Character:** The rule-breakers who march to their own drum
                            """)
                            
                            # Outlier visualization
                            fig = px.box(df, y=story['column'], 
                                       title=f"Outlier Detection: {story['column']}")
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.success("🎉 No significant outliers detected - your data characters are well-behaved!")
            else:
                st.info("No numeric data available for outlier analysis.")
        
        with secret_tabs[1]:
            st.markdown("### 📊 The Pattern Detectives")
            
            # Distribution patterns
            if numeric_cols:
                pattern_col = st.selectbox("Choose a character to analyze", numeric_cols, key="pattern_analysis")
                
                if pattern_col:
                    col_data = df[pattern_col]
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()
                    
                    # Pattern interpretation
                    pattern_story = ""
                    if abs(skewness) < 0.5:
                        pattern_story = "📐 **The Balanced Character** - This variable shows a symmetric, well-balanced distribution."
                    elif skewness > 1:
                        pattern_story = "📈 **The Right-Leaning Character** - Most values cluster on the lower end with some high achievers."
                    elif skewness < -1:
                        pattern_story = "📉 **The Left-Leaning Character** - Most values are high with some low outliers."
                    else:
                        pattern_story = "🤷 **The Slightly Quirky Character** - Shows mild asymmetry in distribution."
                    
                    st.markdown(pattern_story)
                    
                    # Distribution visualization
                    fig_col1, fig_col2 = st.columns(2)
                    
                    with fig_col1:
                        fig1 = px.histogram(df, x=pattern_col, 
                                          title=f"Distribution Story of {pattern_col}")
                        fig1.update_layout(height=300)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with fig_col2:
                        fig2 = px.box(df, y=pattern_col, 
                                    title=f"Shape Analysis of {pattern_col}")
                        fig2.update_layout(height=300)
                        st.plotly_chart(fig2, use_container_width=True)
        
        with secret_tabs[2]:
            st.markdown("### 🔍 The Anomaly Hunters")
            
            # Missing data patterns
            missing_story = []
            missing_cols = df.columns[df.isnull().any()].tolist()
            
            if missing_cols:
                st.markdown("**🕳️ The Missing Data Mystery:**")
                for col in missing_cols:
                    missing_count = df[col].isnull().sum()
                    missing_pct = missing_count / len(df) * 100
                    
                    if missing_pct > 50:
                        mystery_level = "🚨 Critical Mystery"
                    elif missing_pct > 20:
                        mystery_level = "⚠️ Significant Mystery"
                    else:
                        mystery_level = "🔍 Minor Mystery"
                    
                    st.markdown(f"• **{col}**: {mystery_level} - {missing_count} missing values ({missing_pct:.1f}%)")
                
                # Missing data heatmap
                if len(missing_cols) > 1:
                    missing_data = df[missing_cols].isnull()
                    fig = px.imshow(missing_data.T, 
                                  title="The Missing Data Map",
                                  color_continuous_scale=["lightblue", "darkred"])
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No missing data mysteries in this dataset!")
        
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("⬅️ Back to Relationships", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 2
                st.rerun()
        with nav_col2:
            if st.button("➡️ Reach the Climax", type="primary", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 4
                st.rerun()
    
    else:
        # Final Chapter: The Grand Finale
        st.markdown("## 🎆 Final Chapter: The Grand Finale")
        
        st.markdown("""
        Our data story reaches its climax! Let's bring together all the discoveries, insights, 
        and revelations into a comprehensive narrative conclusion.
        """)
        
        # Story summary
        with st.container():
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); 
                        padding: 30px; border-radius: 15px; margin-bottom: 20px;">
                <h3>📚 Our Data Story Summary</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate comprehensive insights
            story_insights = []
            
            # Dataset overview insight
            story_insights.append(f"🌟 Our journey through {len(df):,} records and {len(df.columns)} variables revealed a rich tapestry of information.")
            
            # Data quality insight
            completeness = ((df.size - df.isnull().sum().sum()) / df.size) * 100
            if completeness > 95:
                story_insights.append(f"✨ The dataset showed excellent quality with {completeness:.1f}% completeness.")
            elif completeness > 80:
                story_insights.append(f"📊 The dataset demonstrated good quality with {completeness:.1f}% completeness, with room for improvement.")
            else:
                story_insights.append(f"⚠️ The dataset requires attention with {completeness:.1f}% completeness.")
            
            # Diversity insight
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                story_insights.append(f"🎨 A perfect blend of {len(numeric_cols)} numeric and {len(categorical_cols)} categorical variables created a well-balanced analytical foundation.")
            
            # Key findings
            if numeric_cols:
                # Find the most variable column
                cv_values = []
                for col in numeric_cols:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if mean_val != 0:
                        cv_values.append((col, std_val / abs(mean_val)))
                
                if cv_values:
                    most_variable = max(cv_values, key=lambda x: x[1])
                    story_insights.append(f"📈 '{most_variable[0]}' emerged as the most dynamic variable, showing the greatest relative variation.")
            
            # Display insights
            for insight in story_insights:
                st.markdown(f"• {insight}")
        
        # Interactive finale visualization
        st.markdown("### 🎨 The Grand Visualization")
        
        if len(focus_columns) >= 2:
            finale_type = st.selectbox(
                "Choose your finale style",
                ["Correlation Network", "Multi-dimensional View", "Summary Dashboard"],
                key="finale_viz"
            )
            
            if finale_type == "Correlation Network" and len(numeric_cols) >= 3:
                # Network-style correlation plot
                corr_matrix = df[numeric_cols].corr()
                
                # Create network visualization
                fig = go.Figure()
                
                # Add nodes (variables)
                for i, col in enumerate(numeric_cols):
                    fig.add_trace(go.Scatter(
                        x=[i], y=[0],
                        mode='markers+text',
                        marker=dict(size=50, color='lightblue'),
                        text=col,
                        textposition="middle center",
                        name=col,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="The Data Story Network",
                    height=400,
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif finale_type == "Multi-dimensional View":
                # Multi-dimensional scatter plot
                if len(numeric_cols) >= 3:
                    fig = px.scatter_3d(
                        df.sample(min(1000, len(df))),  # Sample for performance
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        z=numeric_cols[2],
                        color=categorical_cols[0] if categorical_cols else None,
                        title="The Multi-Dimensional Data Universe"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 3 numeric columns for multi-dimensional view")
            
            else:  # Summary Dashboard
                dash_col1, dash_col2 = st.columns(2)
                
                with dash_col1:
                    if numeric_cols:
                        # Summary statistics
                        summary_stats = df[numeric_cols].describe()
                        fig = px.bar(
                            x=summary_stats.columns,
                            y=summary_stats.loc['mean'],
                            title="Average Values Across Variables"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with dash_col2:
                    if categorical_cols:
                        # Category distribution
                        cat_col = categorical_cols[0]
                        cat_counts = df[cat_col].value_counts().head(10)
                        fig = px.pie(
                            values=cat_counts.values,
                            names=cat_counts.index,
                            title=f"Distribution of {cat_col}"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Story completion actions
        st.markdown("### 🎊 Story Complete!")
        
        completion_col1, completion_col2, completion_col3 = st.columns(3)
        
        with completion_col1:
            if st.button("📖 Restart Story", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 0
                st.rerun()
        
        with completion_col2:
            if st.button("📊 Export Story Report", use_container_width=True):
                # Generate story report
                report = {
                    'story_date': datetime.now().isoformat(),
                    'dataset_shape': df.shape,
                    'story_style': st.session_state.story_state['story_style'],
                    'focus_columns': focus_columns,
                    'insights_discovered': story_insights
                }
                
                st.download_button(
                    label="📄 Download Story Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"data_story_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with completion_col3:
            if st.button("🔄 New Story Style", use_container_width=True):
                st.session_state.story_state['current_chapter'] = 0
                st.session_state.story_state['story_style'] = 'analytical'
                st.rerun()

with col2:
    # Story companion panel
    st.markdown("### 🤖 Story Companion")
    
    # AI narrator persona based on style
    narrator_personas = {
        'analytical_detective': {
            'name': 'Detective Data',
            'emoji': '🕵️',
            'personality': 'Methodical and thorough',
            'catchphrase': 'The data never lies...'
        },
        'business_executive': {
            'name': 'Executive Insight',
            'emoji': '💼',
            'personality': 'Strategic and results-focused',
            'catchphrase': 'What does this mean for the bottom line?'
        },
        'scientific_explorer': {
            'name': 'Dr. Discovery',
            'emoji': '🔬',
            'personality': 'Curious and hypothesis-driven',
            'catchphrase': 'Let\'s test this theory...'
        },
        'creative_storyteller': {
            'name': 'Narrative Nancy',
            'emoji': '🎭',
            'personality': 'Imaginative and engaging',
            'catchphrase': 'Once upon a dataset...'
        },
        'data_journalist': {
            'name': 'Reporter Rex',
            'emoji': '📰',
            'personality': 'Fact-finding and investigative',
            'catchphrase': 'What\'s the real story here?'
        }
    }
    
    current_persona = narrator_personas.get(
        st.session_state.story_state['story_style'], 
        narrator_personas['analytical_detective']
    )
    
    with st.container():
        st.markdown(f"""
        <div style="background: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3>{current_persona['emoji']} {current_persona['name']}</h3>
            <p><em>{current_persona['personality']}</em></p>
            <p style="font-style: italic; color: #666;">"{current_persona['catchphrase']}"</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Story navigation
    st.markdown("### 📍 Story Navigation")
    chapter_names = [
        "🌟 Discovery",
        "👥 Characters", 
        "🕸️ Plot",
        "🕵️ Secrets",
        "🎆 Finale"
    ]
    
    for i, name in enumerate(chapter_names):
        if i == st.session_state.story_state['current_chapter']:
            st.markdown(f"**▶️ {name}** (Current)")
        elif i < st.session_state.story_state['current_chapter']:
            st.markdown(f"✅ {name}")
        else:
            st.markdown(f"⏳ {name}")
    
    # Quick stats
    st.markdown("### 📊 Quick Stats")
    stats_data = {
        'Dataset Size': f"{len(df):,} × {len(df.columns)}",
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB",
        'Completeness': f"{((df.size - df.isnull().sum().sum()) / df.size) * 100:.1f}%",
        'Story Progress': f"{st.session_state.story_state['current_chapter']}/4"
    }
    
    for key, value in stats_data.items():
        st.metric(key, value)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎭 Transform your data into compelling narratives • 📊 Discover hidden insights through storytelling</p>
</div>
""", unsafe_allow_html=True)