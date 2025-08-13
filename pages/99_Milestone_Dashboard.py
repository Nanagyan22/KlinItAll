import streamlit as st
from utils.milestone_rewards import milestone_rewards
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Milestone Dashboard", page_icon="🏆", layout="wide")

st.title("🏆 Personalized User Journey & Milestone Rewards")
st.markdown("Track your progress, celebrate achievements, and unlock rewards as you master data preprocessing!")

# Initialize the milestone system
milestone_rewards.initialize_user_progress()

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Progress Overview", "🏆 Achievements", "📊 Analytics", "💡 Recommendations"])

with tab1:
    # Progress Overview Dashboard
    milestone_rewards.show_progress_dashboard()

with tab2:
    st.markdown("## 🎖️ Achievement Gallery & Rewards")
    
    progress = st.session_state.user_progress
    
    if not progress['achievements']:
        st.info("🌱 No achievements yet! Start by uploading your first dataset to begin earning rewards.")
        if st.button("🚀 Go to Upload", type="primary"):
            st.switch_page("pages/01_Upload.py")
    else:
        # Achievement statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Achievements", len(progress['achievements']))
        
        with col2:
            total_points = sum(milestone_rewards.milestones[key]['points'] 
                             for key in progress['achievements'].keys())
            st.metric("⭐ Points Earned", f"{total_points:,}")
        
        with col3:
            categories = set(milestone_rewards.milestones[key]['category'] 
                           for key in progress['achievements'].keys())
            st.metric("📂 Categories Mastered", len(categories))
        
        with col4:
            recent = len([a for a in progress['achievements'].values() 
                         if (datetime.now() - a['timestamp']).days <= 7])
            st.metric("🔥 This Week", recent)
        
        # Achievement timeline
        st.markdown("### 📅 Achievement Timeline")
        
        achievement_data = []
        for milestone_key, data in progress['achievements'].items():
            milestone = milestone_rewards.milestones[milestone_key]
            achievement_data.append({
                'Date': data['timestamp'].date(),
                'Achievement': milestone['name'],
                'Points': milestone['points'],
                'Category': milestone['category'].title(),
                'Badge': milestone['badge']
            })
        
        if achievement_data:
            df_achievements = pd.DataFrame(achievement_data)
            df_achievements = df_achievements.sort_values('Date')
            
            # Timeline chart
            fig = px.scatter(df_achievements, 
                           x='Date', 
                           y='Achievement',
                           size='Points',
                           color='Category',
                           hover_data=['Points'],
                           title="Achievement Timeline")
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed achievement list
            st.markdown("### 📋 Detailed Achievement List")
            st.dataframe(df_achievements, use_container_width=True)

with tab3:
    st.markdown("## 📊 Progress Analytics")
    
    progress = st.session_state.user_progress
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Activity Overview")
        
        # Activity metrics
        activities = {
            'Datasets Processed': progress.get('datasets_processed', 0),
            'Cleaning Operations': progress.get('cleaning_operations', 0),
            'Features Engineered': progress.get('features_engineered', 0),
            'Visualizations Created': progress.get('visualizations_created', 0),
            'Perfect Cleanings': progress.get('perfect_cleanings', 0)
        }
        
        if any(activities.values()):
            fig = px.bar(x=list(activities.keys()), 
                        y=list(activities.values()),
                        title="Activity Summary",
                        labels={'x': 'Activity Type', 'y': 'Count'})
            
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity data yet. Start using KlinItAll to see your analytics!")
    
    with col2:
        st.markdown("### 🎯 Progress by Category")
        
        if progress['achievements']:
            # Achievements by category
            category_counts = {}
            category_points = {}
            
            for milestone_key in progress['achievements']:
                milestone = milestone_rewards.milestones[milestone_key]
                category = milestone['category'].title()
                
                category_counts[category] = category_counts.get(category, 0) + 1
                category_points[category] = category_points.get(category, 0) + milestone['points']
            
            # Create subplot for counts and points
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Achievements', 'Points'),
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )
            
            fig.add_trace(
                go.Pie(labels=list(category_counts.keys()), 
                      values=list(category_counts.values()),
                      name="Achievements"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Pie(labels=list(category_points.keys()), 
                      values=list(category_points.values()),
                      name="Points"),
                row=1, col=2
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete some achievements to see category breakdown!")
    
    # Streak and consistency analytics
    st.markdown("### 🔥 Consistency Tracking")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        streak_days = progress.get('daily_streak', 0)
        st.metric("🔥 Current Streak", f"{streak_days} days")
        
        # Streak visualization
        if streak_days > 0:
            dates = [datetime.now().date() - timedelta(days=i) for i in range(streak_days-1, -1, -1)]
            streak_data = pd.DataFrame({
                'Date': dates,
                'Active': [1] * len(dates)
            })
            
            fig = px.bar(streak_data, x='Date', y='Active', 
                        title=f"{streak_days}-Day Streak",
                        color_discrete_sequence=['#ff6b35'])
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        total_days = len(set(log['timestamp'].date() for log in progress.get('activity_log', [])))
        st.metric("📅 Active Days", total_days)
    
    with col5:
        if progress.get('activity_log'):
            avg_daily = len(progress['activity_log']) / max(total_days, 1)
            st.metric("⚡ Avg Activities/Day", f"{avg_daily:.1f}")

with tab4:
    st.markdown("## 💡 Personalized Recommendations")
    
    recommendations = milestone_rewards.get_personalized_recommendations()
    
    if recommendations:
        st.markdown("Based on your progress, here are some personalized recommendations:")
        
        for rec in recommendations:
            priority_color = {
                'high': '🔴',
                'medium': '🟡', 
                'low': '🟢'
            }
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        background: #f8f9fa;
                    ">
                        <h4>{priority_color.get(rec['priority'], '🔵')} {rec['title']}</h4>
                        <p>{rec['description']}</p>
                        <p><strong>Suggested Action:</strong> {rec['action']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.success("🎉 Great job! You're making excellent progress. Keep exploring KlinItAll's features!")
    
    # Available milestones preview
    st.markdown("### 🎯 Upcoming Milestones")
    st.markdown("Here are some milestones you can work towards:")
    
    # Show next few achievable milestones
    unearned_milestones = [(key, milestone) for key, milestone in milestone_rewards.milestones.items() 
                          if key not in progress['achievements']]
    
    # Sort by points (easier ones first)
    unearned_milestones.sort(key=lambda x: x[1]['points'])
    
    for i, (key, milestone) in enumerate(unearned_milestones[:6]):  # Show first 6
        col = st.columns(3)[i % 3]
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ccc;
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                    height: 150px;
                ">
                    <div style="font-size: 2em; margin-bottom: 5px;">{milestone['badge']}</div>
                    <h5>{milestone['name']}</h5>
                    <p style="font-size: 0.9em;">{milestone['description']}</p>
                    <p><strong>+{milestone['points']} points</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

# Navigation
st.markdown("---")
st.markdown("## 🧭 Continue Your Journey")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("📥 Upload Data", use_container_width=True):
        st.switch_page("pages/01_Upload.py")

with nav_col2:
    if st.button("🧹 Clean Pipeline", use_container_width=True):
        if st.session_state.current_dataset is not None:
            st.switch_page("pages/04_Clean_Pipeline.py")
        else:
            st.warning("Upload data first!")

with nav_col3:
    if st.button("🤖 Auto Clean", use_container_width=True):
        if st.session_state.current_dataset is not None:
            st.switch_page("pages/04_Clean_Pipeline.py")
        else:
            st.warning("Upload data first!")