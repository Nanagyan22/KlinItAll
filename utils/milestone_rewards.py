import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import json

class MilestoneRewards:
    """
    Personalized User Journey Milestone Rewards System
    Tracks user progress, achievements, and provides personalized rewards
    """
    
    def __init__(self):
        self.milestones = {
            # Data Upload Milestones
            'first_upload': {
                'name': 'First Steps',
                'description': 'Upload your first dataset',
                'points': 50,
                'badge': '🚀',
                'category': 'upload',
                'message': "Welcome to KlinItAll! You've taken your first step into data preprocessing. Great job!"
            },
            'multi_upload': {
                'name': 'Data Collector',
                'description': 'Upload 5 different datasets',
                'points': 100,
                'badge': '📁',
                'category': 'upload',
                'message': "You're becoming quite the data collector! Managing multiple datasets shows real dedication."
            },
            'large_dataset': {
                'name': 'Big Data Handler',
                'description': 'Upload a dataset with over 10,000 rows',
                'points': 150,
                'badge': '📊',
                'category': 'upload',
                'message': "Impressive! You're working with substantial datasets. That's some serious data science!"
            },
            
            # Cleaning Milestones
            'first_clean': {
                'name': 'Data Cleaner',
                'description': 'Complete your first cleaning operation',
                'points': 75,
                'badge': '🧹',
                'category': 'cleaning',
                'message': "Excellent! You've cleaned your first dataset. Clean data is the foundation of great analysis."
            },
            'auto_pipeline_master': {
                'name': 'Automation Expert',
                'description': 'Complete 3 auto cleaning pipelines',
                'points': 200,
                'badge': '🤖',
                'category': 'cleaning',
                'message': "You've mastered automation! Smart data scientists let the machines do the heavy lifting."
            },
            'manual_pipeline_expert': {
                'name': 'Manual Control Expert',
                'description': 'Complete 3 manual cleaning pipelines',
                'points': 250,
                'badge': '⚙️',
                'category': 'cleaning',
                'message': "You've shown mastery over manual controls! True expertise means knowing when to take the wheel."
            },
            'outlier_detective': {
                'name': 'Outlier Detective',
                'description': 'Detect and handle outliers in 5 datasets',
                'points': 125,
                'badge': '🔍',
                'category': 'cleaning',
                'message': "Sharp eye! Spotting outliers is crucial for data quality. You're developing keen analytical instincts."
            },
            'ai_auto_fix': {
                'name': 'AI Assistant Master',
                'description': 'Use AI-powered one-click fixes to clean data',
                'points': 200,
                'badge': '🤖',
                'category': 'cleaning',
                'message': "Brilliant! You've leveraged AI automation to clean your data efficiently. This is the future of data science!"
            },
            
            # Feature Engineering Milestones
            'feature_engineer': {
                'name': 'Feature Engineer',
                'description': 'Create new features through engineering',
                'points': 175,
                'badge': '🔧',
                'category': 'engineering',
                'message': "Creative problem-solving! Feature engineering is where art meets science in data work."
            },
            'scaling_specialist': {
                'name': 'Scaling Specialist',
                'description': 'Apply different scaling methods to datasets',
                'points': 100,
                'badge': '📏',
                'category': 'engineering',
                'message': "Well balanced! Proper scaling is essential for many machine learning algorithms."
            },
            
            # Analysis Milestones
            'data_explorer': {
                'name': 'Data Explorer',
                'description': 'Generate comprehensive data overview reports',
                'points': 125,
                'badge': '🗺️',
                'category': 'analysis',
                'message': "Thorough exploration! Understanding your data deeply is the mark of a true data scientist."
            },
            'visualization_artist': {
                'name': 'Visualization Artist',
                'description': 'Create multiple data visualizations',
                'points': 150,
                'badge': '🎨',
                'category': 'analysis',
                'message': "Beautiful work! Great visualizations tell compelling stories with data."
            },
            
            # Consistency Milestones
            'daily_user': {
                'name': 'Daily Dedication',
                'description': 'Use KlinItAll for 7 consecutive days',
                'points': 300,
                'badge': '📅',
                'category': 'consistency',
                'message': "Impressive consistency! Daily practice is the key to mastering any skill."
            },
            'weekly_warrior': {
                'name': 'Weekly Warrior',
                'description': 'Complete data processing tasks for 4 weeks',
                'points': 500,
                'badge': '⚔️',
                'category': 'consistency',
                'message': "Dedication pays off! You're building serious expertise through consistent practice."
            },
            
            # Advanced Milestones
            'efficiency_expert': {
                'name': 'Efficiency Expert',
                'description': 'Complete a full pipeline in under 5 minutes',
                'points': 200,
                'badge': '⚡',
                'category': 'advanced',
                'message': "Lightning fast! Your efficiency shows true mastery of the preprocessing workflow."
            },
            'quality_guardian': {
                'name': 'Quality Guardian',
                'description': 'Achieve 0% missing data in 10 datasets',
                'points': 250,
                'badge': '🛡️',
                'category': 'advanced',
                'message': "Perfection achieved! Your attention to data quality is exemplary."
            },
            'batch_processor': {
                'name': 'Batch Processor',
                'description': 'Process multiple datasets simultaneously',
                'points': 175,
                'badge': '🔄',
                'category': 'advanced',
                'message': "Multitasking mastery! Batch processing shows advanced workflow optimization skills."
            }
        }
        
        self.level_thresholds = {
            1: 0,      # Beginner
            2: 200,    # Novice
            3: 500,    # Apprentice
            4: 1000,   # Practitioner
            5: 2000,   # Expert
            6: 3500,   # Master
            7: 5500,   # Grandmaster
            8: 8000,   # Legend
            9: 12000,  # Sage
            10: 18000  # Data Wizard
        }
        
        self.level_names = {
            1: "🌱 Data Seedling",
            2: "🌿 Data Sprout", 
            3: "🌳 Data Tree",
            4: "🔬 Data Scientist",
            5: "🎯 Data Expert",
            6: "🏆 Data Master",
            7: "👑 Data Grandmaster",
            8: "⚡ Data Legend",
            9: "🧙 Data Sage",
            10: "🌟 Data Wizard"
        }
        
        self.reward_types = [
            'badge_unlock', 'level_up', 'streak_bonus', 'perfect_score', 
            'speed_bonus', 'exploration_bonus', 'consistency_reward'
        ]

    def initialize_user_progress(self):
        """Initialize user progress tracking in session state"""
        if 'user_progress' not in st.session_state:
            st.session_state.user_progress = {
                'total_points': 0,
                'current_level': 1,
                'achievements': {},
                'activity_log': [],
                'daily_streak': 0,
                'last_active_date': None,
                'datasets_processed': 0,
                'cleaning_operations': 0,
                'features_engineered': 0,
                'visualizations_created': 0,
                'perfect_cleanings': 0,
                'fastest_pipeline_time': None,
                'session_stats': defaultdict(int)
            }
    
    def check_milestone_completion(self, milestone_key, context=None):
        """Check if a milestone has been completed and award accordingly"""
        self.initialize_user_progress()
        
        if milestone_key not in self.milestones:
            return False
            
        # Skip if already achieved
        if milestone_key in st.session_state.user_progress['achievements']:
            return False
            
        milestone = self.milestones[milestone_key]
        achieved = False
        
        # Check milestone-specific conditions
        if milestone_key == 'first_upload':
            achieved = st.session_state.user_progress['datasets_processed'] >= 1
            
        elif milestone_key == 'multi_upload':
            achieved = st.session_state.user_progress['datasets_processed'] >= 5
            
        elif milestone_key == 'large_dataset':
            if context and 'dataset_size' in context:
                achieved = context['dataset_size'] >= 10000
                
        elif milestone_key == 'first_clean':
            achieved = st.session_state.user_progress['cleaning_operations'] >= 1
            
        elif milestone_key == 'auto_pipeline_master':
            achieved = st.session_state.user_progress.get('auto_pipelines', 0) >= 3
            
        elif milestone_key == 'manual_pipeline_expert':
            achieved = st.session_state.user_progress.get('manual_pipelines', 0) >= 3
            
        elif milestone_key == 'outlier_detective':
            achieved = st.session_state.user_progress.get('outliers_handled', 0) >= 5
            
        elif milestone_key == 'feature_engineer':
            achieved = st.session_state.user_progress['features_engineered'] >= 1
            
        elif milestone_key == 'data_explorer':
            achieved = st.session_state.user_progress.get('overviews_generated', 0) >= 1
            
        elif milestone_key == 'visualization_artist':
            achieved = st.session_state.user_progress['visualizations_created'] >= 5
            
        elif milestone_key == 'daily_user':
            achieved = st.session_state.user_progress['daily_streak'] >= 7
            
        elif milestone_key == 'efficiency_expert':
            if context and 'pipeline_time' in context:
                achieved = context['pipeline_time'] <= 300  # 5 minutes
                
        elif milestone_key == 'quality_guardian':
            achieved = st.session_state.user_progress['perfect_cleanings'] >= 10
            
        if achieved:
            self.award_milestone(milestone_key, milestone)
            return True
            
        return False
    
    def award_milestone(self, milestone_key, milestone):
        """Award a milestone achievement to the user"""
        self.initialize_user_progress()
        
        # Record achievement
        achievement_data = {
            'timestamp': datetime.now(),
            'points': milestone['points'],
            'badge': milestone['badge'],
            'message': milestone['message']
        }
        
        st.session_state.user_progress['achievements'][milestone_key] = achievement_data
        st.session_state.user_progress['total_points'] += milestone['points']
        
        # Check for level up
        old_level = st.session_state.user_progress['current_level']
        new_level = self.calculate_user_level(st.session_state.user_progress['total_points'])
        
        if new_level > old_level:
            st.session_state.user_progress['current_level'] = new_level
            self.show_level_up_celebration(old_level, new_level)
        
        # Log the achievement
        st.session_state.user_progress['activity_log'].append({
            'timestamp': datetime.now(),
            'type': 'milestone_achieved',
            'milestone': milestone_key,
            'points': milestone['points']
        })
        
        # Show achievement notification
        self.show_achievement_notification(milestone_key, milestone)
    
    def calculate_user_level(self, total_points):
        """Calculate user level based on total points"""
        for level in range(10, 0, -1):
            if total_points >= self.level_thresholds[level]:
                return level
        return 1
    
    def show_achievement_notification(self, milestone_key, milestone):
        """Display an achievement notification"""
        st.balloons()
        
        with st.container():
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                ">
                    <h2 style="margin: 0; color: white;">🎉 Achievement Unlocked! 🎉</h2>
                    <h1 style="font-size: 3em; margin: 10px 0;">{milestone['badge']}</h1>
                    <h3 style="margin: 10px 0; color: white;">{milestone['name']}</h3>
                    <p style="margin: 10px 0; font-size: 1.2em;">+{milestone['points']} points</p>
                    <p style="margin: 15px 0; font-style: italic;">{milestone['message']}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
    
    def show_level_up_celebration(self, old_level, new_level):
        """Show level up celebration"""
        st.snow()
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 30px;
                border-radius: 20px;
                color: white;
                text-align: center;
                margin: 20px 0;
                box-shadow: 0 12px 40px rgba(0,0,0,0.3);
            ">
                <h1 style="margin: 0; color: white;">🌟 LEVEL UP! 🌟</h1>
                <div style="font-size: 4em; margin: 20px 0;">⬆️</div>
                <h2 style="margin: 10px 0; color: white;">{self.level_names[old_level]} → {self.level_names[new_level]}</h2>
                <p style="margin: 15px 0; font-size: 1.3em;">You've reached Level {new_level}!</p>
                <p style="margin: 10px 0;">Your dedication to data quality is paying off!</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    def show_progress_dashboard(self):
        """Display user progress dashboard"""
        self.initialize_user_progress()
        
        progress = st.session_state.user_progress
        current_level = progress['current_level']
        total_points = progress['total_points']
        
        # Progress Overview
        st.markdown("## 🏆 Your Data Journey Progress")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Level", f"Level {current_level}")
            st.markdown(f"**{self.level_names[current_level]}**")
            
        with col2:
            st.metric("Total Points", f"{total_points:,}")
            
        with col3:
            st.metric("Achievements", len(progress['achievements']))
            
        with col4:
            st.metric("Daily Streak", progress['daily_streak'])
        
        # Level Progress Bar
        if current_level < 10:
            next_level_threshold = self.level_thresholds[current_level + 1]
            current_level_threshold = self.level_thresholds[current_level]
            progress_in_level = total_points - current_level_threshold
            points_needed = next_level_threshold - current_level_threshold
            progress_percentage = (progress_in_level / points_needed) * 100
            
            st.markdown("### Progress to Next Level")
            st.progress(progress_percentage / 100)
            st.markdown(f"**{progress_in_level}/{points_needed}** points to {self.level_names[current_level + 1]}")
        else:
            st.markdown("### 🌟 Maximum Level Achieved! 🌟")
            st.markdown("You've reached the pinnacle of data mastery!")
        
        # Achievements Gallery
        if progress['achievements']:
            st.markdown("### 🎖️ Achievement Gallery")
            
            # Group achievements by category
            categories = defaultdict(list)
            for milestone_key, achievement_data in progress['achievements'].items():
                milestone = self.milestones[milestone_key]
                categories[milestone['category']].append((milestone_key, milestone, achievement_data))
            
            for category, achievements in categories.items():
                with st.expander(f"{category.title()} Achievements ({len(achievements)})"):
                    cols = st.columns(min(3, len(achievements)))
                    for i, (key, milestone, data) in enumerate(achievements):
                        with cols[i % 3]:
                            st.markdown(
                                f"""
                                <div style="
                                    text-align: center;
                                    padding: 15px;
                                    border: 2px solid #ddd;
                                    border-radius: 10px;
                                    margin: 5px;
                                ">
                                    <div style="font-size: 3em;">{milestone['badge']}</div>
                                    <h4>{milestone['name']}</h4>
                                    <p><strong>+{milestone['points']} points</strong></p>
                                    <p style="font-size: 0.9em;">{data['timestamp'].strftime('%Y-%m-%d')}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
        
        # Activity Timeline
        if progress['activity_log']:
            st.markdown("### 📈 Recent Activity")
            recent_activities = sorted(progress['activity_log'], 
                                     key=lambda x: x['timestamp'], 
                                     reverse=True)[:10]
            
            for activity in recent_activities:
                timestamp = activity['timestamp'].strftime('%Y-%m-%d %H:%M')
                if activity['type'] == 'milestone_achieved':
                    milestone = self.milestones[activity['milestone']]
                    st.markdown(f"🏆 **{timestamp}** - Achieved '{milestone['name']}' (+{activity['points']} points)")
                else:
                    st.markdown(f"📝 **{timestamp}** - {activity.get('description', 'Activity completed')}")
    
    def show_progress_widget(self):
        """Show a compact progress widget in the sidebar"""
        self.initialize_user_progress()
        
        progress = st.session_state.user_progress
        current_level = progress['current_level']
        total_points = progress['total_points']
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🏆 Your Progress")
            
            # Level and points
            st.markdown(f"**{self.level_names[current_level]}**")
            st.markdown(f"Level {current_level} • {total_points:,} points")
            
            # Recent achievements
            recent_achievements = sorted(
                progress['achievements'].items(), 
                key=lambda x: x[1]['timestamp'], 
                reverse=True
            )[:3]
            
            if recent_achievements:
                st.markdown("**Recent Achievements:**")
                for milestone_key, data in recent_achievements:
                    milestone = self.milestones[milestone_key]
                    st.markdown(f"{milestone['badge']} {milestone['name']}")
            
            # Quick stats
            st.markdown(f"🎯 **{len(progress['achievements'])}** achievements unlocked")
            st.markdown(f"🔥 **{progress['daily_streak']}** day streak")
    
    def track_user_activity(self, activity_type, details=None):
        """Track user activity for milestone calculation"""
        self.initialize_user_progress()
        
        progress = st.session_state.user_progress
        
        # Update daily streak
        today = datetime.now().date()
        last_active = progress.get('last_active_date')
        
        if last_active != today:
            if last_active == today - timedelta(days=1):
                progress['daily_streak'] += 1
            else:
                progress['daily_streak'] = 1
            progress['last_active_date'] = today
        
        # Track specific activities
        if activity_type == 'dataset_uploaded':
            progress['datasets_processed'] += 1
            self.check_milestone_completion('first_upload')
            self.check_milestone_completion('multi_upload')
            
            if details and 'size' in details:
                self.check_milestone_completion('large_dataset', {'dataset_size': details['size']})
        
        elif activity_type == 'cleaning_completed':
            progress['cleaning_operations'] += 1
            self.check_milestone_completion('first_clean')
            
            if details:
                if details.get('type') == 'auto':
                    progress['auto_pipelines'] = progress.get('auto_pipelines', 0) + 1
                    self.check_milestone_completion('auto_pipeline_master')
                elif details.get('type') == 'manual':
                    progress['manual_pipelines'] = progress.get('manual_pipelines', 0) + 1
                    self.check_milestone_completion('manual_pipeline_expert')
                
                if details.get('perfect_clean'):
                    progress['perfect_cleanings'] += 1
                    self.check_milestone_completion('quality_guardian')
                
                if details.get('processing_time'):
                    self.check_milestone_completion('efficiency_expert', 
                                                  {'pipeline_time': details['processing_time']})
        
        elif activity_type == 'outliers_handled':
            progress['outliers_handled'] = progress.get('outliers_handled', 0) + 1
            self.check_milestone_completion('outlier_detective')
        
        elif activity_type == 'features_engineered':
            progress['features_engineered'] += 1
            self.check_milestone_completion('feature_engineer')
        
        elif activity_type == 'visualization_created':
            progress['visualizations_created'] += 1
            self.check_milestone_completion('visualization_artist')
        
        elif activity_type == 'overview_generated':
            progress['overviews_generated'] = progress.get('overviews_generated', 0) + 1
            self.check_milestone_completion('data_explorer')
        
        # Log activity
        progress['activity_log'].append({
            'timestamp': datetime.now(),
            'type': activity_type,
            'details': details or {}
        })
        
        # Check consistency milestones
        self.check_milestone_completion('daily_user')
        self.check_milestone_completion('weekly_warrior')
    
    def get_personalized_recommendations(self):
        """Generate personalized recommendations based on user progress"""
        self.initialize_user_progress()
        
        progress = st.session_state.user_progress
        recommendations = []
        
        # Analyze user patterns and suggest next steps
        if progress['datasets_processed'] == 0:
            recommendations.append({
                'title': 'Start Your Data Journey',
                'description': 'Upload your first dataset to begin exploring KlinItAll features',
                'action': 'Go to Upload page',
                'priority': 'high'
            })
        
        elif progress['cleaning_operations'] == 0:
            recommendations.append({
                'title': 'Clean Your Data',
                'description': 'Try the Auto Clean Pipeline to see the magic of automated preprocessing',
                'action': 'Try Auto Clean Pipeline',
                'priority': 'high'
            })
        
        elif progress.get('manual_pipelines', 0) == 0:
            recommendations.append({
                'title': 'Master Manual Controls',
                'description': 'Explore manual cleaning options for more precise control over your data',
                'action': 'Try Manual Clean Pipeline',
                'priority': 'medium'
            })
        
        elif progress['visualizations_created'] < 3:
            recommendations.append({
                'title': 'Visualize Your Insights',
                'description': 'Create compelling visualizations to better understand your data',
                'action': 'Generate Data Overview',
                'priority': 'medium'
            })
        
        if progress['daily_streak'] < 3:
            recommendations.append({
                'title': 'Build Consistency',
                'description': 'Visit KlinItAll daily to build your data science skills',
                'action': 'Set a daily reminder',
                'priority': 'low'
            })
        
        return recommendations

# Initialize the milestone system
milestone_rewards = MilestoneRewards()