"""
Interactive Guided Tour System with Cute Character Hints
Provides step-by-step guidance throughout the KlinItAll application
"""

import streamlit as st
import json
from datetime import datetime

class GuidedTour:
    """Manages the interactive guided tour system with cute character hints"""
    
    def __init__(self):
        self.character_name = "Klini"
        self.character_emoji = "🤖"
        
        # Initialize tour state
        if 'tour_active' not in st.session_state:
            st.session_state.tour_active = False
        if 'tour_step' not in st.session_state:
            st.session_state.tour_step = 0
        if 'tour_completed_steps' not in st.session_state:
            st.session_state.tour_completed_steps = set()
        if 'show_hints' not in st.session_state:
            st.session_state.show_hints = True
        
        # Define tour steps for each page
        self.tour_steps = {
            "upload": [
                {
                    "title": "Welcome to KlinItAll! 👋",
                    "message": "Hi there! I'm Klini, your friendly data cleaning assistant. Let me show you around! First, let's upload some data to get started.",
                    "hint": "Click the file uploader above to choose your dataset. I support CSV, Excel, and JSON files!",
                    "action": "upload_file"
                },
                {
                    "title": "Great! Your data is uploaded! 🎉",
                    "message": "Awesome! I can see your dataset. Now let's take a quick look at what we're working with.",
                    "hint": "Check out the Dataset Preview section below - it shows you key information about your data quality.",
                    "action": "view_preview"
                },
                {
                    "title": "Ready for some magic? ✨",
                    "message": "Your data looks good! Now we can start cleaning it. I have several options for you.",
                    "hint": "Try the 'Auto Clean Pipeline' button for AI-powered cleaning, or 'Manual Clean Pipeline' for step-by-step control.",
                    "action": "choose_pipeline"
                }
            ],
            "overview": [
                {
                    "title": "Data Overview Time! 📊",
                    "message": "Perfect! Here you can explore your data in detail. I'll help you understand what's in your dataset.",
                    "hint": "Use the tabs above to see statistics, missing values, and data quality metrics. Each tab tells a different story about your data!",
                    "action": "explore_tabs"
                },
                {
                    "title": "Spot any issues? 🔍",
                    "message": "I can help you identify potential problems in your data. Look for red flags in the Data Quality tab!",
                    "hint": "Missing values, duplicates, and inconsistent formats are common issues I can help you fix.",
                    "action": "identify_issues"
                }
            ],
            "clean_pipeline": [
                {
                    "title": "Welcome to the Cleaning Lab! 🧪",
                    "message": "This is where the magic happens! Each tab represents a different cleaning step. I'll guide you through each one.",
                    "hint": "Start with 'Data Types' and work your way through. Each tab has an 'Auto' mode where I do the work for you!",
                    "action": "start_cleaning"
                },
                {
                    "title": "Auto Mode is Your Friend! 🤝",
                    "message": "See those 'Auto' buttons? That's me offering to do the work for you! I'll make smart decisions based on your data.",
                    "hint": "Click 'Auto' in any tab to let me handle that step automatically. You can always review and adjust my choices!",
                    "action": "try_auto_mode"
                },
                {
                    "title": "Step by Step Progress 📈",
                    "message": "Great job! Each completed step makes your data cleaner. Keep going through the tabs in order.",
                    "hint": "The green checkmarks show completed steps. Don't worry if you need to go back and adjust something!",
                    "action": "continue_steps"
                }
            ],
            "results": [
                {
                    "title": "Look at Your Clean Data! ✨",
                    "message": "Fantastic! Your data is now clean and ready for analysis. Look how much better it looks!",
                    "hint": "Compare the before and after stats. See how many issues we fixed together?",
                    "action": "compare_results"
                },
                {
                    "title": "Time to Download! 📥",
                    "message": "Your clean dataset is ready! Don't forget to download it so you can use it in your projects.",
                    "hint": "Use the download button to save your cleaned data. I can also generate a cleaning report for you!",
                    "action": "download_data"
                }
            ]
        }
        
        # Character responses for different situations
        self.character_responses = {
            "encouragement": [
                "You're doing great! Keep it up! 🌟",
                "Excellent choice! I knew you'd figure it out! 🎯",
                "Perfect! You're getting the hang of this! 💪",
                "Wonderful! Your data is looking better already! 📈",
                "Amazing work! You're a natural at this! 🚀"
            ],
            "help": [
                "Don't worry, I'm here to help! Let me guide you through this step. 🤗",
                "No problem! Everyone needs help sometimes. Let's do this together! 👫",
                "That's okay! Data cleaning can be tricky. I'll show you the way! 🗺️",
                "Hey, it's all part of learning! Let me break this down for you. 📚"
            ],
            "tips": [
                "💡 Pro tip: Always check your data quality first before cleaning!",
                "💡 Fun fact: Auto mode uses AI to make smart decisions about your data!",
                "💡 Remember: You can always undo changes if you don't like the results!",
                "💡 Tip: Download your data at each major step to save your progress!"
            ],
            "completion": [
                "🎉 Congratulations! You've mastered data cleaning with KlinItAll!",
                "🌟 Amazing! Your data transformation skills are impressive!",
                "🏆 Well done! You've successfully cleaned your dataset!",
                "✨ Perfect! Your data is now ready for amazing analysis!"
            ]
        }

    def show_character_hint(self, page_name, step_index=None, custom_message=None, hint_type="info"):
        """Display character hint with cute styling"""
        if not st.session_state.show_hints:
            return
            
        # Get appropriate message
        if custom_message:
            message = custom_message
        elif page_name in self.tour_steps and step_index is not None and step_index < len(self.tour_steps[page_name]):
            step = self.tour_steps[page_name][step_index]
            message = step["message"]
            hint = step.get("hint", "")
        else:
            message = "I'm here to help if you need guidance! 😊"
            hint = ""
        
        # Style based on hint type
        if hint_type == "success":
            bg_color = "#d4edda"
            border_color = "#28a745"
            icon = "🎉"
        elif hint_type == "warning":
            bg_color = "#fff3cd"
            border_color = "#ffc107"
            icon = "⚠️"
        elif hint_type == "tip":
            bg_color = "#e7f3ff"
            border_color = "#007bff"
            icon = "💡"
        else:  # info
            bg_color = "#f8f9fa"
            border_color = "#6c757d"
            icon = self.character_emoji
        
        # Create character hint box
        st.markdown(f"""
        <div style="
            background: {bg_color};
            border-left: 4px solid {border_color};
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            position: relative;
        ">
            <div style="display: flex; align-items: flex-start; gap: 10px;">
                <div style="font-size: 24px; flex-shrink: 0;">
                    {icon}
                </div>
                <div style="flex-grow: 1;">
                    <strong style="color: #333; font-size: 14px;">{self.character_name} says:</strong>
                    <p style="margin: 5px 0; color: #555; line-height: 1.4;">{message}</p>
                    {f'<p style="margin: 5px 0; color: #666; font-style: italic; font-size: 13px;">{hint}</p>' if hint else ''}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def start_tour(self):
        """Start the guided tour"""
        st.session_state.tour_active = True
        st.session_state.tour_step = 0
        st.session_state.tour_completed_steps = set()
        
    def end_tour(self):
        """End the guided tour"""
        st.session_state.tour_active = False
        self.show_character_hint("", custom_message="Thanks for taking the tour! I'll still be here with hints if you need help. You can restart the tour anytime from the sidebar! 👋", hint_type="success")
        
    def next_step(self):
        """Move to next tour step"""
        if st.session_state.tour_active:
            st.session_state.tour_step += 1
            
    def complete_step(self, step_id):
        """Mark a step as completed"""
        st.session_state.tour_completed_steps.add(step_id)
        
    def show_tour_controls(self):
        """Show tour control buttons in sidebar"""
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### {self.character_emoji} Tour Guide")
            
            if st.session_state.tour_active:
                st.success("🎯 Tour Active")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⏸️ Pause Tour", use_container_width=True):
                        st.session_state.tour_active = False
                        st.rerun()
                        
                with col2:
                    if st.button("🛑 End Tour", use_container_width=True):
                        self.end_tour()
                        st.rerun()
                        
                # Show progress
                total_steps = sum(len(steps) for steps in self.tour_steps.values())
                completed = len(st.session_state.tour_completed_steps)
                progress = min(completed / max(total_steps, 1), 1.0)
                
                st.progress(progress)
                st.caption(f"Progress: {completed}/{total_steps} steps completed")
                
            else:
                if st.button("🚀 Start Guided Tour", use_container_width=True):
                    self.start_tour()
                    st.rerun()
                    
                if st.button("💡 Restart Tour", use_container_width=True):
                    self.start_tour()
                    st.rerun()
            
            # Hint settings
            st.markdown("#### Settings")
            new_hints = st.checkbox("Show Character Hints", value=st.session_state.show_hints)
            if new_hints != st.session_state.show_hints:
                st.session_state.show_hints = new_hints
                st.rerun()
                
            # Quick tips
            if st.session_state.show_hints:
                st.markdown("#### 💡 Quick Tips")
                import random
                tip = random.choice(self.character_responses["tips"])
                st.info(tip)

    def show_contextual_help(self, page_name, context="general"):
        """Show contextual help based on current page and context"""
        if not st.session_state.show_hints:
            return
            
        help_messages = {
            "upload": {
                "general": "Upload your dataset to get started! I support CSV, Excel, and JSON files up to 200MB.",
                "no_data": "Don't have data? Try the 'Sample Dataset' button in the sidebar to load demo data!",
                "large_file": "Large file? Make sure it's under 200MB. Consider filtering your data first if it's bigger."
            },
            "overview": {
                "general": "Explore your data here! Check each tab to understand your dataset better.",
                "missing_values": "Missing values are normal! I can help you handle them in the Clean Pipeline.",
                "data_quality": "Data quality issues? No worries! That's exactly what I'm here to fix!"
            },
            "clean_pipeline": {
                "general": "This is where the magic happens! Work through each tab from left to right.",
                "auto_mode": "Try Auto mode! I'll make smart decisions and you can always review them.",
                "manual_mode": "Manual mode gives you full control over every cleaning decision."
            }
        }
        
        message = help_messages.get(page_name, {}).get(context, "I'm here to help! Feel free to explore and don't hesitate to try things out!")
        self.show_character_hint(page_name, custom_message=message, hint_type="tip")

    def check_trigger_conditions(self, page_name, trigger_data=None):
        """Check if tour steps should be triggered based on user actions"""
        if not st.session_state.tour_active:
            return
            
        # Define trigger conditions for automatic step progression
        triggers = {
            "upload": {
                "file_uploaded": lambda: 'current_dataset' in st.session_state and st.session_state.current_dataset is not None,
                "preview_viewed": lambda: trigger_data and trigger_data.get("action") == "view_preview",
                "pipeline_selected": lambda: trigger_data and trigger_data.get("action") == "choose_pipeline"
            },
            "overview": {
                "tabs_explored": lambda: trigger_data and trigger_data.get("tab_count", 0) > 2,
                "issues_identified": lambda: trigger_data and trigger_data.get("issues_found", False)
            },
            "clean_pipeline": {
                "auto_used": lambda: trigger_data and trigger_data.get("auto_mode_used", False),
                "step_completed": lambda: trigger_data and trigger_data.get("steps_completed", 0) > 0
            }
        }
        
        # Check triggers and advance tour if conditions are met
        page_triggers = triggers.get(page_name, {})
        for trigger_name, condition in page_triggers.items():
            if condition() and f"{page_name}_{trigger_name}" not in st.session_state.tour_completed_steps:
                self.complete_step(f"{page_name}_{trigger_name}")
                self.next_step()
                break

    def show_celebration(self, achievement):
        """Show celebration message for achievements"""
        celebrations = {
            "first_upload": "🎉 Great job! You've uploaded your first dataset!",
            "first_auto_clean": "🌟 Awesome! You tried auto mode - smart choice!",
            "pipeline_complete": "🏆 Amazing! You've completed the cleaning pipeline!",
            "data_downloaded": "📥 Perfect! Your clean data is ready to use!"
        }
        
        message = celebrations.get(achievement, "🎉 Great work!")
        self.show_character_hint("", custom_message=message, hint_type="success")

# Global tour instance
guided_tour = GuidedTour()