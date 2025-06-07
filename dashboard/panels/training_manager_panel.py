"""Training Manager Panel - Comprehensive training lifecycle monitoring"""

from typing import Optional
from datetime import datetime
from dash import html

class TrainingManagerPanel:
    """Training Manager panel component for dashboard"""
    
    def __init__(self, dark_theme: dict):
        self.DARK_THEME = dark_theme
    
    def create_layout(self) -> html.Div:
        """Create the Training Manager panel layout"""
        return html.Div(
            [
                html.H4(
                    "Training Manager",
                    style={
                        "color": self.DARK_THEME["text_primary"],
                        "marginBottom": "4px",
                        "fontSize": "12px",
                        "fontWeight": "bold",
                    },
                ),
                html.Div(id="training-manager-content"),
            ],
            style=self._card_style(),
        )
    
    def create_content(self, state) -> html.Div:
        """Create the Training Manager panel content"""
        
        # Section 1: Continuous Training
        continuous_section = self._create_continuous_training_section(state)
        
        # Section 2: Training Lifecycle
        training_lifecycle_section = self._create_training_lifecycle_section(state)
        
        # Section 3: Day Lifecycle
        day_lifecycle_section = self._create_day_lifecycle_section(state)
        
        # Section 4: Reset Points
        reset_points_section = self._create_reset_points_section(state)
        
        return html.Div([
            continuous_section,
            self._section_separator(),
            training_lifecycle_section,
            self._section_separator(),
            day_lifecycle_section,
            self._section_separator(),
            reset_points_section,
        ])
    
    def _create_continuous_training_section(self, state) -> html.Div:
        """Section 1: Continuous Training Status"""
        # Get continuous training info from dashboard state
        evaluation_frequency = getattr(state, "evaluation_frequency", 999)
        evaluation_episodes = getattr(state, "evaluation_episodes", 1)
        recommendation_frequency = getattr(state, "recommendation_frequency", 10)
        checkpoint_frequency = getattr(state, "checkpoint_frequency", 25)
        
        # Check if continuous training is active
        continuous_active = getattr(state, "continuous_training_active", False)
        training_mode = getattr(state, "training_mode", "production")
        
        # Current state
        current_updates = getattr(state, "total_updates", 0) or getattr(state, "updates", 0)
        last_evaluation_update = getattr(state, "last_evaluation_update", 0)
        last_checkpoint_update = getattr(state, "last_checkpoint_update", 0)
        
        # Calculate next events
        updates_until_eval = evaluation_frequency - (current_updates % evaluation_frequency) if evaluation_frequency > 0 else "N/A"
        updates_until_checkpoint = checkpoint_frequency - (current_updates % checkpoint_frequency) if checkpoint_frequency > 0 else "N/A"
        
        # Performance analysis from training_manager_update
        performance_trend = getattr(state, "performance_trend", "stable")
        updates_since_improvement = getattr(state, "updates_since_improvement", 0)
        best_performance_episode = getattr(state, "best_performance_episode", 0)
        
        # Calculate confidence based on trend
        confidence = 0.0
        if performance_trend == "improving" or performance_trend == "excelling":
            confidence = 0.8 + (0.2 if updates_since_improvement < 5 else 0.0)
        elif performance_trend == "stable":
            confidence = 0.5
        elif performance_trend == "declining" or performance_trend == "degrading":
            confidence = 0.2
        
        trend_color = self._get_trend_color(performance_trend)
        
        # Current performance metrics
        mean_reward = getattr(state, "mean_episode_reward", 0.0)
        best_reward = getattr(state, "best_reward", mean_reward)
        
        # Training intensity
        training_intensity = getattr(state, "training_intensity", "normal")
        
        return html.Div([
            html.H5("Continuous Training", style=self._section_header_style()),
            self._info_row("Enabled", "✅ Yes" if continuous_active else "❌ No"),
            self._info_row("Mode", training_mode.title()),
            self._info_row("Intensity", training_intensity.title()),
            self._info_row("Performance", performance_trend.title(), color=trend_color),
            self._info_row("Confidence", f"{confidence:.1%}"),
            self._info_row("Since Improve", f"{updates_since_improvement} updates"),
            self._info_row("Next Eval", f"{updates_until_eval} updates" if updates_until_eval != "N/A" else "N/A"),
            self._info_row("Next Checkpoint", f"{updates_until_checkpoint} updates" if updates_until_checkpoint != "N/A" else "N/A"),
            self._info_row("Current Reward", f"{mean_reward:.4f}"),
            self._info_row("Best Reward", f"{best_reward:.4f}"),
            self._info_row("Best Episode", f"#{best_performance_episode}" if best_performance_episode > 0 else "N/A"),
        ])
    
    def _create_training_lifecycle_section(self, state) -> html.Div:
        """Section 2: Training Lifecycle Management"""
        # Get training limits from training_manager_update
        training_max_episodes = getattr(state, "training_max_episodes", float('inf'))
        training_max_updates = getattr(state, "training_max_updates", float('inf')) 
        training_max_cycles = getattr(state, "training_max_cycles", float('inf'))
        
        # Convert inf to None for display
        if training_max_episodes == float('inf'):
            training_max_episodes = None
        if training_max_updates == float('inf'):
            training_max_updates = None
        if training_max_cycles == float('inf'):
            training_max_cycles = None
        
        # Current progress from state
        current_episodes = getattr(state, "total_episodes", 0)
        current_updates = getattr(state, "total_updates", 0) or getattr(state, "updates", 0)
        current_cycles = getattr(state, "cycles_completed", 0) or getattr(state, "cycle_count", 0)
        
        # Get termination tracking from training_manager_update
        episodes_until_termination = getattr(state, "episodes_until_termination", 0)
        updates_until_termination = getattr(state, "updates_until_termination", 0)
        cycles_until_termination = getattr(state, "cycles_until_termination", 0)
        termination_progress_pct = getattr(state, "termination_progress_pct", 0.0)
        
        # Use termination tracking for remaining values if available
        if episodes_until_termination > 0:
            episodes_remaining = episodes_until_termination
        else:
            episodes_remaining = (training_max_episodes - current_episodes) if training_max_episodes else "∞"
            
        if updates_until_termination > 0:
            updates_remaining = updates_until_termination
        else:
            updates_remaining = (training_max_updates - current_updates) if training_max_updates else "∞"
            
        if cycles_until_termination > 0:
            cycles_remaining = cycles_until_termination
        else:
            cycles_remaining = (training_max_cycles - current_cycles) if training_max_cycles else "∞"
        
        # Training time info
        training_hours = getattr(state, "training_hours", 0.0)
        session_start_time = getattr(state, "session_start_time", datetime.now())
        
        # Format start time
        if hasattr(session_start_time, "strftime"):
            start_time_str = session_start_time.strftime("%H:%M")
        elif isinstance(session_start_time, str):
            start_time_str = session_start_time
        else:
            start_time_str = "Unknown"
        
        # Overall progress and next stage info
        overall_progress = getattr(state, "overall_progress", 0.0)
        episodes_to_next_stage = getattr(state, "episodes_to_next_stage", 0)
        next_stage_name = getattr(state, "next_stage_name", "")
        
        # Termination reason if any
        termination_reason = getattr(state, "termination_reason", None)
        
        return html.Div([
            html.H5("Training Lifecycle", style=self._section_header_style()),
            self._info_row("Progress", f"{overall_progress:.1f}%"),
            self._info_row("Next Stage", next_stage_name if next_stage_name else "Continuous"),
            self._info_row("To Next", f"{episodes_to_next_stage} eps" if episodes_to_next_stage > 0 else "N/A"),
            self._info_row("Training Time", f"{training_hours:.1f}h"),
            self._info_row("Started", start_time_str),
            self._info_row("Episodes", f"{current_episodes:,}" + (f"/{training_max_episodes:,}" if training_max_episodes else "")),
            self._info_row("Remaining", str(episodes_remaining)),
            self._info_row("Updates", f"{current_updates:,}" + (f"/{training_max_updates:,}" if training_max_updates else "")),
            self._info_row("Remaining", str(updates_remaining)),
            self._info_row("Cycles", f"{current_cycles}" + (f"/{training_max_cycles}" if training_max_cycles else "")),
            self._info_row("Remaining", str(cycles_remaining)),
            self._info_row("Termination", f"{termination_progress_pct:.1f}%" if not termination_reason else termination_reason),
        ])
    
    def _create_day_lifecycle_section(self, state) -> html.Div:
        """Section 3: Day Lifecycle Management"""
        # Current day information from state
        current_day = getattr(state, "current_momentum_day_date", "N/A")
        current_symbol = getattr(state, "current_symbol", "N/A")
        day_quality = getattr(state, "current_momentum_day_quality", 0.0)
        
        # Day switching info from training_manager_update
        day_switch_in_progress = getattr(state, "day_switch_in_progress", False)
        next_day_date = getattr(state, "next_day_date", "")
        next_day_quality = getattr(state, "next_day_quality", 0.0)
        episodes_until_day_switch = getattr(state, "episodes_until_day_switch", 0)
        updates_until_day_switch = getattr(state, "updates_until_day_switch", 0)
        
        # Day progress tracking
        episodes_on_current_day = getattr(state, "episodes_on_current_day", 0)
        day_switch_progress_pct = getattr(state, "day_switch_progress_pct", 0.0)
        cycles_remaining_for_day_switch = getattr(state, "cycles_remaining_for_day_switch", 0)
        
        # Current cycles from state
        current_cycles = getattr(state, "cycles_completed", 0) or getattr(state, "cycle_count", 0)
        target_cycles_per_day = getattr(state, "target_cycles_per_day", 10)
        
        # Data lifecycle stage
        data_lifecycle_stage = getattr(state, "data_lifecycle_stage", "unknown")
        stage_progress = getattr(state, "stage_progress", 0.0)
        
        # Selection criteria from adaptive data
        day_score_range = getattr(state, "day_score_range", [0.0, 1.0])
        selection_mode = getattr(state, "selection_mode", "adaptive")
        
        # Safely format day score range
        if isinstance(day_score_range, (list, tuple)) and len(day_score_range) >= 2:
            quality_range_str = f"{day_score_range[0]:.2f}-{day_score_range[1]:.2f}"
        else:
            quality_range_str = "0.0-1.0"
        
        # Format next day info
        if next_day_date and next_day_date != "Same day":
            next_day_info = f"{next_day_date} (Q:{next_day_quality:.3f})"
        elif day_switch_in_progress:
            next_day_info = "Selecting..."
        else:
            next_day_info = "Same day"
        
        # Switch status display
        if day_switch_in_progress:
            switch_status = "🔄 Switching"
            switch_color = self.DARK_THEME["accent_orange"]
        elif episodes_until_day_switch <= 5 or updates_until_day_switch <= 10:
            switch_status = "⚠️ Soon"
            switch_color = self.DARK_THEME["accent_orange"]
        else:
            switch_status = "✅ Active"
            switch_color = self.DARK_THEME["accent_green"]
        
        # ETA display
        eta_parts = []
        if episodes_until_day_switch > 0:
            eta_parts.append(f"{episodes_until_day_switch} eps")
        if updates_until_day_switch > 0:
            eta_parts.append(f"{updates_until_day_switch} upd")
        if cycles_remaining_for_day_switch > 0:
            eta_parts.append(f"{cycles_remaining_for_day_switch} cyc")
        eta_str = " / ".join(eta_parts) if eta_parts else "Ready"
        
        return html.Div([
            html.H5("Day Lifecycle", style=self._section_header_style()),
            self._info_row("Current Day", f"{current_day}"),
            self._info_row("Symbol", current_symbol),
            self._info_row("Quality", f"{day_quality:.3f}" if day_quality > 0 else "N/A"),
            self._info_row("Stage", data_lifecycle_stage.title()),
            self._info_row("Stage Progress", f"{stage_progress:.1f}%"),
            self._info_row("Episodes Today", str(episodes_on_current_day)),
            self._info_row("Cycles", f"{current_cycles}/{target_cycles_per_day}"),
            self._info_row("Day Progress", f"{day_switch_progress_pct:.1f}%"),
            self._info_row("Switch Status", switch_status, color=switch_color),
            self._info_row("Next Day", next_day_info),
            self._info_row("Switch In", eta_str),
            self._info_row("Selection", f"{selection_mode} ({quality_range_str})"),
        ])
    
    def _create_reset_points_section(self, state) -> html.Div:
        """Section 4: Reset Points Management"""
        # Reset point cycling info from state
        current_reset_point_index = getattr(state, "current_reset_point_index", 0)
        total_reset_points = getattr(state, "total_reset_points", 0)
        current_cycle = getattr(state, "current_cycle", 0)
        cycle_progress = getattr(state, "cycle_progress", 0.0)
        
        # Reset point info from training_manager_update
        total_available_points = getattr(state, "total_available_points", 0)
        points_used_in_cycle = getattr(state, "points_used_in_cycle", 0)
        points_remaining_in_cycle = getattr(state, "points_remaining_in_cycle", 0)
        
        # Use total_available_points if available (more accurate from training manager)
        if total_available_points > 0:
            total_reset_points = total_available_points
        
        # Reset point quality metrics from state
        current_roc_score = getattr(state, "current_roc_score", 0.0)
        current_activity_score = getattr(state, "current_activity_score", 0.0)
        selected_reset_point_timestamp = getattr(state, "selected_reset_point_timestamp", "")
        
        # Reset point reuse tracking
        reset_point_reuse_count = getattr(state, "reset_point_reuse_count", 0)
        max_reset_point_reuse = getattr(state, "max_reset_point_reuse", 3)
        
        # Reset point status from training_manager_update
        reset_points_exhausted = getattr(state, "reset_points_exhausted", False)
        reset_points_low_warning = getattr(state, "reset_points_low_warning", False)
        
        # Preload status
        preload_in_progress = getattr(state, "preload_in_progress", False)
        preload_ready = getattr(state, "preload_ready", False)
        preload_progress_pct = getattr(state, "preload_progress_pct", 0.0)
        next_stage_preloaded = getattr(state, "next_stage_preloaded", "")
        
        # Selection criteria
        roc_range = getattr(state, "roc_range", [0.0, 1.0])
        activity_range = getattr(state, "activity_range", [0.0, 1.0])
        
        # Format ranges
        if isinstance(roc_range, (list, tuple)) and len(roc_range) >= 2:
            roc_range_str = f"{roc_range[0]:.2f}-{roc_range[1]:.2f}"
        else:
            roc_range_str = "0.0-1.0"
            
        if isinstance(activity_range, (list, tuple)) and len(activity_range) >= 2:
            activity_range_str = f"{activity_range[0]:.2f}-{activity_range[1]:.2f}"
        else:
            activity_range_str = "0.0-1.0"
        
        # Progress display
        if total_reset_points > 0:
            if points_used_in_cycle > 0:
                reset_points_display = f"{points_used_in_cycle}/{total_reset_points}"
            else:
                reset_points_display = f"{current_reset_point_index + 1}/{total_reset_points}"
        else:
            reset_points_display = "N/A"
        
        # Status display
        if reset_points_exhausted:
            status_display = "🔴 Exhausted"
            status_color = self.DARK_THEME["accent_red"]
        elif reset_points_low_warning:
            status_display = "⚠️ Low"
            status_color = self.DARK_THEME["accent_orange"]
        elif preload_in_progress:
            status_display = "📥 Preloading"
            status_color = self.DARK_THEME["accent_blue"]
        elif preload_ready:
            status_display = "✅ Ready"
            status_color = self.DARK_THEME["accent_green"]
        else:
            status_display = "✅ Available"
            status_color = self.DARK_THEME["accent_green"]
        
        # Preload info
        if preload_in_progress:
            preload_info = f"Loading {preload_progress_pct:.0f}%"
        elif next_stage_preloaded:
            preload_info = next_stage_preloaded
        else:
            preload_info = "None"
        
        # Format timestamp
        if selected_reset_point_timestamp:
            # Extract time from timestamp if it's a full datetime
            if "T" in selected_reset_point_timestamp:
                time_part = selected_reset_point_timestamp.split("T")[1].split(".")[0][:5]  # HH:MM
            else:
                time_part = selected_reset_point_timestamp[:5] if len(selected_reset_point_timestamp) >= 5 else selected_reset_point_timestamp
        else:
            time_part = "N/A"
        
        return html.Div([
            html.H5("Reset Points", style=self._section_header_style()),
            self._info_row("Current", reset_points_display),
            self._info_row("Cycle", f"{current_cycle} ({cycle_progress:.1f}%)"),
            self._info_row("Remaining", str(points_remaining_in_cycle)),
            self._info_row("Status", status_display, color=status_color),
            self._info_row("Reuse", f"{reset_point_reuse_count}/{max_reset_point_reuse}"),
            self._info_row("Time", time_part),
            self._info_row("ROC", f"{current_roc_score:.3f}" if current_roc_score > 0 else "N/A"),
            self._info_row("Activity", f"{current_activity_score:.3f}" if current_activity_score > 0 else "N/A"),
            self._info_row("ROC Range", roc_range_str),
            self._info_row("Act Range", activity_range_str),
            self._info_row("Preload", preload_info),
        ])
    
    def _get_trend_color(self, trend: str) -> str:
        """Get color for performance trend"""
        trend_colors = {
            "improving": self.DARK_THEME["accent_green"],
            "excelling": self.DARK_THEME["accent_green"],
            "stable": self.DARK_THEME["accent_blue"],
            "struggling": self.DARK_THEME["accent_orange"],
            "declining": self.DARK_THEME["accent_red"],
            "degrading": self.DARK_THEME["accent_red"],
            "plateau": self.DARK_THEME["accent_orange"],
            "unknown": self.DARK_THEME["text_muted"],
        }
        return trend_colors.get(trend.lower(), self.DARK_THEME["text_primary"])
    
    def _info_row(self, label: str, value: str, color: Optional[str] = None) -> html.Div:
        """Create an info row with label left-aligned and value right-aligned"""
        value_color = color or self.DARK_THEME["text_primary"]
        return html.Div(
            [
                html.Span(label, style={"color": self.DARK_THEME["text_secondary"], "fontSize": "11px"}),
                html.Span(value, style={"color": value_color, "fontWeight": "bold", "fontSize": "11px"}),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-between", 
                "alignItems": "center",
                "marginBottom": "2px",
                "minHeight": "14px",
            },
        )
    
    def _section_header_style(self) -> dict:
        """Style for section headers"""
        return {
            "color": self.DARK_THEME["accent_blue"],
            "fontSize": "10px",
            "fontWeight": "bold",
            "marginBottom": "4px",
            "marginTop": "2px",
        }
    
    def _section_separator(self) -> html.Hr:
        """Create a section separator"""
        return html.Hr(style={
            "margin": "6px 0 4px 0", 
            "borderColor": self.DARK_THEME["border"],
            "opacity": "0.5"
        })
    
    def _card_style(self) -> dict:
        """Standard card styling"""
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }
    
    def _section_header_style(self) -> dict:
        """Style for section headers"""
        return {
            "color": self.DARK_THEME["accent_blue"],
            "fontSize": "10px",
            "fontWeight": "bold",
            "marginBottom": "4px",
            "marginTop": "2px",
        }
    
    def _section_separator(self) -> html.Hr:
        """Create a section separator"""
        return html.Hr(style={
            "margin": "6px 0 4px 0", 
            "borderColor": self.DARK_THEME["border"],
            "opacity": "0.5"
        })
    
    def _card_style(self) -> dict:
        """Standard card styling"""
        return {
            "backgroundColor": self.DARK_THEME["bg_secondary"],
            "border": f"1px solid {self.DARK_THEME['border']}",
            "borderRadius": "6px",
            "padding": "8px",
            "height": "100%",
            "overflow": "auto",
        }