# UI/UX Improvements and Enhancements

## 1. Visual Design Improvements

### Current State
- Basic CustomTkinter theme
- Limited visual feedback
- No animations or transitions
- Minimal use of icons

### Proposed Enhancements

#### A. Modern Visual Theme
```python
class ModernTheme:
    """Enhanced visual theme with gradients and shadows."""
    
    # Color palette
    COLORS = {
        'primary': '#2563eb',      # Blue
        'primary_hover': '#1d4ed8',
        'secondary': '#7c3aed',    # Purple
        'success': '#10b981',      # Green
        'warning': '#f59e0b',      # Amber
        'error': '#ef4444',        # Red
        'background': '#f8fafc',
        'surface': '#ffffff',
        'text': '#1e293b',
        'text_secondary': '#64748b'
    }
    
    # Shadows
    SHADOWS = {
        'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1)',
        'xl': '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
    }
    
    @staticmethod
    def apply_gradient(widget, start_color: str, end_color: str):
        """Apply gradient background to widget."""
        # Implementation would use Canvas for gradient effect
        pass
```

#### B. Icon Integration
```python
class IconManager:
    """Manage and provide icons for UI elements."""
    
    ICONS = {
        'file_open': '📁',
        'save': '💾',
        'export': '📤',
        'settings': '⚙️',
        'help': '❓',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️',
        'play': '▶️',
        'pause': '⏸️',
        'stop': '⏹️',
        'refresh': '🔄'
    }
    
    @classmethod
    def get_icon(cls, name: str, size: int = 16) -> str:
        """Get icon with specified size."""
        icon = cls.ICONS.get(name, '')
        # In real implementation, would return proper icon image
        return icon
```

#### C. Smooth Animations
```python
class AnimationController:
    """Handle UI animations and transitions."""
    
    @staticmethod
    def fade_in(widget, duration: int = 300):
        """Fade in animation."""
        widget.configure(opacity=0)
        
        def animate(step):
            opacity = step / 100
            widget.configure(opacity=opacity)
            
            if step < 100:
                widget.after(duration // 100, lambda: animate(step + 1))
        
        animate(0)
    
    @staticmethod
    def slide_in(widget, direction: str = 'left', duration: int = 300):
        """Slide in animation from specified direction."""
        # Get current position
        x = widget.winfo_x()
        y = widget.winfo_y()
        
        # Set starting position
        if direction == 'left':
            widget.place(x=x - 100, y=y)
        elif direction == 'right':
            widget.place(x=x + 100, y=y)
        
        # Animate to final position
        steps = 30
        dx = 100 / steps if direction == 'left' else -100 / steps
        
        def animate(step):
            if step < steps:
                current_x = widget.winfo_x()
                widget.place(x=current_x + dx)
                widget.after(duration // steps, lambda: animate(step + 1))
        
        animate(0)
```

## 2. User Interaction Improvements

### A. Enhanced Tooltips
```python
class EnhancedTooltip:
    """Rich tooltips with formatting and images."""
    
    def __init__(self, widget, text: str, title: Optional[str] = None,
                 image: Optional[str] = None, delay: int = 500):
        self.widget = widget
        self.text = text
        self.title = title
        self.image = image
        self.delay = delay
        self.tooltip_window = None
        self.show_timer = None
        
        # Bind events
        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)
        widget.bind("<ButtonPress>", self.on_leave)
    
    def show_tooltip(self):
        """Display the tooltip."""
        if self.tooltip_window or not self.text:
            return
        
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        # Create tooltip window
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        # Style the tooltip
        frame = ctk.CTkFrame(tw, corner_radius=8)
        frame.pack()
        
        # Add title if provided
        if self.title:
            title_label = ctk.CTkLabel(
                frame,
                text=self.title,
                font=ctk.CTkFont(weight="bold")
            )
            title_label.pack(padx=10, pady=(10, 5))
        
        # Add image if provided
        if self.image:
            # Load and display image
            pass
        
        # Add text
        text_label = ctk.CTkLabel(
            frame,
            text=self.text,
            wraplength=300
        )
        text_label.pack(padx=10, pady=(5, 10))
```

### B. Context Menus
```python
class ContextMenu:
    """Right-click context menus for widgets."""
    
    def __init__(self, parent):
        self.parent = parent
        self.menu = tk.Menu(parent, tearoff=0)
        self.setup_menu()
    
    def setup_menu(self):
        """Set up menu items."""
        self.menu.add_command(
            label="Cut",
            accelerator="Ctrl+X",
            command=self.cut
        )
        self.menu.add_command(
            label="Copy",
            accelerator="Ctrl+C",
            command=self.copy
        )
        self.menu.add_command(
            label="Paste",
            accelerator="Ctrl+V",
            command=self.paste
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="Select All",
            accelerator="Ctrl+A",
            command=self.select_all
        )
    
    def show(self, event):
        """Show context menu at cursor position."""
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
```

### C. Keyboard Navigation
```python
class KeyboardNavigationMixin:
    """Mixin for keyboard navigation support."""
    
    def setup_keyboard_navigation(self):
        """Set up keyboard shortcuts and navigation."""
        # Tab navigation
        self.bind("<Tab>", self.focus_next_widget)
        self.bind("<Shift-Tab>", self.focus_prev_widget)
        
        # Arrow key navigation for lists
        self.bind("<Up>", self.select_previous_item)
        self.bind("<Down>", self.select_next_item)
        
        # Page navigation
        self.bind("<Prior>", self.page_up)  # Page Up
        self.bind("<Next>", self.page_down)  # Page Down
        
        # Quick actions
        self.bind("<Return>", self.activate_item)
        self.bind("<space>", self.toggle_item)
        self.bind("<Escape>", self.cancel_action)
    
    def focus_next_widget(self, event):
        """Focus next widget in tab order."""
        event.widget.tk_focusNext().focus()
        return "break"
```

## 3. Workflow Enhancements

### A. Guided Workflow Mode
```python
class GuidedWorkflow:
    """Step-by-step guided workflow for new users."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.current_step = 0
        self.steps = [
            WorkflowStep(
                "Welcome",
                "Let's get started with BERTopic analysis!",
                "This guide will walk you through the process."
            ),
            WorkflowStep(
                "Load Data",
                "First, select your dataset",
                "Click 'Browse' to choose a CSV, Excel, or Parquet file."
            ),
            # More steps...
        ]
        
    def start(self):
        """Start the guided workflow."""
        self.show_overlay()
        self.show_current_step()
    
    def show_overlay(self):
        """Show semi-transparent overlay."""
        self.overlay = tk.Toplevel(self.main_window)
        self.overlay.attributes('-alpha', 0.3)
        self.overlay.attributes('-fullscreen', True)
        self.overlay.configure(bg='black')
    
    def highlight_element(self, widget):
        """Highlight the current widget."""
        # Create highlight frame around widget
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        width = widget.winfo_width()
        height = widget.winfo_height()
        
        self.highlight = tk.Toplevel()
        self.highlight.overrideredirect(True)
        self.highlight.attributes('-alpha', 0.5)
        self.highlight.geometry(f"{width+10}x{height+10}+{x-5}+{y-5}")
        
        # Create border effect
        canvas = tk.Canvas(self.highlight, highlightthickness=0)
        canvas.pack(fill='both', expand=True)
        canvas.create_rectangle(
            2, 2, width+8, height+8,
            outline='#2563eb', width=3
        )
```

### B. Quick Actions Bar
```python
class QuickActionsBar(ctk.CTkFrame):
    """Contextual quick actions based on current state."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.actions = []
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the quick actions bar."""
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text="Quick Actions",
            font=ctk.CTkFont(weight="bold")
        )
        self.title_label.pack(side="left", padx=10)
        
        # Actions container
        self.actions_frame = ctk.CTkFrame(self)
        self.actions_frame.pack(side="left", fill="x", expand=True, padx=10)
    
    def update_actions(self, context: Dict[str, Any]):
        """Update available actions based on context."""
        # Clear existing actions
        for widget in self.actions_frame.winfo_children():
            widget.destroy()
        
        # Add contextual actions
        if context.get('data_loaded'):
            self.add_action("Generate Embeddings", "▶️", self.generate_embeddings)
            
        if context.get('embeddings_ready'):
            self.add_action("Run Topic Modeling", "🔍", self.run_modeling)
            
        if context.get('results_available'):
            self.add_action("Export Results", "📤", self.export_results)
            self.add_action("Create Report", "📊", self.create_report)
    
    def add_action(self, text: str, icon: str, command: Callable):
        """Add an action button."""
        btn = ctk.CTkButton(
            self.actions_frame,
            text=f"{icon} {text}",
            command=command,
            width=150
        )
        btn.pack(side="left", padx=5)
```

### C. Smart Suggestions
```python
class SmartSuggestions:
    """Provide intelligent suggestions based on data and user actions."""
    
    def __init__(self):
        self.suggestion_rules = [
            SuggestionRule(
                condition=lambda ctx: ctx['document_count'] > 50000,
                suggestion="Consider using a sample for faster processing",
                action="enable_sampling"
            ),
            SuggestionRule(
                condition=lambda ctx: ctx['avg_text_length'] < 50,
                suggestion="Short texts detected. Consider combining columns",
                action="show_column_combination"
            ),
            # More rules...
        ]
    
    def get_suggestions(self, context: Dict[str, Any]) -> List[Suggestion]:
        """Get relevant suggestions for current context."""
        suggestions = []
        
        for rule in self.suggestion_rules:
            if rule.condition(context):
                suggestions.append(Suggestion(
                    text=rule.suggestion,
                    action=rule.action,
                    priority=rule.priority
                ))
        
        return sorted(suggestions, key=lambda s: s.priority, reverse=True)
```

## 4. Data Visualization Enhancements

### A. Interactive Dashboard
```python
class InteractiveDashboard(ctk.CTkFrame):
    """Multi-panel interactive dashboard for results exploration."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.panels = {}
        self.setup_ui()
    
    def setup_ui(self):
        """Set up dashboard layout."""
        # Create resizable paned window
        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True)
        
        # Left panel - Topic list
        self.topic_panel = self.create_topic_panel()
        self.paned.add(self.topic_panel, weight=1)
        
        # Center panel - Visualization
        self.viz_panel = self.create_viz_panel()
        self.paned.add(self.viz_panel, weight=2)
        
        # Right panel - Details
        self.detail_panel = self.create_detail_panel()
        self.paned.add(self.detail_panel, weight=1)
    
    def create_topic_panel(self):
        """Create interactive topic list panel."""
        panel = ctk.CTkFrame(self.paned)
        
        # Search box
        search_frame = ctk.CTkFrame(panel)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="Search topics..."
        )
        search_entry.pack(fill="x")
        
        # Topic list with virtual scrolling
        self.topic_list = VirtualListbox(
            panel,
            item_height=60,
            render_func=self.render_topic_item
        )
        self.topic_list.pack(fill="both", expand=True, padx=10, pady=10)
        
        return panel
```

### B. Advanced Plot Interactions
```python
class InteractivePlot:
    """Enhanced plot interactions with zoom, pan, and selection."""
    
    def __init__(self, figure, canvas):
        self.figure = figure
        self.canvas = canvas
        self.ax = figure.gca()
        
        # Interaction state
        self.zoom_enabled = False
        self.pan_enabled = False
        self.selection_enabled = False
        self.selected_points = []
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
    
    def enable_lasso_selection(self):
        """Enable lasso selection tool."""
        from matplotlib.widgets import LassoSelector
        
        self.lasso = LassoSelector(
            self.ax,
            onselect=self.on_select,
            useblit=True
        )
        self.selection_enabled = True
    
    def on_select(self, verts):
        """Handle lasso selection."""
        path = Path(verts)
        
        # Get data points
        if hasattr(self.ax, 'collections'):
            for collection in self.ax.collections:
                if hasattr(collection, 'get_offsets'):
                    points = collection.get_offsets()
                    
                    # Find points inside selection
                    ind = path.contains_points(points)
                    self.selected_points = np.where(ind)[0]
                    
                    # Highlight selected points
                    self.highlight_selection()
    
    def highlight_selection(self):
        """Highlight selected points."""
        if self.selected_points:
            # Update point colors
            colors = self.ax.collections[0].get_facecolors()
            colors[self.selected_points] = [1, 0, 0, 1]  # Red
            self.ax.collections[0].set_facecolors(colors)
            self.canvas.draw_idle()
```

### C. Real-time Filtering
```python
class RealTimeFilter:
    """Real-time filtering for visualizations."""
    
    def __init__(self, data_source, update_callback):
        self.data_source = data_source
        self.update_callback = update_callback
        self.filters = {}
        self.debounce_timer = None
    
    def add_filter(self, name: str, filter_type: str, initial_value: Any):
        """Add a filter control."""
        self.filters[name] = Filter(
            name=name,
            type=filter_type,
            value=initial_value,
            callback=self.on_filter_change
        )
    
    def on_filter_change(self, filter_name: str, new_value: Any):
        """Handle filter value change with debouncing."""
        # Cancel previous timer
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        # Set new timer
        self.debounce_timer = threading.Timer(
            0.3,  # 300ms debounce
            self.apply_filters
        )
        self.debounce_timer.start()
    
    def apply_filters(self):
        """Apply all active filters."""
        filtered_data = self.data_source.copy()
        
        for name, filter_obj in self.filters.items():
            if filter_obj.is_active:
                filtered_data = filter_obj.apply(filtered_data)
        
        # Update visualization
        self.update_callback(filtered_data)
```

## 5. Accessibility Improvements

### A. Screen Reader Support
```python
class AccessibilityManager:
    """Manage accessibility features."""
    
    def __init__(self):
        self.screen_reader_enabled = self.detect_screen_reader()
        self.high_contrast_enabled = self.detect_high_contrast()
    
    def make_accessible(self, widget, role: str, label: str, 
                       description: Optional[str] = None):
        """Add accessibility attributes to widget."""
        # Set ARIA-like attributes
        widget.configure(name=f"accessible_{role}")
        
        # Add screen reader text
        if self.screen_reader_enabled:
            widget.bind("<FocusIn>", 
                       lambda e: self.announce(label, description))
    
    def announce(self, text: str, description: Optional[str] = None):
        """Announce text to screen reader."""
        announcement = text
        if description:
            announcement += f". {description}"
        
        # Platform-specific announcement
        if sys.platform == "win32":
            # Use Windows SAPI
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(announcement)
        elif sys.platform == "darwin":
            # Use macOS say command
            os.system(f'say "{announcement}"')
```

### B. Keyboard-Only Navigation
```python
class KeyboardNavigationManager:
    """Comprehensive keyboard navigation support."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.focusable_widgets = []
        self.current_focus_index = 0
        
        # Set up global key bindings
        self.setup_global_bindings()
    
    def setup_global_bindings(self):
        """Set up application-wide keyboard shortcuts."""
        bindings = {
            '<Control-Tab>': self.next_tab,
            '<Control-Shift-Tab>': self.previous_tab,
            '<F6>': self.focus_next_pane,
            '<Shift-F6>': self.focus_previous_pane,
            '<Alt-Key>': self.handle_access_key,
        }
        
        for key, handler in bindings.items():
            self.main_window.bind(key, handler)
    
    def register_widget(self, widget, navigation_group: str = "main"):
        """Register widget for keyboard navigation."""
        widget.navigation_group = navigation_group
        self.focusable_widgets.append(widget)
        
        # Add visual focus indicator
        widget.bind("<FocusIn>", 
                   lambda e: self.show_focus_indicator(widget))
        widget.bind("<FocusOut>", 
                   lambda e: self.hide_focus_indicator(widget))
```

## 6. Responsive Design

### A. Adaptive Layouts
```python
class ResponsiveContainer(ctk.CTkFrame):
    """Container that adapts layout based on window size."""
    
    def __init__(self, parent, breakpoints: Dict[str, int]):
        super().__init__(parent)
        self.breakpoints = breakpoints
        self.current_layout = None
        
        # Monitor size changes
        self.bind("<Configure>", self.on_resize)
    
    def on_resize(self, event):
        """Handle window resize."""
        width = event.width
        
        # Determine layout based on width
        if width < self.breakpoints['mobile']:
            new_layout = 'mobile'
        elif width < self.breakpoints['tablet']:
            new_layout = 'tablet'
        else:
            new_layout = 'desktop'
        
        # Apply layout if changed
        if new_layout != self.current_layout:
            self.current_layout = new_layout
            self.apply_layout(new_layout)
    
    def apply_layout(self, layout: str):
        """Apply responsive layout."""
        if layout == 'mobile':
            # Stack elements vertically
            self.configure_mobile_layout()
        elif layout == 'tablet':
            # 2-column layout
            self.configure_tablet_layout()
        else:
            # Full desktop layout
            self.configure_desktop_layout()
```

## Conclusion

These UI/UX improvements focus on:

1. **Visual Appeal**: Modern design with animations and consistent theming
2. **Usability**: Intuitive interactions and helpful guidance
3. **Accessibility**: Full keyboard navigation and screen reader support
4. **Responsiveness**: Adaptive layouts for different screen sizes
5. **Productivity**: Quick actions and smart suggestions

Implementation priority should focus on core usability improvements first, followed by visual enhancements and advanced features.