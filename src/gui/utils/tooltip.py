"""
Tooltip utility for creating informative tooltips in the GUI.
"""

import tkinter as tk
from typing import Optional


class ToolTip:
    """
    A tooltip widget that shows informative text when hovering over widgets.
    """
    
    def __init__(self, widget, text: str, delay: int = 500, wrap_length: int = 300):
        """
        Initialize tooltip for a widget.
        
        Args:
            widget: The widget to attach tooltip to
            text: Tooltip text to display
            delay: Delay before showing tooltip (ms)
            wrap_length: Maximum width before text wraps
        """
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wrap_length = wrap_length
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.after_id = None
        
        # Bind events
        self.widget.bind("<Enter>", self._on_enter)
        self.widget.bind("<Leave>", self._on_leave)
        self.widget.bind("<Motion>", self._on_motion)
    
    def _on_enter(self, event=None):
        """Handle mouse enter event."""
        self._schedule_tooltip()
    
    def _on_leave(self, event=None):
        """Handle mouse leave event."""
        self._cancel_tooltip()
        self._hide_tooltip()
    
    def _on_motion(self, event=None):
        """Handle mouse motion event."""
        self._cancel_tooltip()
        self._schedule_tooltip()
    
    def _schedule_tooltip(self):
        """Schedule tooltip to appear after delay."""
        self._cancel_tooltip()
        self.after_id = self.widget.after(self.delay, self._show_tooltip)
    
    def _cancel_tooltip(self):
        """Cancel scheduled tooltip."""
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
    
    def _show_tooltip(self):
        """Show the tooltip window."""
        if self.tooltip_window:
            return
        
        # Get widget position
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        # Create tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        
        # Style the tooltip
        self.tooltip_window.configure(bg="#2b2b2b", relief="solid", borderwidth=1)
        
        # Create label with text
        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            background="#2b2b2b",
            foreground="white",
            font=("Arial", 10),
            wraplength=self.wrap_length,
            justify="left",
            padx=8,
            pady=6
        )
        label.pack()
    
    def _hide_tooltip(self):
        """Hide the tooltip window."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def add_tooltip(widget, text: str, delay: int = 500, wrap_length: int = 300) -> ToolTip:
    """
    Convenience function to add tooltip to a widget.
    
    Args:
        widget: Widget to add tooltip to
        text: Tooltip text
        delay: Delay before showing (ms)
        wrap_length: Text wrap length
        
    Returns:
        ToolTip instance
    """
    return ToolTip(widget, text, delay, wrap_length)


# Parameter descriptions for tooltips
PARAMETER_DESCRIPTIONS = {
    # HDBSCAN Clustering Parameters
    "min_cluster_size": (
        "Minimum Cluster Size (HDBSCAN)\n"
        "The minimum number of documents required to form a cluster. "
        "Smaller values create more, smaller clusters. Larger values create fewer, larger clusters. "
        "Typical range: 5-50 documents."
    ),
    "min_samples": (
        "Minimum Samples (HDBSCAN)\n"
        "The minimum number of documents in a neighborhood for a point to be considered core. "
        "Lower values make clustering more aggressive, higher values make it more conservative. "
        "Should be ≤ min_cluster_size."
    ),
    "cluster_selection_epsilon": (
        "Cluster Selection Epsilon (HDBSCAN)\n"
        "Distance threshold for cluster extraction. Clusters separated by less than this distance "
        "will be merged. Set to 0.0 to use default HDBSCAN behavior."
    ),
    "cluster_selection_method": (
        "Cluster Selection Method (HDBSCAN)\n"
        "'eom' (Excess of Mass): More stable, creates fewer clusters\n"
        "'leaf': More sensitive, creates more smaller clusters"
    ),
    
    # UMAP Dimensionality Reduction Parameters  
    "n_neighbors": (
        "Number of Neighbors (UMAP)\n"
        "Controls local vs global structure preservation. "
        "Lower values (5-15) preserve local structure, higher values (30-100) preserve global structure. "
        "Affects how tightly clustered the reduced dimensions are."
    ),
    "n_components": (
        "Number of Components (UMAP)\n"
        "Dimensionality of the reduced embedding space. "
        "Lower values (2-5) for visualization, higher values (10-50) for clustering. "
        "More components preserve more information but increase computation."
    ),
    "min_dist": (
        "Minimum Distance (UMAP)\n"
        "Controls how tightly points are packed in the low-dimensional space. "
        "Lower values (0.0-0.1) create tighter clusters, higher values (0.3-0.8) spread points out more. "
        "Affects cluster separation."
    ),
    "metric": (
        "Distance Metric (UMAP)\n"
        "Method for measuring distance between documents:\n"
        "• cosine: Good for text (default)\n"
        "• euclidean: Standard geometric distance\n"
        "• manhattan: Sum of absolute differences"
    ),
    
    # Vectorization Parameters
    "min_df": (
        "Minimum Document Frequency\n"
        "Ignore terms that appear in fewer than this many documents. "
        "Higher values remove rare words, reducing noise but potentially losing specific topics. "
        "Can be integer (absolute count) or float (proportion)."
    ),
    "max_df": (
        "Maximum Document Frequency\n"
        "Ignore terms that appear in more than this proportion of documents. "
        "Helps remove very common words that don't distinguish topics. "
        "Typical values: 0.8-0.95 (80%-95% of documents)."
    ),
    "ngram_range": (
        "N-gram Range\n"
        "Range of n-values for different n-grams to extract. "
        "(1,1): Only single words, (1,2): Single words and bigrams, (2,2): Only bigrams. "
        "Larger ranges capture more context but increase complexity."
    ),
    "max_features": (
        "Maximum Features\n"
        "Maximum number of terms to include in the vocabulary. "
        "Keeps only the most frequent terms. Higher values preserve more vocabulary "
        "but increase memory usage and computation time."
    ),
    
    # BERTopic Parameters
    "top_n_words": (
        "Top N Words per Topic\n"
        "Number of most important words to extract for each topic. "
        "More words provide better topic understanding but can include noise. "
        "Typical range: 5-20 words."
    ),
    "nr_topics": (
        "Number of Topics\n"
        "Target number of topics to reduce to. Leave empty for automatic detection. "
        "Lower values create broader topics, higher values create more specific topics. "
        "Consider your document collection size and desired granularity."
    ),
    "low_memory": (
        "Low Memory Mode\n"
        "Reduces memory usage during training by computing embeddings in smaller batches. "
        "Enable for large datasets or limited memory systems. May increase processing time."
    ),
    "calculate_probabilities": (
        "Calculate Topic Probabilities\n"
        "Compute the probability of each document belonging to each topic. "
        "Useful for uncertainty analysis but increases computation time significantly."
    ),
    
    # Representation Model Parameters
    "use_representation": (
        "Use Representation Models\n"
        "Enhance topic descriptions using advanced language models. "
        "Improves topic coherence and interpretability but requires more computation time."
    ),
    "diversity": (
        "Representation Diversity\n"
        "Controls diversity of words in topic representations. "
        "Higher values (0.7-0.9) encourage more diverse words, "
        "lower values (0.1-0.3) focus on most similar words."
    )
}