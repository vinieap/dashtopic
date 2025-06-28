# BERTopic Desktop Application - UI/UX Design Specification

## Design Philosophy

The application prioritizes **ease of use**, **discoverability**, and **professional appearance**. Using CustomTkinter, the design features a modern, clean interface with rounded corners, smooth animations, and contemporary styling that guides users through the topic modeling workflow.

## Overall Layout

### Main Window Structure
```
┌─────────────────────────────────────────────────────────────────┐
│ File Edit View Tools Help                                      │
├─────────────────────────────────────────────────────────────────┤
│ [Toolbar: Quick Actions]                                        │
├─────────────────────────────────────────────────────────────────┤
│ ┌─ Data Import ─┐ ┌─ Model Config ─┐ ┌─ Topic Modeling ─┐      │
│ │               │ │                │ │                  │ ...  │
│ │   Tab Content │ │   Tab Content  │ │    Tab Content   │      │
│ │               │ │                │ │                  │      │
│ │               │ │                │ │                  │      │
│ └───────────────┘ └────────────────┘ └──────────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│ Status: Ready | Progress: ████████░░ 80% | Memory: 2.1GB       │
└─────────────────────────────────────────────────────────────────┘
```

## Tab Design Specifications

### 1. Data Import Tab

**Purpose**: Dataset loading, column selection, and data preview

#### Layout
```
┌─ Data Source ────────────────────────────────────────────────────┐
│ File Path: [Browse...] [C:\Users\...\dataset.csv]              │
│ File Type: [Auto-detect ▼] Format: CSV                         │
└─────────────────────────────────────────────────────────────────┘

┌─ Column Selection ──────────────────────────────────────────────┐
│ Available Columns:          Selected Columns:                  │
│ ┌─────────────────────┐    ┌─────────────────────┐            │
│ │ ☐ id                │    │ ☑ title             │ [↑] [↓]     │
│ │ ☑ title             │ >> │ ☑ description       │ [Remove]    │
│ │ ☑ description       │ << │                     │            │
│ │ ☐ category          │    │                     │            │
│ │ ☐ date             │    │                     │            │
│ └─────────────────────┘    └─────────────────────┘            │
│                                                                │
│ Text Column Name: [title_description        ] (concatenated)   │
│ Separator: [space ▼]                                          │
└─────────────────────────────────────────────────────────────────┘

┌─ Data Preview ──────────────────────────────────────────────────┐
│ Showing first 100 rows of concatenated text column             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Row | Text Column (title_description)                       │ │
│ │ 1   | Machine Learning Fundamentals A comprehensive guide...│ │
│ │ 2   | Python Programming Best practices for writing clea...│ │
│ │ ... │ ...                                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ Total Rows: 15,432 | Non-empty: 15,401 | Empty: 31            │
└─────────────────────────────────────────────────────────────────┘

[Validate Data] [Next: Model Configuration →]
```

#### Key Features
- **Drag & Drop**: Support for dragging files into the interface
- **Format Auto-detection**: Automatically detect file formats
- **Real-time Preview**: Update preview as columns are selected
- **Data Validation**: Show statistics and warnings for empty/invalid data
- **Column Reordering**: Allow reordering of selected columns

### 2. Model Configuration Tab

**Purpose**: Embedding model selection and clustering algorithm configuration

#### Layout
```
┌─ Embedding Model ───────────────────────────────────────────────┐
│ Model Source: ● Local Models ○ Custom Path                     │
│                                                                │
│ Available Models:                    Selected Model:           │
│ ┌─────────────────────────────────┐  ┌─────────────────────┐    │
│ │ all-MiniLM-L6-v2               │  │ all-MiniLM-L6-v2   │    │
│ │ all-mpnet-base-v2              │  │ Size: 90MB          │    │
│ │ distilbert-base-nli-mean-token │  │ Dimensions: 384     │    │
│ │ sentence-t5-base               │  │ Language: Multi     │    │
│ └─────────────────────────────────┘  └─────────────────────┘    │
│                                                                │
│ [Refresh Models] [Test Model]                                  │
└─────────────────────────────────────────────────────────────────┘

┌─ Clustering Algorithm ──────────────────────────────────────────┐
│ Algorithm: [HDBSCAN ▼]                                         │
│                                                                │
│ Parameters:                                                    │
│ Min Cluster Size:     [15        ] ⓘ                          │
│ Min Samples:          [10        ] ⓘ                          │
│ Cluster Selection:    [eom ▼     ] ⓘ                          │
│ Metric:              [euclidean ▼] ⓘ                          │
│                                                                │
│ [Advanced Parameters...] [Reset to Defaults]                  │
└─────────────────────────────────────────────────────────────────┘

┌─ Dimensionality Reduction (UMAP) ───────────────────────────────┐
│ ☑ Use UMAP for dimensionality reduction                        │
│                                                                │
│ For Topic Modeling:        For Visualization:                 │
│ N Components: [5    ]      N Components: [2 ▼] (2D/3D)       │
│ N Neighbors:  [15   ] ⓘ    N Neighbors:  [15   ] ⓘ           │
│ Min Distance: [0.1  ] ⓘ    Min Distance: [0.1  ] ⓘ           │
│ Metric:       [cosine▼] ⓘ  Metric:       [cosine▼] ⓘ         │
│                                                                │
│ [Copy Parameters →] [Advanced Settings...]                     │
└─────────────────────────────────────────────────────────────────┘

[← Previous] [Generate/Load Embeddings] [Next: Topic Modeling →]
```

#### Key Features
- **Model Information**: Display model size, dimensions, and capabilities
- **Parameter Tooltips**: Contextual help for all parameters
- **Parameter Validation**: Real-time validation with helpful error messages
- **Preset Configurations**: Quick access to common parameter combinations
- **Advanced Panels**: Collapsible sections for advanced options

### 3. Topic Modeling Tab

**Purpose**: Main topic modeling configuration and execution

#### Layout
```
┌─ Embedding Status ──────────────────────────────────────────────┐
│ ● Embeddings Ready: title_description_all-MiniLM-L6-v2.pkl     │
│ Cache Size: 156MB | Generated: 2024-01-15 14:23:45            │
│ [View Cache Info] [Regenerate] [Clear Cache]                   │
└─────────────────────────────────────────────────────────────────┘

┌─ Vectorization ─────────────────────────────────────────────────┐
│ Vectorizer: [CountVectorizer ▼]                               │
│                                                                │
│ ☑ Remove Stop Words     ☑ Use N-grams: (1,2)                 │
│ Min DF: [5     ] ⓘ     Max DF: [0.95  ] ⓘ                    │
│ Max Features: [1000 ] ⓘ                                       │
│                                                                │
│ [Advanced Vectorizer Settings...]                             │
└─────────────────────────────────────────────────────────────────┘

┌─ Representation Models ─────────────────────────────────────────┐
│ ☑ KeyBERT            ☑ Maximal Marginal Relevance             │
│ ☐ Part of Speech     ☐ Zero-Shot Classification              │
│                                                                │
│ Representation per Topic: [5 ▼] words                         │
└─────────────────────────────────────────────────────────────────┘

┌─ Guided Topic Modeling ─────────────────────────────────────────┐
│ ☐ Enable Guided Modeling                                      │
│                                                                │
│ Seed Topics:                                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Topic 1: [machine, learning, model, algorithm]             │ │
│ │ Topic 2: [python, code, programming, development]          │ │
│ │ [Add Topic] [Remove Selected]                              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─ Training Progress ─────────────────────────────────────────────┐
│ Status: Ready to train                                         │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ [████████████████████████████████████████████░░░░░] 90%     │ │
│ │ Current Step: Clustering documents...                      │ │
│ │ Elapsed: 00:02:34 | Estimated Remaining: 00:00:18         │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ [Start Training] [Cancel] [Save Model]                        │
└─────────────────────────────────────────────────────────────────┘

[← Previous] [Next: Visualization →]
```

#### Key Features
- **Status Indicators**: Clear visual feedback on process status
- **Progress Tracking**: Detailed progress bars with time estimates
- **Guided Topics**: Easy-to-use interface for adding seed topics
- **Model Persistence**: Save and load trained models
- **Real-time Validation**: Parameter validation before training

### 4. Visualization Tab

**Purpose**: Interactive visualization of topics and results

#### Layout
```
┌─ Topic Overview ────────────────────────────────────────────────┐
│ Total Topics: 42 | Total Documents: 15,432 | Outliers: 1,234   │
│                                                                │
│ ┌─ Topic Distribution ──────────────────┐ ┌─ Topic Info ─────┐ │
│ │                                      │ │ Selected Topic: 5│ │
│ │    [Interactive Bar Chart]           │ │ Documents: 892   │ │
│ │                                      │ │ Keywords:        │ │
│ │                                      │ │ • machine (0.23) │ │
│ │                                      │ │ • learning (0.19)│ │
│ │                                      │ │ • model (0.15)   │ │
│ │                                      │ │ • algorithm(0.12)│ │
│ └──────────────────────────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─ Document Visualization ────────────────────────────────────────┐
│ View: ● 2D Plot ○ 3D Plot    Color by: [Topic ▼]              │
│                                                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                                                             │ │
│ │        [Interactive UMAP Scatter Plot]                     │ │
│ │                                                             │ │
│ │   • Hover for document preview                             │ │
│ │   • Click to select topic                                  │ │
│ │   • Zoom/pan controls                                      │ │
│ │                                                             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ Selected Documents: 15 | [Export Selection] [Clear Selection] │
└─────────────────────────────────────────────────────────────────┘

┌─ Topic Deep Dive ──────────────────────────────────────────────┐
│ Selected Topic: [5 ▼] Machine Learning                        │
│                                                                │
│ ┌─ Word Cloud ─────────┐ ┌─ Top N-grams ─────────────────────┐ │
│ │                     │ │ N-gram Size: [1 ▼] [2] [3]       │ │
│ │  [Word Cloud Plot]  │ │                                   │ │
│ │                     │ │ 1. machine learning (245)         │ │
│ │                     │ │ 2. neural network (198)          │ │
│ └─────────────────────┘ │ 3. deep learning (187)           │ │
│                        │ 4. data science (156)            │ │
│ Representative Docs:    │ 5. artificial intelligence (142) │ │
│ 1. "Introduction to... │ ...                               │ │
│ 2. "Machine Learning.. │ [Export N-grams]                 │ │
│ 3. "Deep Neural Net... │                                   │ │
│ [View All Documents]   └───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

[← Previous] [Regenerate Plots] [Next: Export →]
```

#### Key Features
- **Interactive Plots**: Zoom, pan, hover, and click interactions
- **Real-time Updates**: Plots update as parameters change
- **Multiple Views**: 2D/3D scatter plots, bar charts, word clouds
- **Topic Selection**: Click on plots to explore specific topics
- **Export Options**: Save individual plots or complete visualizations

### 5. Hyperparameter Optimization Tab

**Purpose**: Automated parameter tuning and performance evaluation

#### Layout
```
┌─ Optimization Configuration ────────────────────────────────────┐
│ Optimization Method: [Grid Search ▼]                          │
│ Cross Validation: [3 ▼] folds | Parallel Jobs: [4 ▼]         │
│                                                                │
│ ┌─ Parameter Search Space ─────────────────────────────────────┐ │
│ │ HDBSCAN:                     UMAP:                         │ │
│ │ Min Cluster Size: [10,15,20] N Components: [3,5,7]        │ │
│ │ Min Samples: [5,10]          N Neighbors: [10,15,20]      │ │
│ │                              Min Distance: [0.1,0.2,0.3]  │ │
│ │                                                            │ │
│ │ [Add Parameter] [Remove] [Load Preset] [Save Preset]      │ │
│ └────────────────────────────────────────────────────────────┘ │
│                                                                │
│ Evaluation Metrics:                                            │
│ ☑ Silhouette Score    ☑ Calinski-Harabasz Index              │
│ ☑ Davies-Bouldin Score ☐ Custom Metric                       │
└─────────────────────────────────────────────────────────────────┘

┌─ Optimization Progress ─────────────────────────────────────────┐
│ Status: Running optimization... (Trial 23/72)                 │
│                                                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Overall Progress: [████████████░░░░░░░░░] 32%              │ │
│ │ Current Trial:    [██████████████████░░] 90%               │ │
│ │                                                             │ │
│ │ Best Score So Far: 0.847 (Silhouette)                     │ │
│ │ Best Parameters: {min_cluster_size: 15, n_components: 5}   │ │
│ │                                                             │ │
│ │ Elapsed: 00:15:32 | Estimated Total: 00:48:15             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ [Start Optimization] [Pause] [Stop] [Export Current Results]  │
└─────────────────────────────────────────────────────────────────┘

┌─ Results Visualization ─────────────────────────────────────────┐
│ ┌─ Parameter Importance ──────┐ ┌─ Score Distribution ─────────┐ │
│ │                            │ │                             │ │
│ │  [Horizontal Bar Chart]    │ │    [Histogram/Box Plot]     │ │
│ │                            │ │                             │ │
│ │  Shows which parameters    │ │  Distribution of scores     │ │
│ │  have the most impact      │ │  across all trials          │ │
│ └────────────────────────────┘ └─────────────────────────────┘ │
│                                                                │
│ ┌─ Parameter Correlation Heatmap ──────────────────────────────┐ │
│ │                                                             │ │
│ │           [Interactive Heatmap]                            │ │
│ │                                                             │ │
│ │  Shows correlation between parameters and performance       │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─ Best Results ──────────────────────────────────────────────────┐
│ Rank | Parameters                              | Score          │
│ 1    | min_cluster_size: 15, n_components: 5  | 0.847         │
│ 2    | min_cluster_size: 20, n_components: 3  | 0.832         │
│ 3    | min_cluster_size: 10, n_components: 7  | 0.821         │
│                                                                │
│ [Apply Best Parameters] [Export All Results] [View Details]   │
└─────────────────────────────────────────────────────────────────┘

[← Previous] [Run with Best Parameters] [Next: Export →]
```

#### Key Features
- **Flexible Search Space**: Easy parameter range definition
- **Real-time Progress**: Live updates during optimization
- **Multiple Metrics**: Support for various evaluation metrics
- **Visualization**: Parameter importance and correlation analysis
- **Results Export**: Save optimization results for later analysis

### 6. Export Tab

**Purpose**: Export results, visualizations, and model artifacts

#### Layout
```
┌─ Export Data ───────────────────────────────────────────────────┐
│ ☑ Topic Assignments     ☑ Topic Information                    │
│ ☑ Document Probabilities ☑ Representative Documents           │
│ ☑ Topic Keywords        ☐ Raw Embeddings                      │
│                                                                │
│ Export Format: [Excel (.xlsx) ▼]                              │
│ Output File: [Browse...] [C:\Users\...\results.xlsx]          │
│                                                                │
│ Excel Options:                                                 │
│ ☑ Create separate sheets for each data type                   │
│ ☑ Include metadata and parameters                             │
│ ☑ Add conditional formatting                                  │
└─────────────────────────────────────────────────────────────────┘

┌─ Export Visualizations ─────────────────────────────────────────┐
│ Available Plots:                                               │
│ ☑ Topic Distribution (Bar Chart)                              │
│ ☑ Document Scatter Plot (2D)                                  │
│ ☑ Document Scatter Plot (3D)                                  │
│ ☑ Topic Word Clouds (All Topics)                              │
│ ☑ N-gram Analysis Charts                                      │
│ ☐ Hyperparameter Optimization Results                         │
│                                                                │
│ Format: [PNG ▼] | Resolution: [300 DPI ▼] | Size: [Large ▼]  │
│ Output Folder: [Browse...] [C:\Users\...\visualizations\]     │
│                                                                │
│ ☑ Create HTML dashboard with interactive plots                │
└─────────────────────────────────────────────────────────────────┘

┌─ Export Model ──────────────────────────────────────────────────┐
│ ☑ Trained BERTopic Model                                       │
│ ☑ Model Configuration (JSON)                                  │
│ ☑ Training Log and Metrics                                    │
│ ☐ Cached Embeddings                                           │
│                                                                │
│ Model Name: [bertopic_model_20240115] (.pkl)                  │
│ Output Path: [Browse...] [C:\Users\...\models\]               │
│                                                                │
│ ☑ Include documentation and usage instructions                │
└─────────────────────────────────────────────────────────────────┘

┌─ Export Progress ───────────────────────────────────────────────┐
│ Status: Ready to export                                        │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Export Progress: [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0%     │ │
│ │ Current Task: Preparing export...                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                │
│ Estimated Size: 45.2 MB                                       │
│ [Start Export] [Open Output Folder] [View Export Log]         │
└─────────────────────────────────────────────────────────────────┘

[← Previous] [New Analysis] [Exit Application]
```

#### Key Features
- **Flexible Selection**: Choose specific data types to export
- **Multiple Formats**: Support for various output formats
- **Batch Export**: Export all visualizations at once
- **Size Estimation**: Preview export size before starting
- **Progress Tracking**: Real-time export progress

## Common UI Elements

### Design System

#### Colors (CustomTkinter Theme)
- **Primary**: #1f538d (CustomTkinter Blue)
- **Secondary**: #144870 (Dark Blue)
- **Success**: #2d8659 (Green)
- **Warning**: #d67e00 (Orange)
- **Error**: #c53030 (Red)
- **Background**: #212121 / #f0f0f0 (Dark/Light Mode)
- **Text**: #ffffff / #000000 (Dark/Light Mode)
- **Widget Background**: #2b2b2b / #ffffff (Dark/Light Mode)

#### Typography (CustomTkinter Fonts)
- **Headers**: Default system font, 16-20pt, bold
- **Body**: Default system font, 12pt
- **Code/Data**: Monospace font, 11pt
- **Tooltips**: Default system font, 10pt

#### Icons
- **File Operations**: Document, folder, download icons
- **Actions**: Play, pause, stop, refresh icons
- **Status**: Check, warning, error, info icons
- **Navigation**: Arrow, chevron icons

### Interactive Elements

#### Buttons (CustomTkinter)
- **CTkButton**: Modern rounded buttons with hover effects
- **CTkSegmentedButton**: For grouped button selections
- **CTkCheckBox**: Modern checkboxes with smooth animations
- **CTkSwitch**: Toggle switches for on/off states

#### Input Fields (CustomTkinter)
- **CTkEntry**: Clean text inputs with rounded corners
- **CTkComboBox**: Modern dropdown with search capability
- **CTkOptionMenu**: Simple dropdown selections
- **CTkSlider**: Smooth sliders for numerical ranges
- **CTkTextbox**: Multi-line text areas with scrolling

#### Progress Indicators (CustomTkinter)
- **CTkProgressBar**: Smooth progress bars with animations
- **Custom Progress Dialogs**: Modal dialogs with cancellation
- **Status Labels**: Text-based status indicators

### Responsive Behavior

#### Window Resizing
- **Minimum Size**: 1200x800 pixels
- **Responsive Tabs**: Content adapts to available space
- **Collapsible Panels**: Advanced options can be hidden
- **Scrollable Areas**: Vertical scroll for long content

#### High DPI Support
- **Vector Icons**: Scale cleanly on high-DPI displays
- **Font Scaling**: Respect system font scaling settings
- **Layout Adaptation**: Maintain proportions across DPI settings

## Accessibility Features

### Keyboard Navigation
- **Tab Order**: Logical tab sequence through all controls
- **Keyboard Shortcuts**: Common shortcuts (Ctrl+O, Ctrl+S, etc.)
- **Focus Indicators**: Clear visual focus states
- **Screen Reader**: ARIA labels and descriptions

### Visual Accessibility
- **High Contrast**: Support for high contrast themes
- **Color Blindness**: Don't rely solely on color for information
- **Font Sizes**: Respect system font size preferences
- **Clear Labels**: Descriptive labels for all interactive elements

## Error States and Feedback

### Error Messages
- **Contextual**: Show errors near the relevant input
- **Actionable**: Provide clear steps to resolve issues
- **Non-blocking**: Don't prevent users from continuing when possible
- **Persistent**: Keep error messages visible until resolved

### Loading States
- **Progress Feedback**: Show progress for all long operations
- **Cancellation**: Allow users to cancel long-running tasks
- **Status Updates**: Provide meaningful status messages
- **Error Recovery**: Graceful handling of failed operations

### Success Feedback
- **Confirmation**: Clear indication when operations complete
- **Next Steps**: Guide users to logical next actions
- **Results Summary**: Show key metrics and outcomes
- **Quick Access**: Easy access to results and exports 