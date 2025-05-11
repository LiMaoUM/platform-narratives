# Platform Narratives Analysis

This repository contains code for analyzing social media platform narratives using text analysis and graph-based methods. The project provides a modular structure with components that work together to identify significant content, connect similar posts across platforms, analyze narrative structures, and classify content using LLMs.

## Overview

This project provides tools to:

1. Identify significant content using FastLexRank algorithm
2. Build cross-platform similarity graphs to connect related content
3. Analyze narrative trees in social media conversations
4. Classify narratives using LLM-based analysis
5. Analyze reply chains to understand discourse dynamics

## Project Structure

```
.
├── README.md                     # Project documentation
├── requirements.txt              # Dependencies
├── data/                         # Data directory (you need to add your data here)
├── example_narrative_analysis.py # Example script demonstrating the basic workflow
├── example_narrative_workflow.py # Complete workflow with all components
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── analysis_workflow.ipynb   # Analysis workflow notebook
│   └── matching.ipynb            # Original analysis notebook
└── src/                          # Source code
    ├── __init__.py               # Package initialization
    ├── text_processing.py        # Text cleaning and processing functions
    ├── lexrank.py                # FastLexRank implementation for significant content
    ├── similarity_graph.py       # Cross-platform similarity graph builder
    ├── narrative_trees.py        # Narrative tree analysis with conversation structure
    ├── narrative_classification.py # LLM-based narrative classification
    ├── reply_chain_analysis.py   # Analysis of reply chains and discourse dynamics
    ├── ranking.py                # Content ranking utilities
    ├── graph_analysis.py         # Graph building and analysis functions
    └── utils.py                  # Utility functions
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

The project includes two example scripts that demonstrate the workflow:

### Basic Workflow

`example_narrative_analysis.py` shows the basic components:

```python
# Import the components
from src.lexrank import FastLexRank
from src.similarity_graph import SimilarityGraphBuilder
from src.narrative_trees import NarrativeTreeAnalyzer

# 1. Identify significant content
lexrank = FastLexRank()
ranked_posts = lexrank.rank(posts_df)

# 2. Build cross-platform similarity graph
graph_builder = SimilarityGraphBuilder()
platform_data = {'platform1': posts1, 'platform2': posts2, 'platform3': posts3}
tripartite_graph = graph_builder.build_tripartite_graph(platform_data)

# 3. Analyze narrative trees
tree_analyzer = NarrativeTreeAnalyzer(llm_analyzer=your_llm_function)
analysis_results = tree_analyzer.analyze_tree_narratives(graph, root_ids, id_to_post_map)
```

### Complete Workflow with LLM Integration

`example_narrative_workflow.py` demonstrates the full pipeline including narrative classification and reply chain analysis:

```python
# Import all components
from src.lexrank import FastLexRank
from src.similarity_graph import SimilarityGraphBuilder
from src.narrative_classification import NarrativeClassifier
from src.reply_chain_analysis import ReplyChainAnalyzer
from src.narrative_trees import NarrativeTreeAnalyzer

# Define your LLM pipeline
def your_llm_pipeline(messages, max_new_tokens=200):
    # Implement your LLM integration here
    pass

# Initialize components
classifier = NarrativeClassifier(llm_pipeline=your_llm_pipeline)
reply_analyzer = ReplyChainAnalyzer(llm_pipeline=your_llm_pipeline)

# Classify narratives in posts
classification_results = classifier.classify_component_posts(component_posts)

# Analyze reply chains
reply_analysis = reply_analyzer.analyze_reply_chain(root_post, replies)
```

You can run either example script to see the workflows in action:

```bash
python example_narrative_analysis.py  # Basic workflow
python example_narrative_workflow.py  # Complete workflow with LLM integration
```

Alternatively, you can follow the analysis workflow in the notebooks directory.

## Project Components

### 1. FastLexRank

The `FastLexRank` component identifies significant content within a collection of posts using an efficient implementation of the LexRank algorithm. It uses sentence embeddings to calculate centrality scores for posts, helping to identify the most representative or important content.

**Key features:**
- Text cleaning to remove mentions, hashtags, and URLs
- Sentence embedding generation using transformer models
- Centrality scoring based on embedding similarity to the centroid

### 2. Similarity Graph Analysis

The `SimilarityGraphBuilder` component creates connections between posts across different platforms based on semantic similarity. It enables cross-platform narrative analysis by building a tripartite graph structure.

**Key features:**
- Cross-platform post similarity calculation
- Tripartite graph construction
- Threshold-based edge creation

### 3. Narrative Tree Analysis

The `NarrativeTreeAnalyzer` component extracts and analyzes conversation trees from social media interactions. It provides methods for calculating tree statistics and analyzing discourse patterns using external analyzers, including LLM-based analysis.

**Key features:**
- Tree structure extraction from conversation graphs
- Tree statistics calculation (depth, breadth, node count)
- Integration with LLM-based discourse analysis
- Narrative pattern summarization

### 4. Narrative Classification

The `NarrativeClassifier` component uses LLM-based analysis to classify social media posts based on narrative frames, subjects, stances, and topic focus.

**Key features:**
- LLM integration for narrative analysis
- Classification of posts by narrative elements
- Batch processing of posts from different platforms
- JSON-formatted classification results

### 5. Reply Chain Analysis

The `ReplyChainAnalyzer` component analyzes how replies relate to the original post's narrative, determining whether they reinforce, challenge, or shift the narrative.

**Key features:**
- Analysis of discourse dynamics in reply chains
- Classification of replies into reinforcement, challenge, or shift categories
- Batch processing of multiple reply chains
- Summary statistics on narrative dynamics

## Data

This project was originally developed using social media data. To use this project, you'll need to:

1. Place your data files in the `data/` directory
2. Format your data to include post content and relationship information

### Data Availability

We are currently working on publishing the datasets used in this research. Due to non-clear policy guidelines, the data is not publicly available at this time. For data requests, please reach out to Mao Li (maolee@umich.edu).

## Extending the Project

### Adding New Platforms

To add support for a new platform, format the data to include at minimum:
- A unique post ID
- The post content text
- Parent/reply relationships (if analyzing conversation trees)

### Custom LLM Integration

To use a custom LLM for narrative analysis:

1. Create a function that takes messages in the OpenAI API format and returns generated text
2. Pass this function to the appropriate component constructor

Example:

```python
def my_llm_pipeline(messages, max_new_tokens=200):
    # Call your LLM API here
    return [{
        "generated_text": [
            {"role": "assistant", "content": "Your LLM response here"}
        ]
    }]

classifier = NarrativeClassifier(llm_pipeline=my_llm_pipeline)
reply_analyzer = ReplyChainAnalyzer(llm_pipeline=my_llm_pipeline)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
[Your citation information here]
```