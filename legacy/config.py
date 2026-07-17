"""
Configuration file for narrative divergence analysis experiments.
"""

# Model configurations
MODELS = {
    "semantic": "all-MiniLM-L6-v2",      # For semantic similarity
    "style": "all-mpnet-base-v2",        # For stylistic analysis  
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"  # For multilingual content
}

# Similarity thresholds for different analysis types
SIMILARITY_THRESHOLDS = {
    "strict": 0.8,    # High similarity - very similar content
    "moderate": 0.7,  # Medium similarity - related content
    "loose": 0.6      # Low similarity - broadly related content
}

# Narrative frame taxonomy with detailed definitions
NARRATIVE_FRAMES = {
    "persecution": {
        "definition": "Presents political actor as unfairly targeted or victimized by powerful forces",
        "keywords": ["witch hunt", "deep state", "targeted", "persecuted", "unfair"],
        "examples": [
            "This is a witch hunt by the deep state elites",
            "They're persecuting him for standing up to the establishment"
        ]
    },
    "corruption": {
        "definition": "Emphasizes unethical, illegal, or morally questionable behavior",
        "keywords": ["corrupt", "illegal", "bribes", "unethical", "criminal"],
        "examples": [
            "Hunter's laptop proves they've been selling access for years",
            "This is clear evidence of corruption at the highest levels"
        ]
    },
    "accountability": {
        "definition": "Portrays legal/political consequences as necessary for justice or democracy",
        "keywords": ["justice", "accountable", "rule of law", "consequences", "democracy"],
        "examples": [
            "No one is above the law. Justice must be served",
            "This is what accountability looks like in a democracy"
        ]
    },
    "irony_detachment": {
        "definition": "Uses sarcasm, cynicism, or emotional distancing as a critique style",
        "keywords": ["oh great", "of course", "totally", "sure", "obviously"],
        "examples": [
            "Oh great, another indictment. Totally going to fix everything",
            "Sure, this will definitely make things better"
        ]
    },
    "heroism": {
        "definition": "Frames political figure as savior, patriot, or self-sacrificing fighter",
        "keywords": ["hero", "patriot", "fighting for us", "savior", "champion"],
        "examples": [
            "He's fighting the system for us. A true hero",
            "Trump is the only one willing to take on the establishment"
        ]
    },
    "civic_critique": {
        "definition": "Highlights structural or democratic failures of systems or institutions",
        "keywords": ["system broken", "reform needed", "institutional failure", "democratic crisis"],
        "examples": [
            "This trial is just a performance. The system needs reform",
            "Our democratic institutions are failing us"
        ]
    },
    "moral_decay": {
        "definition": "Depicts societal collapse in values, ethics, or collective morality",
        "keywords": ["country lost", "no values", "moral collapse", "society broken"],
        "examples": [
            "This country is lost. No one has honor anymore",
            "We've completely lost our moral compass as a society"
        ]
    },
    "media_manipulation": {
        "definition": "Frames mainstream media as biased, deceptive, or agenda-driven",
        "keywords": ["fake news", "media bias", "twisted", "propaganda", "lies"],
        "examples": [
            "Fake news again. They twist everything to fit their narrative",
            "The media is completely biased and can't be trusted"
        ]
    },
    "strategic_pragmatism": {
        "definition": "Focuses on political calculation or practical consequences",
        "keywords": ["damage control", "lesser evil", "strategic vote", "practical"],
        "examples": [
            "Voting for Biden is just damage control at this point",
            "We need to be strategic about this election"
        ]
    },
    "cultural_identity": {
        "definition": "Invokes national, religious, or group identity to frame meaning",
        "keywords": ["real Americans", "our values", "patriotic", "traditional"],
        "examples": [
            "Real Americans know who the enemy is",
            "This goes against everything we stand for as a nation"
        ]
    }
}

# Platform-specific configurations
PLATFORM_CONFIGS = {
    "truth_social": {
        "name": "Truth Social",
        "color": "#FF6B6B",
        "expected_frames": ["persecution", "heroism", "media_manipulation", "cultural_identity"],
        "api_limits": {"posts_per_request": 100, "requests_per_hour": 300}
    },
    "bluesky": {
        "name": "Bluesky", 
        "color": "#4DABF7",
        "expected_frames": ["irony_detachment", "civic_critique", "strategic_pragmatism"],
        "api_limits": {"posts_per_request": 100, "requests_per_hour": 300}
    },
    "mastodon": {
        "name": "Mastodon",
        "color": "#51CF66", 
        "expected_frames": ["accountability", "civic_critique", "moral_decay"],
        "api_limits": {"posts_per_request": 100, "requests_per_hour": 300}
    }
}

# Analysis configurations
ANALYSIS_CONFIGS = {
    "quick_test": {
        "similarity_threshold": 0.7,
        "model": "all-MiniLM-L6-v2",
        "max_posts_per_platform": 100,
        "min_component_size": 2,
        "classification_method": "embedding_similarity"
    },
    "full_analysis": {
        "similarity_threshold": 0.65,
        "model": "all-mpnet-base-v2", 
        "max_posts_per_platform": 10000,
        "min_component_size": 3,
        "classification_method": "llm_prompt"
    },
    "high_precision": {
        "similarity_threshold": 0.8,
        "model": "all-mpnet-base-v2",
        "max_posts_per_platform": 5000,
        "min_component_size": 2,
        "classification_method": "hybrid"
    }
}

# Visualization settings
VIZ_CONFIGS = {
    "figure_size": (15, 12),
    "dpi": 300,
    "color_palette": "Set2",
    "font_size": 12,
    "save_formats": ["png", "pdf"],
    "plot_types": [
        "divergence_distribution",
        "platform_comparison", 
        "component_analysis",
        "frame_heatmap"
    ]
}

# LLM prompt templates for frame classification
FRAME_CLASSIFICATION_PROMPTS = {
    "system_prompt": """You are an expert in political communication and framing analysis. Your task is to identify the narrative frame used in social media posts about political events. 

A narrative frame is the interpretive lens through which a political event is presented, emphasizing certain aspects while de-emphasizing others.

You will classify posts into one of these frames: {frame_list}

For each frame, consider:
- The underlying interpretation of events
- The emotional tone and stance
- The implicit values being invoked
- The suggested cause-effect relationships""",
    
    "user_prompt": """Analyze this social media post and identify its primary narrative frame:

Post: "{post_text}"

Available frames:
{frame_definitions}

Respond with:
1. Primary frame: [frame_name]
2. Confidence: [high/medium/low] 
3. Brief explanation: [1-2 sentences explaining why this frame fits]

If the post contains multiple frames, identify the most prominent one.""",
    
    "few_shot_examples": [
        {
            "post": "This indictment is a complete witch hunt by the deep state trying to stop Trump.",
            "frame": "persecution",
            "explanation": "Frames the legal action as unfair targeting by powerful hidden forces."
        },
        {
            "post": "Finally, some accountability. No one should be above the law.",
            "frame": "accountability", 
            "explanation": "Emphasizes justice and equal treatment under democratic institutions."
        },
        {
            "post": "Oh great, another political circus. This helps absolutely nobody.",
            "frame": "irony_detachment",
            "explanation": "Uses sarcasm to express cynicism about the entire political process."
        }
    ]
}

# Evaluation metrics and thresholds
EVALUATION_CONFIGS = {
    "divergence_thresholds": {
        "low": 0.1,
        "medium": 0.3, 
        "high": 0.5
    },
    "significance_tests": {
        "js_divergence": {"min_sample_size": 10, "alpha": 0.05},
        "centroid_distance": {"min_sample_size": 5, "alpha": 0.05}
    },
    "quality_filters": {
        "min_post_length": 10,  # minimum characters
        "max_post_length": 2000,  # maximum characters
        "min_platforms_per_component": 2,
        "min_posts_per_platform": 1
    }
}

# File paths and directory structure
FILE_CONFIGS = {
    "data_dir": "data",
    "results_dir": "results", 
    "figures_dir": "figures",
    "logs_dir": "logs",
    "temp_dir": "temp",
    "required_subdirs": ["raw", "processed", "analysis", "exports"]
}
