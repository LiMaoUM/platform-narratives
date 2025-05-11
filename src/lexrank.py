import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

class FastLexRank:
    """
    Implementation of a fast version of LexRank algorithm for extractive summarization.
    This uses sentence embeddings to identify the most central/important posts in a dataset.
    """
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the FastLexRank model.
        
        Args:
            model_name: The name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
    
    def clean_text(self, text_series):
        """
        Clean text by removing mentions, hashtags, and URLs.
        
        Args:
            text_series: Pandas Series containing text to clean
            
        Returns:
            Cleaned text series
        """
        # Remove @mentions
        cleaned = text_series.str.replace(r'@\w+', '', regex=True)
        # Remove hashtags
        cleaned = cleaned.str.replace(r'#\w+', '', regex=True)
        # Remove URLs
        cleaned = cleaned.str.replace(r'http\S+', '', regex=True)
        return cleaned
    
    def rank(self, df, text_column="post", show_progress=True):
        """
        Apply FastLexRank algorithm to identify important posts.
        
        Args:
            df: DataFrame containing posts
            text_column: Column name containing the text to analyze
            show_progress: Whether to show progress bar during encoding
            
        Returns:
            DataFrame with added 'ap' column containing importance scores
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Get and clean the text
        posts = result_df[text_column]
        posts = self.clean_text(posts)
        
        # Generate embeddings
        embeddings = self.model.encode(posts, show_progress_bar=show_progress)
        
        # Sum in column to get the centroid
        z = embeddings.sum(axis=0)
        
        # Normalize the centroid
        z = z / np.sqrt((z**2).sum(axis=0))
        
        # Calculate affinity propagation scores (dot product with centroid)
        ap = np.dot(embeddings, z)
        
        # Add scores to dataframe
        result_df["ap"] = ap
        
        # Sort by score (descending)
        return result_df.sort_values(by="ap", ascending=False)