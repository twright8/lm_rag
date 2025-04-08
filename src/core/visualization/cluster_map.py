"""
Cluster map visualization module for Anti-Corruption RAG System.
Creates interactive topic clusters from document embeddings.
"""
import sys
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import re
import ast
import time
from typing import List, Dict, Tuple
import os
import streamlit as st
import requests
import tempfile
import io
import matplotlib.pyplot as plt
from fontTools import ttLib
import matplotlib.font_manager as fm

# Force GPU visibility (WSL-specific fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib/"
os.environ["NUMBA_CUDA_DRIVER"] = "/usr/lib/wsl/lib/libcuda.so.1"
# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import utils
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Load configuration
CONFIG_PATH = ROOT_DIR / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Check if clustering configuration exists, if not, create default
if "clustering" not in CONFIG:
    CONFIG["clustering"] = {
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine"
        },
        "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "prediction_data": True
        },
        "visualization": {
            "umap": {
                "n_neighbors": 15,
                "n_components": 2,
                "min_dist": 0.7,
                "spread": 1.5,
                "metric": "cosine"
            }
        },
        "vectorizer": {
            "stop_words": "english",
            "ngram_range": [1, 2]
        },
        "topics": {
            "nr_topics": "auto",
            "seed_topic_list": []
        },
        "visualization_type": "plotly",
        "static_datamapplot": {
            "darkmode": False,
            "cvd_safer": True,
            "color_label_text": True,
            "marker_size": 8,
            "font_family": "Oswald",
            "dpi": 300,
        }
    }
    logger.info("Creating default clustering configuration")

# Optional imports with error handling
try:
    # Import clustering and visualization libraries
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from bertopic.vectorizers import ClassTfidfTransformer
    import plotly.express as px
    import plotly.graph_objects as go

    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required dependencies for cluster mapping not available: {e}")
    DEPS_AVAILABLE = False


def check_dependencies():
    """
    Check if required dependencies are available.

    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    if not DEPS_AVAILABLE:
        logger.error("Required dependencies are not available. Please install the required packages.")
        logger.error("pip install cuml umap-learn hdbscan bertopic plotly scikit-learn")
        return False
    return True


def get_google_font(fontname):
    """
    Download and register a Google Font for matplotlib.

    Args:
        fontname: Name of the Google Font to load

    Returns:
        str: Font family name or None if failed
    """
    try:
        fontname = fontname.replace(" ", "+")
        if fontname == "Oswald":
            fontname = "Oswald:wght@700&display=swap"

        api_response = requests.get(f"https://fonts.googleapis.com/css?family={fontname}")
        font_urls = re.findall(r'(https?://[^\)]+)', str(api_response.content))

        if not font_urls:
            logger.warning(f"No font URLs found for {fontname}")
            return None

        for font_url in font_urls:
            font_data = requests.get(font_url)
            f = tempfile.NamedTemporaryFile(delete=False, suffix='.ttf')
            f.write(font_data.content)
            f.close()

            try:
                font = ttLib.TTFont(f.name)
                font_family_name = font['name'].getDebugName(1)
                fm.fontManager.addfont(f.name)
                logger.info(f"Added new font as {font_family_name}")
                return font_family_name
            except Exception as font_error:
                logger.error(f"Error loading font {fontname}: {font_error}")
                return None
    except Exception as e:
        logger.error(f"Error downloading Google Font {fontname}: {e}")
        return None


def extract_embeddings_from_qdrant(query_engine) -> Tuple[np.ndarray, List[str], List[Dict]]:
    """
    Extract embeddings and text data from Qdrant.

    Args:
        query_engine: QueryEngine instance with access to Qdrant

    Returns:
        tuple: (embeddings_array, texts, metadata)
    """
    start_time = time.time()
    logger.info("Extracting embeddings and text data from Qdrant")

    try:
        # Get the Qdrant client from the query engine
        from qdrant_client import QdrantClient

        # Connect to Qdrant
        client = QdrantClient(
            host=query_engine.qdrant_host,
            port=query_engine.qdrant_port
        )

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if query_engine.qdrant_collection not in collection_names:
            logger.error(f"Collection {query_engine.qdrant_collection} does not exist in Qdrant")
            return np.array([]), [], []

        # Get collection info to determine vector size
        collection_info = client.get_collection(query_engine.qdrant_collection)
        vector_size = collection_info.config.params.vectors.size

        logger.info(f"Collection found with vector size: {vector_size}")

        # Use scroll to get all points with vectors
        batch_size = 100
        offset = None
        all_embeddings = []
        all_texts = []
        all_metadata = []

        total_points = 0

        while True:
            result = client.scroll(
                collection_name=query_engine.qdrant_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )

            # Unpack returned data (depends on client version)
            points = result[0] if isinstance(result, tuple) else result

            if not points:
                break

            # Extract data from points
            for point in points:
                # Get embedding vector
                vector = point.vector

                # Ensure we have a valid vector
                if vector is None or len(vector) == 0:
                    logger.warning(f"Point {point.id} has no vector, skipping")
                    continue

                # Convert to numpy if needed
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()

                # Get text from payload
                text = point.payload.get('original_text', point.payload.get('text', ''))

                # Extract metadata
                metadata = {
                    'id': point.id,
                    'chunk_id': point.payload.get('chunk_id', point.id),
                    'document_id': point.payload.get('document_id', ''),
                    'file_name': point.payload.get('file_name', ''),
                    'page_num': point.payload.get('page_num', None)
                }

                # Append to our collections
                all_embeddings.append(vector)
                all_texts.append(text)
                all_metadata.append(metadata)

            # Update total count
            total_points += len(points)
            logger.info(f"Extracted {total_points} points so far")

            # Get next offset
            offset = result[1] if isinstance(result, tuple) else None

            # If no offset, we're done
            if offset is None:
                break

        # Convert to numpy array
        embeddings_array = np.array(all_embeddings, dtype=np.float32)

        elapsed_time = time.time() - start_time
        logger.info(f"Extracted {len(all_embeddings)} embeddings in {elapsed_time:.2f}s")
        logger.info(f"Embeddings shape: {embeddings_array.shape}")
        logger.info(f"Sample embedding: {embeddings_array[0][:5]}...")

        return embeddings_array, all_texts, all_metadata

    except Exception as e:
        logger.error(f"Error extracting embeddings from Qdrant: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return np.array([]), [], []


# Add this at the top of the cluster_map.py file, after the imports
# Global variables to store model instances for reuse
_topic_model_instance = None
_representation_model_instance = None
_last_embedding_shape = None


def create_topic_model(embeddings, texts, metadata_list, config=None):
    """
    Create and fit a BERTopic model on the embeddings and texts.
    Handles pre-filtering of short docs and ensures fresh components for stability.
    MODIFIED: Reuses the AphroditeRepresentation instance if available.

    Args:
        embeddings: numpy array of embeddings
        texts: list of text data
        metadata_list: list of metadata dictionaries corresponding to texts
        config: configuration dictionary

    Returns:
        tuple: (topic_model, topics_mapped_full, topic_info, docs_df)
               Returns (None, None, None, None) on failure.
    """
    global _representation_model_instance  # Access the global variable

    if not check_dependencies():
        return None, None, None, None

    logger.info("Creating topic model")
    start_time = time.time()

    try:
        # --- Pre-filtering ---
        MIN_TEXT_LENGTH = 10  # Minimum characters threshold
        original_indices = list(range(len(texts)))
        filtered_data = [
            (emb, txt, meta, idx) for emb, txt, meta, idx in zip(embeddings, texts, metadata_list, original_indices)
            if isinstance(txt, str) and len(txt.strip()) >= MIN_TEXT_LENGTH
        ]

        if not filtered_data:
            logger.error(
                f"No documents remaining after filtering texts shorter than {MIN_TEXT_LENGTH} chars. Cannot create topic model.")
            return None, None, None, None

        filtered_embeddings, filtered_texts, filtered_metadata, filtered_indices = zip(*filtered_data)
        filtered_embeddings = np.array(filtered_embeddings, dtype=np.float32)
        filtered_texts = list(filtered_texts)

        num_filtered = len(texts) - len(filtered_texts)
        if num_filtered > 0:
            logger.warning(f"Filtered out {num_filtered} documents shorter than {MIN_TEXT_LENGTH} characters.")
        logger.info(f"Processing {len(filtered_texts)} documents for BERTopic fitting.")
        # --- End Pre-filtering ---

        # Use provided config or get from global CONFIG
        if config is None:
            config = CONFIG.get("clustering", {})

        # --- Always create new component instances ---
        logger.info("Instantiating fresh model components for this run.")
        # Load UMAP parameters
        umap_params = config.get("umap", {})
        umap_model = UMAP(
            n_neighbors=int(umap_params.get("n_neighbors", 15)),
            n_components=int(umap_params.get("n_components", 5)),
            min_dist=float(umap_params.get("min_dist", 0.0)),
            metric=umap_params.get("metric", "cosine"),
            random_state=42
        )

        # Load HDBSCAN parameters
        hdbscan_params = config.get("hdbscan", {})
        hdbscan_model = HDBSCAN(
            min_cluster_size=int(hdbscan_params.get("min_cluster_size", 10)),
            min_samples=int(hdbscan_params.get("min_samples", 5)),
            prediction_data=True  # Keep true if needed elsewhere
        )

        # Load vectorizer parameters
        vectorizer_params = config.get("vectorizer", {})
        vectorizer_model = CountVectorizer(
            stop_words=vectorizer_params.get("stop_words", "english"),
            ngram_range=tuple(vectorizer_params.get("ngram_range", [1, 2]))
        )

        # Load ClassTfidfTransformer parameters
        ctfidf_model = ClassTfidfTransformer(
            reduce_frequent_words=True  # Keep this setting for now
        )

        # Load topic parameters
        topic_params = config.get("topics", {})
        nr_topics_config = topic_params.get("nr_topics", "auto")
        nr_topics = None if nr_topics_config == "auto" else int(nr_topics_config)
        seed_topic_list_config = topic_params.get("seed_topic_list", [])
        seed_topic_list = [[]] if not seed_topic_list_config else seed_topic_list_config

        # === START MODIFIED CODE: Check for existing representation model ===
        # Get representation model instance from session state if it exists
        if 'aphrodite_representation_instance' in st.session_state:
            representation_instance = st.session_state.aphrodite_representation_instance
            logger.info("Reusing existing AphroditeRepresentation instance from session state")
        elif _representation_model_instance is not None:
            representation_instance = _representation_model_instance
            logger.info("Reusing existing AphroditeRepresentation instance from global variable")
        else:
            # Create new representation model if not exists
            from src.core.visualization.aphrodite_representation import AphroditeRepresentation
            logger.info("Creating new AphroditeRepresentation instance for topic refinement")
            representation_instance = AphroditeRepresentation(
                model_name="Qwen/Qwen2.5-3B-Instruct",
                nr_docs=3,
                doc_length=150,
                direct_load=False,
                tokenizer="whitespace"
            )
            # Store for future reuse
            _representation_model_instance = representation_instance
            st.session_state.aphrodite_representation_instance = representation_instance
        # === END MODIFIED CODE ===

        representation_model_list = [
            KeyBERTInspired(),
            representation_instance
        ]
        # --- End Component Creation ---

        # --- Always create a new BERTopic instance ---
        logger.info("Creating new BERTopic model instance.")
        topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,  # Use the fresh instance
            ctfidf_model=ctfidf_model,  # Use the fresh instance
            representation_model=representation_model_list,
            embedding_model="intfloat/multilingual-e5-large-instruct",
            # Ensure this matches your embeddings if precomputed
            seed_topic_list=seed_topic_list,
            nr_topics=nr_topics,
            verbose=True
        )
        # --- End BERTopic Creation ---

        # Fit the model using FILTERED data
        logger.info(f"Fitting BERTopic model on {len(filtered_texts)} documents with precomputed embeddings.")
        # Ensure embeddings are float32
        filtered_embeddings_array = filtered_embeddings.astype(np.float32)

        # Log shapes for debugging just before fit
        logger.info(f"Embeddings shape for fit: {filtered_embeddings_array.shape}")
        logger.info(f"Number of texts for fit: {len(filtered_texts)}")

        # Perform fit_transform
        topics, probs = topic_model.fit_transform(filtered_texts, filtered_embeddings_array)

        logger.info("BERTopic fit_transform complete.")

        # --- Map results back to original document size ---
        logger.info("Mapping topic results back to original document indices.")
        # Initialize arrays for all original documents
        # Use a distinct value (e.g., -2) for docs filtered out before BERTopic fitting
        topics_mapped_full = np.full(len(texts), -2, dtype=int)
        probs_mapped_full = np.zeros(len(texts), dtype=float)

        # Populate results for documents that were actually processed
        for i, original_idx in enumerate(filtered_indices):
            topics_mapped_full[original_idx] = topics[i]
            # Handle probabilities (might be None if BERTopic doesn't calculate them)
            if probs is not None and i < len(probs):
                probs_mapped_full[original_idx] = probs[i]
            else:
                # Assign a default probability if needed (e.g., 1 for assigned topics, 0 for outliers)
                probs_mapped_full[original_idx] = 1.0 if topics[i] != -1 else 0.0

        # Get topic info from the fitted model
        logger.info("Getting topic information from BERTopic model.")
        topic_info = topic_model.get_topic_info()

        # FIXED: Clean up topic names (remove "Topic Name:" and underscores)
        # Clean up topic names to remove any "Topic Name:" prefix and underscores
        for i, row in topic_info.iterrows():
            if isinstance(row['Name'], str):
                # Remove "Topic Name:" and underscores
                name = row['Name'].replace("Topic Name:", "").replace("_", " ").strip()
                # If the name is too long or contains problematic patterns, simplify it
                if len(name) > 150 or "_type_" in name:
                    name = f"Topic {row['Topic']}"
                topic_info.at[i, 'Name'] = name

        # Create the final DataFrame aligned with the original texts and metadata
        logger.info("Creating final document information DataFrame.")
        docs_df = pd.DataFrame({
            'Document': texts,  # Original full list of texts
            'Topic': topics_mapped_full,  # Topics mapped to original indices (-2 for filtered)
            'Probability': probs_mapped_full,  # Probabilities mapped to original indices
        })

        # Add Topic Names/Labels
        topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()
        # Apply the cleaned topic names
        docs_df['Topic_Name'] = docs_df['Topic'].map(lambda x: topic_name_map.get(x, f"Topic {x}"))
        # Assign labels for special topics
        docs_df.loc[docs_df['Topic'] == -1, 'Topic_Name'] = "Outlier"
        docs_df.loc[docs_df['Topic'] == -2, 'Topic_Name'] = "Filtered Document (Short)"  # Label for filtered docs

        # Add original metadata back, ensuring alignment
        meta_df = pd.DataFrame(metadata_list)
        # Reset index if needed to ensure clean join, assuming order is preserved
        docs_df = pd.concat([docs_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1)

        elapsed_time = time.time() - start_time
        logger.info(f"Created topic model and mapped results in {elapsed_time:.2f}s")
        logger.info(f"Found {len(topic_info[topic_info['Topic'] != -1])} topics (excluding outliers).")

        # Return the fitted model, the mapped topics array, topic info, and the final comprehensive DataFrame
        return topic_model, topics_mapped_full, topic_info, docs_df

    except ValueError as ve:
        # Catch the specific dimension mismatch error if it still occurs
        if "Incompatible dimension" in str(ve):
            logger.error(f"Caught ValueError during topic model creation (likely vocab mismatch): {ve}")
            logger.error("This might indicate issues with vectorizer/TF-IDF internal consistency.")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None
        else:  # Re-raise other ValueErrors
            logger.error(f"Caught unexpected ValueError: {ve}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None, None, None
    except Exception as e:
        logger.error(f"Error creating topic model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None


def create_visualization_embeddings(embeddings, config=None):
    """
    Create 2D embeddings for visualization using UMAP.

    Args:
        embeddings: numpy array of embeddings
        config: configuration dictionary

    Returns:
        numpy array: 2D embeddings for visualization
    """
    if not check_dependencies():
        return None

    logger.info("Creating visualization embeddings")
    start_time = time.time()

    try:
        # Use provided config or get from global CONFIG
        if config is None:
            config = CONFIG.get("clustering", {}).get("visualization", {}).get("umap", {})

        # Load visualization UMAP parameters
        vis_n_neighbors = int(config.get("n_neighbors", 15))
        vis_n_components = int(config.get("n_components", 2))
        vis_min_dist = float(config.get("min_dist", 0.7))
        vis_spread = float(config.get("spread", 1.5))
        vis_metric = config.get("metric", "cosine")

        logger.info(
            f"Visualization UMAP parameters: n_neighbors={vis_n_neighbors}, n_components={vis_n_components}, min_dist={vis_min_dist}, spread={vis_spread}, metric={vis_metric}")

        # Create UMAP model for visualization
        vis_umap = UMAP(
            n_neighbors=vis_n_neighbors,
            n_components=vis_n_components,
            min_dist=vis_min_dist,
            spread=vis_spread,
            metric=vis_metric,
            random_state=42
        )

        # Fit and transform the embeddings
        logger.info(f"Fitting UMAP on {embeddings.shape[0]} embeddings for visualization")
        vis_embeddings = vis_umap.fit_transform(embeddings.astype(np.float32))

        elapsed_time = time.time() - start_time
        logger.info(f"Created visualization embeddings in {elapsed_time:.2f}s")
        logger.info(f"Visualization embeddings shape: {vis_embeddings.shape}")

        return vis_embeddings

    except Exception as e:
        logger.error(f"Error creating visualization embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# Add this function to cluster_map.py
def create_download_dataframe(docs_df, topic_info):
    """
    Create a DataFrame for download that includes topics and chunks.

    Args:
        docs_df: DataFrame with document info including topics
        topic_info: DataFrame with topic information

    Returns:
        DataFrame formatted for download
    """
    try:
        # Create a mapping of topic ID to representative words
        topic_words_map = {}
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            # FIXED: Clean the topic name to remove "Topic Name:" and underscores
            topic_name = row['Name']
            if isinstance(topic_name, str):
                topic_name = topic_name.replace("Topic Name:", "").replace("_", " ").strip()

            # For outliers and filtered docs, just use the name
            if topic_id < 0:
                topic_words_map[topic_id] = topic_name
            else:
                # Get topic representation (top words) if available
                if 'Representation' in row and row['Representation']:
                    # Format may vary, handle both string and list
                    if isinstance(row['Representation'], str):
                        try:
                            # Try to parse as list if it looks like one
                            if row['Representation'].startswith('[') and row['Representation'].endswith(']'):
                                words_list = ast.literal_eval(row['Representation'])
                                top_words = ', '.join(words_list[:5]) if isinstance(words_list, list) else row[
                                    'Representation']
                            else:
                                top_words = row['Representation']
                        except:
                            top_words = row['Representation']
                    else:
                        top_words = str(row['Representation'])

                    # Combine topic ID, name and top words
                    topic_words_map[topic_id] = f"{topic_name} ({top_words})"
                else:
                    topic_words_map[topic_id] = topic_name

        # Create a copy of the docs_df with necessary columns
        download_df = docs_df.copy()

        # Add topic representation column
        download_df['Topic_Representation'] = download_df['Topic'].map(topic_words_map)

        # Select and reorder columns for download
        columns_to_include = [
            'file_name', 'page_num', 'Topic', 'Topic_Name', 'Topic_Representation',
            'Probability', 'Document'  # Document contains the text
        ]

        # Only include columns that exist
        existing_columns = [col for col in columns_to_include if col in download_df.columns]

        # Add any missing but important columns
        if 'Document' not in existing_columns and 'text' in download_df.columns:
            download_df['Document'] = download_df['text']
            existing_columns.append('Document')

        # Create final dataframe with selected columns
        result_df = download_df[existing_columns].copy()

        # Rename columns for clarity
        column_renames = {
            'file_name': 'Document Name',
            'page_num': 'Page Number',
            'Topic': 'Topic ID',
            'Topic_Name': 'Topic Name',
            'Topic_Representation': 'Topic Keywords',
            'Probability': 'Topic Assignment Probability',
            'Document': 'Text Content'
        }

        # Apply renames for columns that exist
        rename_dict = {old: new for old, new in column_renames.items() if old in result_df.columns}
        result_df = result_df.rename(columns=rename_dict)

        return result_df

    except Exception as e:
        logger.error(f"Error creating download dataframe: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return a basic DataFrame if there's an error
        return pd.DataFrame({"Error": [f"Failed to create download data: {str(e)}"]})


def create_interactive_cluster_map(vis_embeddings, docs_df, texts, metadata, topic_info, include_outliers=False):
    """
    Create an interactive cluster map visualization.
    MODIFIED: Fixes hover_data ambiguity with data_frame columns.

    Args:
        vis_embeddings: 2D embeddings for visualization (filtered)
        docs_df: DataFrame with document info (filtered, must include 'Topic_Name', 'Topic')
        texts: list of document texts (filtered)
        metadata: list of document metadata (filtered)
        topic_info: DataFrame with topic info (original)
        include_outliers: (Not directly used as filtering is assumed done beforehand)

    Returns:
        plotly figure: Interactive cluster map
    """
    if not check_dependencies():
        return None

    logger.info("Creating interactive cluster map")
    if 'Topic_Name' not in docs_df.columns:
         logger.error("'Topic_Name' column missing in docs_df for create_interactive_cluster_map. Cannot create labels.")
         # Fallback or error
         # Option: Try to reconstruct from Topic ID and topic_info if possible
         if 'Topic' in docs_df.columns:
             logger.warning("Attempting to map Topic ID to Name as 'Topic_Name' is missing.")
             topic_dict = topic_info.set_index('Topic')['Name'].to_dict()
             docs_df['Topic_Name'] = docs_df['Topic'].map(lambda x: topic_dict.get(x, f"Topic {x}"))
             docs_df.loc[docs_df['Topic'] == -1, 'Topic_Name'] = "Outlier" # Explicitly handle outlier
         else:
             return None # Cannot proceed without topic labels

    try:
        # Create a DataFrame for visualization from the *already filtered* data
        vis_df = pd.DataFrame({
            'x': vis_embeddings[:, 0],
            'y': vis_embeddings[:, 1],
            'text': texts, # Use filtered texts
            'topic': docs_df['Topic'].values, # Use topic ID from filtered df
            'topic_label': docs_df['Topic_Name'].values # Directly use the pre-computed Topic_Name
        })

        # Add metadata (ensure metadata list corresponds to the filtered data)
        meta_df_filtered = pd.DataFrame(metadata) # Convert filtered metadata list to DF
        # Check length alignment
        if len(vis_df) != len(meta_df_filtered):
             logger.error(f"Length mismatch between vis_df ({len(vis_df)}) and filtered metadata ({len(meta_df_filtered)})")
             # Handle error or proceed with caution
        else:
             # Join or assign columns carefully - assuming order is preserved
             vis_df = pd.concat([vis_df.reset_index(drop=True), meta_df_filtered.reset_index(drop=True)], axis=1)


        # Create custom hover text using data available in vis_df
        vis_df['hover_text'] = vis_df.apply(
            lambda row: f"<b>Topic:</b> {row['topic_label']}<br>" +
                        f"<b>Document:</b> {row.get('file_name', 'Unknown')}<br>" +
                        f"<b>Page:</b> {row.get('page_num', 'N/A')}<br>" +
                        f"<b>Text:</b> {str(row['text'])[:100]}...", # Ensure text is string
            axis=1
        )

        # Create figure using 'topic_label' for color
        # FIXED: Removed duplicate references to file_name and page_num in hover_data
        fig = px.scatter(
            vis_df,
            x='x',
            y='y',
            color='topic_label', # Use the pre-defined label column
            hover_data={
                'x': False,
                'y': False,
                'topic_label': True,
                # Removed file_name and page_num from hover_data as they exist in dataframe
            },
            custom_data=['hover_text', 'text', 'topic'], # Removed file_name to avoid duplicates
            title="Document Cluster Map",
            labels={'topic_label': 'Topic'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7
        )

        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_white",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                title=dict(text=""),
                bordercolor="LightGrey",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(title="", showticklabels=False, showgrid=False, zeroline=False)
        )

        # Update hover template to use custom data
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b>",
            marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )

        # Create topic filter dropdown
        unique_topics = sorted(vis_df['topic_label'].unique())
        buttons = [
            dict(
                args=[{"visible": [True] * len(fig.data)}],
                label="All Topics",
                method="update"
            )
        ]

        # For each topic, create a button that shows only that topic
        topic_to_trace_idx = {}
        for i, trace in enumerate(fig.data):
            # Extract topic from the name (assumes format "topic=X")
            topic_name = trace.name.split("=")[-1] if trace.name and "=" in trace.name else ""
            if topic_name:
                topic_to_trace_idx[topic_name] = i

        for topic in unique_topics:
            visible = [False] * len(fig.data)
            if topic in topic_to_trace_idx:
                visible[topic_to_trace_idx[topic]] = True
                buttons.append(
                    dict(
                        args=[{"visible": visible}],
                        label=topic,
                        method="update"
                    )
                )

        # Add dropdown menu to layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.98,
                    y=1.02,
                    xanchor="right",
                    yanchor="bottom",
                    buttons=buttons,
                    showactive=True,
                    bgcolor="white",
                    bordercolor="lightgray",
                    font=dict(color="black"),
                )
            ]
        )

        logger.info("Interactive cluster map created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating interactive cluster map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_datamap_visualization(vis_embeddings, docs_df, texts, metadata, topic_info,
                                 include_outliers=False, config=None):
    """
    Create a DataMapPlot visualization from cluster data with improved tooltips and styling.

    Args:
        vis_embeddings: 2D embeddings for visualization
        docs_df: DataFrame with document info
        texts: list of document texts
        metadata: list of document metadata
        topic_info: DataFrame with topic info
        include_outliers: Whether to include outliers
        config: Configuration dict with DataMapPlot settings

    Returns:
        DataMapPlot interactive figure object
    """
    import datamapplot
    import pandas as pd

    logger.info("Creating DataMapPlot visualization")

    try:
        # Use provided config or get from global CONFIG
        if config is None:
            config = CONFIG.get("clustering", {}).get("datamapplot", {})

        # Filter out outliers if not included
        if not include_outliers:
            # Create mask for non-outlier documents
            non_outlier_mask = docs_df['Topic'] != -1
            filtered_embeddings = vis_embeddings[non_outlier_mask]
            filtered_texts = [texts[i] for i, include in enumerate(non_outlier_mask) if include]
            filtered_metadata = [metadata[i] for i, include in enumerate(non_outlier_mask) if include]
            filtered_docs_df = docs_df[non_outlier_mask].reset_index(drop=True)
        else:
            filtered_embeddings = vis_embeddings
            filtered_texts = texts
            filtered_metadata = metadata
            filtered_docs_df = docs_df

        # Create a mapping of topic ID to topic name from topic_info
        topic_name_map = dict(zip(topic_info['Topic'], topic_info['Name']))

        # Create improved hover text with better formatting and more information
        hover_text = []
        for i, (meta, text, topic) in enumerate(zip(filtered_metadata, filtered_texts, filtered_docs_df['Topic'])):
            # Get topic name
            topic_id = topic
            topic_name = topic_name_map.get(topic_id, f"Topic {topic_id}")
            if topic_id == -1:
                topic_name = "Noise/Outlier"

            # FIXED: Clean up topic names (remove "Topic Name:" and underscores)
            if isinstance(topic_name, str):
                topic_name = topic_name.replace("Topic Name:", "").replace("_", " ").strip()

            # Format document information
            doc_info = f"Document: {meta.get('file_name', 'Unknown')}"
            if meta.get('page_num') is not None:
                doc_info += f" (Page {meta.get('page_num')})"

            # Format text snippet (truncate to 200 chars)
            text_snippet = text[:200] + ("..." if len(text) > 200 else "")

            # Combine all information into a well-formatted hover text
            hover_info = f"{doc_info}\nTopic: {topic_name}\nText: {text_snippet}"

            hover_text.append(hover_info)

        # Create a DataFrame with extra point data for improved tooltips
        extra_point_data = pd.DataFrame({
            'topic_id': filtered_docs_df['Topic'].values,
            'topic_name': [topic_name_map.get(t, f"Topic {t}") if t != -1 else "Noise/Outlier"
                           for t in filtered_docs_df['Topic'].values],
            'document': [meta.get('file_name', 'Unknown') for meta in filtered_metadata],
            'page': [meta.get('page_num', 'N/A') for meta in filtered_metadata],
            'text_snippet': [text[:200] + ("..." if len(text) > 200 else "") for text in filtered_texts],
        })

        # Clean up extra_point_data topic names
        extra_point_data['topic_name'] = extra_point_data['topic_name'].apply(
            lambda x: x.replace("Topic Name:", "").replace("_", " ").strip() if isinstance(x, str) else x
        )

        # Add chunk IDs if available
        extra_point_data['chunk_id'] = [meta.get('chunk_id', 'Unknown') for meta in filtered_metadata]

        # Extract config parameters
        height = config.get("height", 800)
        width = config.get("width", "100%")
        marker_size = config.get("marker_size", 8)
        darkmode = config.get("darkmode", False)
        cvd_safer = config.get("cvd_safer", True)
        cluster_boundaries = config.get("cluster_boundary_polygons", True)  # Enable by default
        color_label_text = config.get("color_label_text", True)
        polygon_alpha = float(config.get("polygon_alpha", 2.5))  # Slightly more visible
        font_family = config.get("font_family", "Oswald")
        enable_toc = config.get("enable_table_of_contents", True)

        # Check if we need to load a Google font
        if font_family in ["Oswald", "Open Sans"]:
            logger.info(f"Loading Google Font for interactive plot: {font_family}")
            loaded_font = get_google_font(font_family)
            if loaded_font:
                font_family = loaded_font
                logger.info(f"Using loaded font: {font_family}")
            else:
                font_family = "Oswald"
                logger.warning(f"Failed to load {font_family}, falling back to Oswald")

        # FIX: Customize tooltip using HTML template with proper structure
        hover_template = """
        <div style="max-width: 400px; padding: 10px; font-family: Open Sans, Arial, sans-serif;">
            <h3 style="margin: 0 0 5px 0; color: white; font-size: 16px; font-weight: bold;">{document}</h3>
            <p style="margin: 0 0 8px 0;"><b>Page:</b> {page}</p>
            <p style="margin: 0 0 8px 0;"><b>Topic:</b> {topic_name}</p>
            <p style="margin: 0 0 8px 0;"><b>Chunk ID:</b> {chunk_id}</p>
            <hr style="margin: 5px 0; border: 0; border-top: 1px solid #ddd;">
            <p style="margin: 0; font-style: italic; font-size: 14px;">{text_snippet}</p>
        </div>
        """

        # FIX: Customize tooltip CSS with more explicit rules
        tooltip_css = """
        .datamapplot-tooltip {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            font-family: Open Sans, Arial, sans-serif;
            max-width: 420px;
            padding: 10px !important;
            pointer-events: none;
            z-index: 9999;
            color: #333;
            font-size: 14px;
            line-height: 1.4;
            opacity: 0.95;
        }
        .datamapplot-tooltip h3 {
            font-size: 16px !important;
            font-weight: bold !important;
            color: #333 !important;
            margin-top: 0 !important;
            margin-bottom: 5px !important;
        }
        .datamapplot-tooltip p {
            font-size: 14px !important;
            line-height: 1.3 !important;
            margin: 0 0 5px 0 !important;
        }
        .datamapplot-tooltip hr {
            border: 0;
            border-top: 1px solid #ddd;
            margin: 5px 0;
        }
        """

        def ensure_string_labels(labels):
            """Convert any non-string labels to strings to prevent expandtabs() errors.
            Also clean up any 'Topic Name:' prefixes and replace underscores with spaces."""
            # For DataMapPlot, we need to map the numeric topic IDs to their text representations
            cleaned_labels = []
            for label in labels:
                if label == -1:
                    cleaned_label = "Noise/Outlier"
                else:
                    text = topic_name_map.get(label, f"Topic {label}")
                    # Clean up the topic name
                    if isinstance(text, str):
                        text = text.replace("Topic Name:", "").replace("_", " ").strip()
                    cleaned_label = str(text)
                cleaned_labels.append(cleaned_label)
            return cleaned_labels

        # Then when creating the label layer:
        label_layer = filtered_docs_df['Topic'].values
        # Convert to strings with proper names before passing to DataMapPlot
        label_layer = ensure_string_labels(label_layer)
        marker_size_array = np.ones(len(filtered_embeddings)) * marker_size

        # FIX: Add a clear progress container
        progress_container = st.empty()

        with progress_container:
            # Create interactive plot with improved settings
            plot = datamapplot.create_interactive_plot(
                filtered_embeddings,
                label_layer,  # Single layer implementation
                hover_text=hover_text,
                height=height,
                width=width,
                noise_label="Noise/Outlier",
                darkmode=darkmode,
                cvd_safer=cvd_safer,
                cluster_boundary_polygons=cluster_boundaries,
                color_label_text=color_label_text,
                polygon_alpha=polygon_alpha,
                font_family=font_family,
                marker_size_array=marker_size_array,
                # Tooltip improvements
                extra_point_data=extra_point_data,
                hover_text_html_template=hover_template,
                tooltip_css=tooltip_css,
                # Additional visual improvements
                text_outline_width=10,  # Thicker text outline for better visibility
                point_hover_color='#FF5500BB',  # More vibrant hover color
                text_min_pixel_size=14,  # Slightly larger minimum text size
                text_max_pixel_size=42,  # Larger maximum text size
                text_collision_size_scale=4,  # Adjusted collision detection
                cluster_boundary_line_width=1.5,  # Thicker cluster boundaries
                enable_search=True,  # Enable search functionality
                search_field="text_snippet",  # Search by text content
                #add_table_of_contents=enable_toc,  # Add table of contents if enabled
            )

        # FIX: Clear the progress container
        progress_container.empty()

        logger.info("DataMapPlot visualization created successfully")

        return plot
    except Exception as e:
        logger.error(f"Error creating DataMapPlot visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_static_datamap_visualization(vis_embeddings, docs_df, texts, metadata, topic_info,
                                        include_outliers=False, config=None):
    """
    Create a static DataMapPlot visualization from cluster data.
    This is specifically for 'static_datamapplot' visualization type.

    Args:
        vis_embeddings: 2D embeddings for visualization
        docs_df: DataFrame with document info
        texts: list of document texts
        metadata: list of document metadata
        topic_info: DataFrame with topic info
        include_outliers: Whether to include outliers
        config: Configuration dict with DataMapPlot settings

    Returns:
        matplotlib figure object
    """
    try:
        # Import datamapplot for static plotting
        import datamapplot
        from datamapplot.create_plots import create_plot
    except ImportError:
        logger.error("DataMapPlot not installed. Please run: pip install datamapplot")
        return None

    logger.info("Creating static DataMapPlot visualization")

    try:
        # Use provided config or get from global CONFIG
        if config is None:
            config = CONFIG.get("clustering", {}).get("static_datamapplot", {})

        # Filter out outliers if not included
        if not include_outliers:
            # Create mask for non-outlier documents
            non_outlier_mask = docs_df['Topic'] != -1
            filtered_embeddings = vis_embeddings[non_outlier_mask]
            filtered_docs_df = docs_df[non_outlier_mask].reset_index(drop=True)

            # Filter texts and metadata accordingly
            filtered_indices = np.where(non_outlier_mask)[0]
            filtered_texts = [texts[i] for i in filtered_indices]
            filtered_metadata = [metadata[i] for i in filtered_indices]
        else:
            filtered_embeddings = vis_embeddings
            filtered_docs_df = docs_df
            filtered_texts = texts
            filtered_metadata = metadata

        # Make sure Google Fonts are loaded if specified
        font_family = config.get("font_family", "Oswald")
        if font_family in ["Oswald", "Open Sans"]:
            logger.info(f"Loading Google Font: {font_family}")
            loaded_font = get_google_font(font_family)
            if loaded_font:
                font_family = loaded_font
                logger.info(f"Using loaded font: {font_family}")
            else:
                font_family = "Oswald"
                logger.warning(f"Failed to load {font_family}, falling back to Oswald")

        # Extract parameters from config with defaults
        darkmode = config.get("darkmode", False)
        cvd_safer = config.get("cvd_safer", True)
        marker_size = config.get("marker_size", 8)
        label_wrap_width = config.get("label_wrap_width", 16)
        color_label_text = config.get("color_label_text", True)
        color_label_arrows = config.get("color_label_arrows", True)
        dpi = config.get("dpi", 300)
        noise_label = config.get("noise_label", "Noise/Outlier")
        title = config.get("title", "Document Cluster Map")

        # Create a mapping of topic IDs to labels
        topic_name_map = topic_info.set_index('Topic')['Name'].to_dict()

        # Get the labels array for the plot
        labels = filtered_docs_df['Topic'].map(lambda x:
                                               topic_name_map.get(x, f"Topic {x}")
                                               if x != -1 else noise_label).values

        # FIXED: Clean up topic labels - remove "Topic Name:" and underscores from labels
        labels = [label.replace("Topic Name:", "").replace("_", " ").strip() if isinstance(label, str) else label for
                  label in labels]

        # Create the static plot
        fig, ax = create_plot(
            filtered_embeddings,
            labels=labels,
            title=title,
            figsize=(16,10),  # Convert list to tuple
            darkmode=darkmode,
            cvd_safer=cvd_safer,
            noise_label=noise_label,
            label_wrap_width=label_wrap_width,
            color_label_text=color_label_text,
            color_label_arrows=color_label_arrows,
            dpi=dpi,
            force_matplotlib=True,  # Always use matplotlib for static plots
            #marker_size=marker_size,
            font_family=font_family
        )

        logger.info("Static DataMapPlot visualization created successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating static DataMapPlot visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def check_datamapplot_dependencies():
    """Check if DataMapPlot dependencies are available"""
    try:
        import datamapplot
        return True
    except ImportError:
        logger.error("DataMapPlot not installed. Please run: pip install datamapplot")
        return False


def generate_cluster_map(query_engine, include_outliers=False):
    """
    Generate an interactive cluster map from document embeddings.
    FIXED: Uses updated topic modeling, handles visualization filtering.

    Args:
        query_engine: QueryEngine instance with access to Qdrant
        include_outliers: Whether to include outliers (topic -1) in the visualization

    Returns:
        dict with visualization data or None if failed, and a message
    """
    try:
        with open(CONFIG_PATH, "r") as f:
            fresh_config = yaml.safe_load(f)
        visualization_type = fresh_config.get("clustering", {}).get("visualization_type", "plotly").lower()
        logger.info(f"VISUALIZATION TYPE FROM CONFIG: {visualization_type}")
    except Exception as e:
        logger.error(f"Error reading visualization type from config: {e}")
        visualization_type = "plotly"

    if visualization_type == "datamapplot":
        logger.info("⭐️⭐️⭐️ USING INTERACTIVE DATAMAPPLOT VISUALIZATION ⭐️⭐️⭐️")
    elif visualization_type == "static_datamapplot":
        logger.info("⭐️⭐️⭐️ USING STATIC DATAMAPPLOT VISUALIZATION ⭐️⭐️⭐️")
    else:
        logger.info("⭐️⭐️⭐️ USING PLOTLY VISUALIZATION ⭐️⭐️⭐️")

    # Check dependencies
    if not check_dependencies():
        return None, "Required dependencies for clustering not available"
    if ("datamapplot" in visualization_type) and not check_datamapplot_dependencies():
        return None, "DataMapPlot dependencies not available. Please install datamapplot."

    try:
        # Extract embeddings from Qdrant
        embeddings, texts, metadata = extract_embeddings_from_qdrant(query_engine)

        if len(embeddings) == 0 or len(texts) == 0:
            logger.error("No embeddings or text data found in Qdrant")
            return None, "No data found. Please process documents first."
        if len(embeddings) != len(texts) or len(texts) != len(metadata):
            logger.error(
                f"Data length mismatch: Embeddings ({len(embeddings)}), Texts ({len(texts)}), Metadata ({len(metadata)})")
            return None, "Data inconsistency found during extraction."

        # Create topic model using the updated function
        # Pass the original metadata list along with embeddings and texts
        topic_model, topics_mapped, topic_info, docs_df = create_topic_model(embeddings, texts, metadata)

        if topic_model is None or docs_df is None:
            logger.error("Failed to create topic model (returned None)")
            return None, "Failed to create topic model. Check logs for details."

        # Create visualization embeddings using ORIGINAL embeddings
        vis_embeddings = create_visualization_embeddings(embeddings)

        if vis_embeddings is None:
            logger.error("Failed to create visualization embeddings")
            return None, "Failed to create visualization embeddings. Check logs for details."
        if len(vis_embeddings) != len(docs_df):
            logger.error(f"Mismatch between vis_embeddings ({len(vis_embeddings)}) and docs_df ({len(docs_df)})")
            return None, "Dimension mismatch after creating visualization embeddings."

        # --- Prepare data for visualization ---
        # Filter out documents marked as filtered (-2) and potentially outliers (-1)
        # based on the `include_outliers` flag.
        viz_mask = (docs_df['Topic'] != -2)  # Always exclude docs that were pre-filtered
        if not include_outliers:
            viz_mask &= (docs_df['Topic'] != -1)  # Exclude outliers if flag is false

        # Apply the mask to get data specifically for the plot
        # Ensure indices align correctly! Use boolean indexing.
        filtered_vis_embeddings = vis_embeddings[viz_mask]
        filtered_docs_df = docs_df[viz_mask].reset_index(drop=True)  # Reset index for plotting libs
        # Get corresponding original texts and metadata for hover info etc.
        # Need to apply the same mask to the original lists
        original_indices_to_keep = np.where(viz_mask)[0]
        filtered_texts_for_viz = [texts[i] for i in original_indices_to_keep]
        filtered_metadata_for_viz = [metadata[i] for i in original_indices_to_keep]

        logger.info(f"Preparing visualization for {len(filtered_docs_df)} documents.")
        if len(filtered_docs_df) == 0:
            logger.warning(
                "No documents left to visualize after filtering. Check 'include_outliers' setting and topic results.")
            # Return an empty state or message instead of erroring?
            return None, "No documents to display based on current filters."

        # Choose visualization based on selected type
        fig = None
        is_datamap = False
        is_static = False

        # FIXED: Specific handling for static_datamapplot
        if visualization_type == "static_datamapplot":
            logger.info("Creating Static DataMapPlot visualization")
            static_config = fresh_config.get("clustering", {}).get("static_datamapplot", {})
            fig = create_static_datamap_visualization(
                filtered_vis_embeddings,  # Use data filtered for viz
                filtered_docs_df,  # Use data filtered for viz
                filtered_texts_for_viz,  # Use data filtered for viz
                filtered_metadata_for_viz,  # Use data filtered for viz
                topic_info,
                include_outliers=True,  # Pass true as filtering is done *before* this call
                config=static_config
            )
            if fig is not None:
                is_static = True
                logger.info("Static DataMapPlot visualization created successfully")
            else:
                logger.error("Failed to create Static DataMapPlot visualization")

        # Interactive DataMapPlot
        elif visualization_type == "datamapplot":
            logger.info("Creating Interactive DataMapPlot visualization")
            dmp_config = fresh_config.get("clustering", {}).get("datamapplot", {})
            fig = create_datamap_visualization(
                filtered_vis_embeddings,  # Use data filtered for viz
                filtered_docs_df,  # Use data filtered for viz
                filtered_texts_for_viz,  # Use data filtered for viz
                filtered_metadata_for_viz,  # Use data filtered for viz
                topic_info,
                include_outliers=True,  # Pass true as filtering is done *before* this call
                config=dmp_config
            )
            if fig is not None:
                is_datamap = True
                logger.info("Interactive DataMapPlot visualization created successfully")
            else:
                logger.error("Failed to create Interactive DataMapPlot visualization")

        # Plotly visualization (default fallback)
        if fig is None and visualization_type not in ["datamapplot", "static_datamapplot"] or (
                fig is None and visualization_type in ["datamapplot", "static_datamapplot"]):
            # Create Plotly visualization as default or fallback
            logger.info("Creating Plotly visualization")
            fig = create_interactive_cluster_map(
                filtered_vis_embeddings,  # Use data filtered for viz
                filtered_docs_df,  # Use data filtered for viz (needs Topic_Name)
                filtered_texts_for_viz,  # Use data filtered for viz
                filtered_metadata_for_viz,  # Use data filtered for viz
                topic_info,
                include_outliers=True  # Pass true as filtering is done *before* this call
            )
            if fig is not None:
                logger.info("Plotly visualization created successfully")
            else:
                logger.error("Failed to create Plotly visualization.")

        if fig is None:
            logger.error("Failed to create any visualization.")
            return None, "Failed to generate visualization. Check logs."

        # Calculate final statistics based on the *original* full docs_df
        total_docs = len(docs_df)
        outlier_count = (docs_df['Topic'] == -1).sum()
        filtered_count = (docs_df['Topic'] == -2).sum()  # Count how many were initially filtered
        clustered_count = total_docs - outlier_count - filtered_count
        outlier_percentage = (outlier_count / total_docs) * 100 if total_docs > 0 else 0
        filtered_percentage = (filtered_count / total_docs) * 100 if total_docs > 0 else 0

        # Return the result
        result = {
            "figure": fig,
            "topic_info": topic_info,
            "docs_df": filtered_docs_df,  # Include docs_df for download functionality
            "document_count": total_docs,  # Total original docs
            "clustered_count": clustered_count,  # Docs assigned to actual topics
            "outlier_count": outlier_count,  # Docs assigned to -1
            "pre_filtered_count": filtered_count,  # Docs filtered out before BERTopic
            "outlier_percentage": outlier_percentage,
            "pre_filtered_percentage": filtered_percentage,
            "topic_count": len(topic_info[topic_info['Topic'] >= 0]),  # Count actual topics (>=0)
            "is_datamap": is_datamap,
            "is_static": is_static,  # Flag for static matplotlib figure
            "include_outliers_setting": include_outliers,  # The setting requested
            "visualization_type": visualization_type
        }

        return result, f"Cluster map generated successfully using {visualization_type} visualization"

    except Exception as e:
        logger.error(f"Error generating cluster map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error: {str(e)}"