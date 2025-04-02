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


def create_topic_model(embeddings, texts, config=None):
    """
    Create and fit a BERTopic model on the embeddings and texts.
    Reuses existing model instances when possible to avoid memory issues.

    Args:
        embeddings: numpy array of embeddings
        texts: list of text data
        config: configuration dictionary

    Returns:
        tuple: (topic_model, topics, topic_info, docs_df)
    """
    global _topic_model_instance, _representation_model_instance, _last_embedding_shape

    if not check_dependencies():
        return None, None, None, None

    logger.info("Creating topic model")
    start_time = time.time()

    try:
        # Use provided config or get from global CONFIG
        if config is None:
            config = CONFIG.get("clustering", {})

        # Load UMAP parameters
        umap_params = config.get("umap", {})
        umap_n_neighbors = int(umap_params.get("n_neighbors", 15))
        umap_n_components = int(umap_params.get("n_components", 5))
        umap_min_dist = float(umap_params.get("min_dist", 0.0))
        umap_metric = umap_params.get("metric", "cosine")

        logger.info(
            f"UMAP parameters: n_neighbors={umap_n_neighbors}, n_components={umap_n_components}, min_dist={umap_min_dist}, metric={umap_metric}")

        # Load HDBSCAN parameters
        hdbscan_params = config.get("hdbscan", {})
        min_cluster_size = int(hdbscan_params.get("min_cluster_size", 10))
        min_samples = int(hdbscan_params.get("min_samples", 5))

        logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

        # Load vectorizer parameters
        vectorizer_params = config.get("vectorizer", {})
        stop_words = vectorizer_params.get("stop_words", "english")
        ngram_range = vectorizer_params.get("ngram_range", [1, 2])

        # Load topic parameters
        topic_params = config.get("topics", {})
        nr_topics_config = topic_params.get("nr_topics", "auto")
        nr_topics = None if nr_topics_config == "auto" else int(nr_topics_config)

        # Create seed topic list if provided
        seed_topic_list_config = topic_params.get("seed_topic_list", [])
        seed_topic_list = [[]] if not seed_topic_list_config else seed_topic_list_config

        # Create UMAP model for dimensionality reduction
        logger.info("Creating UMAP model for dimensionality reduction")
        umap_model = UMAP(
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=42
        )

        # Create HDBSCAN model for clustering
        logger.info("Creating HDBSCAN model for clustering")
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True
        )

        # Create CountVectorizer for tokenization
        logger.info("Creating CountVectorizer for tokenization")
        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=tuple(ngram_range)
        )

        # Create ClassTfidfTransformer for topic representation
        logger.info("Creating ClassTfidfTransformer for topic representation")
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        # Create or reuse a representation model for topic naming
        from src.core.visualization.aphrodite_representation import AphroditeRepresentation
        if _representation_model_instance is None:
            logger.info("Creating new AphroditeRepresentation instance")
            _representation_model_instance = AphroditeRepresentation(
                model_name="Qwen/Qwen2.5-3B-Instruct",  # Specify a valid model name
                nr_docs=3,
                doc_length=150,
                direct_load=False,
                tokenizer="whitespace"
            )
        else:
            logger.info("Reusing existing AphroditeRepresentation instance")

        # Create representation model list
        representation_model = [
            KeyBERTInspired(),
            _representation_model_instance
        ]

        # Check if we need to create a new BERTopic model
        if _topic_model_instance is None or _last_embedding_shape != embeddings.shape:
            logger.info(
                f"Creating new BERTopic model instance (previous shape: {_last_embedding_shape}, current: {embeddings.shape})")
            # Create BERTopic model
            _topic_model_instance = BERTopic(
                umap_model=umap_model,
                embedding_model="intfloat/multilingual-e5-large-instruct",
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                ctfidf_model=ctfidf_model,
                seed_topic_list=seed_topic_list,
                nr_topics=nr_topics,
                representation_model=representation_model,
                verbose=True
            )
            # Update the stored embedding shape
            _last_embedding_shape = embeddings.shape
        else:
            logger.info("Reusing existing BERTopic model instance")
            # Update parameters if needed
            _topic_model_instance.umap_model = umap_model
            _topic_model_instance.hdbscan_model = hdbscan_model
            _topic_model_instance.vectorizer_model = vectorizer_model
            _topic_model_instance.ctfidf_model = ctfidf_model
            _topic_model_instance.seed_topic_list = seed_topic_list
            _topic_model_instance.nr_topics = nr_topics
            _topic_model_instance.representation_model = representation_model

        # Fit the model
        logger.info(f"Fitting BERTopic model on {len(texts)} documents with precomputed embeddings")
        # Track how many documents are empty or very short
        empty_docs = sum(1 for text in texts if len(text.strip()) < 10)
        if empty_docs > 0:
            logger.warning(f"{empty_docs} documents are empty or very short (< 10 chars)")

        # Convert embeddings to float32 if not already
        embeddings_array = embeddings.astype(np.float32)

        # Log embeddings shape and sample values for debugging
        logger.info(f"Embeddings shape before fit: {embeddings_array.shape}")
        if len(embeddings_array) > 0:
            logger.info(f"Sample embedding: {embeddings_array[0][:5]}... (showing first 5 values)")
            logger.info(f"Embedding type: {type(embeddings_array[0])}")
            logger.info(f"Any NaN values: {np.isnan(embeddings_array).any()}")

        # Fit transform with embeddings
        topics, probs = _topic_model_instance.fit_transform(texts, embeddings_array)

        # Get topic info
        logger.info("Getting topic information")
        topic_info = _topic_model_instance.get_topic_info()

        # Get document info
        logger.info("Getting document information")
        document_info = _topic_model_instance.get_document_info(texts)

        # Create a DataFrame from document_info for easier handling
        docs_df = pd.DataFrame(document_info)

        elapsed_time = time.time() - start_time
        logger.info(f"Created topic model in {elapsed_time:.2f}s")
        logger.info(f"Found {len(topic_info) - 1} topics (excluding -1 for outliers)")

        return _topic_model_instance, topics, topic_info, docs_df

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


# This patch should be applied to the create_interactive_cluster_map function
# in src/core/visualization/cluster_map.py

def create_interactive_cluster_map(vis_embeddings, docs_df, texts, metadata, topic_info, include_outliers=False):
    """
    Create an interactive cluster map visualization.

    Args:
        vis_embeddings: 2D embeddings for visualization
        docs_df: DataFrame with document info
        texts: list of document texts
        metadata: list of document metadata
        topic_info: DataFrame with topic info
        include_outliers: Whether to include outliers (topic -1) in the visualization

    Returns:
        plotly figure: Interactive cluster map
    """
    if not check_dependencies():
        return None

    logger.info("Creating interactive cluster map")

    try:
        # Create a DataFrame with visualization data
        vis_df = pd.DataFrame({
            'x': vis_embeddings[:, 0],
            'y': vis_embeddings[:, 1],
            'text': texts,
            'topic': docs_df['Topic'].values
        })

        # Add metadata to the DataFrame
        for i, meta in enumerate(metadata):
            for key, value in meta.items():
                vis_df.loc[i, key] = value

        # Filter out outliers (topic -1) if include_outliers is False
        if not include_outliers:
            logger.info("Filtering out outliers (topic -1) from visualization")
            before_count = len(vis_df)
            vis_df = vis_df[vis_df['topic'] != -1]
            after_count = len(vis_df)
            logger.info(
                f"Removed {before_count - after_count} outlier points ({(before_count - after_count) / before_count * 100:.1f}% of data)")

            # Check if we have points left to visualize
            if len(vis_df) == 0:
                logger.warning("No data points remain after filtering out outliers. Cannot create visualization.")
                return None

        # Create topic labels
        if 'KeyBERT' in docs_df.columns:
            # Use KeyBERT labels if available
            try:
                logger.info("Using KeyBERT labels for topics")
                keybert_labels = []
                for label in docs_df['KeyBERT'].values:
                    try:
                        # Try to parse the KeyBERT label
                        parsed = ast.literal_eval(label)[0] if label else "Unlabelled"
                        # Clean up the label
                        cleaned = re.sub(r'\W+', ' ', str(parsed).split("\n")[0].replace('"', ''))
                        keybert_labels.append(cleaned if cleaned else "Unlabelled")
                    except:
                        keybert_labels.append("Unlabelled")

                vis_df['topic_label'] = keybert_labels
            except Exception as e:
                logger.warning(f"Error creating KeyBERT labels: {e}, using default labels")
                # Use topic_info for labels
                topic_dict = topic_info.set_index('Topic')['Name'].to_dict()
                vis_df['topic_label'] = vis_df['topic'].map(lambda x: topic_dict.get(x, "Outlier"))
        else:
            # Use topic_info for labels
            logger.info("Using topic_info for topic labels")
            topic_dict = topic_info.set_index('Topic')['Name'].to_dict()
            vis_df['topic_label'] = vis_df['topic'].map(lambda x: topic_dict.get(x, "Outlier"))

        # Create plotly figure
        logger.info("Creating plotly figure")

        # Create a categorical color scale based on topics
        unique_topics = vis_df['topic'].unique()

        # Create custom hover text
        vis_df['hover_text'] = vis_df.apply(
            lambda row: f"<b>Topic:</b> {row['topic_label']}<br>" +
                        f"<b>Document:</b> {row.get('file_name', 'Unknown')}<br>" +
                        f"<b>Page:</b> {row.get('page_num', 'N/A')}<br>" +
                        f"<b>Text:</b> {row['text'][:100]}...",
            axis=1
        )

        # Create figure
        fig = px.scatter(
            vis_df,
            x='x',
            y='y',
            color='topic_label',
            hover_data={'x': False, 'y': False, 'topic_label': True, 'file_name': True, 'page_num': True},
            custom_data=['hover_text', 'text', 'file_name', 'topic'],
            title="Document Cluster Map",
            labels={'topic_label': 'Topic'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7
        )

        # Update layout for better visualization
        fig.update_layout(
            plot_bgcolor='white',
            legend=dict(
                title_font_size=14,
                font=dict(size=12),
                itemsizing='constant',
                tracegroupgap=0
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            ),
            height=900,
            width=1000,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        # Update traces
        fig.update_traces(
            marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate='%{customdata[0]}<extra></extra>'
        )

        # Create buttons for topic filtering
        topic_buttons = []

        # Add button for All topics
        topic_buttons.append(
            dict(
                label="All Topics",
                method="update",
                args=[{"visible": [True] * len(fig.data)}]
            )
        )

        # Add buttons for each topic
        for i, topic_label in enumerate(set(vis_df['topic_label'])):
            # Create visibility list
            visible = [topic == topic_label for topic in vis_df['topic_label'].unique()]

            # Add button
            topic_buttons.append(
                dict(
                    label=topic_label,
                    method="update",
                    args=[{"visible": visible}]
                )
            )

        # Add buttons to layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    showactive=True,
                    buttons=topic_buttons,
                    x=0.1,
                    xanchor="left",
                    y=1.15,
                    yanchor="top"
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

        # Extract config parameters
        height = config.get("height", 800)
        width = config.get("width", "100%")
        marker_size = config.get("marker_size", 8)
        darkmode = config.get("darkmode", False)
        cvd_safer = config.get("cvd_safer", True)
        cluster_boundaries = config.get("cluster_boundary_polygons", True)  # Enable by default
        color_label_text = config.get("color_label_text", True)
        polygon_alpha = float(config.get("polygon_alpha", 2.5))  # Slightly more visible
        font_family = config.get("font_family", "Arial")

        # Customize tooltip using HTML template
        hover_template = """
        <div style="max-width: 400px; padding: 10px;">
            <h3 style="margin: 0 0 5px 0; color: white;">{document}</h3>
            <p style="margin: 0 0 8px 0;"><b>Page:</b> {page}</p>
            <p style="margin: 0 0 8px 0;"><b>Topic:</b> {topic_name}</p>
            <hr style="margin: 5px 0;">
            <p style="margin: 0; font-style: italic;">{text_snippet}</p>
        </div>
        """

        # Customize tooltip CSS
        tooltip_css = """
        .datamapplot-tooltip {
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
            font-family: Arial, sans-serif;
            max-width: 420px;
            padding: 0;
            pointer-events: none;
            z-index: 9999;
        }
        .datamapplot-tooltip h3 {
            font-size: 14px;
            font-weight: bold;
            color:white;
        }
        .datamapplot-tooltip p {
            font-size: 12px;
            line-height: 1.3;
        }
        """

        def ensure_string_labels(labels):
            """Convert any non-string labels to strings to prevent expandtabs() errors."""
            # For DataMapPlot, we need to map the numeric topic IDs to their text representations
            # to ensure the labels are more meaningful
            return [str(topic_name_map.get(label, f"Topic {label}") if label != -1 else "Noise/Outlier")
                    for label in labels]

        # Then when creating the label layer:
        label_layer = filtered_docs_df['Topic'].values
        # Convert to strings with proper names before passing to DataMapPlot
        label_layer = ensure_string_labels(label_layer)
        label_layer = [x.replace("_Topic name:",":").replace("_","")for x in label_layer]
        marker_size_array = np.ones(len(filtered_embeddings)) * marker_size

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
            search_field="text_snippet"  # Search by text content
        )

        logger.info("DataMapPlot visualization created successfully")

        return plot
    except Exception as e:
        logger.error(f"Error creating DataMapPlot visualization: {e}")
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
# This patch should be applied to the generate_cluster_map function
# in src/core/visualization/cluster_map.py

def generate_cluster_map(query_engine, include_outliers=False):
    """
    Generate an interactive cluster map from document embeddings.

    Args:
        query_engine: QueryEngine instance with access to Qdrant
        include_outliers: Whether to include outliers (topic -1) in the visualization

    Returns:
        dict with visualization data or None if failed, and a message
    """
    # Get visualization type from config
    visualization_type = CONFIG.get("clustering", {}).get("visualization_type", "plotly").lower()

    # Check dependencies based on visualization type
    if not check_dependencies():
        return None, "Required dependencies for clustering not available"

    if visualization_type == "datamapplot" and not check_datamapplot_dependencies():
        return None, "DataMapPlot dependencies not available. Please install datamapplot."

    try:
        # Extract embeddings from Qdrant
        embeddings, texts, metadata = extract_embeddings_from_qdrant(query_engine)

        if len(embeddings) == 0:
            logger.error("No embeddings found in Qdrant")
            return None, "No embeddings found. Please process documents first."

        # Create topic model
        topic_model, topics, topic_info, docs_df = create_topic_model(embeddings, texts)

        if topic_model is None:
            logger.error("Failed to create topic model")
            return None, "Failed to create topic model. Check logs for details."

        # Create visualization embeddings
        vis_embeddings = create_visualization_embeddings(embeddings)

        if vis_embeddings is None:
            logger.error("Failed to create visualization embeddings")
            return None, "Failed to create visualization embeddings. Check logs for details."

        # Count outliers before any filtering
        total_docs = len(docs_df)
        outlier_mask = docs_df['Topic'] == -1
        outlier_count = outlier_mask.sum()
        clustered_count = total_docs - outlier_count
        outlier_percentage = (outlier_count / total_docs) * 100 if total_docs > 0 else 0

        # Choose visualization based on selected type
        if visualization_type == "datamapplot":
            logger.info("Using DataMapPlot for visualization")

            # Get DataMapPlot config
            dmp_config = CONFIG.get("clustering", {}).get("datamapplot", {})

            # Create DataMapPlot visualization
            fig = create_datamap_visualization(
                vis_embeddings,
                docs_df,
                texts,
                metadata,
                topic_info,
                include_outliers=include_outliers,
                config=dmp_config
            )

            if fig is None:
                logger.error("Failed to create DataMapPlot visualization")
                return None, "Failed to create DataMapPlot visualization. Check logs for details."

            is_datamap = True
        else:
            # Default to Plotly
            logger.info("Using Plotly for visualization")
            fig = create_interactive_cluster_map(
                vis_embeddings,
                docs_df,
                texts,
                metadata,
                topic_info,
                include_outliers=include_outliers
            )

            if fig is None:
                logger.error("Failed to create interactive cluster map")
                return None, "Failed to create interactive cluster map. Check logs for details."

            is_datamap = False

        # Return the result with common structure
        result = {
            "figure": fig,
            "topic_info": topic_info,
            "document_count": total_docs,
            "clustered_count": clustered_count,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "topic_count": len(topic_info[topic_info['Topic'] != -1]),  # Excluding -1 topic
            "is_datamap": is_datamap,  # Flag to identify visualization type
            "include_outliers": include_outliers  # Store the outlier inclusion setting
        }

        return result, "Cluster map generated successfully"

    except Exception as e:
        logger.error(f"Error generating cluster map: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error: {str(e)}"