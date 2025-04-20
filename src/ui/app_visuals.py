# app_visuals.py
import streamlit as st
import networkx as nx
from pyvis.network import Network
import pandas as pd
import json
import math
import random
import traceback
import uuid
from collections import deque, Counter
import numpy as np # For Kamada-Kawai layout handling
from typing import Dict
# Import necessary functions/variables from other modules
from app_setup import ROOT_DIR, CONFIG, logger

# Conditional imports for optional features
try:
    import community as community_louvain
    louvain_available = True
except ImportError:
    louvain_available = False
    logger.warning("Python-louvain library not found. Community detection metrics will be unavailable.")

try:
    from node2vec import Node2Vec
    node2vec_available = True
except ImportError:
    node2vec_available = False
    logger.warning("Node2Vec library not found. Node Similarity metrics will be unavailable.")


# --- Helper Functions ---

def _build_pyvis_options(physics_enabled: bool, solver: str, spring_length: int, spring_constant: float, central_gravity: float, grav_constant: int) -> Dict:
    """
    Builds the PyVis physics options dictionary based on user settings and selected solver.
    (Copied from original app.py)
    """
    if not physics_enabled:
        return {
            "physics": {"enabled": False},
            "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 300, "keyboard": {"enabled": True}},
            "edges": {"smooth": False, "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}}},
            "nodes": {"font": {"size": 14, "face": "Arial"}}
        }

    physics_config = {
        "enabled": True,
        "solver": solver,
        "stabilization": {"enabled": True, "iterations": 2000, "updateInterval": 50, "onlyDynamicEdges": False, "fit": True},
        "adaptiveTimestep": True,
        "minVelocity": 0.75
    }

    if solver == "forceAtlas2Based":
        physics_config["forceAtlas2Based"] = {
            "gravitationalConstant": grav_constant, "centralGravity": central_gravity,
            "springLength": spring_length, "springConstant": spring_constant,
            "damping": 0.4, "avoidOverlap": 0.6
        }
    elif solver == "barnesHut":
        physics_config["barnesHut"] = {
            "gravitationalConstant": grav_constant, "centralGravity": central_gravity,
            "springLength": spring_length, "springConstant": spring_constant,
            "damping": 0.09, "avoidOverlap": 0.1
        }
    elif solver == "repulsion":
        physics_config["repulsion"] = {
            "centralGravity": central_gravity, "springLength": spring_length,
            "springConstant": spring_constant, "nodeDistance": int(spring_length * 1.5),
            "damping": 0.09
        }

    # Remove unused solver keys (safer approach)
    valid_solvers = ["forceAtlas2Based", "barnesHut", "repulsion"]
    for s in valid_solvers:
        if s != solver and s in physics_config:
            del physics_config[s]

    options = {
        "physics": physics_config,
        "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 300, "keyboard": {"enabled": True}},
        "edges": {"smooth": {"enabled": True, "type": "dynamic"}, "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}}},
        "nodes": {"font": {"size": 14, "face": "Arial"}}
    }
    return options

@st.cache_data(ttl=3600, show_spinner=False) # Cache Node2Vec model for an hour
def train_node2vec_model(_UG, dimensions=64, walk_length=30, num_walks=100, window=5, workers=4):
    """Trains a Node2Vec model on the graph using the 'node2vec' library. Cached function."""
    if not node2vec_available:
        logger.error("Node2Vec library is not available. Cannot train model.")
        st.error("Node Similarity feature requires 'node2vec'. Please install it (`pip install node2vec`) and restart.")
        return None, None

    logger.info(f"Preparing graph for Node2Vec training...")
    if _UG.number_of_nodes() == 0 or _UG.number_of_edges() == 0:
         logger.warning("Graph is empty or has no edges. Cannot train Node2Vec.")
         return None, None

    logger.info("Converting input graph to simple Graph for Node2Vec compatibility.")
    graph_to_train = nx.Graph(_UG) # Use a simple, undirected graph
    if graph_to_train.number_of_nodes() == 0 or graph_to_train.number_of_edges() == 0:
        logger.warning("Simple graph became empty after conversion. Cannot train Node2Vec.")
        return None, None

    logger.info(f"Training Node2Vec model (dim={dimensions}, walk_length={walk_length}, num_walks={num_walks})...")
    try:
        logger.info(f"Using {workers} workers for Node2Vec. (May require workers=1 on Windows if issues occur)")
        node2vec_model = Node2Vec(
            graph_to_train, dimensions=dimensions, walk_length=walk_length,
            num_walks=num_walks, p=1, q=1, workers=workers, seed=42, quiet=True
        )
        model = node2vec_model.fit(window=window, min_count=1, sg=1, epochs=10, batch_words=4)
        logger.info("Node2Vec model training complete.")
        # Map original IDs to string representation for KeyedVectors compatibility
        final_node_map = {orig_id: str(orig_id) for orig_id in graph_to_train.nodes()}
        return model.wv, final_node_map

    except Exception as e:
         logger.error(f"Node2Vec training failed: {e}", exc_info=True)
         st.error(f"Node2Vec training failed: {e}")
         return None, None


# --- Graph Rendering Functions ---

def render_network_overview_tab(entities, relationships):
    """
    Render the overview network graph tab using MultiDiGraph.
    Enhanced with layout algorithm choice, better styling, physics controls, and tooltips.
    (Copied and adapted from original app.py)
    """
    st.markdown("#### Global Network Visualization")
    st.markdown("Visualize the connections between the most prominent entities based on relationship frequency.")

    # --- Controls ---
    st.markdown("**Visualization Controls**")
    control_col1, control_col2, control_col3 = st.columns(3)
    with control_col1:
        max_nodes = len(entities)
        top_entities_count = st.slider(
            "Max Entities to Display", min_value=10, max_value=max(max_nodes, 300),
            value=min(100, max_nodes) if max_nodes > 0 else 10, # Adjusted default
            step=5, key="overview_max_entities",
            help=f"Adjust the maximum number of entities shown (Total available: {max_nodes}). Lower values improve performance."
        )
    with control_col2:
        entity_types = sorted(list(set(entity.get("type", "Unknown") for entity in entities)))
        # Ensure default is valid if entity_types is empty
        default_types_val = entity_types if entity_types else []
        selected_graph_types = st.multiselect(
            "Filter by Entity Type", options=entity_types, default=default_types_val,
            key="overview_entity_type_filter", help="Select entity types to include in the visualization."
        )
    with control_col3:
         layout_algorithm = st.selectbox(
              "Layout Algorithm",
              options=["PyVis Physics", "Kamada-Kawai (Static)"], index=0, key="overview_layout_algo",
              help="'PyVis Physics' uses interactive simulation. 'Kamada-Kawai' pre-calculates positions (best for < 150 nodes)."
         )
         disable_physics_controls = (layout_algorithm == "Kamada-Kawai (Static)")

    # --- Physics Controls ---
    st.markdown("**Physics & Layout Controls** (Only active for 'PyVis Physics' layout)")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
         physics_enabled_toggle = st.toggle("Enable Physics Simulation", value=True, key="overview_physics_toggle", disabled=disable_physics_controls)
         physics_enabled = physics_enabled_toggle and not disable_physics_controls
         physics_solver = st.selectbox(
                "Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=0,
                key="overview_physics_solver", disabled=disable_physics_controls or not physics_enabled,
                help="Algorithm for PyVis physics simulation."
            )
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion", min_value=-30000, max_value=-100, value=-8000, step=500, key="overview_grav_constant", disabled=disable_physics_controls or not physics_enabled)
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.1, step=0.05, key="overview_central_gravity", disabled=disable_physics_controls or not physics_enabled)
    with physics_col3:
        spring_length = st.slider( "Edge Length", min_value=50, max_value=600, value=150, step=10, key="overview_spring_length", disabled=disable_physics_controls or not physics_enabled)
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.04, step=0.005, format="%.3f", key="overview_spring_constant", disabled=disable_physics_controls or not physics_enabled)

    # --- Build Graph ---
    if not entities:
        st.warning("No entities available based on current filters.")
        return

    with st.spinner("Building graph..."):
        entity_lookup = {entity.get("id"): entity for entity in entities if entity.get("id")}
        entity_mentions = Counter() # Use Counter for efficiency
        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            if source_id in entity_lookup: entity_mentions[source_id] += 1
            if target_id in entity_lookup: entity_mentions[target_id] += 1

        # Filter entities by type first, then add mention count, then sort and take top N
        filtered_entities_by_type = [
            entity for entity in entities
            if entity.get("type", "Unknown") in selected_graph_types and entity.get("id") in entity_lookup
        ]
        # Add mention count (default to 1 if no mentions found but entity exists)
        for entity in filtered_entities_by_type:
            entity["mention_count"] = entity_mentions.get(entity.get("id"), 1)

        # Sort by mention count and take top N
        top_entities = sorted(filtered_entities_by_type, key=lambda e: e["mention_count"], reverse=True)[:top_entities_count]
        top_entity_ids = {entity.get("id") for entity in top_entities}

        if not top_entity_ids:
             st.warning("No entities selected based on type filter and count limit.")
             return

        G = nx.MultiDiGraph()
        node_attributes_added = set()

        for entity in top_entities:
             node_id = entity.get("id")
             if node_id and node_id not in node_attributes_added: # Ensure ID exists
                 G.add_node(
                    node_id,
                    label=entity.get("name", "Unknown"),
                    type=entity.get("type", "Unknown"),
                    mention_count=entity.get("mention_count", 1),
                    title=f"{entity.get('name', 'Unknown')}\nType: {entity.get('type', 'Unknown')}\nMentions: {entity.get('mention_count', 1)}"
                 )
                 node_attributes_added.add(node_id)

        edge_count = 0
        skipped_edges = 0
        for rel in relationships:
            source_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            target_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            if source_id in top_entity_ids and target_id in top_entity_ids: # Check against top_entity_ids
                rel_type = rel.get("type", rel.get("relationship_type"))
                description = rel.get("description", None)
                rel_id_key = rel.get("id", str(uuid.uuid4())) # Unique key for multi-edge

                if rel_type and isinstance(rel_type, str) and rel_type.strip() and rel_type.upper() != "UNKNOWN":
                     G.add_edge(source_id, target_id, key=rel_id_key, type=rel_type, description=description)
                     edge_count += 1
                else:
                    skipped_edges += 1

    if skipped_edges > 0:
         st.caption(f"ℹ️ Skipped {skipped_edges} relationships with missing/invalid types.")

    # --- Kamada-Kawai Layout ---
    node_positions = None
    if layout_algorithm == "Kamada-Kawai (Static)":
         if G.number_of_nodes() == 0: st.warning("Graph is empty, cannot calculate layout.")
         elif G.number_of_nodes() > 150: st.warning("Kamada-Kawai layout may be slow for > 150 nodes.")

         if G.number_of_nodes() > 1:
             with st.spinner("Calculating Kamada-Kawai layout..."):
                 try:
                     # Use largest weakly connected component for layout
                     largest_cc_nodes = max(nx.weakly_connected_components(G), key=len)
                     subgraph_for_layout = G.subgraph(largest_cc_nodes)

                     # Convert to simple graph for layout robustness
                     subgraph_simple = nx.Graph(subgraph_for_layout)

                     if subgraph_simple.number_of_nodes() > 1:
                         node_positions_comp = nx.kamada_kawai_layout(subgraph_simple)
                         # Map positions back to original graph nodes, place others at origin
                         node_positions = {
                             node: node_positions_comp.get(node, np.array([0.0, 0.0])) for node in G.nodes()
                         }
                         logger.info(f"Kamada-Kawai layout calculated for {len(node_positions_comp)} nodes.")
                         if len(largest_cc_nodes) < G.number_of_nodes():
                              st.caption(f"Layout applied to the largest component ({len(largest_cc_nodes)} nodes). Other {G.number_of_nodes() - len(largest_cc_nodes)} nodes placed near origin.")
                     else:
                         st.warning("Largest connected component has <= 1 node. Cannot apply Kamada-Kawai layout.")
                 except Exception as layout_err:
                     st.error(f"Failed to compute Kamada-Kawai layout: {layout_err}")
                     logger.error(f"Kamada-Kawai layout error: {traceback.format_exc()}")
                     node_positions = None # Fallback to physics
                     layout_algorithm = "PyVis Physics" # Force fallback
                     physics_enabled = True # Ensure physics is enabled if falling back
                     st.warning("Falling back to PyVis Physics layout due to error.")
         else:
             st.info("Graph has <= 1 node, layout calculation skipped.")

    # --- Create PyVis Network ---
    if G.number_of_nodes() > 0:
        with st.spinner("Rendering visualization..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # Node styling setup
            default_colors = {"PERSON": "#3B82F6", "ORGANIZATION": "#10B981", "GOVERNMENT_BODY": "#60BD68", "COMMERCIAL_COMPANY": "#F17CB0", "LOCATION": "#F59E0B", "POSITION": "#8B5CF6", "MONEY": "#EC4899", "ASSET": "#EF4444", "EVENT": "#6366F1", "Unknown": "#9CA3AF"}
            default_shapes = {"PERSON": "dot", "ORGANIZATION": "square", "GOVERNMENT_BODY": "triangle", "COMMERCIAL_COMPANY": "diamond", "LOCATION": "star", "POSITION": "ellipse", "MONEY": "hexagon", "ASSET": "box", "EVENT": "database", "Unknown": "dot"}
            colors = CONFIG.get("visualization", {}).get("node_colors", default_colors)
            shapes = CONFIG.get("visualization", {}).get("node_shapes", default_shapes)

            # Add nodes
            for node_id, attrs in G.nodes(data=True):
                entity_type = attrs.get("type", "Unknown")
                mentions = attrs.get("mention_count", 1)
                size = max(10, min(35, 10 + 5 * math.log1p(mentions)))
                color = colors.get(entity_type, colors.get("Unknown", "#9CA3AF"))
                shape = shapes.get(entity_type, shapes.get("Unknown", "dot"))
                pos_x, pos_y = None, None
                if node_positions is not None and node_id in node_positions:
                     pos_val = node_positions[node_id]
                     if isinstance(pos_val, (np.ndarray, list)) and len(pos_val) >= 2:
                         try:
                             pos_x = float(pos_val[0]) * 1000 # Scale for PyVis canvas
                             pos_y = float(pos_val[1]) * 1000
                         except (ValueError, TypeError):
                              logger.warning(f"Invalid position data for node {node_id}: {pos_val}. Skipping.")
                              pos_x, pos_y = None, None

                net.add_node(
                    node_id, label=attrs.get("label", "Unknown"), title=attrs.get("title", ""),
                    color=color, shape=shape, size=size,
                    font={'size': max(10, min(18, 11 + int(math.log1p(mentions))))},
                    x=pos_x, y=pos_y, # PyVis handles None positions
                    physics=(layout_algorithm != "Kamada-Kawai (Static)") # Disable physics for individual nodes if static layout
                )

            # Edge styling setup
            rel_type_styles = {
                "WORKS_FOR": {"color": "#ff5733", "dashes": False, "width": 2}, "OWNS": {"color": "#33ff57", "dashes": [5, 5], "width": 2},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False, "width": 1.5}, "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2], "width": 1.5},
                "MET_WITH": {"color": "#f4f70a", "dashes": False, "width": 1.5}, "DEFAULT": {"color": "#A0A0A0", "dashes": False, "width": 1.0}
            }

            # Add edges
            for source, target, attrs in G.edges(data=True):
                rel_type = attrs.get("type", "UNKNOWN")
                description = attrs.get("description", "N/A")
                edge_title = f"Type: {rel_type}\nDescription: {description}"
                style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"])
                net.add_edge(
                    source, target, title=edge_title, label="",
                    color=style.get("color"), width=style.get("width"),
                    dashes=style.get("dashes", False), opacity=0.7,
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                )

            # Set PyVis options
            pyvis_options = _build_pyvis_options(
                physics_enabled and layout_algorithm != "Kamada-Kawai (Static)", # Only enable physics if selected AND not static layout
                physics_solver, spring_length, spring_constant, central_gravity, grav_constant
            )
            net.set_options(json.dumps(pyvis_options))

            # Save and display graph
            graph_html_path = ROOT_DIR / "temp" / "overview_graph.html"
            try:
                 net.save_graph(str(graph_html_path))
                 with open(graph_html_path, "r", encoding="utf-8") as f: html_content = f.read()
                 st.components.v1.html(html_content, height=710, scrolling=False)
                 st.caption(f"Displaying {G.number_of_nodes()} entities and {G.number_of_edges()} relationships. Layout: {layout_algorithm}.")
            except Exception as render_err:
                 st.error(f"Failed to render graph: {render_err}")
                 logger.error(f"PyVis rendering failed: {traceback.format_exc()}")
    else:
        st.info("No nodes to display based on current filters.")


def render_connection_explorer_tab(entities, relationships):
    """
    Render the entity connection explorer tab using MultiDiGraph.
    Visualizes paths between two entities.
    (Copied and adapted from original app.py)
    """
    st.markdown("#### Entity-to-Entity Connection Path")
    st.markdown("Find and visualize the shortest paths connecting two specific entities.")

    if not entities:
        st.warning("No entities available for selection.")
        return

    # Entity lookups
    entity_name_to_id = {e.get("name"): e.get("id") for e in entities if e.get("name") and e.get("id")}
    entity_id_to_name = {v: k for k, v in entity_name_to_id.items()}
    entity_id_to_type = {e.get("id"): e.get("type", "Unknown") for e in entities if e.get("id")}
    entity_names = sorted(entity_name_to_id.keys())

    if len(entity_names) < 2:
        st.warning("Need at least two entities to find a connection.")
        return

    # --- Controls ---
    st.markdown("**Connection Controls**")
    col1, col2, col3 = st.columns(3)
    with col1: source_entity_name = st.selectbox("Source Entity", options=entity_names, index=0, key="conn_source")
    with col2:
        default_target_index = 1 if len(entity_names) > 1 else 0
        if entity_names[default_target_index] == source_entity_name and len(entity_names) > 1: default_target_index = 0
        target_entity_name = st.selectbox("Target Entity", options=entity_names, index=default_target_index, key="conn_target")
    with col3: degrees_of_separation = st.slider("Max Path Length (hops)", min_value=1, max_value=10, value=3, key="conn_degrees")

    st.markdown("**Physics & Layout Controls**")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
        physics_enabled = st.toggle("Enable Physics Simulation", value=True, key="conn_physics_toggle")
        physics_solver = st.selectbox("Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=1, key="conn_physics_solver")
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion", min_value=-20000, max_value=-100, value=-5000, step=500, key="conn_grav_constant")
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="conn_central_gravity")
    with physics_col3:
        spring_length = st.slider( "Edge Length", min_value=50, max_value=500, value=100, step=10, key="conn_spring_length")
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.06, step=0.005, format="%.3f", key="conn_spring_constant")

    # --- Visualization Button ---
    if st.button("Visualize Connection Path", key="visualize_connection_btn", type="primary"):
        if source_entity_name == target_entity_name:
            st.warning("Source and Target entities must be different.")
            return

        source_id = entity_name_to_id.get(source_entity_name)
        target_id = entity_name_to_id.get(target_entity_name)

        if not source_id or not target_id:
            st.error("Could not find IDs for selected entities.")
            return

        # --- Build Full Graph for Path Finding ---
        with st.spinner("Finding connection paths..."):
            G_path = nx.MultiDiGraph()
            nodes_in_rels = set()

            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 if s_id and t_id:
                     nodes_in_rels.add(s_id)
                     nodes_in_rels.add(t_id)

            for entity in entities:
                e_id = entity.get("id")
                if e_id in nodes_in_rels and not G_path.has_node(e_id):
                     G_path.add_node(e_id, label=entity.get("name"), type=entity.get("type"))

            for rel in relationships:
                s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                r_desc = rel.get("description", "N/A")
                rel_id_key = rel.get("id", str(uuid.uuid4()))
                if G_path.has_node(s_id) and G_path.has_node(t_id):
                     G_path.add_edge(s_id, t_id, key=rel_id_key, type=r_type, description=r_desc)

            # --- Path Finding (Undirected BFS on Simple Graph) ---
            UG_simple = nx.Graph(G_path) # Convert to simple undirected graph
            paths_found = []
            if UG_simple.has_node(source_id) and UG_simple.has_node(target_id):
                try:
                    # Find all shortest paths within the cutoff length
                    shortest_paths_gen = nx.all_simple_paths(UG_simple, source=source_id, target=target_id, cutoff=degrees_of_separation)
                    # Find the length of the absolute shortest path first
                    shortest_len = -1
                    try:
                        shortest_len = nx.shortest_path_length(UG_simple, source=source_id, target=target_id)
                    except nx.NetworkXNoPath:
                        pass # No path exists

                    # Collect only paths that match the shortest length AND are within cutoff
                    if shortest_len != -1 and shortest_len <= degrees_of_separation:
                        paths_found = [p for p in nx.all_shortest_paths(UG_simple, source=source_id, target=target_id) if len(p)-1 == shortest_len]
                        if not paths_found: # Should not happen if shortest_len was found, but safety check
                             logger.warning("Shortest path length found, but all_shortest_paths returned empty.")
                    elif shortest_len == -1:
                         logger.info(f"No path found between {source_entity_name} and {target_entity_name}.")
                    else: # Shortest path exists but exceeds cutoff
                         logger.info(f"Shortest path ({shortest_len} hops) exceeds limit ({degrees_of_separation}).")

                except nx.NetworkXNoPath:
                    logger.info(f"No path found between {source_entity_name} and {target_entity_name}.")
                except Exception as path_err:
                    st.error(f"Error finding paths: {path_err}")
                    logger.error(f"Path finding error: {traceback.format_exc()}")
            else:
                st.warning(f"Source or target entity not found in the graph after filtering/processing.")

            if not paths_found:
                st.warning(f"No connection path found between '{source_entity_name}' and '{target_entity_name}' within {degrees_of_separation} hops.")
                return

            # --- Create Visualization Subgraph (MultiDiGraph) ---
            path_nodes = set(node for path in paths_found for node in path)
            viz_graph = nx.MultiDiGraph()

            for node_id in path_nodes:
                 if node_id in entity_id_to_name:
                    viz_graph.add_node(
                        node_id, label=entity_id_to_name[node_id],
                        type=entity_id_to_type.get(node_id, "Unknown"),
                        is_source=(node_id == source_id), is_target=(node_id == target_id)
                    )

            # Add all edges (including parallels) between nodes that are part of any path
            for u, v, key, data in G_path.edges(data=True, keys=True):
                 if u in path_nodes and v in path_nodes:
                     viz_graph.add_edge(u, v, key=key, type=data.get("type"), description=data.get("description"))

        # --- Create PyVis Network ---
        with st.spinner("Rendering connection path..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # Node styling setup
            default_colors = {"PERSON": "#3B82F6", "ORGANIZATION": "#10B981", "GOVERNMENT_BODY": "#60BD68", "COMMERCIAL_COMPANY": "#F17CB0", "LOCATION": "#F59E0B", "POSITION": "#8B5CF6", "MONEY": "#EC4899", "ASSET": "#EF4444", "EVENT": "#6366F1", "Unknown": "#9CA3AF"}
            default_shapes = {"PERSON": "dot", "ORGANIZATION": "square", "GOVERNMENT_BODY": "triangle", "COMMERCIAL_COMPANY": "diamond", "LOCATION": "star", "POSITION": "ellipse", "MONEY": "hexagon", "ASSET": "box", "EVENT": "database", "Unknown": "dot"}
            colors = CONFIG.get("visualization", {}).get("node_colors", default_colors)
            shapes = CONFIG.get("visualization", {}).get("node_shapes", default_shapes)
            source_color, target_color = "#FF4444", "#44FF44" # Bright Red/Green

            # Add nodes
            for node_id, attrs in viz_graph.nodes(data=True):
                 entity_type = attrs.get("type", "Unknown")
                 is_source, is_target = attrs.get("is_source", False), attrs.get("is_target", False)
                 if is_source: color, border_width, size, title_suffix = source_color, 3, 30, " (Source)"
                 elif is_target: color, border_width, size, title_suffix = target_color, 3, 30, " (Target)"
                 else: color, border_width, size, title_suffix = colors.get(entity_type, colors.get("Unknown")), 1, 20, ""
                 shape = shapes.get(entity_type, shapes.get("Unknown"))
                 label = attrs.get("label", "Unknown")
                 title = f"{label}\nType: {entity_type}{title_suffix}"
                 net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size, borderWidth=border_width)

            # Edge styling setup
            rel_type_styles = {
                "WORKS_FOR": {"color": "#ff5733", "dashes": False}, "OWNS": {"color": "#33ff57", "dashes": [5, 5]},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False}, "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2]},
                "MET_WITH": {"color": "#f4f70a", "dashes": False}, "DEFAULT": {"color": "#AAAAAA", "dashes": False}
            }
            # Identify edges belonging to the *first* shortest path found for highlighting
            shortest_path_segments = set()
            if paths_found:
                shortest_p = paths_found[0]
                for i in range(len(shortest_p) - 1):
                    shortest_path_segments.add((shortest_p[i], shortest_p[i+1]))
                    # Add reverse segment if path was found undirected but graph is directed
                    shortest_path_segments.add((shortest_p[i+1], shortest_p[i]))


            # Add edges
            for source, target, attrs in viz_graph.edges(data=True):
                 rel_type = attrs.get("type", "RELATED_TO")
                 description = attrs.get("description", "N/A")
                 edge_title = f"Type: {rel_type}\nDescription: {description}"
                 is_shortest = (source, target) in shortest_path_segments
                 style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"]).copy()
                 if is_shortest:
                     style['color'], style['width'], style['dashes'] = "#FF6347", 3, False # Tomato color, thicker
                 else:
                     style['width'] = style.get('width', 1.5) # Use default width or 1.5
                 net.add_edge(
                    source, target, title=edge_title, label="",
                    width=style.get('width'), color=style.get('color'), dashes=style.get('dashes'),
                    arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                 )

            # Set options and display
            pyvis_options = _build_pyvis_options(physics_enabled, physics_solver, spring_length, spring_constant, central_gravity, grav_constant)
            net.set_options(json.dumps(pyvis_options))
            graph_html_path = ROOT_DIR / "temp" / "connection_graph.html"
            try:
                net.save_graph(str(graph_html_path))
                with open(graph_html_path, "r", encoding="utf-8") as f: html_content = f.read()

                st.markdown(f"**Found {len(paths_found)} shortest path(s) (length: {len(paths_found[0])-1} hops):**")
                path_desc = " → ".join([entity_id_to_name.get(node_id, "?") for node_id in paths_found[0]])
                st.success(f"Path: {path_desc}")
                if len(paths_found) > 1:
                     with st.expander(f"Show {len(paths_found)-1} other shortest path(s)"):
                         for i, path in enumerate(paths_found[1:], 1):
                             path_desc_other = " → ".join([entity_id_to_name.get(node_id, "?") for node_id in path])
                             st.text(f"Path {i+1}: {path_desc_other}")

                st.components.v1.html(html_content, height=710, scrolling=False)
                st.caption(f"Displaying {viz_graph.number_of_nodes()} entities and {viz_graph.number_of_edges()} relationships involved.")

            except Exception as render_err:
                st.error(f"Failed to render connection graph: {render_err}")
                logger.error(f"PyVis rendering failed for connection graph: {traceback.format_exc()}")


def render_entity_centered_tab(entities, relationships):
    """
    Render the entity-centered explorer tab using MultiDiGraph.
    Visualizes connections around a specific entity.
    (Copied and adapted from original app.py)
    """
    st.markdown("#### Entity-Centered Network View")
    st.markdown("Explore the immediate neighborhood around a selected entity.")

    if not entities:
        st.warning("No entities available for selection.")
        return

    # Entity lookups
    entity_name_to_id = {e.get("name"): e.get("id") for e in entities if e.get("name") and e.get("id")}
    entity_id_to_name = {v: k for k, v in entity_name_to_id.items()}
    entity_id_to_type = {e.get("id"): e.get("type", "Unknown") for e in entities if e.get("id")}
    entity_names = sorted(entity_name_to_id.keys())

    if not entity_names:
         st.warning("No valid entity names found for selection.")
         return

    # --- Controls ---
    st.markdown("**View Controls**")
    col1, col2, col3 = st.columns(3)
    with col1: center_entity_name = st.selectbox("Center Entity", options=entity_names, index=0, key="center_entity")
    with col2: connection_depth = st.slider("Connection Depth (hops)", min_value=1, max_value=5, value=1, key="center_depth")
    with col3:
        rel_types_available = sorted(list(set(rel.get("type", rel.get("relationship_type", "Unknown")) for rel in relationships)))
        selected_rel_types = st.multiselect("Filter Relationship Types", options=rel_types_available, default=rel_types_available, key="center_rel_filter")

    st.markdown("**Physics & Layout Controls**")
    physics_col1, physics_col2, physics_col3 = st.columns(3)
    with physics_col1:
        physics_enabled = st.toggle("Enable Physics Simulation", value=True, key="center_physics_toggle")
        physics_solver = st.selectbox("Physics Solver", options=["barnesHut", "forceAtlas2Based", "repulsion"], index=0, key="center_physics_solver")
    with physics_col2:
        grav_constant = st.slider( "Node Repulsion", min_value=-20000, max_value=-100, value=-6000, step=500, key="center_grav_constant")
        central_gravity = st.slider( "Central Gravity", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="center_central_gravity")
    with physics_col3:
        spring_length = st.slider( "Ideal Edge Length", min_value=50, max_value=500, value=150, step=10, key="center_spring_length")
        spring_constant = st.slider( "Edge Stiffness", min_value=0.005, max_value=0.5, value=0.07, step=0.005, format="%.3f", key="center_spring_constant")

    # --- Visualization Button ---
    if st.button("Visualize Centered Network", key="visualize_centered_btn", type="primary"):
        center_id = entity_name_to_id.get(center_entity_name)
        if not center_id:
            st.error("Could not find ID for the selected center entity.")
            return

        # --- Build Full Graph (Filtered by selected relationship types) ---
        with st.spinner(f"Building neighborhood graph around '{center_entity_name}'..."):
            G_full = nx.MultiDiGraph()
            nodes_in_rels = set()

            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                 if r_type in selected_rel_types and s_id and t_id:
                     nodes_in_rels.add(s_id)
                     nodes_in_rels.add(t_id)

            for entity in entities:
                 e_id = entity.get("id")
                 if e_id in nodes_in_rels and not G_full.has_node(e_id):
                     G_full.add_node(e_id, label=entity.get("name"), type=entity.get("type"))

            for rel in relationships:
                 s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
                 t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
                 r_type = rel.get("type", rel.get("relationship_type", "RELATED_TO"))
                 r_desc = rel.get("description", "N/A")
                 rel_id_key = rel.get("id", str(uuid.uuid4()))
                 if r_type in selected_rel_types and G_full.has_node(s_id) and G_full.has_node(t_id):
                      G_full.add_edge(s_id, t_id, key=rel_id_key, type=r_type, description=r_desc)

            # --- Find Neighborhood (Undirected BFS on Simple Graph) ---
            UG_simple = nx.Graph(G_full)
            nodes_in_neighborhood = {} # node_id: distance
            if UG_simple.has_node(center_id):
                 nodes_in_neighborhood = {center_id: 0}
                 queue = deque([(center_id, 0)])
                 while queue:
                    curr_node, dist = queue.popleft()
                    if dist >= connection_depth: continue
                    for neighbor in UG_simple.neighbors(curr_node):
                        if neighbor not in nodes_in_neighborhood:
                            nodes_in_neighborhood[neighbor] = dist + 1
                            queue.append((neighbor, dist + 1))
            else:
                 st.warning(f"Center entity '{center_entity_name}' not found in the graph after filtering.")

            if len(nodes_in_neighborhood) <= 1 and connection_depth > 0:
                st.warning(f"No connections found for '{center_entity_name}' within {connection_depth} hop(s) with the selected relationship types.")
                return

            # --- Create Visualization Subgraph (MultiDiGraph) ---
            viz_graph = nx.MultiDiGraph()
            for node_id, distance in nodes_in_neighborhood.items():
                 if node_id in entity_id_to_name:
                    viz_graph.add_node(
                        node_id, label=entity_id_to_name[node_id],
                        type=entity_id_to_type.get(node_id, "Unknown"),
                        distance=distance, is_center=(node_id == center_id)
                    )

            edge_count = 0
            for u, v, key, data in G_full.edges(data=True, keys=True):
                 if u in nodes_in_neighborhood and v in nodes_in_neighborhood:
                     viz_graph.add_edge(u, v, key=key, type=data.get("type"), description=data.get("description"))
                     edge_count +=1

        # --- Create PyVis Network ---
        with st.spinner("Rendering centered visualization..."):
            net = Network(height="700px", width="100%", directed=True, notebook=True, cdn_resources='remote', heading="")

            # Node styling setup
            default_colors = {"PERSON": "#3B82F6", "ORGANIZATION": "#10B981", "GOVERNMENT_BODY": "#60BD68", "COMMERCIAL_COMPANY": "#F17CB0", "LOCATION": "#F59E0B", "POSITION": "#8B5CF6", "MONEY": "#EC4899", "ASSET": "#EF4444", "EVENT": "#6366F1", "Unknown": "#9CA3AF"}
            default_shapes = {"PERSON": "dot", "ORGANIZATION": "square", "GOVERNMENT_BODY": "triangle", "COMMERCIAL_COMPANY": "diamond", "LOCATION": "star", "POSITION": "ellipse", "MONEY": "hexagon", "ASSET": "box", "EVENT": "database", "Unknown": "dot"}
            colors = CONFIG.get("visualization", {}).get("node_colors", default_colors)
            shapes = CONFIG.get("visualization", {}).get("node_shapes", default_shapes)
            center_color = "#FF0000" # Bright Red

            # Add nodes
            for node_id, attrs in viz_graph.nodes(data=True):
                 entity_type = attrs.get("type", "Unknown"); distance = attrs.get("distance", 0); is_center = attrs.get("is_center", False)
                 size = max(12, 35 - (distance * 6))
                 if is_center: color, border_width, title_suffix = center_color, 3, " (Center)"
                 else: color, border_width, title_suffix = colors.get(entity_type, colors.get("Unknown")), 1, f" ({distance} hop{'s' if distance != 1 else ''})"
                 shape = shapes.get(entity_type, shapes.get("Unknown"))
                 label = attrs.get("label", "Unknown")
                 title = f"{label}\nType: {entity_type}{title_suffix}"
                 net.add_node(node_id, label=label, title=title, color=color, shape=shape, size=size, borderWidth=border_width)

            # Edge styling setup
            rel_type_styles = {
                "WORKS_FOR": {"color": "#ff5733", "dashes": False, "width": 2}, "OWNS": {"color": "#33ff57", "dashes": [5, 5], "width": 2},
                "LOCATED_IN": {"color": "#3357ff", "dashes": False, "width": 1.5}, "CONNECTED_TO": {"color": "#ff33a1", "dashes": [2, 2], "width": 1.5},
                "MET_WITH": {"color": "#f4f70a", "dashes": False, "width": 1.5}, "DEFAULT": {"color": "#B0B0B0", "dashes": False, "width": 1.0}
            }

            # Add edges
            for source, target, attrs in viz_graph.edges(data=True):
                 rel_type = attrs.get("type", "RELATED_TO")
                 description = attrs.get("description", "N/A")
                 edge_title = f"Type: {rel_type}\nDescription: {description}"
                 style = rel_type_styles.get(rel_type, rel_type_styles["DEFAULT"]).copy()
                 is_direct_connection = (source == center_id or target == center_id)
                 if is_direct_connection:
                     style['color'], style['width'], style['dashes'] = "#FF6A6A", max(style.get('width', 1.0), 2.0), False # Light red, thicker
                 net.add_edge(
                     source, target, title=edge_title, label="",
                     width=style.get('width'), color=style.get('color'), dashes=style.get('dashes'),
                     arrows={'to': {'enabled': True, 'scaleFactor': 0.6}}
                 )

            # Set options and display
            pyvis_options = _build_pyvis_options(physics_enabled, physics_solver, spring_length, spring_constant, central_gravity, grav_constant)
            net.set_options(json.dumps(pyvis_options))
            graph_html_path = ROOT_DIR / "temp" / "centered_graph.html"
            try:
                 net.save_graph(str(graph_html_path))
                 with open(graph_html_path, "r", encoding="utf-8") as f: html_content = f.read()
                 st.components.v1.html(html_content, height=710, scrolling=False)
                 st.caption(f"Displaying neighborhood: {viz_graph.number_of_nodes()} entities and {viz_graph.number_of_edges()} relationships within {connection_depth} hop(s) of '{center_entity_name}'.")
            except Exception as render_err:
                 st.error(f"Failed to render centered graph: {render_err}")
                 logger.error(f"PyVis rendering failed for centered graph: {traceback.format_exc()}")


# --- Network Metrics ---

# (Helper functions for metrics calculation remain largely the same, ensure they use the passed graphs/lookup)
def render_centrality_metrics(G: nx.DiGraph, UG: nx.Graph, entity_lookup: dict):
    """Renders the Centrality & Influence analysis sub-tab."""
    st.markdown("##### Identify Key Entities")
    st.markdown("""
    Centrality metrics help pinpoint influential nodes within the network.
    - **Degree (In/Out/Total):** Number of direct connections. High degree nodes are local hubs.
    - **Betweenness:** Measures how often a node lies on the shortest paths between other nodes. High betweenness nodes act as bridges or brokers.
    - **Eigenvector:** Measures influence based on connections to other influential nodes. High eigenvector nodes are connected to well-connected nodes.
    - **PageRank:** Google's algorithm, similar to Eigenvector, measures influence based on link structure.
    """)

    if G.number_of_nodes() == 0:
        st.warning("Graph is empty. Cannot calculate centrality.")
        return

    with st.spinner("Calculating centrality metrics..."):
        try:
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            # Use approximate betweenness for larger graphs if needed
            k_betweenness = min(100, G.number_of_nodes() // 2) if G.number_of_nodes() > 200 else None
            if k_betweenness is not None and k_betweenness <= 0: k_betweenness = None
            betweenness = nx.betweenness_centrality(G, k=k_betweenness, normalized=True, weight=None)

            eigenvector = {}
            if G.number_of_edges() > 0:
                 try:
                     # Calculate on largest weakly connected component for stability
                     largest_cc = max(nx.weakly_connected_components(G), key=len)
                     G_conn = G.subgraph(largest_cc)
                     if G_conn.number_of_nodes() > 1:
                         eigenvector_conn = nx.eigenvector_centrality_numpy(G_conn, max_iter=1000, tol=1e-03)
                         eigenvector = {node: eigenvector_conn.get(node, 0.0) for node in G.nodes()}
                     else: eigenvector = {node: 0.0 for node in G.nodes()}
                 except (nx.PowerIterationFailedConvergence, nx.NetworkXError, Exception) as eig_err:
                     logger.warning(f"Eigenvector centrality failed: {eig_err}. Assigning 0.")
                     eigenvector = {node: 0.0 for node in G.nodes()}
            else: eigenvector = {node: 0.0 for node in G.nodes()}

            pagerank = nx.pagerank(G, alpha=0.85)

            centrality_data = []
            for node_id in G.nodes():
                entity = entity_lookup.get(node_id)
                if entity:
                    entity_label = entity.get('label', f"Unknown {node_id[:4]}")
                    entity_type = entity.get('type', 'Unknown')
                else:
                    logger.warning(f"Node ID {node_id} found in graph but not in entity_lookup!")
                    entity_label, entity_type = f"Missing {node_id[:4]}", "Missing"

                in_d, out_d = in_degree.get(node_id, 0), out_degree.get(node_id, 0)
                centrality_data.append({
                    'Entity': entity_label, 'Type': entity_type,
                    'In-Degree': in_d, 'Out-Degree': out_d, 'Total Degree': in_d + out_d,
                    'Betweenness': round(betweenness.get(node_id, 0), 5),
                    'Eigenvector': round(eigenvector.get(node_id, 0), 5),
                    'PageRank': round(pagerank.get(node_id, 0), 5),
                    'Node ID': node_id
                })
            df_centrality = pd.DataFrame(centrality_data)

            if df_centrality.empty and G.number_of_nodes() > 0:
                 st.warning("Centrality calculations resulted in an empty dataframe.")
                 return

        except Exception as e:
            st.error(f"Failed to calculate centrality metrics: {e}")
            logger.error(f"Centrality calculation error: {traceback.format_exc()}")
            return

    # Display Key Players & Full Table
    st.markdown("**Key Players Summary (Top 5)**")
    if not df_centrality.empty:
        kp_col1, kp_col2, kp_col3 = st.columns(3)
        with kp_col1:
            top_degree = df_centrality.nlargest(5, 'Total Degree')
            st.metric("Highest Total Degree", top_degree.iloc[0]['Entity'] if not top_degree.empty else "N/A", f"{top_degree.iloc[0]['Total Degree'] if not top_degree.empty else 0} connections")
            st.dataframe(top_degree[['Entity', 'Type', 'Total Degree']], hide_index=True, use_container_width=True)
        with kp_col2:
            top_betweenness = df_centrality.nlargest(5, 'Betweenness')
            st.metric("Top Broker (Betweenness)", top_betweenness.iloc[0]['Entity'] if not top_betweenness.empty else "N/A", f"{top_betweenness.iloc[0]['Betweenness']:.3f}" if not top_betweenness.empty else "0.000")
            st.dataframe(top_betweenness[['Entity', 'Type', 'Betweenness']], hide_index=True, use_container_width=True, column_config={"Betweenness": st.column_config.NumberColumn(format="%.5f")})
        with kp_col3:
            top_pagerank = df_centrality.nlargest(5, 'PageRank')
            st.metric("Most Influential (PageRank)", top_pagerank.iloc[0]['Entity'] if not top_pagerank.empty else "N/A", f"{top_pagerank.iloc[0]['PageRank']:.3f}" if not top_pagerank.empty else "0.000")
            st.dataframe(top_pagerank[['Entity', 'Type', 'PageRank']], hide_index=True, use_container_width=True, column_config={"PageRank": st.column_config.NumberColumn(format="%.5f")})
    else: st.info("No centrality data to display.")

    st.markdown("**Full Centrality Data**")
    if not df_centrality.empty:
        filt_col1, filt_col2 = st.columns([2,1])
        with filt_col1: search_entity = st.text_input("Search Entity Name", key="cent_search")
        with filt_col2: min_degree = st.number_input("Min Total Degree", min_value=0, value=0, step=1, key="cent_min_degree")
        filtered_df = df_centrality[df_centrality['Total Degree'] >= min_degree]
        if search_entity:
            filtered_df = filtered_df[filtered_df['Entity'].astype(str).str.contains(search_entity, case=False, na=False)]
        st.dataframe(
            filtered_df.sort_values(by='Total Degree', ascending=False), hide_index=True, use_container_width=True,
            column_config={ "Betweenness": st.column_config.NumberColumn(format="%.5f"), "Eigenvector": st.column_config.NumberColumn(format="%.5f"), "PageRank": st.column_config.NumberColumn(format="%.5f"), "Node ID": None }
        )
        st.caption(f"Displaying {len(filtered_df)} of {len(df_centrality)} entities.")
    else: st.info("No centrality data to display.")


def render_community_metrics(UG: nx.Graph, entity_lookup: dict):
    """Renders the Communities & Groups analysis sub-tab."""
    if not louvain_available:
        st.warning("Community detection requires 'python-louvain'. Install with `pip install python-louvain`.")
        return

    st.markdown("##### Detect Cohesive Groups")
    st.markdown("""
    Community detection algorithms identify groups of nodes that are more densely connected internally than with the rest of the network.
    - **Louvain Method:** A popular algorithm for finding high-modularity partitions.
    - **Modularity:** Score indicating the quality of the detected community structure (higher is generally better, > 0.3).
    """)

    if UG.number_of_nodes() == 0:
        st.warning("Graph is empty. Cannot detect communities.")
        return

    with st.spinner("Detecting communities using Louvain method..."):
        try:
            # Use MultiGraph directly if community_louvain supports it, otherwise convert
            # partition = community_louvain.best_partition(UG) # Try direct first
            # If error, convert to simple graph:
            UG_simple = nx.Graph(UG)
            partition = community_louvain.best_partition(UG_simple)
            modularity = community_louvain.modularity(partition, UG_simple)

            community_sizes = Counter(partition.values())
            num_communities = len(community_sizes)
            st.success(f"Detected {num_communities} communities with a modularity score of {modularity:.4f}.")

            community_data = []
            for node_id, comm_id in partition.items():
                entity = entity_lookup.get(node_id, {})
                community_data.append({
                    'Entity': entity.get('label', 'Unknown'), 'Type': entity.get('type', 'Unknown'),
                    'Community ID': comm_id, 'Community Size': community_sizes[comm_id], 'Node ID': node_id
                })
            df_communities = pd.DataFrame(community_data)

        except Exception as e:
            st.error(f"Failed to detect communities: {e}")
            logger.error(f"Community detection error: {traceback.format_exc()}")
            return

    # Display Community Summary
    st.markdown("**Community Overview**")
    min_comm_size_display = st.slider("Min Community Size to Display", min_value=1, max_value=max(50, max(community_sizes.values()) // 2 if community_sizes else 50), value=max(3, min(5, max(community_sizes.values()) if community_sizes else 5)), key="comm_min_size")

    summary_data = []
    displayed_community_ids = set()
    for comm_id, size in sorted(community_sizes.items(), key=lambda item: item[1], reverse=True):
         if size >= min_comm_size_display:
            displayed_community_ids.add(comm_id)
            comm_nodes = df_communities[df_communities['Community ID'] == comm_id]
            type_counts = Counter(comm_nodes['Type'])
            most_common_types = ", ".join([f"{t} ({c})" for t, c in type_counts.most_common(3)])

            # Find most central node within the community (using degree within community subgraph)
            subgraph = UG.subgraph(comm_nodes['Node ID'].tolist())
            central_entity_name = "N/A"
            if subgraph.number_of_nodes() > 0:
                degrees_in_subgraph = dict(subgraph.degree())
                if degrees_in_subgraph:
                    central_node_id = max(degrees_in_subgraph, key=degrees_in_subgraph.get)
                    central_entity_name = entity_lookup.get(central_node_id, {}).get('label', 'Unknown')
                else: central_entity_name = "N/A (isolated)"

            summary_data.append({
                 'ID': comm_id, 'Size': size, 'Top Types': most_common_types,
                 'Most Connected Internal Node': central_entity_name
            })

    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

    # Explore Specific Community
    st.markdown("**Explore Community Members**")
    if displayed_community_ids:
        selected_comm_id = st.selectbox("Select Community ID", options=sorted(list(displayed_community_ids)), key="comm_select")
        if selected_comm_id is not None:
             members_df = df_communities[df_communities['Community ID'] == selected_comm_id][['Entity', 'Type']].sort_values(by='Type')
             st.dataframe(members_df, hide_index=True, use_container_width=True)
    else: st.info("No communities large enough to display based on the current size filter.")


def render_link_prediction_hints(UG: nx.Graph, entity_lookup: dict):
    """Renders the Link Prediction Hints sub-tab."""
    st.markdown("##### Suggest Potential Connections")
    st.markdown("""
    Link prediction algorithms suggest pairs of nodes that are *not* currently connected but are likely to be, based on the network structure.
    - **Adamic-Adar Index:** Predicts links based on shared neighbors, weighting rarer neighbors more heavily.
    """)

    num_nodes = UG.number_of_nodes()
    limit_nodes = False
    if num_nodes > 1500:
        st.warning(f"Graph has {num_nodes} nodes. Link prediction calculation is limited to top 1500 nodes by degree.")
        limit_nodes = True

    nodes_to_consider = list(UG.nodes())
    if limit_nodes:
        degrees = dict(UG.degree())
        nodes_to_consider = sorted(degrees, key=degrees.get, reverse=True)[:1500]

    num_suggestions = st.slider("Number of Potential Links to Suggest", min_value=10, max_value=200, value=50, step=10, key="lp_num")

    if st.button("Calculate Potential Links", key="lp_calc_btn"):
        with st.spinner("Calculating Adamic-Adar scores..."):
            try:
                # Convert MultiGraph to simple Graph for Adamic-Adar
                UG_simple = nx.Graph()
                UG_simple.add_nodes_from(UG.nodes())
                # Add edges without parallels for calculation
                simple_edges = set()
                for u, v in UG.edges():
                    # Ensure consistent order for undirected edges
                    edge = tuple(sorted((u, v)))
                    simple_edges.add(edge)
                UG_simple.add_edges_from(list(simple_edges))

                # Generate pairs of non-connected nodes within the considered set
                potential_pairs = []
                considered_set = set(nodes_to_consider)
                for i, u in enumerate(nodes_to_consider):
                    for j in range(i + 1, len(nodes_to_consider)):
                        v = nodes_to_consider[j]
                        if not UG_simple.has_edge(u, v):
                            potential_pairs.append((u, v))

                if not potential_pairs:
                    st.info("No potential links to evaluate (graph might be fully connected or too small).")
                    if 'link_prediction_results' in st.session_state: del st.session_state.link_prediction_results
                    return

                predictions = nx.adamic_adar_index(UG_simple, potential_pairs)

                link_suggestions = []
                for u, v, score in predictions:
                    entity_u = entity_lookup.get(u, {})
                    entity_v = entity_lookup.get(v, {})
                    link_suggestions.append({
                        'Entity 1': entity_u.get('label', 'Unknown'), 'Type 1': entity_u.get('type', 'Unknown'),
                        'Entity 2': entity_v.get('label', 'Unknown'), 'Type 2': entity_v.get('type', 'Unknown'),
                        'Adamic-Adar Score': round(score, 4), 'Node ID 1': u, 'Node ID 2': v
                    })

                df_suggestions = pd.DataFrame(link_suggestions).nlargest(num_suggestions, 'Adamic-Adar Score')
                st.session_state.link_prediction_results = df_suggestions
                st.success(f"Calculated scores. Showing top {min(num_suggestions, len(df_suggestions))} potential links.")

            except Exception as e:
                st.error(f"Failed to calculate link prediction scores: {e}")
                logger.error(f"Link prediction error: {traceback.format_exc()}")
                if 'link_prediction_results' in st.session_state: del st.session_state.link_prediction_results

    # Display Results
    if 'link_prediction_results' in st.session_state:
        df_results = st.session_state.link_prediction_results
        st.markdown("**Top Potential Links (Higher score suggests higher likelihood)**")
        st.dataframe(
            df_results[['Entity 1', 'Type 1', 'Entity 2', 'Type 2', 'Adamic-Adar Score']],
            hide_index=True, use_container_width=True,
            column_config={"Adamic-Adar Score": st.column_config.NumberColumn(format="%.4f")}
        )
    else: st.info("Click 'Calculate Potential Links' to generate suggestions.")


def render_node_similarity(UG: nx.Graph, entity_lookup: dict):
    """Renders the Node Similarity (Embeddings) sub-tab using the 'node2vec' library."""
    if not node2vec_available:
        st.warning("Node similarity requires the 'node2vec' library. Install with `pip install node2vec`.")
        return

    st.markdown("##### Find Structurally Similar Entities")
    st.markdown("""
    Node Embeddings learn vector representations of entities based on their network neighborhood. Entities with similar vectors often play similar roles. This uses Node2Vec.
    - **Node2Vec:** Learns embeddings by simulating random walks.
    - **Cosine Similarity:** Measures similarity between entity vectors (closer to 1 is more similar).
    """)

    if UG.number_of_nodes() < 5:
         st.warning("Graph is too small (< 5 nodes) for meaningful Node2Vec training.")
         return

    # Node2Vec Parameters
    st.markdown("**Node2Vec Parameters**")
    n2v_col1, n2v_col2, n2v_col3 = st.columns(3)
    with n2v_col1:
         n2v_dims = st.slider("Embedding Dimensions", 16, 128, 64, 16, key="n2v_dims")
         n2v_walk_len = st.slider("Walk Length", 10, 80, 30, 5, key="n2v_walklen")
    with n2v_col2:
         n2v_num_walks = st.slider("Walks / Node", 10, 200, 50, 10, key="n2v_numwalks")
         n2v_window = st.slider("Window Size", 2, 10, 5, 1, key="n2v_window")
    # with n2v_col3: # Placeholder for future params like p, q, workers

    # Similarity Search
    st.markdown("**Similarity Search**")
    search_col1, search_col2 = st.columns([3,1])
    with search_col1:
         entity_names = sorted([data['label'] for _, data in UG.nodes(data=True) if data and 'label' in data])
         if not entity_names:
              st.warning("No entity names found in the graph nodes.")
              return
         target_entity_name = st.selectbox("Find entities similar to:", options=entity_names, key="n2v_target")
    with search_col2:
         top_n_similar = st.number_input("Number of results", 1, 50, 10, 1, key="n2v_topn")

    if st.button("Find Similar Entities", key="n2v_find_btn", type="primary"):
        target_node_id = None
        for node_id, data in UG.nodes(data=True):
            if data and data.get('label') == target_entity_name:
                target_node_id = node_id
                break
        if target_node_id is None:
            st.error(f"Could not find node ID for entity: {target_entity_name}")
            return

        with st.spinner(f"Calculating embeddings and finding entities similar to '{target_entity_name}'..."):
            # Use cached training function
            wv, node_map = train_node2vec_model(
                UG, dimensions=n2v_dims, walk_length=n2v_walk_len,
                num_walks=n2v_num_walks, window=n2v_window, workers=4 # Default workers
            )

            if wv is None or node_map is None:
                 st.error("Failed to get Node2Vec model. Cannot perform similarity search.")
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
                 return

            # Find the string representation of the target node used in the model
            target_node_str = node_map.get(target_node_id)

            if target_node_str is None or target_node_str not in wv:
                st.error(f"'{target_entity_name}' (ID: {target_node_id}) not found in the Node2Vec model vocabulary. It might be isolated or training failed.")
                logger.warning(f"Node ID {target_node_id} mapped to '{target_node_str}', which is not in WV keys. WV keys sample: {list(wv.index_to_key[:10])}...")
                if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
                return

            try:
                 similar_nodes = wv.most_similar(target_node_str, topn=top_n_similar + 5)
                 similarity_results = []
                 reverse_node_map = {v: k for k, v in node_map.items()}
                 count = 0
                 for node_str, similarity_score in similar_nodes:
                     if count >= top_n_similar: break
                     original_id = reverse_node_map.get(node_str)
                     if original_id == target_node_id: continue # Skip self
                     if original_id is None: continue # Skip if mapping fails

                     if original_id in entity_lookup:
                         entity_data = entity_lookup[original_id]
                         similarity_results.append({
                             'Similar Entity': entity_data.get('label', 'Unknown'),
                             'Type': entity_data.get('type', 'Unknown'),
                             'Similarity Score': round(similarity_score, 4),
                             'Node ID': original_id
                         })
                         count += 1

                 df_similar = pd.DataFrame(similarity_results)
                 st.session_state.node_similarity_results = df_similar
                 st.session_state.node_similarity_target = target_entity_name
                 st.success(f"Found {len(df_similar)} entities similar to '{target_entity_name}'.")

            except KeyError as ke:
                 st.error(f"Entity key '{target_node_str}' not found in the embedding model's vocabulary: {ke}")
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results
            except Exception as e:
                 st.error(f"An error occurred during similarity search: {e}")
                 logger.error(f"Node similarity search error: {traceback.format_exc()}")
                 if 'node_similarity_results' in st.session_state: del st.session_state.node_similarity_results

    # Display Results
    if 'node_similarity_results' in st.session_state:
        df_results = st.session_state.node_similarity_results
        target_display_name = st.session_state.get('node_similarity_target', 'the selected entity')
        st.markdown(f"**Entities Structurally Similar to '{target_display_name}'**")
        if not df_results.empty:
            st.dataframe(
                df_results[['Similar Entity', 'Type', 'Similarity Score']], hide_index=True, use_container_width=True,
                column_config={ "Similarity Score": st.column_config.NumberColumn(format="%.4f")}
            )
        else: st.info(f"No similar entities found for '{target_display_name}'.")
    else: st.info("Select an entity and click 'Find Similar Entities'.")


def render_network_metrics_tab(entities, relationships):
    """
    Render the network metrics analysis tab with sub-tabs for different metric categories.
    (Copied and adapted from original app.py)
    """
    st.markdown("#### Advanced Network Analysis")
    st.markdown("""
    Dive deeper into the network structure with quantitative metrics. Identify key players,
    cohesive groups, potential hidden connections, and structural anomalies.
    """)

    if not entities or not relationships:
        st.info("Insufficient data for network metrics calculation (need both entities and relationships).")
        return

    # Use caching for graph building
    @st.cache_data(ttl=3600)
    def build_analysis_graphs(_entities, _relationships):
        logger.info("Building NetworkX multi-graphs for analysis...")
        G = nx.MultiDiGraph()
        UG = nx.MultiGraph()
        entity_lookup = {}
        valid_entity_count = 0
        for entity in _entities:
            e_id = entity.get("id")
            if e_id:
                entity_data = {"id": e_id, "label": entity.get("name", f"Unknown {e_id[:4]}"), "type": entity.get("type", "Unknown")}
                entity_data.update(entity) # Include all original entity data
                entity_lookup[e_id] = entity_data
                valid_entity_count += 1
        logger.info(f"Created entity lookup with {valid_entity_count} valid entities.")

        # Add nodes
        for e_id, entity_data in entity_lookup.items():
            node_attrs = {"label": entity_data["label"], "type": entity_data["type"]}
            if not G.has_node(e_id): G.add_node(e_id, **node_attrs)
            if not UG.has_node(e_id): UG.add_node(e_id, **node_attrs)

        # Add edges
        valid_edge_count, skipped_edge_count = 0, 0
        for rel in _relationships:
            s_id = rel.get("source_entity_id", rel.get("from_entity_id"))
            t_id = rel.get("target_entity_id", rel.get("to_entity_id"))
            if s_id in entity_lookup and t_id in entity_lookup:
                r_type = rel.get("type", rel.get("relationship_type"))
                if r_type and isinstance(r_type, str) and r_type.strip() and r_type.upper() != "UNKNOWN":
                    edge_attrs = {"type": r_type, "description": rel.get("description")}
                    rel_id_key = rel.get("id", str(uuid.uuid4()))
                    G.add_edge(s_id, t_id, key=rel_id_key, **edge_attrs)
                    UG.add_edge(s_id, t_id, key=rel_id_key, **edge_attrs)
                    valid_edge_count += 1
                else: skipped_edge_count += 1
            else: skipped_edge_count += 1
        logger.info(f"Analysis graphs built: Directed ({G.number_of_nodes()}N, {G.number_of_edges()}E), Undirected ({UG.number_of_nodes()}N, {UG.number_of_edges()}E). Skipped {skipped_edge_count} edges.")
        return G, UG, entity_lookup

    try:
        # Build graphs using caching
        G, UG, entity_lookup = build_analysis_graphs(tuple(entities), tuple(relationships)) # Use tuples for caching

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
             st.warning("The filtered data resulted in an empty graph. Cannot calculate metrics.")
             return

        # Create Sub-Tabs for Metrics
        centrality_tab, community_tab, links_tab, similarity_tab = st.tabs([
            "👑 Centrality & Influence", "👥 Communities & Groups",
            "🔮 Link Prediction Hints", "🤝 Node Similarity"
        ])

        with centrality_tab: render_centrality_metrics(G, UG, entity_lookup)
        with community_tab: render_community_metrics(UG, entity_lookup) # Use undirected
        with links_tab: render_link_prediction_hints(UG, entity_lookup) # Use undirected
        with similarity_tab: render_node_similarity(UG, entity_lookup) # Use undirected

    except Exception as e:
        st.error(f"An error occurred during network metrics calculation: {e}")
        logger.error(f"Network metrics calculation failed: {traceback.format_exc()}")