import dash
from dash import Input, Output, State, callback_context, html, dcc
import numpy as np
import plotly.graph_objs as go
from .projection import _interpolate_hyperbolic

from .image_utils import _encode_image, _create_content_element
import json


def _compute_hyperbolic_distances(point, all_points, curv=1.0):
    """
    Compute hyperbolic distances from a point to all points using Lorentzian metric.
    
    Args:
        point: numpy array of shape (D,) -a the reference point
        all_points: numpy array of shape (N, D) - all points
        curv: curvature parameter (default 1.0)
    
    Returns:
        numpy array of shape (N,) with hyperbolic distances
    """
    # Compute time coordinates for Lorentz model
    point_time = np.sqrt(1.0 / curv + np.sum(point**2))
    all_times = np.sqrt(1.0 / curv + np.sum(all_points**2, axis=1))
    
    # Compute Lorentzian inner product: <x, y>_L = x·y - x_time * y_time
    spatial_product = np.dot(all_points, point)
    lorentzian_inner = spatial_product - point_time * all_times
    
    # Hyperbolic distance: d(x,y) = acosh(-<x,y>_L)
    # Clamp to avoid numerical issues with acosh
    lorentzian_inner = np.clip(lorentzian_inner, None, -1.0)
    distances = np.arccosh(-lorentzian_inner)
    
    return distances

#-------------------------------------------------------------------------------------
def _create_hover_text(idx, points=None, meta=None):
    """Create meaningful hover text for a point based on its content."""
    # If we have points data, use the embedding type to show relevant content
    if points and idx < len(points):
        point = points[idx]
        embedding_type = point.get("embedding_type", "")
        synset_id = point.get("synset_id", "")
        
        # For text embeddings, show the actual text content
        if embedding_type in ["parent_text", "child_text"]:
            if meta and synset_id in meta:
                if embedding_type == "parent_text":
                    text_content = meta[synset_id].get("name", "No parent text available")
                    return f"Parent Text: {text_content[:100]}{'...' if len(text_content) > 100 else ''}"
                else:  # child_text
                    text_content = meta[synset_id].get("description", "No child text available")
                    return f"Child Text: {text_content[:100]}{'...' if len(text_content) > 100 else ''}"
        
        # For image embeddings, show image type and synset info
        elif embedding_type in ["parent_image", "child_image"]:
            if meta and synset_id in meta:
                synset_name = meta[synset_id].get("name", "Unknown")
                return f"{embedding_type.replace('_', ' ').title()}: {synset_name}"
    
    # Fallback - just show the index
    return f"Point {idx}"

# All callback functions and register_callbacks go here
# ... (move all @app.callback functions and register_callbacks from main.py) ... 
#-------------------------------------------------------------------------------------

def _create_simple_scatter(x, y, labels, target_names, emb_labels, title, points=None, meta=None):
    """Create a simple scatter plot without interactions for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'parent_text':  '#e41a1c',   # red
    'child_text':   '#377eb8',   # blue  
    'child_image':  '#4daf4a',   # green
    'parent_image': '#984ea3',   # purple
    }
    
    traces = []
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                x_coords = [x[i] for i in indices]
                y_coords = [y[i] for i in indices]
                hover_text = [
                    _create_hover_text(i, points, meta)
                    for i in indices
                ]
                
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    marker=dict(
                        size=6, 
                        opacity=0.7, 
                        color=colors.get(label_type, 'gray'),
                        line=dict(width=0.5, color='black')
                    ),
                    name=label_type.replace('_', ' ').title(),
                    showlegend=True,
                )
                traces.append(trace)
    else:
        # Fallback to single trace with colorscale
        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=[
                _create_hover_text(i, points, meta)
                for i in range(len(x))
            ],
            hoverinfo="text",
            marker=dict(size=6, opacity=0.7, color=labels, colorscale="Viridis"),
            name="Data points",
            showlegend=False,
        )
        traces = [trace]
    
    fig = go.Figure(data=traces)
    fig.update_layout(
            title=title,
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            margin=dict(l=0, r=0, b=60, t=40),
            uirevision="embedding",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            dragmode='pan',
        )
    return fig

#-------------------------------------------------------------------------------------
def _create_interactive_scatter(x, y, labels, target_names, emb_labels, title, sel, points=None, meta=None):
    """Create an interactive scatter plot with selection highlighting for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'parent_text':  '#e41a1c',   # red
    'child_text':   '#377eb8',   # blue  
    'child_image':  '#4daf4a',   # green
    'parent_image': '#984ea3',   # purple
    }
    
    traces = []
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                x_coords = [x[i] for i in indices]
                y_coords = [y[i] for i in indices]
                hover_text = [
                    _create_hover_text(i, points, meta)
                    for i in indices
                ]
                
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    text=hover_text,
                    hoverinfo="text",
                    customdata=indices,  # Store original indices for clicking
                    marker=dict(
                        size=8, 
                        opacity=0.7, 
                        color=colors.get(label_type, 'gray'),
                        line=dict(width=0.5, color='black')
                    ),
                    name=label_type.replace('_', ' ').title(),
                    showlegend=True,
                )
                traces.append(trace)
    else:
        # Fallback to single trace with colorscale
        trace = go.Scatter(
            x=x,
            y=y,
            mode="markers",
            text=[
                _create_hover_text(i, points, meta)
                for i in range(len(x))
            ],
            hoverinfo="text",
            customdata=list(range(len(x))),  # Store original indices for clicking
            marker=dict(size=8, opacity=0.7, color=labels, colorscale="Viridis"),
            name="Data points",
            showlegend=False,
        )
        traces = [trace]
    
    # Add selected points as a separate trace
    if sel:
        selected_x = [x[i] for i in sel if i < len(x)]
        selected_y = [y[i] for i in sel if i < len(x)]
        
        if selected_x:
            selected_trace = go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=3)),
                name="Selected points",
                showlegend=False,
                hoverinfo="skip",
            )
            traces.append(selected_trace)
    
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=60, t=40),
        uirevision="embedding",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        dragmode='pan',
    )
    return fig


#-------------------------------------------------------------------------------------
def _create_full_interactive_scatter(x, y, labels, target_names, emb_labels, title, sel, neighbor_indices, tree_connections, interp_point, mode, points=None, meta=None):
    """Create a full interactive scatter plot with all mode features for comparison views"""
    # Define colors matching plotting_utils.py
    colors = {
        'parent_text':  '#e41a1c',   # red
    'child_text':   '#377eb8',   # blue  
    'child_image':  '#4daf4a',   # green
    'parent_image': '#984ea3',   # purple
    }
    
    traces = []
    neighbor_set = set(neighbor_indices) if neighbor_indices is not None else set()
    
    # Add interpolated path FIRST (like single mode) so colored points appear on top
    if interp_point is not None:
        if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
            # Multiple points forming a path - match single mode styling exactly
            # Plot orange diamond markers for all traversal points
            traces.append(
                go.Scatter(
                    x=interp_point[:, 0],
                    y=interp_point[:, 1],
                    mode="markers",
                    marker=dict(size=12, color="orange", symbol="diamond"),
                    name="Traversal Path Points",
                    text=[f"Traversal {i}" for i in range(len(interp_point))],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            # Plot a dashed orange line between every two adjacent points
            for i in range(len(interp_point) - 1):
                traces.append(
                    go.Scatter(
                        x=[interp_point[i, 0], interp_point[i+1, 0]],
                        y=[interp_point[i, 1], interp_point[i+1, 1]],
                        mode="lines",
                        line=dict(color="orange", width=2, dash="dash"),
                        name="Traversal Segment" if i == 0 else None,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        else:
            # Single interpolated point
            traces.append(
                go.Scatter(
                    x=[interp_point[0]],
                    y=[interp_point[1]],
                    mode="markers",
                    marker=dict(size=12, color="orange", symbol="diamond"),
                    name="Interpolated point",
                    text=["Interpolated point"],
                    hoverinfo="text",
                    showlegend=False,
                )
            )
    
    # If we have emb_labels, create separate traces for each label type
    if emb_labels and len(emb_labels) == len(x):
        unique_label_types = sorted(set(emb_labels))
        
        for label_type in unique_label_types:
            # Find indices for this label type
            indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
            
            if indices:
                # Separate regular points and neighbor points
                regular_indices = [i for i in indices if i not in neighbor_set]
                neighbor_indices_for_type = [i for i in indices if i in neighbor_set]
                
                # Regular points trace
                if regular_indices:
                    x_coords = [x[i] for i in regular_indices]
                    y_coords = [y[i] for i in regular_indices]
                    hover_text = [
                        f"{i}"
                        for i in regular_indices
                    ]
                    
                    trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="markers",
                        text=hover_text,
                        hoverinfo="text",
                        customdata=regular_indices,  # Store original indices for clicking
                        marker=dict(
                            size=8, 
                            opacity=0.7, 
                            color=colors.get(label_type, 'gray'),
                            line=dict(width=0.5, color='black')
                        ),
                        name=label_type.replace('_', ' ').title(),
                        showlegend=True,
                    )
                    traces.append(trace)
                
                # Neighbor points trace (larger and brighter)
                if neighbor_indices_for_type:
                    x_coords_neighbors = [x[i] for i in neighbor_indices_for_type]
                    y_coords_neighbors = [y[i] for i in neighbor_indices_for_type]
                    hover_text_neighbors = [
                        f"{i} (neighbor)"
                        for i in neighbor_indices_for_type
                    ]
                    
                    neighbor_trace = go.Scatter(
                        x=x_coords_neighbors,
                        y=y_coords_neighbors,
                        mode="markers",
                        text=hover_text_neighbors,
                        hoverinfo="text",
                        customdata=neighbor_indices_for_type,
                        marker=dict(
                            size=12,  # Larger size for neighbors
                            opacity=1.0,  # Full opacity for neighbors
                            color=colors.get(label_type, 'gray'),
                            line=dict(width=2, color='purple')  # Purple border to make them stand out
                        ),
                        name=f"{label_type.replace('_', ' ').title()} (Neighbors)",
                        showlegend=False,  # Don't show in legend to avoid clutter
                    )
                    traces.append(neighbor_trace)
    else:
        # Fallback to single trace with colorscale
        regular_indices = [i for i in range(len(x)) if i not in neighbor_set]
        neighbor_indices_list = [i for i in range(len(x)) if i in neighbor_set]
        
        # Regular points trace
        if regular_indices:
            base = go.Scatter(
                x=[x[i] for i in regular_indices],
                y=[y[i] for i in regular_indices],
                mode="markers",
                text=[
                    f"{i}"
                    for i in regular_indices
                ],
                hoverinfo="text",
                customdata=regular_indices,
                marker=dict(size=8, opacity=0.7, color=[labels[i] for i in regular_indices], colorscale="Viridis"),
                name="Data points",
                showlegend=False,
            )
            traces = [base]
        else:
            traces = []
        
        # Neighbor points trace (larger and brighter)
        if neighbor_indices_list:
            neighbor_trace = go.Scatter(
                x=[x[i] for i in neighbor_indices_list],
                y=[y[i] for i in neighbor_indices_list],
                mode="markers",
                text=[
                    f"{i} (neighbor)"
                    for i in neighbor_indices_list
                ],
                hoverinfo="text",
                customdata=neighbor_indices_list,
                marker=dict(
                    size=12,  # Larger size for neighbors
                    opacity=1.0,  # Full opacity for neighbors
                    color=[labels[i] for i in neighbor_indices_list], 
                    colorscale="Viridis",
                    line=dict(width=2, color='purple')  # Purple border to make them stand out
                ),
                name="Neighbors",
                showlegend=False,
            )
            traces.append(neighbor_trace)

    # Store tree connections for later arrow annotation (like single view)
    tree_arrow_annotations = []
    if tree_connections and points:
        for conn in tree_connections:
            idx1, idx2 = conn
            if idx1 < len(x) and idx2 < len(x):
                # Create line trace
                x1, y1 = x[idx1], y[idx1]
                x2, y2 = x[idx2], y[idx2]
                
                line_trace = go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(color="gold", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Tree connections"
                )
                traces.append(line_trace)
                
                # Store arrow annotation info (same as single view)
                tree_arrow_annotations.append({
                    'x': x2, 'y': y2,
                    'ax': x1, 'ay': y1,
                    'xref': 'x', 'yref': 'y',
                    'axref': 'x', 'ayref': 'y',
                    'arrowhead': 2,
                    'arrowsize': 1.5,
                    'arrowwidth': 2,
                    'arrowcolor': 'gold',
                    'showarrow': True,
                    'text': '',
                })

    # Add selected points as a separate trace
    if sel:
        selected_x = [x[i] for i in sel if i < len(x)]
        selected_y = [y[i] for i in sel if i < len(x)]
        
        if selected_x:
            selected_trace = go.Scatter(
                x=selected_x,
                y=selected_y,
                mode="markers",
                marker=dict(size=12, color="red", symbol="circle-open", line=dict(width=3)),
                name="Selected points",
                showlegend=False,
                hoverinfo="skip",
            )
            traces.append(selected_trace)
    
    # Calculate the maximum distance from origin to any point for boundary circle
    max_distance = 0
    for trace in traces:
        if hasattr(trace, 'x') and hasattr(trace, 'y') and len(trace.x) > 0 and len(trace.y) > 0:
            distances = np.sqrt(np.array(trace.x)**2 + np.array(trace.y)**2)
            max_distance = max(max_distance, np.max(distances))
    
    # Add boundary circle if we have data points
    if max_distance > 0:
        circle_radius = 1.1 * max_distance
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = circle_radius * np.cos(theta)
        circle_y = circle_radius * np.sin(theta)
        
        # Add circle trace (make sure it's first so it renders behind other traces)
        circle_trace = go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        )
        traces.insert(0, circle_trace)

    fig = go.Figure(data=traces)
    
    # Add arrow annotations for tree connections (same as single view)
    annotations = tree_arrow_annotations
    
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, b=60, t=40),
        uirevision="embedding",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        annotations=annotations,
        dragmode='pan',
    )
    return fig


#-------------------------------------------------------------------------------------
def register_callbacks(app: dash.Dash) -> None:
    CONE_COLORS = [
        {"fill": "rgba(230, 126, 34, 0.20)", "line": "rgba(230, 126, 34, 0.9)"},
        {"fill": "rgba(211, 84,   0, 0.20)", "line": "rgba(211, 84,   0, 0.9)"},
        {"fill": "rgba(243, 156, 18, 0.20)", "line": "rgba(243, 156, 18, 0.9)"},
        {"fill": "rgba(192, 57,  43, 0.20)", "line": "rgba(192, 57,  43, 0.9)"},
        {"fill": "rgba(127, 40,   0, 0.20)", "line": "rgba(127, 40,   0, 0.9)"},
    ]
    # Dataset loading callback
    @app.callback(
        Output("data-store", "data"),
        Output("labels-store", "data"),
        Output("feature-names-store", "data"),
        Output("target-names-store", "data"),
        Output("images-store", "data"),
        Output("meta-store", "data"),
        Output("points-store", "data"),
        Output("sel", "data", allow_duplicate=True),
        Output("interpolated-point", "data", allow_duplicate=True),
        Output("emb", "data", allow_duplicate=True),
        Input("dataset-dropdown", "value"),
        Input("proj", "data"),
        State("sel", "data"),
        prevent_initial_call=False,
    )
    def _update_dataset_stores(dataset_name, projection_method, current_sel):
        if not dataset_name or not projection_method:
            return dash.no_update
        if dataset_name == "imagenet":
            try:
                import pickle
                with open("hierchical_datasets/ImageNet/meta_data_trees.json", "r", encoding="utf-8") as f_meta:
                    meta_data_trees = json.load(f_meta)
                
                # Load embeddings based on selected projection method
                emb_file = f"hierchical_datasets/ImageNet/{projection_method}_embeddings.pkl"
                with open(emb_file, "rb") as f_emb:
                    emb_data = pickle.load(f_emb)
            except FileNotFoundError as e:
                print(f"Error loading ImageNet files: {e}")
                return dash.no_update
            except Exception as e:
                print(f"ERROR: ImageNet loading failed: {e}")
                return dash.no_update
            
            # Extract embeddings
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            print(f"Loaded ImageNet with {projection_method}: {embeddings.shape} embeddings")
            
            # Create points list from embeddings, grouping them properly by tree
            points = []
            synset_ids = []
            images_list = []
            
            # Get the embedding labels to understand what each embedding represents
            embedding_labels = emb_data.get("labels", [])
            
            # Create a mapping from tree components to actual trees
            # Distribute embeddings to ensure each tree has different types
            trees_list = list(meta_data_trees["trees"].items())
            
            # Group embeddings by type first
            embeddings_by_type = {}
            for i, label in enumerate(embedding_labels):
                if label not in embeddings_by_type:
                    embeddings_by_type[label] = []
                embeddings_by_type[label].append(i)
            
            # Distribute each type across trees to create mixed trees
            # Use the number of parent_text embeddings to determine how many trees we need
            num_trees = len(embeddings_by_type.get('parent_text', []))
            
            for i, (embedding_label) in enumerate(embedding_labels):
                # For each embedding type, distribute across trees
                type_embeddings = embeddings_by_type[embedding_label]
                position_in_type = type_embeddings.index(i)
                tree_idx = position_in_type % num_trees
                tree_id, tree_data = trees_list[tree_idx]
                
                synset_id = tree_data["synset_id"]
                synset_ids.append(synset_id)
                
                # Get appropriate image path based on embedding type
                image_path = None
                if embedding_label == "child_image" and "child_images" in tree_data and tree_data["child_images"]:
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label in ["parent_text", "child_text"]:
                    # For text embeddings, still use child image for display
                    if "child_images" in tree_data and tree_data["child_images"]:
                        original_path = tree_data["child_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": synset_id,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree",
                    "embedding_type": embedding_label
                })
                images_list.append(image_path)
            
            # Create meta dict for compatibility
            meta = {}
            for tree_id, tree_data in meta_data_trees["trees"].items():
                synset_id = tree_data["synset_id"]
                # Fix the image path here too
                first_image_path = None
                if tree_data.get("child_images"):
                    original_path = tree_data["child_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                
                meta[synset_id] = {
                    "name": tree_data["parent_text"]["text"],
                    "description": tree_data["child_text"]["text"],
                    "first_image_path": first_image_path
                }
            
            unique_synsets = sorted({sid for sid in synset_ids})
            syn_to_int = {sid: i for i, sid in enumerate(unique_synsets)}
            labels = np.array([syn_to_int[s] for s in synset_ids], dtype=int)
            feature_names = [f"dim{i}" for i in range(embeddings.shape[1])]
            target_names = unique_synsets
            
            return (
                embeddings.tolist(),  # data-store - needed for neighbor calculations
                labels.tolist(),
                feature_names,
                target_names,
                images_list,
                meta,
                points,
                current_sel or [],  # Preserve current selection
                None,
                embeddings.tolist(),  # emb store - this is what the scatter callback needs!
            )
        if dataset_name == "grit":
            try:
                import pickle
                with open("hierchical_datasets/GRIT/meta_data_trees.json", "r", encoding="utf-8") as f_meta:
                    meta_data_trees = json.load(f_meta)
                
                # Load embeddings based on selected projection method
                emb_file = f"hierchical_datasets/GRIT/{projection_method}_embeddings.pkl"
                with open(emb_file, "rb") as f_emb:
                    emb_data = pickle.load(f_emb)
            except FileNotFoundError as e:
                print(f"Error loading GRIT files: {e}")
                return dash.no_update
            except Exception as e:
                print(f"ERROR: GRIT loading failed: {e}")
                return dash.no_update
            
            # Extract embeddings
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            print(f"Loaded GRIT with {projection_method}: {embeddings.shape} embeddings")
            
            # Create points list from embeddings, grouping them properly by tree
            points = []
            synset_ids = []
            images_list = []
            
            # Get the embedding labels to understand what each embedding represents
            embedding_labels = emb_data.get("labels", [])
            
            # Create a mapping from tree components to actual trees
            # Distribute embeddings to ensure each tree has different types
            trees_list = list(meta_data_trees["trees"].items())
            
            # Group embeddings by type first
            embeddings_by_type = {}
            for i, label in enumerate(embedding_labels):
                if label not in embeddings_by_type:
                    embeddings_by_type[label] = []
                embeddings_by_type[label].append(i)
            
            # Distribute each type across trees to create mixed trees
            # Use the number of parent_text embeddings to determine how many trees we need
            num_trees = len(embeddings_by_type.get('parent_text', []))
            
            for i, (embedding_label) in enumerate(embedding_labels):
                # For each embedding type, distribute across trees
                type_embeddings = embeddings_by_type[embedding_label]
                position_in_type = type_embeddings.index(i)
                tree_idx = position_in_type % num_trees
                tree_id, tree_data = trees_list[tree_idx]
                
                # GRIT uses sample_key instead of synset_id
                sample_key = tree_data.get("sample_key", tree_id)
                synset_ids.append(sample_key)
                
                # Get appropriate image path based on embedding type
                image_path = None
                if embedding_label == "child_image" and "child_images" in tree_data and tree_data["child_images"]:
                    original_path = tree_data["child_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label == "parent_image" and "parent_images" in tree_data and tree_data["parent_images"]:
                    original_path = tree_data["parent_images"][0]["path"]
                    image_path = original_path.replace("/data/", "/trees/")
                elif embedding_label in ["parent_text", "child_text"]:
                    # For text embeddings, use child image for display if available
                    if "child_images" in tree_data and tree_data["child_images"]:
                        original_path = tree_data["child_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                    elif "parent_images" in tree_data and tree_data["parent_images"]:
                        original_path = tree_data["parent_images"][0]["path"]
                        image_path = original_path.replace("/data/", "/trees/")
                
                points.append({
                    "synset_id": sample_key,
                    "tree_id": tree_id,
                    "image_path": image_path,
                    "kind": "tree",
                    "embedding_type": embedding_label
                })
                images_list.append(image_path)
            
            # Create meta dict for compatibility
            meta = {}
            for tree_id, tree_data in meta_data_trees["trees"].items():
                sample_key = tree_data.get("sample_key", tree_id)
                # Fix the image path here too
                first_image_path = None
                if tree_data.get("child_images"):
                    original_path = tree_data["child_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                elif tree_data.get("parent_images"):
                    original_path = tree_data["parent_images"][0]["path"]
                    first_image_path = original_path.replace("/data/", "/trees/")
                
                meta[sample_key] = {
                    "name": tree_data["parent_text"]["text"],
                    "description": tree_data["child_text"]["text"],
                    "first_image_path": first_image_path
                }
            
            unique_synsets = sorted({sid for sid in synset_ids})
            syn_to_int = {sid: i for i, sid in enumerate(unique_synsets)}
            labels = np.array([syn_to_int[s] for s in synset_ids], dtype=int)
            feature_names = [f"dim{i}" for i in range(embeddings.shape[1])]
            target_names = unique_synsets
            
            return (
                embeddings.tolist(),  # data-store - needed for neighbor calculations
                labels.tolist(),
                feature_names,
                target_names,
                images_list,
                meta,
                points,
                current_sel or [],  # Preserve current selection
                None,
                embeddings.tolist(),  # emb store - this is what the scatter callback needs!
            )
        return dash.no_update

############################ CALLBACKS #############################################
    @app.callback(
        Output("scatter-disk", "figure"),
        Input("emb", "data"),
        Input("sel", "data"),
        Input("proj", "data"),
        Input("interpolated-point", "data"),
        Input("cone-direction", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        State("data-store", "data"),
        Input("dataset-dropdown", "value"),
        State("points-store", "data"),
        State("meta-store", "data"),
    )
    def _scatter(edata, sel, proj, traversal_path, cone_direction,
                 labels_data, target_names, mode, k_neighbors,
                 data_store, dataset_name, points, meta):
        if edata is None or labels_data is None:
            print("Warning: No embedding or label data available for plotting")
            return {}
        emb = np.asarray(edata, dtype=np.float32)
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        highlight = sel
        neighbor_indices = []
        selected_idx = sel
        tree_connections = []
        interpolation_lines = []
        interpolation_highlight = []
        traversal_points = None


        if mode == "interpolate" and traversal_path is not None and len(traversal_path) > 0:
            # traversal_path is a list of indices; convert to actual points
            traversal_points = np.asarray([emb[idx] for idx in traversal_path if idx < len(emb)])

        if mode == "neighbors" and sel and len(sel) == 1:
            selected_idx = sel[:1]
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = _compute_hyperbolic_distances(data_np[sel[0]], data_np)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            selected_idx = sel[:1]
            tree_connections = []
            try:
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                tree_point_indices = []
                tree_points_by_type = {}
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        tree_point_indices.append(i)
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                neighbor_indices = tree_point_indices
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        else:
            neighbor_indices = []

        def _fig_disk(x, y, sel, labels, target_names, traversal_points=None,
                      neighbor_indices=None, emb_labels=None,
                      tree_connections=None, points=None, meta=None,
                      hidden_types=None, plot_range=0.85):
            colors = {
                'parent_text':  '#e41a1c',   # red
    'child_text':   '#377eb8',   # blue  
    'child_image':  '#4daf4a',   # green
    'parent_image': '#984ea3',   # purple
            }
            traces = []
            neighbor_set = set(neighbor_indices) if neighbor_indices is not None else set()
            tree_arrow_annotations = []

            # Highlight traversal_points in interpolation mode
            if traversal_points is not None and len(traversal_points) > 0:
                # Project traversal_points to disk coordinates
                traversal_points = np.asarray(traversal_points)
                if traversal_points.shape[1] > 2:
                    traversal_x = traversal_points[:, 0] / (1.0 + traversal_points[:, 2])
                    traversal_y = traversal_points[:, 1] / (1.0 + traversal_points[:, 2])
                else:
                    traversal_x = traversal_points[:, 0]
                    traversal_y = traversal_points[:, 1]
                # Plot orange diamond markers for all traversal points
                traces.append(
                    go.Scatter(
                        x=traversal_x,
                        y=traversal_y,
                        mode="markers",
                        marker=dict(size=12, color="orange", symbol="diamond"),
                        name="Traversal Path Points",
                        text=[f"Traversal {i}" for i in range(len(traversal_points))],
                        hoverinfo="text",
                        showlegend=False,
                    )
                )
                # Plot a dashed orange line between every two adjacent points
                for i in range(len(traversal_x) - 1):
                    traces.append(
                        go.Scatter(
                            x=[traversal_x[i], traversal_x[i+1]],
                            y=[traversal_y[i], traversal_y[i+1]],
                            mode="lines",
                            line=dict(color="orange", width=2, dash="dash"),
                            name="Traversal Segment" if i == 0 else None,
                            showlegend=False,
                        )
                    )

            # If we have emb_labels, create separate traces for each label type
            if emb_labels and len(emb_labels) == len(x):
                unique_label_types = sorted(set(emb_labels))
                
                for label_type in unique_label_types:
                    # Find indices for this label type
                    indices = [i for i, lbl in enumerate(emb_labels) if lbl == label_type]
                    
                    if indices:
                        # Separate regular points and neighbor points (neighbors can be tree points in tree mode)
                        regular_indices = [i for i in indices if i not in neighbor_set]
                        neighbor_indices_for_type = [i for i in indices if i in neighbor_set]
                        
                        # Regular points trace
                        if regular_indices:
                            x_coords = [x[i] for i in regular_indices]
                            y_coords = [y[i] for i in regular_indices]
                            hover_text = [
                                _create_hover_text(i, points, meta)
                                for i in regular_indices
                            ]
                            
                            trace = go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode="markers",
                                text=hover_text,
                                hoverinfo="text",
                                customdata=regular_indices,  # Store original indices directly
                                marker=dict(
                                    size=8, 
                                    opacity=0.7, 
                                    color=colors.get(label_type, 'gray'),
                                    line=dict(width=0.5, color='black')
                                ),
                                name=label_type.replace('_', ' ').title(),
                                showlegend=True,
                            )
                            traces.append(trace)
                        

                        
                        # Neighbor points trace (larger and brighter)
                        if neighbor_indices_for_type:
                            x_coords_neighbors = [x[i] for i in neighbor_indices_for_type]
                            y_coords_neighbors = [y[i] for i in neighbor_indices_for_type]
                            hover_text_neighbors = [
                                f"{_create_hover_text(i, points, meta)} (neighbor)"
                                for i in neighbor_indices_for_type
                            ]
                            
                            neighbor_trace = go.Scatter(
                                x=x_coords_neighbors,
                                y=y_coords_neighbors,
                                mode="markers",
                                text=hover_text_neighbors,
                                hoverinfo="text",
                                customdata=neighbor_indices_for_type,
                                marker=dict(
                                    size=12,  # Larger size for neighbors
                                    opacity=1.0,  # Full opacity for neighbors
                                    color=colors.get(label_type, 'gray'),
                                    line=dict(width=2, color='purple')  # Purple border to make them stand out
                                ),
                                name=f"{label_type.replace('_', ' ').title()} (Neighbors)",
                                showlegend=False,  # Don't show in legend to avoid clutter
                            )
                            traces.append(neighbor_trace)
            else:
                # Fallback to single trace with colorscale
                
                # Separate regular points and neighbor points
                regular_indices = [i for i in range(len(x)) if i not in neighbor_set]
                neighbor_indices_list = [i for i in range(len(x)) if i in neighbor_set]
                
                # Regular points trace
                if regular_indices:
                    base = go.Scatter(
                        x=[x[i] for i in regular_indices],
                        y=[y[i] for i in regular_indices],
                        mode="markers",
                        text=[
                            _create_hover_text(i, points, meta)
                            for i in regular_indices
                        ],
                        hoverinfo="text",
                        customdata=regular_indices,
                        marker=dict(size=8, opacity=0.7, color=[labels[i] for i in regular_indices], colorscale="Viridis"),
                        name="Data points",
                        showlegend=False,
                    )
                    traces = [base]
                else:
                    traces = []
                

                
                # Neighbor points trace (larger and brighter)
                if neighbor_indices_list:
                    neighbor_trace = go.Scatter(
                        x=[x[i] for i in neighbor_indices_list],
                        y=[y[i] for i in neighbor_indices_list],
                        mode="markers",
                        text=[
                            f"{_create_hover_text(i, points, meta)} (neighbor)"
                            for i in neighbor_indices_list
                        ],
                        hoverinfo="text",
                        customdata=neighbor_indices_list,
                        marker=dict(
                            size=12,  # Larger size for neighbors
                            opacity=1.0,  # Full opacity for neighbors
                            color=[labels[i] for i in neighbor_indices_list], 
                            colorscale="Viridis",
                            line=dict(width=2, color='purple')  # Purple border to make them stand out
                        ),
                        name="Neighbors",
                        showlegend=False,
                    )
                    traces.append(neighbor_trace)

            # Store tree connections for later arrow annotation
            tree_arrow_annotations = []
            if tree_connections and points:
                for conn in tree_connections:
                    idx1, idx2 = conn
                    if idx1 < len(x) and idx2 < len(x):
                        # Create line trace
                        x1, y1 = x[idx1], y[idx1]
                        x2, y2 = x[idx2], y[idx2]
                        
                        line_trace = go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line=dict(color="gold", width=2),
                            hoverinfo="skip",
                            showlegend=False,
                            name="Tree connections"
                        )
                        traces.append(line_trace)
                        
                        # Store arrow annotation info
                        tree_arrow_annotations.append({
                            'x': x2, 'y': y2,
                            'ax': x1, 'ay': y1,
                            'xref': 'x', 'yref': 'y',
                            'axref': 'x', 'ayref': 'y',
                            'arrowhead': 2,
                            'arrowsize': 1.5,
                            'arrowwidth': 2,
                            'arrowcolor': 'gold',
                            'showarrow': True,
                            'text': '',
                        })

            # if traversal_points is not None:
            #     traces.append(
            #         go.Scatter(
            #             x=traversal_points[:, 0],
            #             y=traversal_points[:, 1],
            #             mode="markers",
            #             marker=dict(size=12, color="orange", symbol="diamond"),
            #             name="Interpolated point",
            #             text=["Interpolated point"],
            #             hoverinfo="text",
            #         )
            #     )

            if sel:
                traces.append(
                    go.Scatter(
                        x=x[sel],
                        y=y[sel],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color="rgba(0,0,0,0)",
                            symbol="circle",
                            line=dict(width=3, color="#FFD700"),
                        ),
                        name="Selected point",
                    )
                )
            
            # Calculate the maximum distance from origin to any point for boundary circle
            max_distance = 0
            for trace in traces:
                if hasattr(trace, 'x') and hasattr(trace, 'y') and len(trace.x) > 0 and len(trace.y) > 0:
                    distances = np.sqrt(np.array(trace.x)**2 + np.array(trace.y)**2)
                    max_distance = max(max_distance, np.max(distances))
            
            # Add boundary circle if we have data points
            if max_distance > 0:
                circle_radius = 1.1 * max_distance
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = circle_radius * np.cos(theta)
                circle_y = circle_radius * np.sin(theta)
                
                # Add circle trace (make sure it's first so it renders behind other traces)
                circle_trace = go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode='lines',
                    line=dict(color='lightgray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                )
                traces.insert(0, circle_trace)

            fig = go.Figure(data=traces)
            
            # Add arrow annotations for tree connections
            annotations = tree_arrow_annotations
            
            fig.update_layout(
                xaxis=dict(
                    range=[-plot_range, plot_range],
                    fixedrange=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showline=False,
                    scaleanchor="y",
                    scaleratio=1,
                ),
                yaxis=dict(
                    range=[-plot_range, plot_range],
                    fixedrange=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showline=False,
                    scaleanchor="x",
                    scaleratio=1,
                ),
                margin=dict(l=0, r=0, b=60, t=30),
                uirevision="embedding",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.1,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#dee2e6",
                    borderwidth=1,
                    itemclick="toggle",
                    itemdoubleclick="toggleothers",
                    title=dict(
                        text="Click to filter ↓",
                        font=dict(size=10, color="#6c757d")
                    ),
                ),
                annotations=annotations,
                dragmode='pan',
                plot_bgcolor="white",
            )
            return fig

        # Handle 2D embeddings for disk projection
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])  # For 2D embeddings, use z=0
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        # Dynamic disk radius and plot range
        all_norms = np.sqrt(dx**2 + dy**2)
        disk_radius = float(np.max(all_norms)) + 0.08
        plot_range = float(np.max(all_norms)) * 1.15
        emb_labels = None
        if dataset_name and proj:
            try:
                import pickle
                dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
                emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
                with open(emb_file, "rb") as f:
                    emb_data_loaded = pickle.load(f)
                emb_labels = emb_data_loaded.get("labels", [])
            except Exception as e:
                emb_labels = []
        fig_disk = _fig_disk(
            dx, dy,
            selected_idx if mode == "neighbors" else highlight,
            labels, target_names,
            traversal_points=traversal_points,
            neighbor_indices=neighbor_indices,
            emb_labels=emb_labels,
            tree_connections=tree_connections,
            points=points,
            meta=meta,
            hidden_types=None,
            plot_range=plot_range,       
        )

        # ── Cone mode: draw wedge + highlight 512D members ────────────
        # ── Cone mode: one wedge per selected point ────────────────────
        if mode == "cones" and sel and len(sel) >= 1:
            from .cone_utils import (
                compute_cone_aperture_2d,
                compute_cone_wedge_path,
                compute_inward_cone_wedge_path,
                CONE_SCALE_2D,
            )

            shapes = []
            for i, anchor_idx in enumerate(sel):
                if anchor_idx >= len(dx):
                    continue
                color = CONE_COLORS[i % len(CONE_COLORS)]
                anchor_2d = np.array([dx[anchor_idx], dy[anchor_idx]])
                aperture_deg = compute_cone_aperture_2d(
                    anchor_2d, scale=CONE_SCALE_2D)

                if cone_direction in ("outward", "both"):
                    shapes.append(dict(
                        type="path",
                        path=compute_cone_wedge_path(
                            anchor_2d, aperture_deg,
                            disk_radius=disk_radius),
                        fillcolor=color["fill"],
                        line=dict(color=color["line"], width=1.5),
                        layer="below",
                    ))
                if cone_direction in ("inward", "both"):
                    shapes.append(dict(
                        type="path",
                        path=compute_inward_cone_wedge_path(
                            anchor_2d, aperture_deg,
                            disk_radius=disk_radius),
                        fillcolor=color["fill"],
                        line=dict(color=color["line"], width=1.5,
                                  dash="dot"),
                        layer="below",
                    ))

            fig_disk.update_layout(shapes=shapes)

        return fig_disk

#####################################################################################
    @app.callback(
        Output("proj", "data", allow_duplicate=True),
        Output("proj-horopca-btn", "style", allow_duplicate=True),
        Output("proj-cosne-btn", "style", allow_duplicate=True),
        Output("proj-umap-btn", "style", allow_duplicate=True),
        Output("proj-trimap-btn", "style", allow_duplicate=True),
        [
            Input("scatter-disk-1", "clickData"),
            Input("scatter-disk-2", "clickData"),
            Input("scatter-disk-3", "clickData"),
            Input("scatter-disk-4", "clickData"),
        ],
        State("comparison-mode", "data"),
        prevent_initial_call=True,
    )
    def _auto_switch_projection_visual(click_disk_1, click_disk_2, click_disk_3, click_disk_4, comparison_mode):
        ctx = callback_context
        if not ctx.triggered or not comparison_mode:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered_id = ctx.triggered_id

        # Button styles
        active_style = {
            "backgroundColor": "#28a745",
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "100%",
            "minWidth": "0",
            "flex": "1 1 0",
            "boxSizing": "border-box",
            "transition": "background-color 0.2s",
        }
        inactive_style = {
            **active_style,
            "backgroundColor": "#6c757d"
        }

        if triggered_id == "scatter-disk-1":
            # Clicked on HoroPCA plot - switch to HoroPCA
            return "horopca", active_style, inactive_style, inactive_style, inactive_style
        elif triggered_id == "scatter-disk-2":
            # Clicked on CO-SNE plot - switch to CO-SNE
            return "cosne", inactive_style, active_style, inactive_style, inactive_style
        elif triggered_id == "scatter-disk-3":
            # Clicked on UMAP plot - switch to UMAP
            return "umap", inactive_style, inactive_style, active_style, inactive_style
        elif triggered_id == "scatter-disk-4":
            # Clicked on TriMap plot - switch to TriMap
            return "trimap", inactive_style, inactive_style, inactive_style, active_style

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
#####################################################################################
    @app.callback(
        Output("sel", "data"),
        [
            Input("scatter-disk", "clickData"),
            Input("scatter-disk-1", "clickData"),
            Input("scatter-disk-2", "clickData"),
            Input("scatter-disk-3", "clickData"),
            Input("scatter-disk-4", "clickData"),
            Input({"type": "close-button", "index": dash.ALL}, "n_clicks")
        ],
        State("sel", "data"),
        State("mode", "data"),
        State("interpolated-point", "data"),
        State("cone-multi-mode", "data"),
        prevent_initial_call=True,
    )
    def _select(click_disk, click_disk_1, click_disk_2, click_disk_3, click_disk_4, close_clicks, sel, mode, traversal_path, cone_multi_mode):
        ctx = callback_context
        if not ctx.triggered or not ctx.triggered_id:
            return dash.no_update
        triggered_id = ctx.triggered_id
        current_sel = sel or []
        
        def _clicked(click):
            try:
                pt = click["points"][0]
                curve_number = pt.get("curveNumber", 0)
                point_index = int(pt.get("pointIndex", pt["pointNumber"]))
                
                # Check if we have customdata with original indices
                if "customdata" in pt and pt["customdata"] is not None:
                    return int(pt["customdata"])
                
                # Fallback to original logic for single trace
                if curve_number != 0:
                    return None
                return point_index
            except (TypeError, KeyError, IndexError):
                return None
        
        if triggered_id in ["scatter-disk", "scatter-disk-1", "scatter-disk-2", "scatter-disk-3", "scatter-disk-4"]:
            # In interpolate mode, prevent new selections if there's already a traversal path
            if mode == "interpolate" and traversal_path is not None and len(traversal_path) > 0:
                return dash.no_update

            # Handle clicks from any of the scatter plots
            click_data = ctx.inputs[f"{triggered_id}.clickData"]
            
            idx = _clicked(click_data)
            if idx is None:
                return dash.no_update
            if idx in current_sel:
                new_sel = [i for i in current_sel if i != idx]
            else:
                new_sel = current_sel + [idx]
                max_points = 1
                if mode == "compare": max_points = 5
                elif mode == "interpolate": max_points = 2
                elif mode == "neighbors": max_points = 1
                elif mode == "cones":
                    max_points = 5 if cone_multi_mode else 1
                if len(new_sel) > max_points:
                    new_sel = new_sel[-max_points:]
            return new_sel
        elif isinstance(triggered_id, dict) and triggered_id.get("type") == "close-button":
            if not ctx.triggered[0]['value']:
                return dash.no_update
            idx_to_remove = triggered_id.get("index")
            if idx_to_remove in current_sel:
                new_sel = [i for i in current_sel if i != idx_to_remove]
                return new_sel
        return dash.no_update

#####################################################################################
    @app.callback(
        Output("run-interpolate-btn", "disabled"),
        Output("run-interpolate-btn", "style"),
        Input("sel", "data"),
        Input("mode", "data"),
    )
    def _update_button_state(sel, mode):
        base_style = {
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "width": "100%",
            "marginBottom": "0.5rem",
        }
        
        if mode == "interpolate":
            is_disabled = not (sel and len(sel) == 2)
            if is_disabled:
                # Grayed out style
                style = {
                    **base_style,
                    "backgroundColor": "#6c757d",
                    "color": "white",
                    "cursor": "not-allowed",
                }
            else:
                # Active style
                style = {
                    **base_style,  
                    "backgroundColor": "#007bff",
                    "color": "white",
                    "cursor": "pointer",
                }
            return is_disabled, style
        else:
            # Disabled when not in interpolate mode
            style = {
                **base_style,
                "backgroundColor": "#6c757d", 
                "color": "white",
                "cursor": "not-allowed",
            }
            return True, style
        
#####################################################################################
    @app.callback(
        Output("interpolated-point", "data"),
        Input("run-interpolate-btn", "n_clicks"),
        Input("interpolation-slider", "value"),
        State("sel", "data"),
        State("proj", "data"),
        State("dataset-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _interpolate(n_clicks, t, sel, proj, dataset_name):
        if not (n_clicks and sel and len(sel) == 2 and proj and dataset_name):
            return None
        
        # Load the embeddings for the selected projection method
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            emb = np.array(emb_data["embeddings"], dtype=np.float32)
        except Exception as e:
            print(f"Error loading embeddings for interpolation: {e}")
            return None
        
        i, j = sel[:2]
        p1, p2 = emb[i], emb[j]
        traversal_path = _interpolate_hyperbolic(p1, p2, emb, model='tmp', steps=t)
        
        # Ensure the path always starts and ends with the originally selected points
        if traversal_path and len(traversal_path) > 0:
            # Remove the original start/end points if they appear in the middle
            # and ensure they're at the correct positions
            if i in traversal_path:
                traversal_path.remove(i)
            if j in traversal_path:
                traversal_path.remove(j)
            
            # Add the original start and end points
            traversal_path = [i] + traversal_path + [j]
        else:
            # Fallback if interpolation fails
            traversal_path = [i, j]
            
        return traversal_path
        
#####################################################################################
    @app.callback(
        [Output("tree-levels-above", "children"), Output("tree-selected-level", "children"), Output("tree-levels-below", "children")],
        Input("sel", "data"),
        Input("mode", "data"),
        State("meta-store", "data"),
        State("points-store", "data"),
        Input("dataset-dropdown", "value"),
        Input("proj", "data"),
    )
    def _update_tree_view(sel, mode, meta, points, dataset_name, proj):
        if mode != "tree" or not sel or len(sel) != 1 or meta is None or points is None:
            return html.Span(), html.Span(), html.Span()
        
        idx = sel[0]
        
        # Load embedding labels to determine the level of the selected point
        emb_labels = None
        if dataset_name and proj:
            try:
                import pickle
                dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
                emb_file = f"hierchical_datasets/{dataset_dir}/{proj}_embeddings.pkl"
                with open(emb_file, "rb") as f:
                    emb_data_loaded = pickle.load(f)
                emb_labels = emb_data_loaded.get("labels", [])
            except Exception as e:
                emb_labels = []
        
        if not emb_labels or idx >= len(emb_labels):
            return html.Span(), html.Span(), html.Span()
        
        # Load the tree data to get proper image paths
        tree_data = None
        try:
            import json
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            meta_file = f"hierchical_datasets/{dataset_dir}/meta_data_trees.json"
            with open(meta_file, "r") as f:
                meta_data_trees = json.load(f)
            
            # Find the tree for this point
            pt = points[idx]
            synset_id = pt.get("synset_id", "?")
            
            # Find the tree that matches this synset_id
            for tree_id, tree_info in meta_data_trees["trees"].items():
                if (dataset_name == "imagenet" and tree_info.get("synset_id") == synset_id) or \
                   (dataset_name == "grit" and tree_info.get("sample_key") == synset_id):
                    tree_data = tree_info
                    break
                    
        except Exception as e:
            print(f"Error loading tree data: {e}")
        
        # Define the hierarchy levels based on dataset
        if dataset_name == "imagenet":
            # ImageNet only has 3 levels (no parent_image)
            level_mapping = {
                'parent_text': 1,
                'child_text': 2, 
                'child_image': 3,
            }
            level_names = {
                1: "Level 1",
                2: "Level 2", 
                3: "Level 3",
            }
            max_level = 3
        else:  # GRIT
            level_mapping = {
                'parent_text': 1,
                'child_text': 2, 
                'parent_image': 3,
                'child_image': 4
            }
            level_names = {
                1: "Level 1",
                2: "Level 2", 
                3: "Level 3",
                4: "Level 4"
            }
            max_level = 4
        
        # Get the level of the selected point
        selected_label = emb_labels[idx]
        selected_level = level_mapping.get(selected_label, 0)
        
        if selected_level == 0:
            return html.Span(), html.Span(), html.Span()
        
        try:
            pt = points[idx]
            synset_id = pt.get("synset_id", "?")
            meta_row = meta.get(synset_id, {}) if isinstance(meta, dict) else {}
        except (IndexError, TypeError):
            return html.Span(), html.Span(), html.Span()
        
        # Find all points in the same tree that are actually plotted
        tree_point_indices = []
        tree_points_by_level = {}
        
        for i, point in enumerate(points):
            if i < len(emb_labels):
                # For ImageNet: points in same tree share same synset_id
                # For GRIT: points in same tree might have different sample_key but same tree structure
                point_synset = point.get("synset_id", "?")
                
                # Check if this point belongs to the same tree
                is_same_tree = False
                if dataset_name == "imagenet":
                    is_same_tree = (point_synset == synset_id)
                else:  # GRIT
                    is_same_tree = (point_synset == synset_id)
                
                if is_same_tree:
                    # This point is in the same tree and is plotted
                    tree_point_indices.append(i)
                    
                    # Group by level
                    point_level_label = emb_labels[i]
                    point_level = level_mapping.get(point_level_label, 0)
                    if point_level > 0:
                        if point_level not in tree_points_by_level:
                            tree_points_by_level[point_level] = []
                        tree_points_by_level[point_level].append(i)
        
        # Create level components
        def create_level_component(level, content, is_selected=False):
            border_color = "#28a745" if is_selected else "#6c757d"
            bg_color = "#d4edda" if is_selected else "#f8f9fa"
            
            return html.Div([
                html.H5(level_names[level], style={
                    "margin": "0 0 0.5rem 0", 
                    "color": "#28a745" if is_selected else "#6c757d",
                    "fontWeight": "bold" if is_selected else "normal"
                }),
                content
            ], style={
                "padding": "0.75rem", 
                "backgroundColor": bg_color,
                "border": f"2px solid {border_color}",
                "borderRadius": "6px", 
                "marginBottom": "0.5rem",
                "marginLeft": f"{(level-1) * 1}rem"
            })
        
        # Build the hierarchy display
        levels_above = []
        current_level = None
        levels_below = []
        
        # Level 1: Parent Text
        if selected_level == 1:
            current_level = create_level_component(1, html.P(meta_row.get("name", synset_id), style={"margin": 0, "fontWeight": "bold"}), True)
        elif selected_level > 1:
            levels_above.append(create_level_component(1, html.P(meta_row.get("name", synset_id), style={"margin": 0})))
        
        # Level 2: Child Text  
        if selected_level == 2:
            current_level = create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0, "fontWeight": "bold"}), True)
        elif selected_level > 2:
            levels_above.append(create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0})))
        elif selected_level < 2:
            levels_below.append(create_level_component(2, html.P(meta_row.get("description", "(no description)"), style={"margin": 0, "color": "#6c757d"})))
        
        # Level 3: Parent Image (GRIT) or Child Image (ImageNet)
        if dataset_name == "imagenet":
            # For ImageNet, level 3 is child image - show only child images that are actually plotted
            child_img_components = []
            
            # Get the plotted points at level 3 (child_image)
            level_3_points = tree_points_by_level.get(3, [])
            
            # Instead of complex path matching, just use the point indices directly
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("child_images") and len(level_3_points) > 0:
                # Take the first len(level_3_points) images from the tree data
                num_images_to_show = min(len(level_3_points), len(tree_data["child_images"]))
                
                for i in range(num_images_to_show):
                    child_img = tree_data["child_images"][i]
                    child_img_path = child_img["path"]
                    child_img_rel = child_img_path.replace("/data/", "/trees/")
                    child_img_src = _encode_image(child_img_rel)
                    
                    if child_img_src:
                        child_img_components.append(
                            html.Div([
                                html.Img(src=child_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not child_img_components:
                child_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            child_img_container = html.Div(child_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 3:
                current_level = create_level_component(3, child_img_container, True)
            elif selected_level < 3:
                levels_below.append(create_level_component(3, child_img_container))
        else:
            # For GRIT, level 3 is parent image - show only parent images that are actually plotted
            parent_img_components = []
            
            # Get the plotted points at level 3 (parent_image)
            level_3_points = tree_points_by_level.get(3, [])
            
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("parent_images") and len(level_3_points) > 0:
                # Take the first len(level_3_points) images from the tree data
                num_images_to_show = min(len(level_3_points), len(tree_data["parent_images"]))
                
                for i in range(num_images_to_show):
                    parent_img = tree_data["parent_images"][i]
                    parent_img_path = parent_img["path"]
                    parent_img_rel = parent_img_path.replace("/data/", "/trees/")
                    parent_img_src = _encode_image(parent_img_rel)
                    
                    if parent_img_src:
                        parent_img_components.append(
                            html.Div([
                                html.Img(src=parent_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not parent_img_components:
                parent_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            parent_img_container = html.Div(parent_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 3:
                current_level = create_level_component(3, parent_img_container, True)
            elif selected_level > 3:
                levels_above.append(create_level_component(3, parent_img_container))
            elif selected_level < 3:
                levels_below.append(create_level_component(3, parent_img_container))
        
        # Level 4: Child Image (GRIT only) - show only child images that are actually plotted
        if dataset_name == "grit" and max_level >= 4:
            child_img_components = []
            
            # Get the plotted points at level 4 (child_image)
            level_4_points = tree_points_by_level.get(4, [])
            
            # Show the first N images from tree data where N = number of plotted points
            if tree_data and tree_data.get("child_images") and len(level_4_points) > 0:
                # Take the first len(level_4_points) images from the tree data
                num_images_to_show = min(len(level_4_points), len(tree_data["child_images"]))
                
                for i in range(num_images_to_show):
                    child_img = tree_data["child_images"][i]
                    child_img_path = child_img["path"]
                    child_img_rel = child_img_path.replace("/data/", "/trees/")
                    child_img_src = _encode_image(child_img_rel)
                    
                    if child_img_src:
                        child_img_components.append(
                            html.Div([
                                html.Img(src=child_img_src, style={"maxWidth": "160px", "maxHeight": "160px", "objectFit": "contain", "border": "1px solid #ccc"}),
                                html.P(f"Image {i+1}/{num_images_to_show}", style={"fontSize": "0.7rem", "color": "#666", "margin": "0.2rem 0 0 0", "textAlign": "center"})
                            ], style={"margin": "0.5rem 0"})
                        )
            
            if not child_img_components:
                child_img_components = [html.P("No images available", style={"color": "#6c757d", "fontStyle": "italic"})]
            
            child_img_container = html.Div(child_img_components, style={"margin": 0, "display": "flex", "flexDirection": "column", "alignItems": "center"})
            
            if selected_level == 4:
                current_level = create_level_component(4, child_img_container, True)
            elif selected_level < 4:
                levels_below.append(create_level_component(4, child_img_container))
        
        # Combine all levels
        all_levels = levels_above + ([current_level] if current_level else []) + levels_below
        
        # Split into three sections for the layout
        if len(all_levels) <= 3:
            # Pad with empty divs if needed
            while len(all_levels) < 3:
                all_levels.append(html.Div())
            return all_levels[0], all_levels[1], all_levels[2]
        else:
            # If more than 3 levels, combine some
            return (
                html.Div(all_levels[:2]) if len(all_levels) > 3 else all_levels[0],
                all_levels[2] if len(all_levels) > 3 else all_levels[1], 
                html.Div(all_levels[3:]) if len(all_levels) > 3 else all_levels[2]
            )
        
#####################################################################################
    @app.callback(
    Output("mode", "data"),
    Output("compare-btn", "style"),
    Output("interpolate-mode-btn", "style"),
    Output("tree-mode-btn", "style"),
    Output("neighbors-mode-btn", "style"),
    Output("cones-mode-btn", "style"),
    Output("interpolate-controls", "style"),
    Output("neighbors-controls", "style"),
    Output("cones-controls", "style"),
    Output("tree-traversal-section", "style"),
    Output("cone-panel", "style"),
    Output("mode-instructions", "children"),
    Output("sel", "data", allow_duplicate=True),
    Output("interpolated-point", "data", allow_duplicate=True),
    Input("compare-btn", "n_clicks"),
    Input("interpolate-mode-btn", "n_clicks"),
    Input("tree-mode-btn", "n_clicks"),
    Input("neighbors-mode-btn", "n_clicks"),
    Input("cones-mode-btn", "n_clicks"),
    prevent_initial_call=True,
    )
    def _update_mode(compare_clicks, interpolate_clicks, tree_clicks,
                 neighbors_clicks, cones_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        base_style = {
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "100%",
            "minWidth": "0",
            "flex": "1 1 0",
            "boxSizing": "border-box",
            "transition": "background-color 0.2s",
        }
        inactive = {**base_style, "backgroundColor": "#007bff"}
        active   = {**base_style, "backgroundColor": "green"}
        cones_inactive = {**base_style, "backgroundColor": "#e67e22"}

        # Default all inactive
        compare_style   = inactive
        interpolate_style = inactive
        tree_style      = inactive
        neighbors_style = inactive
        cones_style     = cones_inactive

        # Default all controls hidden
        interpolate_controls_style = {"display": "none"}
        neighbors_controls_style   = {"display": "none"}
        cones_controls_style       = {"display": "none"}
        tree_traversal_style       = {"display": "none"}
        cone_panel_style           = {"display": "none"}

        mode = "compare"
        instructions = "Select up to 5 points to compare."

        if triggered_id == "interpolate-mode-btn":
            mode = "interpolate"
            interpolate_style = active
            interpolate_controls_style = {"display": "block"}
            instructions = "Select 2 points to traverse between."

        elif triggered_id == "tree-mode-btn":
            mode = "tree"
            tree_style = active
            tree_traversal_style = {"display": "block"}
            instructions = "Select 1 point to view its lineage."

        elif triggered_id == "neighbors-mode-btn":
            mode = "neighbors"
            neighbors_style = active
            neighbors_controls_style = {"display": "block"}
            instructions = "Select 1 point to view its neighbors."

        elif triggered_id == "cones-mode-btn":
            mode = "cones"
            cones_style = active
            cones_controls_style = {"display": "block"}
            cone_panel_style = {"display": "block"}
            instructions = "Select 1-5 point to draw its entailment cone."

        else:
            compare_style = active

        return (
            mode,
            compare_style, interpolate_style, tree_style,
            neighbors_style, cones_style,
            interpolate_controls_style, neighbors_controls_style,
            cones_controls_style, tree_traversal_style,
            cone_panel_style,
            instructions,
            [], None
        )
    
    #####################################################################################
    @app.callback(
        Output("cone-multi-mode", "data"),
        Output("cone-single-btn", "style"),
        Output("cone-multi-btn", "style"),
        Input("cone-single-btn", "n_clicks"),
        Input("cone-multi-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_cone_multi_mode(single_clicks, multi_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        active = {
            "backgroundColor": "#e67e22", "color": "white",
            "border": "none", "padding": "0.3rem 0.8rem",
            "borderRadius": "6px", "cursor": "pointer",
            "flex": "1 1 0", "fontSize": "0.82rem",
        }
        inactive = {**active, "backgroundColor": "#6c757d"}

        if triggered_id == "cone-multi-btn":
            return True, inactive, active
        return False, active, inactive
    
#####################################################################################
    @app.callback(
        Output("cmp", "children"),
        Output("cmp-instructions", "children"),
        Input("sel", "data"),
        Input("interpolated-point", "data"),
        Input("interpolation-slider", "value"),
        Input("mode", "data"),
        Input("comparison-mode", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("images-store", "data"),
        State("emb", "data"),
        Input("neighbors-slider", "value"),
        State("points-store", "data"),
        State("meta-store", "data"),
    )
    def _compare(sel, traversal_path, t_value, mode, comparison_mode, labels_data, target_names, images, emb_data, k_neighbors, points, meta):
        if labels_data is None:
            return html.Div(), html.P("Select a dataset to begin.")
        labels = np.asarray(labels_data)
        sel = sel or []
        instructions = None
        components = []
        if mode == "tree":
            instructions = html.P("Select 1 point to view its lineage.")
            if sel:
                pass
            return html.Div(components), instructions
        if mode == "compare":
            instructions = html.P("Select up to 5 points to compare.")
            for idx in sel:
                # Get the content to display based on embedding type
                content_element = _create_content_element(idx, images, points, meta)
                
                components.append(
                    html.Div([
                        html.Button(
                            "×",
                            id={"type": "close-button", "index": idx},
                            style={
                                "position": "absolute",
                                "top": "0.25rem",
                                "right": "0.25rem",
                                "width": "1.5rem",
                                "height": "1.5rem",
                                "borderRadius": "50%",
                                "border": "none",
                                "backgroundColor": "#ff4444",
                                "color": "white",
                                "fontSize": "1rem",
                                "lineHeight": "1",
                                "cursor": "pointer",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "padding": "0",
                                "transition": "background-color 0.2s",
                                "hover": {"backgroundColor": "#cc0000"}
                            }
                        ),
                        content_element
                    ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                )
            return html.Div(components), instructions
        if mode == "interpolate":
            instructions = html.P("Select two distinct points to traverse between.")
            
            # Show selected points immediately as start/end points (only if no path created yet)
            if (traversal_path is None or len(traversal_path) == 0) and sel and len(sel) >= 1:
                def create_endpoint_display(idx, label, border_color, bg_color):
                    content_element = _create_content_element(idx, images, points, meta)
                    return html.Div([
                        html.H6(label, style={
                            "margin": "0 0 0.5rem 0", 
                            "color": border_color,
                            "fontWeight": "bold",
                            "fontSize": "1rem"
                        }),
                        html.Button(
                            "×",
                            id={"type": "close-button", "index": idx},
                            style={
                                "position": "absolute",
                                "top": "0.25rem",
                                "right": "0.25rem",
                                "width": "1.5rem",
                                "height": "1.5rem",
                                "borderRadius": "50%",
                                "border": "none",
                                "backgroundColor": "#ff4444",
                                "color": "white",
                                "fontSize": "1rem",
                                "lineHeight": "1",
                                "cursor": "pointer",
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "padding": "0",
                                "transition": "background-color 0.2s",
                                "hover": {"backgroundColor": "#cc0000"}
                            }
                        ),
                        content_element
                    ], style={
                        "padding": "0.75rem", 
                        "backgroundColor": bg_color,
                        "border": f"2px solid {border_color}",
                        "borderRadius": "6px", 
                        "marginBottom": "0.5rem",
                        "position": "relative"
                    })
                
                # Show starting point
                components.append(create_endpoint_display(sel[0], "Starting Point", "#007bff", "#e3f2fd"))
                
                # Show end point if selected
                if len(sel) >= 2:
                    components.append(create_endpoint_display(sel[1], "End Point", "#007bff", "#e3f2fd"))
            
            # Show full traversal path if it exists (after clicking Create Path)
            if traversal_path is not None and len(traversal_path) > 0:
                # Create a hierarchical traversal path visualization similar to tree
                def create_traversal_step(step_idx, content, is_endpoint=False):
                    # Calculate size based on position (start/end slightly bigger, middle all same smaller size)
                    if is_endpoint:
                        # Start and end points are slightly bigger
                        size_factor = 0.95
                        border_color = "#007bff"
                        bg_color = "#e3f2fd"
                        step_label = "Start Point" if step_idx == 0 else "End Point"
                    else:
                        # All middle points are the same smaller size
                        size_factor = 0.8
                        border_color = "#6c757d"
                        bg_color = "#f8f9fa"
                        step_label = f"Step {step_idx}"
                    
                    # Calculate padding and margin based on size
                    base_padding = 0.75
                    padding = f"{base_padding * size_factor}rem"
                    margin_left = f"{(1.0 - size_factor) * 2}rem"
                    
                    return html.Div([
                        html.H6(step_label, style={
                            "margin": "0 0 0.5rem 0", 
                            "color": border_color,
                            "fontWeight": "bold" if is_endpoint else "normal",
                            "fontSize": f"{0.8 + 0.2 * size_factor}rem"
                        }),
                        content
                    ], style={
                        "padding": padding, 
                        "backgroundColor": bg_color,
                        "border": f"2px solid {border_color}",
                        "borderRadius": "6px", 
                        "marginBottom": "0.5rem",
                        "marginLeft": margin_left,
                        "transform": f"scale({size_factor})",
                        "transformOrigin": "left center",
                        "transition": "all 0.3s ease"
                    })
                
                # Create traversal steps
                for i, idx in enumerate(traversal_path):
                    content_element = _create_content_element(idx, images, points, meta)
                    is_endpoint = (i == 0 or i == len(traversal_path) - 1)
                    step_component = create_traversal_step(i, content_element, is_endpoint)
                    components.append(step_component)
                    
                    # Add connecting line between steps (except after the last one)
                    if i < len(traversal_path) - 1:
                        components.append(
                            html.Div(style={
                                "height": "1rem",
                                "width": "2px",
                                "backgroundColor": "#007bff",
                                "margin": "0 auto",
                                "position": "relative",
                            })
                        )
            return html.Div(components), instructions
        if mode == "neighbors":
            instructions = html.P("Select one point to view its neighbors.")
            if not sel or len(sel) != 1:
                return html.Div(), instructions
            if emb_data is None:
                components.append(html.P("Embedding not available."))
                return html.Div(components), instructions
            emb = np.asarray(emb_data, dtype=np.float32)
            dists = _compute_hyperbolic_distances(emb[sel[0]], emb)
            neighbors = np.argsort(dists)
            neighbors = neighbors[neighbors != sel[0]][:k_neighbors]
            
            # Get the content to display based on embedding type
            content_element = _create_content_element(sel[0], images, points, meta)
            
            components.append(
                html.Div([
                    html.Button(
                        "×",
                        id={"type": "close-button", "index": sel[0]},
                        style={
                            "position": "absolute",
                            "top": "0.25rem",
                            "right": "0.25rem",
                            "width": "1.5rem",
                            "height": "1.5rem",
                            "borderRadius": "50%",
                            "border": "none",
                            "backgroundColor": "#ff4444",
                            "color": "white",
                            "fontSize": "1rem",
                            "lineHeight": "1",
                            "cursor": "pointer",
                            "display": "flex",
                            "alignItems": "center",
                            "justifyContent": "center",
                            "padding": "0",
                            "transition": "background-color 0.2s",
                            "hover": {"backgroundColor": "#cc0000"}
                        }
                    ),
                    content_element
                ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "position": "relative", "backgroundColor": "#e0f7fa", "marginBottom": "0.5rem"})
            )
            if len(neighbors) > 0:
                components.append(html.H6("Neighbors:", style={"margin": "1rem 0 0.5rem 0", "color": "#666"}))
                for nidx in neighbors:
                    # Get the content to display based on embedding type
                    neighbor_content_element = _create_content_element(nidx, images, points, meta)
                    
                    components.append(
                        html.Div([
                            neighbor_content_element
                        ], style={"display": "flex", "alignItems": "center", "padding": "0.5rem", "borderRadius": "4px", "backgroundColor": "#f8f9fa", "marginBottom": "0.5rem"})
                    )
            else:
                components.append(html.P("No neighbors found.", style={"color": "#666", "fontStyle": "italic"}))
            return html.Div(components), instructions
        return html.Div(), instructions

#####################################################################################
    @app.callback(
        Output("cmp-header", "children"),
        Input("mode", "data"),
    )
    def _cmp_header(mode):
        if mode == "compare":
            return html.H4("Point comparison")
        elif mode == "interpolate":
            return html.H4("Traverse Path")
        elif mode == "tree":
            return html.H4("Tree Traversal")
        elif mode == "neighbors":
            return html.H4("Neighbors")
        elif mode == "cones":
            return html.H4("Entailment Cones")
        else:
            return html.H4("Point comparison")

#####################################################################################
    @app.callback(
        Output("cone-direction", "data"),
        Output("cone-outward-btn", "style"),
        Output("cone-inward-btn", "style"),
        Output("cone-both-btn", "style"),
        Input("cone-outward-btn", "n_clicks"),
        Input("cone-inward-btn", "n_clicks"),
        Input("cone-both-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _update_cone_direction(outward_clicks, inward_clicks, both_clicks):
        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        active = {
            "backgroundColor": "#e67e22",
            "color": "white",
            "border": "none",
            "padding": "0.4rem 0.8rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "flex": "1 1 0",
            "fontSize": "0.85rem",
        }
        inactive = {**active, "backgroundColor": "#6c757d"}

        if triggered_id == "cone-outward-btn":
            return "outward", active, inactive, inactive
        elif triggered_id == "cone-inward-btn":
            return "inward", inactive, active, inactive
        elif triggered_id == "cone-both-btn":
            return "both", inactive, inactive, active

        return "outward", active, inactive, inactive

#####################################################################################
    @app.callback(
        Output("cone-tab-content", "children"),
        Input("sel", "data"),
        Input("cone-direction", "data"),
        Input("cone-active-tab", "data"),
        Input("mode", "data"),
        State("emb", "data"),
        State("points-store", "data"),
        State("dataset-dropdown", "value"),
        State("proj", "data"),
    )
    def _update_cone_panel(sel, cone_direction, active_tab, mode,
                           emb_data, points, dataset_name, proj):
        if mode != "cones" or not sel:
            return html.P(
                "Select 1-5 points to draw entailment cones.",
                style={"color": "#6c757d", "fontStyle": "italic"}
            )

        if emb_data is None or points is None:
            return html.P("No data available.",
                          style={"color": "#dc3545"})

        # Load 2D coordinates
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet",
                           "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = (f"hierchical_datasets/{dataset_dir}"
                        f"/{proj}_embeddings.pkl")
            with open(emb_file, "rb") as f:
                emb_pkl = pickle.load(f)
            emb_np = np.array(emb_pkl["embeddings"], dtype=np.float32)
            labels_2d = emb_pkl.get("labels", [])
        except Exception as e:
            return html.P(f"Error: {e}", style={"color": "#dc3545"})

        xh, yh = emb_np[:, 0], emb_np[:, 1]
        zh = (emb_np[:, 2] if emb_np.shape[1] > 2
              else np.zeros(len(emb_np)))
        dx = xh / (1.0 + zh)
        dy = yh / (1.0 + zh)
        coords_2d = np.stack([dx, dy], axis=1)

        umap_warning = None
        if proj == "umap":
            umap_warning = html.Div(
                "⚠️ UMAP is Euclidean — cone wedges are shown for reference only and "
                "don't carry hyperbolic meaning. The 512D highlights (purple rings) "
                "are still geometrically valid.",
                style={
                    "backgroundColor": "#fff3cd",
                    "border": "1px solid #ffc107",
                    "borderRadius": "4px",
                    "padding": "0.4rem 0.6rem",
                    "fontSize": "0.75rem",
                    "color": "#856404",
                    "marginBottom": "0.5rem",
                }
            )

        from .cone_utils import compute_cone_data

        type_colors = {
            'parent_text': '#e41a1c',
            'child_text':  '#377eb8',
            'child_image': '#4daf4a',
            'parent_image':'#984ea3',
        }

        CONE_LINE_COLORS = [
            "#e67e22", "#d35400", "#f39c12", "#c0392b", "#7f2800"
        ]

        def _point_pill(idx):
            lbl = labels_2d[idx] if idx < len(labels_2d) else "?"
            color = type_colors.get(lbl, "#6c757d")
            return html.Span(
                f"#{idx} {lbl.replace('_', ' ')}",
                style={
                    "backgroundColor": color,
                    "color": "white",
                    "padding": "0.15rem 0.4rem",
                    "borderRadius": "3px",
                    "fontSize": "0.75rem",
                    "marginRight": "0.3rem",
                    "marginBottom": "0.3rem",
                    "display": "inline-block",
                }
            )

        # ── Intersection tab ─────────────────────────────────────────
        if active_tab == 99 and len(sel) >= 2:
            # Compute cone data for all selected points
            all_cone_data = []
            for anchor_idx in sel:
                cd = compute_cone_data(
                    anchor_idx=anchor_idx,
                    coords_2d=coords_2d,
                    points=points,
                    labels_2d=labels_2d,
                    dataset_name=dataset_name,
                )
                all_cone_data.append(cd)

            # Compute outward cone intersection
            outward_sets = [
                set(cd["outward_indices"]) for cd in all_cone_data
            ]
            inward_sets = [
                set(cd["inward_indices"]) for cd in all_cone_data
            ]

            outward_intersection = outward_sets[0]
            for s in outward_sets[1:]:
                outward_intersection &= s

            inward_intersection = inward_sets[0]
            for s in inward_sets[1:]:
                inward_intersection &= s

            # GT children intersection
            gt_children_sets = [
                set(cd["gt_children"]) for cd in all_cone_data
            ]
            gt_children_union = set()
            for s in gt_children_sets:
                gt_children_union |= s
            gt_in_outward = gt_children_union & outward_intersection

            return html.Div([
                umap_warning,
                html.H6("∩ Cone Intersection",
                        style={"margin": "0 0 0.5rem 0",
                               "color": "#2c3e50"}),
                html.Hr(style={"margin": "0.3rem 0"}),

                html.P("Outward cone intersection:",
                       style={"fontSize": "0.8rem", "fontWeight": "600",
                               "margin": "0.5rem 0 0.2rem 0",
                               "color": "#495057"}),
                html.P(
                    f"{len(outward_intersection)} points inside ALL "
                    f"{len(sel)} outward cones simultaneously",
                    style={"fontSize": "0.8rem", "color": "#6c757d",
                           "margin": "0"}
                ),

                html.P("Inward cone intersection:",
                       style={"fontSize": "0.8rem", "fontWeight": "600",
                               "margin": "0.5rem 0 0.2rem 0",
                               "color": "#495057"}),
                html.P(
                    f"{len(inward_intersection)} points inside ALL "
                    f"{len(sel)} inward cones simultaneously",
                    style={"fontSize": "0.8rem", "color": "#6c757d",
                           "margin": "0"}
                ),

                html.Hr(style={"margin": "0.5rem 0"}),

                html.P("GT children in outward intersection:",
                       style={"fontSize": "0.8rem", "fontWeight": "600",
                               "margin": "0.3rem 0 0.2rem 0",
                               "color": "#495057"}),
                html.Div([_point_pill(i) for i in gt_in_outward])
                if gt_in_outward else
                html.P("None", style={"fontSize": "0.8rem",
                                      "color": "#6c757d",
                                      "fontStyle": "italic"}),

                html.Hr(style={"margin": "0.5rem 0"}),

                html.P(
                    "Points in the intersection of all outward cones "
                    "are concepts entailed by ALL selected anchors "
                    "simultaneously.",
                    style={"fontSize": "0.73rem", "color": "#6c757d",
                           "fontStyle": "italic", "margin": "0"}
                ),
            ])

        # ── Per-point tab ────────────────────────────────────────────
        tab_idx = active_tab if active_tab < len(sel) else 0
        anchor_idx = sel[tab_idx]
        cone_color = CONE_LINE_COLORS[tab_idx % len(CONE_LINE_COLORS)]

        cd = compute_cone_data(
            anchor_idx=anchor_idx,
            coords_2d=coords_2d,
            points=points,
            labels_2d=labels_2d,
            dataset_name=dataset_name,
        )

        aperture    = cd["aperture_deg"]
        anchor_type = cd["anchor_type"]
        anchor_norm = cd["anchor_norm"]
        gt_children = cd["gt_children"]
        gt_parents  = cd["gt_parents"]
        coverage    = cd["coverage"]
        outward_idx = cd["outward_indices"]
        inward_idx  = cd["inward_indices"]

        type_color = type_colors.get(anchor_type, "#6c757d")

        coverage_pct = coverage * 100
        coverage_color = (
            "#28a745" if coverage_pct >= 60 else
            "#ffc107" if coverage_pct >= 30 else
            "#dc3545"
        )

        return html.Div([
            umap_warning,
            # Point info
            html.Div([
                html.Div(style={
                    "width": "12px", "height": "12px",
                    "borderRadius": "50%",
                    "backgroundColor": cone_color,
                    "display": "inline-block",
                    "marginRight": "0.4rem",
                    "verticalAlign": "middle",
                }),
                html.Span(
                    anchor_type.replace("_", " ").title(),
                    style={
                        "backgroundColor": type_color,
                        "color": "white",
                        "padding": "0.2rem 0.5rem",
                        "borderRadius": "4px",
                        "fontSize": "0.8rem",
                        "marginRight": "0.5rem",
                    }
                ),
                html.Span(
                    f"norm: {anchor_norm:.3f}",
                    style={"fontSize": "0.8rem", "color": "#6c757d"}
                ),
            ], style={"marginBottom": "0.3rem",
                      "display": "flex", "alignItems": "center"}),

            html.P(
                f"Aperture: {aperture:.1f}°",
                style={"margin": "0 0 0.2rem 0",
                       "fontSize": "0.85rem", "color": "#495057"}
            ),
            html.P(
                "Wide = general  |  Narrow = specific",
                style={"margin": "0 0 0.5rem 0", "fontSize": "0.73rem",
                       "color": "#6c757d", "fontStyle": "italic"}
            ),

            html.Hr(style={"margin": "0.3rem 0"}),

            # Parents
            html.H6("↑ Parents",
                    style={"margin": "0.4rem 0 0.3rem 0",
                           "color": "#6c757d", "fontSize": "0.82rem"}),
            html.Div([_point_pill(i) for i in gt_parents])
            if gt_parents else
            html.P("Root concept — no parents.",
                   style={"color": "#6c757d", "fontStyle": "italic",
                           "fontSize": "0.8rem", "margin": "0"}),

            html.Hr(style={"margin": "0.4rem 0"}),

            # Children
            html.H6("↓ Children",
                    style={"margin": "0.4rem 0 0.3rem 0",
                           "color": "#6c757d", "fontSize": "0.82rem"}),
            html.Div([_point_pill(i) for i in gt_children])
            if gt_children else
            html.P("Leaf — no children.",
                   style={"color": "#6c757d", "fontStyle": "italic",
                           "fontSize": "0.8rem", "margin": "0"}),

            html.P(
                f"Cone captures {len(outward_idx)} points.",
                style={"fontSize": "0.73rem", "color": "#6c757d",
                       "margin": "0.3rem 0 0 0"}
            ),

            html.Hr(style={"margin": "0.4rem 0"}),

            # Coverage
            html.H6("Coverage",
                    style={"margin": "0 0 0.3rem 0",
                           "color": "#333", "fontSize": "0.82rem"}),
            html.Div([
                html.Span(
                    f"{coverage_pct:.0f}%",
                    style={
                        "fontSize": "1.4rem",
                        "fontWeight": "bold",
                        "color": coverage_color,
                        "marginRight": "0.4rem",
                    }
                ),
                html.Span(
                    f"{len(set(gt_children) & set(outward_idx))}"
                    f"/{len(gt_children)} GT children inside cone"
                    if gt_children else "No children to evaluate",
                    style={"fontSize": "0.78rem", "color": "#6c757d"}
                ),
            ]),
        ])
    
####################################################################################
    @app.callback(
        Output("comparison-mode", "data"),
        Output("compare-projections-btn", "style"),
        Output("toggle-knob", "style"),
        Output("single-plot-container", "style"),
        Output("comparison-plot-container", "style"),
        Output("config-panel", "style"),
        Output("right-panel", "style"),
        Output("centre-panel", "style"),
        Output("exit-comparison-btn", "style"),
        Input("compare-projections-btn", "n_clicks"),
        Input("exit-comparison-btn", "n_clicks"),
        State("comparison-mode", "data"),
        prevent_initial_call=True,
    )
    def _toggle_comparison_mode(toggle_clicks, exit_clicks, current_mode):
        new_mode = not current_mode

        exit_btn_visible = {
            "display": "inline-block",
            "alignSelf": "flex-start",
            "marginBottom": "0.5rem",
            "backgroundColor": "#4a5568",
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "fontWeight": "600",
            "transition": "background-color 0.2s",
        }
        exit_btn_hidden = {**exit_btn_visible, "display": "none"}

        track_on = {
            "width": "44px", "height": "24px",
            "backgroundColor": "#41ae76",    # green when on
            "borderRadius": "12px", "border": "none",
            "cursor": "pointer", "padding": "3px",
            "display": "flex", "alignItems": "center",
            "transition": "background-color 0.2s",
        }
        track_off = {**track_on, "backgroundColor": "#4a5568"}

        knob_on = {
            "width": "18px", "height": "18px",
            "backgroundColor": "white", "borderRadius": "50%",
            "transition": "transform 0.2s",
            "transform": "translateX(20px)",    # slid right = ON
        }
        knob_off = {**knob_on, "transform": "translateX(0px)"}

        # Normal (single-view) panel styles — mirror layout.py
        config_panel_normal = {
            "width": "20vw", "minWidth": "240px", "maxWidth": "300px",
            "padding": "1rem", "backgroundColor": "#2d3748", "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "flexShrink": 0, "overflowY": "auto",
        }
        right_panel_normal = {
            "width": "18vw", "minWidth": "300px", "maxWidth": "350px",
            "padding": "1rem", "backgroundColor": "white", "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "flexShrink": 0, "overflowY": "auto",
        }
        centre_panel_normal = {
            "flex": 1, "width": "60vw", "padding": "1rem", "backgroundColor": "white",
            "borderRadius": "8px", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "display": "flex", "flexDirection": "column", "justifyContent": "center",
            "alignItems": "center", "minHeight": 0, "overflow": "visible",
        }
        centre_panel_compare = {**centre_panel_normal, "width": "100%", "maxWidth": "100%"}

        if new_mode:
            # Compare-All: hide both side panels, give the 4 plots the full width
            return (
                True,
                track_on,
                knob_on,
                {"display": "none"},
                {
                    "display": "grid", "width": "100%", "height": "100%",
                    "gridTemplateColumns": "1fr 1fr", "gridTemplateRows": "1fr 1fr",
                    "gap": "0.5rem", "minHeight": "0",
                },
                {"display": "none"},
                {"display": "none"},
                centre_panel_compare,
                exit_btn_visible,
            )
        else:
            return (
                False,
                track_off,
                knob_off,
                {
                    "display": "flex",
                    "width": "min(85vh, 50vw)",
                    "height": "min(85vh, 50vw)",
                    "aspectRatio": "1 / 1",
                    "margin": "auto",
                    "maxWidth": "100%",
                    "maxHeight": "100%",
                    "flexShrink": 0,
                    "flexGrow": 0,
                },
                {"display": "none"},
                config_panel_normal,
                right_panel_normal,
                centre_panel_normal,
                exit_btn_hidden,
            )

#####################################################################################
    @app.callback(
        Output("scatter-disk-2", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
        State("proj", "data"),
    )
    def _scatter_plot_2(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode, selected_proj):
        if not comparison_mode or labels_data is None or not dataset_name:
            return {}
        
        # Always load CO-SNE for right plot
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/cosne_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            emb = embeddings
        except Exception as e:
            print(f"Error loading CO-SNE embeddings: {e}")
            return {}
        
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        
        # Handle interpolated points - they are calculated in the space of selected_proj
        # traversal_path contains indices into the dataset, not coordinates
        interp_point = None
        if traversal_path is not None and isinstance(traversal_path, list):
            if selected_proj == "cosne":
                # Path was calculated in CO-SNE space, get the coordinates for these indices
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
            else:
                # Path was calculated in HoroPCA space, but we're showing CO-SNE
                # Show the same indices but in CO-SNE coordinates
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
        
        # Handle 2D embeddings for disk projection (same as main scatter)
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        
        # Transform interpolated path if exists
        interp_transformed = None
        if interp_point is not None:
            if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
                # Multiple points forming a path - transform each point
                interp_coords = []
                for pt in interp_point:
                    if len(pt) > 2:
                        interp_dx = pt[0] / (1.0 + pt[2])
                        interp_dy = pt[1] / (1.0 + pt[2])
                    else:
                        interp_dx, interp_dy = pt[0], pt[1]
                    interp_coords.append([interp_dx, interp_dy])
                interp_transformed = np.array(interp_coords)
            else:
                # Single point
                if len(interp_point) > 2:
                    interp_dx = interp_point[0] / (1.0 + interp_point[2])
                    interp_dy = interp_point[1] / (1.0 + interp_point[2])
                else:
                    interp_dx, interp_dy = interp_point[0], interp_point[1]
                interp_transformed = np.array([interp_dx, interp_dy])
        
        # Calculate neighbors and tree connections based on mode
        neighbor_indices = []
        tree_connections = []
        
        if mode == "neighbors" and sel and len(sel) == 1:
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = _compute_hyperbolic_distances(data_np[sel[0]], data_np)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            # Find all points in the same tree and create connections between adjacent levels
            tree_connections = []
            try:
                # Get the selected point's tree
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                
                # Find all points that belong to the same tree
                tree_point_indices = []
                tree_points_by_type = {}
                
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        # Include all tree points in highlighting (including selected)
                        tree_point_indices.append(i)
                        
                        # Group by embedding type for creating connections
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                
                # Create connections between adjacent hierarchical levels
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:  # GRIT
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                
                # Connect consecutive levels
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                
                # Use tree points as neighbor_indices for highlighting
                neighbor_indices = tree_point_indices
                
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        
        # Load the original label types for color coding (always CO-SNE for right plot)
        emb_labels = emb_data.get("labels", [])
        
        # Create the figure with all mode features
        fig = _create_full_interactive_scatter(dx, dy, labels, target_names, emb_labels, "", sel, neighbor_indices, tree_connections, interp_transformed, mode, points=points)
        return fig

#####################################################################################
    @app.callback(
        Output("scatter-disk-1", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
        State("proj", "data"),
    )
    def _scatter_plot_1(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode, selected_proj):
        if not comparison_mode or labels_data is None or not dataset_name:
            return {}
        
        # Always load HoroPCA for left plot
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/horopca_embeddings.pkl"
            
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            
            embeddings = np.array(emb_data["embeddings"], dtype=np.float32)
            emb = embeddings
        except Exception as e:
            print(f"Error loading HoroPCA embeddings: {e}")
            return {}
        
        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []
        
        # Handle interpolated points - they are calculated in the space of selected_proj
        # traversal_path contains indices into the dataset, not coordinates
        interp_point = None
        if traversal_path is not None and isinstance(traversal_path, list):
            if selected_proj == "horopca":
                # Path was calculated in HoroPCA space, get the coordinates for these indices
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
            else:
                # Path was calculated in CO-SNE space, but we're showing HoroPCA
                # Show the same indices but in HoroPCA coordinates
                interp_coords = []
                for idx in traversal_path:
                    if idx < len(emb):
                        interp_coords.append(emb[idx])
                if interp_coords:
                    interp_point = np.array(interp_coords)
        
        # Handle 2D embeddings for disk projection (same as main scatter)
        xh, yh = emb[:, 0], emb[:, 1]
        if emb.shape[1] > 2:
            zh = emb[:, 2]
        else:
            zh = np.zeros(emb.shape[0])
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)
        
        # Transform interpolated path if exists
        interp_transformed = None
        if interp_point is not None:
            if len(interp_point.shape) == 2 and interp_point.shape[0] > 1:
                # Multiple points forming a path - transform each point
                interp_coords = []
                for pt in interp_point:
                    if len(pt) > 2:
                        interp_dx = pt[0] / (1.0 + pt[2])
                        interp_dy = pt[1] / (1.0 + pt[2])
                    else:
                        interp_dx, interp_dy = pt[0], pt[1]
                    interp_coords.append([interp_dx, interp_dy])
                interp_transformed = np.array(interp_coords)
            else:
                # Single point
                if len(interp_point) > 2:
                    interp_dx = interp_point[0] / (1.0 + interp_point[2])
                    interp_dy = interp_point[1] / (1.0 + interp_point[2])
                else:
                    interp_dx, interp_dy = interp_point[0], interp_point[1]
                interp_transformed = np.array([interp_dx, interp_dy])
        
        # Calculate neighbors and tree connections based on mode
        neighbor_indices = []
        tree_connections = []
        
        if mode == "neighbors" and sel and len(sel) == 1:
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = _compute_hyperbolic_distances(data_np[sel[0]], data_np)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
            else:
                neighbor_indices = []
        elif mode == "tree" and sel and len(sel) == 1:
            # Find all points in the same tree and create connections between adjacent levels
            tree_connections = []
            try:
                # Get the selected point's tree
                selected_pt = points[sel[0]]
                selected_tree_id = selected_pt.get("tree_id", "?")
                
                # Find all points that belong to the same tree
                tree_point_indices = []
                tree_points_by_type = {}
                
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        # Include all tree points in highlighting (including selected)
                        tree_point_indices.append(i)
                        
                        # Group by embedding type for creating connections
                        emb_type = pt.get("embedding_type", "unknown")
                        if emb_type not in tree_points_by_type:
                            tree_points_by_type[emb_type] = []
                        tree_points_by_type[emb_type].append(i)
                
                # Create connections between adjacent hierarchical levels
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:  # GRIT
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                
                # Connect consecutive levels
                for i in range(len(level_order) - 1):
                    current_level = level_order[i]
                    next_level = level_order[i + 1]
                    
                    if current_level in tree_points_by_type and next_level in tree_points_by_type:
                        for curr_pt in tree_points_by_type[current_level]:
                            for next_pt in tree_points_by_type[next_level]:
                                tree_connections.append((curr_pt, next_pt))
                
                # Use tree points as neighbor_indices for highlighting
                neighbor_indices = tree_point_indices
                
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []
        
        # Load the original label types for color coding (always HoroPCA for left plot)
        emb_labels = emb_data.get("labels", [])

        # Create the figure with all mode features
        fig = _create_full_interactive_scatter(dx, dy, labels, target_names, emb_labels, "", sel, neighbor_indices, tree_connections, interp_transformed, mode, points=points)
        return fig

#####################################################################################
    def _build_compare_scatter(proj_name, dataset_name, sel, mode, k_neighbors,
                               traversal_path, labels_data, target_names, data_store, points):
        """Shared loader/renderer for an extra projection panel in Compare-All view.

        Mirrors _scatter_plot_1/_scatter_plot_2 but loads {proj_name}_embeddings.pkl,
        so it works for UMAP (scatter-disk-3) and hyperbolic TriMap (scatter-disk-4).
        Returns {} (empty figure) if the projection file is missing.
        """
        if labels_data is None or not dataset_name:
            return {}
        try:
            import pickle
            dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
            emb_file = f"hierchical_datasets/{dataset_dir}/{proj_name}_embeddings.pkl"
            with open(emb_file, "rb") as f_emb:
                emb_data = pickle.load(f_emb)
            emb = np.array(emb_data["embeddings"], dtype=np.float32)
        except Exception as e:
            print(f"Error loading {proj_name} embeddings: {e}")
            return {}

        labels = np.asarray(labels_data, dtype=int)
        sel = sel or []

        # Interpolated path holds dataset indices → look up this projection's coords
        interp_point = None
        if traversal_path is not None and isinstance(traversal_path, list):
            interp_coords = [emb[idx] for idx in traversal_path if idx < len(emb)]
            if interp_coords:
                interp_point = np.array(interp_coords)

        # 2D inputs pass through unchanged; 3D Lorentz coords → Poincaré
        xh, yh = emb[:, 0], emb[:, 1]
        zh = emb[:, 2] if emb.shape[1] > 2 else np.zeros(emb.shape[0])
        dx, dy = xh / (1.0 + zh), yh / (1.0 + zh)

        interp_transformed = None
        if interp_point is not None:
            if interp_point.ndim == 2 and interp_point.shape[0] > 1:
                coords = []
                for pt in interp_point:
                    if len(pt) > 2:
                        coords.append([pt[0] / (1.0 + pt[2]), pt[1] / (1.0 + pt[2])])
                    else:
                        coords.append([pt[0], pt[1]])
                interp_transformed = np.array(coords)
            else:
                pt = interp_point if interp_point.ndim == 1 else interp_point[0]
                if len(pt) > 2:
                    interp_transformed = np.array([pt[0] / (1.0 + pt[2]), pt[1] / (1.0 + pt[2])])
                else:
                    interp_transformed = np.array([pt[0], pt[1]])

        neighbor_indices = []
        tree_connections = []
        if mode == "neighbors" and sel and len(sel) == 1:
            if data_store is not None:
                data_np = np.asarray(data_store, dtype=np.float32)
                dists = _compute_hyperbolic_distances(data_np[sel[0]], data_np)
                neighbor_indices = np.argsort(dists)
                neighbor_indices = neighbor_indices[neighbor_indices != sel[0]][:k_neighbors]
        elif mode == "tree" and sel and len(sel) == 1:
            try:
                selected_tree_id = points[sel[0]].get("tree_id", "?")
                tree_point_indices = []
                tree_points_by_type = {}
                for i, pt in enumerate(points):
                    if pt.get("tree_id") == selected_tree_id:
                        tree_point_indices.append(i)
                        emb_type = pt.get("embedding_type", "unknown")
                        tree_points_by_type.setdefault(emb_type, []).append(i)
                if dataset_name == "imagenet":
                    level_order = ['parent_text', 'child_text', 'child_image']
                else:
                    level_order = ['parent_text', 'child_text', 'parent_image', 'child_image']
                for i in range(len(level_order) - 1):
                    cur, nxt = level_order[i], level_order[i + 1]
                    if cur in tree_points_by_type and nxt in tree_points_by_type:
                        for c in tree_points_by_type[cur]:
                            for nb in tree_points_by_type[nxt]:
                                tree_connections.append((c, nb))
                neighbor_indices = tree_point_indices
            except Exception as e:
                print(f"Error finding tree points: {e}")
                neighbor_indices = []
                tree_connections = []

        emb_labels = emb_data.get("labels", [])
        return _create_full_interactive_scatter(dx, dy, labels, target_names, emb_labels, "", sel, neighbor_indices, tree_connections, interp_transformed, mode, points=points)

#####################################################################################
    @app.callback(
        Output("scatter-disk-3", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
    )
    def _scatter_plot_3(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode):
        if not comparison_mode:
            return {}
        return _build_compare_scatter("umap", dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points)

#####################################################################################
    @app.callback(
        Output("scatter-disk-4", "figure"),
        Input("dataset-dropdown", "value"),
        Input("sel", "data"),
        Input("mode", "data"),
        Input("neighbors-slider", "value"),
        Input("interpolated-point", "data"),
        State("labels-store", "data"),
        State("target-names-store", "data"),
        State("data-store", "data"),
        State("points-store", "data"),
        Input("comparison-mode", "data"),
    )
    def _scatter_plot_4(dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points, comparison_mode):
        if not comparison_mode:
            return {}
        return _build_compare_scatter("trimap", dataset_name, sel, mode, k_neighbors, traversal_path, labels_data, target_names, data_store, points)

#####################################################################################
    @app.callback(
        Output("hyperparams-table", "children"),
        Input("proj", "data"),
    )
    def _update_hyperparams_display(projection_method):
        """Update hyperparameters display based on selected projection method."""
        if not projection_method:
            return html.Div()
        
        if projection_method == "horopca":
            # HoroPCA hyperparameters from create_projections.py
            params = [
                {"param": "Components", "value": "2", "description": "Output dimensions"},
                {"param": "Learning Rate", "value": "0.05", "description": "Optimization step size"},
                {"param": "Max Steps", "value": "500", "description": "Maximum iterations"},
            ]
        elif projection_method == "cosne":
            # CO-SNE hyperparameters from create_projections.py
            params = [
                {"param": "Learning Rate", "value": "0.5", "description": "Main learning rate"},
                {"param": "Hyperbolic LR", "value": "0.01", "description": "Hyperbolic learning rate"},
                {"param": "Perplexity", "value": "30", "description": "Local neighborhood size"},
                {"param": "Exaggeration", "value": "12.0", "description": "Early exaggeration factor"},
                {"param": "Gamma", "value": "0.1", "description": "Student-t distribution parameter"},
            ]
        elif projection_method == "umap":
            params = [
                {"param": "n_neighbors", "value": "15", "description": "Local neighborhood size"},
                {"param": "min_dist", "value": "0.1", "description": "Minimum distance in 2D"},
                {"param": "metric", "value": "euclidean", "description": "Distance metric (Euclidean)"},
                {"param": "n_components", "value": "2", "description": "Output dimensions"},
            ]
        elif projection_method == "trimap":
            params = [
                {"param": "Geometry", "value": "Fully hyperbolic", "description": "Lorentz-distance triplets, optimised on the Poincaré ball"},
                {"param": "Optimiser", "value": "Riemannian SGD", "description": "Poincaré student-t similarity 1/(1+d_P²)"},
                {"param": "n_inliers", "value": "10", "description": "Nearest hyperbolic neighbours per point"},
                {"param": "n_outliers", "value": "5", "description": "Outlier neighbours per point"},
                {"param": "n_random", "value": "5", "description": "Random triplets per point"},
                {"param": "n_iters", "value": "400", "description": "Riemannian SGD iterations"},
            ]
        else:
            return html.Div("Unknown projection method")
        
        # Create table rows
        table_rows = []
        for param in params:
            table_rows.append(
                html.Tr([
                    html.Td(param["param"], style={
                        "fontWeight": "600", 
                        "color": "#e2e8f0",
                        "fontSize": "0.8rem",
                        "padding": "0.25rem 0.5rem 0.25rem 0",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "35%"
                    }),
                    html.Td(param["value"], style={
                        "color": "#007bff", 
                        "fontFamily": "monospace",
                        "fontSize": "0.8rem",
                        "padding": "0.25rem 0.5rem",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "25%",
                        "textAlign": "center"
                    }),
                    html.Td(param["description"], style={
                        "color": "#a0aec0", 
                        "fontSize": "0.75rem",
                        "padding": "0.25rem 0 0.25rem 0.5rem",
                        "borderBottom": "1px solid #e9ecef",
                        "width": "40%"
                    }),
                ])
            )
        
        return html.Table(
            [html.Tbody(table_rows)],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "0.8rem"
            }
        )

#####################################################################################
    @app.callback(
        Output("interpolated-point", "data", allow_duplicate=True),
        Output("sel", "data", allow_duplicate=True),
        Input("clear-path-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _clear_interpolated_point(n_clicks):
        if n_clicks:
            return None, []  # Clear both interpolated path and selected points
        return dash.no_update, dash.no_update


#####################################################################################
    @app.callback(
        Output("proj", "data"),
        Output("proj-horopca-btn", "style"),
        Output("proj-cosne-btn", "style"),
        Output("proj-umap-btn", "style"),
        Output("proj-trimap-btn", "style"),
        Input("proj-horopca-btn", "n_clicks"),
        Input("proj-cosne-btn", "n_clicks"),
        Input("proj-umap-btn", "n_clicks"),
        Input("proj-trimap-btn", "n_clicks"),
        State("proj", "data"),
        prevent_initial_call=True,
    )
    def _update_projection_selection(horopca_clicks, cosne_clicks, umap_clicks, trimap_clicks, current_proj):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

        # Button styles
        active_style = {
            "backgroundColor": "#28a745",
            "color": "white",
            "border": "none",
            "padding": "0.5rem 1rem",
            "borderRadius": "6px",
            "cursor": "pointer",
            "width": "100%",
            "minWidth": "0",
            "flex": "1 1 0",
            "boxSizing": "border-box",
            "transition": "background-color 0.2s",
        }
        inactive_style = {
            **active_style,
            "backgroundColor": "#6c757d"
        }

        if triggered_id == "proj-horopca-btn":
            return "horopca", active_style, inactive_style, inactive_style, inactive_style
        elif triggered_id == "proj-cosne-btn":
            return "cosne", inactive_style, active_style, inactive_style, inactive_style
        elif triggered_id == "proj-umap-btn":
            return "umap", inactive_style, inactive_style, active_style, inactive_style
        elif triggered_id == "proj-trimap-btn":
            return "trimap", inactive_style, inactive_style, inactive_style, active_style

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

#####################################################################################
    # Interpolation number input callbacks
    @app.callback(
        Output("interpolation-slider", "value"),
        Input("interpolation-increase-btn", "n_clicks"),
        Input("interpolation-decrease-btn", "n_clicks"),
        State("interpolation-slider", "value"),
        prevent_initial_call=True,
    )
    def _update_interpolation_value(increase_clicks, decrease_clicks, current_value):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if triggered_id == "interpolation-increase-btn":
            return current_value + 1
        elif triggered_id == "interpolation-decrease-btn":
            return max(1, current_value - 1)  # Don't go below 1
        
        return dash.no_update

    # End of callbacks