
############################# HIVE-C EXTENSION ###############################

import json
import torch
import numpy as np
import pickle
from pathlib import Path
from . import lorentz as L

# Level mapping — same across the whole project
LEVEL_MAP = {
    "parent_text": 1,
    "child_text": 2,
    "parent_image": 3,
    "child_image": 4
}

# Scale for cone aperture — calibrated for HyCoCLIP 2D projections
CONE_SCALE_2D = 5.0

#---------------------------------------------------------------------
def _in_unit_disk(point_2d, eps=1e-6): 
    """True if point_2d genuinely lies inside the Poincaré disk. Helper func for """
    return float(np.linalg.norm(point_2d)) < 1.0 - eps
#---------------------------------------------------------------------
def _load_hd_embeddings(dataset_name: str) -> tuple[np.ndarray, list]:
    """
    Load high-dimensional Poincaré embeddings aligned with the 2D projections.
    Returns (embeddings array, labels list).
    """
    dataset_dir = {"imagenet": "ImageNet", "grit": "GRIT"}.get(dataset_name, dataset_name)
    path = Path(f"hierchical_datasets/{dataset_dir}/embeddings_subset.pkl")

    with open(path, "rb") as f:
        data = pickle.load(f)

    embeddings = np.array(data["embeddings"], dtype=np.float32)
    labels = data["labels"]
    return embeddings, labels

#---------------------------------------------------------------------
def _get_tree_id(point: dict, dataset_name: str) -> str:
    """Extract tree identifier from a point's metadata."""
    return point.get("tree_id", "")

#---------------------------------------------------------------------
def _get_gt_relatives(
    anchor_idx: int,
    points: list,
    labels_2d: list,
    dataset_name: str
) -> tuple[list, list]:
    """
    Get ground truth parents and children of the anchor point
    using tree structure from points metadata.

    Returns:
        gt_children: list of indices that are children of anchor
        gt_parents:  list of indices that are parents of anchor
    """
    anchor_point = points[anchor_idx]
    anchor_tree_id = _get_tree_id(anchor_point, dataset_name)
    anchor_label = labels_2d[anchor_idx]
    anchor_level = LEVEL_MAP.get(anchor_label, 0)

    gt_children = []
    gt_parents = []

    for i, point in enumerate(points):
        if i == anchor_idx:
            continue
        if _get_tree_id(point, dataset_name) != anchor_tree_id:
            continue

        point_label = labels_2d[i]
        point_level = LEVEL_MAP.get(point_label, 0)

        if point_level > anchor_level:
            gt_children.append(i)
        elif point_level < anchor_level:
            gt_parents.append(i)

    return gt_children, gt_parents

#---------------------------------------------------------------------
def get_direct_relatives(
    anchor_idx: int,
    points: list,
    labels_2d: list,
    dataset_name: str
) -> tuple[list, list]:
    """
    Direct (adjacent-level) parents/children of the anchor — the immediate
    edges in the tree, as opposed to the full transitive taxonomy returned by
    :func:`_get_gt_relatives`.

    HyCoCLIP is trained on specific parent-child pairs, not the whole
    taxonomy, so evaluating recall against these direct pairs is the fair
    test of whether the model encodes the relationships it actually saw.

    Uses the levels *present in the anchor's tree* (not LEVEL_MAP directly) so
    ImageNet — which has no parent_image level — still resolves the correct
    immediate child/parent instead of leaving a gap.

    Returns:
        direct_children: indices one level deeper than the anchor
        direct_parents:  indices one level shallower than the anchor
    """
    anchor_tree_id = _get_tree_id(points[anchor_idx], dataset_name)
    anchor_level = LEVEL_MAP.get(labels_2d[anchor_idx], 0)

    tree_indices = [
        i for i, p in enumerate(points)
        if _get_tree_id(p, dataset_name) == anchor_tree_id
    ]
    levels_present = sorted(
        {LEVEL_MAP.get(labels_2d[i], 0) for i in tree_indices} - {0}
    )

    deeper = [lv for lv in levels_present if lv > anchor_level]
    shallower = [lv for lv in levels_present if lv < anchor_level]
    child_level = min(deeper) if deeper else None
    parent_level = max(shallower) if shallower else None

    direct_children = [
        i for i in tree_indices
        if i != anchor_idx and LEVEL_MAP.get(labels_2d[i], 0) == child_level
    ] if child_level is not None else []
    direct_parents = [
        i for i in tree_indices
        if i != anchor_idx and LEVEL_MAP.get(labels_2d[i], 0) == parent_level
    ] if parent_level is not None else []

    return direct_children, direct_parents

#---------------------------------------------------------------------
def compute_cone_aperture_2d(
    anchor_2d: np.ndarray,
    scale: float = CONE_SCALE_2D
) -> float:
    """
    Compute cone aperture angle in degrees for a 2D Poincaré point.

    Args:
        anchor_2d: 2D Poincaré coordinates of the selected point
        scale: Scaling factor for the aperture formula

    Returns:
        Aperture angle in degrees
    """
    if not _in_unit_disk(anchor_2d): #guard for points outside the disk
        return None
    x = torch.tensor(anchor_2d, dtype=torch.float32).unsqueeze(0)
    x_lorentz = L.poincare_to_lorentz(x)
    aperture_rad = L.cone_aperture(x_lorentz, scale=scale).squeeze(0)
    return float(torch.rad2deg(aperture_rad).item())

#---------------------------------------------------------------------
def compute_cone_members_2d(
    anchor_2d: np.ndarray,
    all_coords_2d: np.ndarray,
    scale: float = CONE_SCALE_2D
    ) -> tuple[list, list]:
    """
    Find which 2D points fall inside the outward and inward cones
    of the anchor point.

    Used for highlighting points on the Poincaré disk.

    Args:
        anchor_2d: shape (2,) - anchor in 2D Poincaré coordinates
        all_coords_2d: shape (N, 2) - all points in 2D Poincaré
        scale: aperture scaling factor

    Returns:
        outward_indices: indices of points inside outward cone (children direction)
        inward_indices:  indices of points inside inward cone (parents direction)
    """
    if not _in_unit_disk(anchor_2d):
        return [], [] #guard for points outside the disk
    x = torch.tensor(anchor_2d, dtype=torch.float32)
    candidates = torch.tensor(all_coords_2d, dtype=torch.float32)

    inside_out = L.is_inside_cone(x, candidates, scale=scale)
    inside_in = L.is_inside_cone_inward(x, candidates, scale=scale)

    outward_indices = torch.where(inside_out)[0].tolist()
    inward_indices = torch.where(inside_in)[0].tolist()

    return outward_indices, inward_indices

#---------------------------------------------------------------------
def compute_coverage(
    anchor_idx: int,
    gt_children: list,
    outward_indices_2d: list
) -> float:
    """
    Compute what percentage of ground truth children
    fall inside the geometric outward cone.

    This is the key evaluation metric — measures how well
    the 2D projection preserves hierarchical cone structure.

    Args:
        anchor_idx: index of selected point
        gt_children: ground truth child indices
        outward_indices_2d: indices inside geometric outward cone

    Returns:
        Coverage as float between 0 and 1
    """
    if not gt_children:
        return 0.0

    gt_set = set(gt_children)
    cone_set = set(outward_indices_2d)
    overlap = gt_set & cone_set
    return len(overlap) / len(gt_set)

#---------------------------------------------------------------------
def compute_cone_data(
    anchor_idx: int,
    coords_2d: np.ndarray,
    points: list,
    labels_2d: list,
    dataset_name: str,
    scale: float = CONE_SCALE_2D
) -> dict:
    """
    Main function — computes everything needed for the cone mode.

    Called by callbacks.py when user clicks a point in cone mode.

    Args:
        anchor_idx: index of the selected point
        coords_2d: shape (N, 2) - all 2D Poincaré coordinates
        points: list of point metadata dicts from callbacks
        labels_2d: list of embedding type labels
        dataset_name: "imagenet" or "grit"
        scale: aperture scaling factor

    Returns:
        dict with keys:
            aperture_deg    : float - cone aperture in degrees
            anchor_type     : str   - embedding type of selected point
            anchor_norm     : float - distance from origin in Poincaré disk
            outward_indices : list  - indices inside geometric outward cone
            inward_indices  : list  - indices inside geometric inward cone
            gt_children     : list  - ground truth child indices
            gt_parents      : list  - ground truth parent indices
            coverage        : float - % of gt_children inside geometric cone
    """
    anchor_2d = coords_2d[anchor_idx]
    anchor_type = labels_2d[anchor_idx] if anchor_idx < len(labels_2d) else ""
    anchor_norm = float(np.linalg.norm(anchor_2d))
    out_of_disk = anchor_norm >= 1.0 - 1e-6

    # Compute aperture angle
    aperture_deg = compute_cone_aperture_2d(anchor_2d, scale=scale) #none if outside disk

    # Find geometric cone members in 2D
    outward_indices, inward_indices = compute_cone_members_2d(
        anchor_2d, coords_2d, scale=scale
    )
    # Get ground truth relatives from tree structure
    gt_children, gt_parents = _get_gt_relatives(
        anchor_idx, points, labels_2d, dataset_name
    )

    # Coverage metric (vs 2D geometric cone, kept for comparison)
    coverage = None if out_of_disk else compute_coverage(anchor_idx, gt_children, outward_indices)

    # 512D geometric cone membership (the meaningful one)
    hl = compute_cone_highlights_512d(
        anchor_idx, dataset_name, scale=1.0, band=None
    )
    outward_512d = hl["outward_512d"]
    inward_512d = hl["inward_512d"]

    # Precision/recall of the 512D cone against ground truth children
    gt_set = set(gt_children)
    cone_set = set(outward_512d)
    if cone_set:
        precision_512d = len(gt_set & cone_set) / len(cone_set)
    else:
        precision_512d = 0.0
    if gt_set:
        recall_512d = len(gt_set & cone_set) / len(gt_set)
    else:
        recall_512d = 0.0

    print(f"[cone_data] anchor={anchor_idx} type={anchor_type} "
          f"aperture={aperture_deg:.1f} "
          f"outward_2d={len(outward_indices)} "
          f"gt_children={gt_children} "
          f"gt_children_in_2d_cone={set(gt_children) & set(outward_indices)}")
    return {
        "out_of_disk": out_of_disk,
        "aperture_deg": aperture_deg,
        "anchor_type": anchor_type,
        "anchor_norm": anchor_norm,
        "outward_indices": outward_indices,      # 2D (direction wedge)
        "inward_indices": inward_indices,         # 2D
        "outward_512d": outward_512d,             # 512D highlights
        "inward_512d": inward_512d,               # 512D highlights
        "gt_children": gt_children,
        "gt_parents": gt_parents,
        "coverage": coverage,
        "precision_512d": precision_512d,
        "recall_512d": recall_512d,
    }

#---------------------------------------------------------------------
def compute_cone_wedge_path(
    anchor_2d: np.ndarray,
    aperture_deg: float,
    disk_radius: float = 1.0,
    n_points: int = 50
    ) -> str:
    """
    Return an SVG path for the OUTWARD entailment-cone wedge (children).

    The wedge opens from the anchor, pointing away from the origin, with
    half-angle aperture_deg/2. Each boundary ray is extended to the disk
    rim via ray-circle intersection so the wedge always reaches the
    Poincaré boundary at the correct aperture.
    """

    cx, cy = float(anchor_2d[0]), float(anchor_2d[1])
    anchor_norm = np.sqrt(cx**2 + cy**2)

    if anchor_norm < 1e-8:
        theta_center = 0.0
    else:
        theta_center = np.arctan2(cy, cx)

    half_aperture = np.deg2rad(min(aperture_deg / 2.0, 89.0))
    theta_start = theta_center - half_aperture
    theta_end   = theta_center + half_aperture
    thetas = np.linspace(theta_start, theta_end, n_points)

    # For each ray direction, find where it hits the disk boundary
    arc_x = []
    arc_y = []
    for t in thetas:
        dx = np.cos(t)
        dy = np.sin(t)
        # Ray from anchor: P(s) = (cx + s*dx, cy + s*dy)
        # Hit disk when |P(s)|^2 = disk_radius^2
        # s^2 + 2s*(cx*dx + cy*dy) + (cx^2 + cy^2 - disk_radius^2) = 0
        a = 1.0
        b = 2.0 * (cx * dx + cy * dy)
        c = cx**2 + cy**2 - disk_radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            s = disk_radius  # fallback
        else:
            s1 = (-b + np.sqrt(discriminant)) / 2.0
            s2 = (-b - np.sqrt(discriminant)) / 2.0
            s = max(s1, s2)  # take the forward intersection
        arc_x.append(cx + s * dx)
        arc_y.append(cy + s * dy)

    path = f"M {cx},{cy} "
    path += f"L {arc_x[0]},{arc_y[0]} "
    for ax, ay in zip(arc_x[1:], arc_y[1:]):
        path += f"L {ax},{ay} "
    path += "Z"
    return path

#---------------------------------------------------------------------
def compute_inward_cone_wedge_path(
    anchor_2d: np.ndarray,
    aperture_deg: float,
    disk_radius: float = 1.0,
    n_points: int = 50
    ) -> str:
    """Return an SVG path for the INWARD entailment-cone wedge (parents).

    Mirror of compute_cone_wedge_path but opening toward the origin,
    representing the more-general concepts that could entail the anchor.
    """

    cx, cy = float(anchor_2d[0]), float(anchor_2d[1])
    anchor_norm = np.sqrt(cx**2 + cy**2)

    if anchor_norm < 1e-8:
        theta_center = 0.0
    else:
        theta_center = np.arctan2(-cy, -cx)  # toward origin

    half_aperture = np.deg2rad(min(aperture_deg / 2.0, 89.0))
    theta_start = theta_center - half_aperture
    theta_end   = theta_center + half_aperture
    thetas = np.linspace(theta_start, theta_end, n_points)

    # Inward cone rays hit the origin side — cap at anchor_norm distance
    arc_x = []
    arc_y = []
    for t in thetas:
        dx = np.cos(t)
        dy = np.sin(t)
        # Extend ray inward until it either hits origin area or exits disk
        a = 1.0
        b = 2.0 * (cx * dx + cy * dy)
        c = cx**2 + cy**2 - disk_radius**2
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            s = anchor_norm
        else:
            s1 = (-b + np.sqrt(discriminant)) / 2.0
            s2 = (-b - np.sqrt(discriminant)) / 2.0
            # Take the smaller positive value (inward direction)
            candidates = [s for s in [s1, s2] if s > 0]
            s = min(candidates) if candidates else anchor_norm
        arc_x.append(cx + s * dx)
        arc_y.append(cy + s * dy)

    path = f"M {cx},{cy} "
    path += f"L {arc_x[0]},{arc_y[0]} "
    for ax, ay in zip(arc_x[1:], arc_y[1:]):
        path += f"L {ax},{ay} "
    path += "Z"
    return path

#---------------------------------------------------------------------
def compute_cone_highlights_512d(
    anchor_idx: int,
    dataset_name: str,
    scale: float = 1.0,
    band: float | None = None,
) -> dict:
    """
    Compute cone membership in the ORIGINAL 512D Lorentz space.
    Returns indices to highlight as purple rings on the 2D disk.

    The 2D projection degenerates (apertures ~90°), so cone membership
    computed in 2D is unreliable. Computing in 512D gives the geometrically
    correct answer, which we then map back to 2D point indices via the
    aligned subset file.

    Returns:
        outward_512d : list[int]  indices inside 512D outward cone
        inward_512d  : list[int]  indices inside 512D inward cone
        n_total      : int        number of 512D points
    """
    import torch
    import src.lorentz as L

    hd_embeddings, _ = _load_hd_embeddings(dataset_name)  # _ = labels unused
    hd_poincare = torch.tensor(hd_embeddings, dtype=torch.float32)
    hd_lorentz = L.poincare_to_lorentz(hd_poincare)

    n_total = hd_lorentz.shape[0]
    if anchor_idx >= n_total:
        return {"outward_512d": [], "inward_512d": [], "n_total": n_total}

    anchor = hd_lorentz[anchor_idx]

    out_mask = L.cone_members_truncated_lorentz(
        anchor, hd_lorentz, scale=scale, direction="outward", band=band
    )
    in_mask = L.cone_members_truncated_lorentz(
        anchor, hd_lorentz, scale=scale, direction="inward", band=band
    )

    outward_512d = [i for i in torch.where(out_mask)[0].tolist()
                    if i != anchor_idx]
    inward_512d  = [i for i in torch.where(in_mask)[0].tolist()
                    if i != anchor_idx]

    return {
        "outward_512d": outward_512d,
        "inward_512d":  inward_512d,
        "n_total":      n_total,
    }

#----------------------------------------------------------------------------
def cone_wedge_polygon(
    anchor_2d, aperture_deg, disk_radius=1.0,
    direction="outward", n_points=50
):
    """Return the wedge as a list of (x, y) vertices (same geometry as
    compute_cone_wedge_path / compute_inward_cone_wedge_path)."""
    cx, cy = float(anchor_2d[0]), float(anchor_2d[1])
    anchor_norm = np.sqrt(cx**2 + cy**2)

    if anchor_norm < 1e-8:
        theta_center = 0.0
    elif direction == "inward":
        theta_center = np.arctan2(-cy, -cx)
    else:
        theta_center = np.arctan2(cy, cx)

    half = np.deg2rad(min(aperture_deg / 2.0, 89.0))
    thetas = np.linspace(theta_center - half, theta_center + half, n_points)

    verts = [(cx, cy)]
    for t in thetas:
        ddx, ddy = np.cos(t), np.sin(t)
        a = 1.0
        b = 2.0 * (cx * ddx + cy * ddy)
        c = cx**2 + cy**2 - disk_radius**2
        disc = b**2 - 4 * a * c
        if disc < 0:
            s = disk_radius
        else:
            s1 = (-b + np.sqrt(disc)) / 2.0
            s2 = (-b - np.sqrt(disc)) / 2.0
            if direction == "inward":
                cand = [s for s in (s1, s2) if s > 0]
                s = min(cand) if cand else anchor_norm
            else:
                s = max(s1, s2)
        verts.append((cx + s * ddx, cy + s * ddy))
    return verts