# HIVE-C — Code Handoff & UMAP Integration Guide

This document describes every change the cone team made to the original HIVE
codebase, so you can review, comment, and build on it. It also gives a concrete
plan for adding **Euclidean UMAP** as a third projection.

Repo: https://github.com/GoncaloBFM/HIVE
Local: `C:\Projects\MM4AI\HIVE`
Run: `python src/main.py` → http://127.0.0.1:8081

---

## 0. Quick map of what changed

| File | Status | Summary |
|------|--------|---------|
| `src/main.py` | untouched | entry point |
| `src/projection.py` | untouched | traversal engine |
| `src/image_utils.py` | untouched | image/text display |
| `src/__init__.py` | untouched | package init |
| `src/lorentz.py` | **edited (append-only)** | added all cone math; original 3 functions untouched |
| `src/cone_utils.py` | **new file** | cone orchestration, ground-truth alignment, coverage/precision, wedge geometry |
| `src/layout.py` | **edited** | title, dark theme, Cones button, cone panel + tabs, new stores |
| `src/callbacks.py` | **edited** | cone drawing, multi-cone selection, tab callbacks, colour scheme |
| `build_aligned_subset.py` | **new file (root)** | one-off script to align the 512D subset with the 2D ordering |

---

## 1. `src/lorentz.py` — math layer (append-only)

The original three functions are **unchanged**:
`exp_map0`, `log_map0`, `pairwise_inner`.

Everything below was appended for HIVE-C.

### `poincare_to_lorentz(x, curv=1.0, eps=1e-8)`
Converts 2D Poincaré disk coordinates to Lorentz space components
(`x_lorentz = 2x / (1 - ||x||²)`), so cone math runs in the geometrically
correct model. Used by every cone function that takes 2D input.

### `cone_aperture(x, curv=1.0, scale=1.0, eps=1e-8)`
Computes the entailment-cone half-angle ψ(x) (MERU/Ganea formula)
`ψ = arcsin( sinh(scale) / sinh(‖x‖_H · scale) )`, where ‖x‖_H = arccosh(x₀)
is the **hyperbolic** norm. The `scale` parameter (we use 5.0 in 2D) was added
because the projected 2D embeddings sit too close to the origin and the raw
formula saturates every aperture at 90°. Scale=1.0 is correct for the original
512D space.

### `cone_aperture_degrees(x, curv=1.0, scale=5.0, eps=1e-8)`
Convenience wrapper that returns ψ in degrees.

### `lorentz_norm(x, curv=1.0, eps=1e-8)`
Hyperbolic norm from origin = arccosh(x₀). Used by the exact membership
condition.

### `is_inside_cone(x, candidates, curv=1.0, scale=5.0, eps=1e-8)`
**Outward cone (children)** for 2D Poincaré input. Converts to Lorentz, then
applies the MERU condition `-⟨x,y⟩_L < ‖x‖_L·cosh(ψ)` AND a radial check
(candidate must be further from origin than the anchor).

### `is_inside_cone_inward(x, candidates, curv=1.0, scale=5.0, eps=1e-8)`
**Inward cone (parents)** for 2D Poincaré input. Same idea, reversed: candidate
must be closer to origin, and uses `-⟨x,y⟩_L >= ‖x‖_L·cosh(ψ)`.

### `is_inside_cone_lorentz(x, candidates, curv=1.0, scale=1.0, eps=1e-8)`
Same as `is_inside_cone` but for points **already in 512D Lorentz space** (no
Poincaré conversion). Uses the hyperbolic norm for the radial check. This is the
"compute in 512D where the geometry is intact" path.

### `is_inside_cone_inward_lorentz(x, candidates, curv=1.0, scale=1.0, eps=1e-8)`
512D inward-cone counterpart.

### `cone_members_truncated_lorentz(x, candidates, curv=1.0, scale=1.0, direction="outward", band=None, eps=1e-8)`
Truncated cone in 512D Lorentz space. Combines three checks:
1. angular MERU condition,
2. radial direction (`outward` = further out / `inward` = closer in),
3. optional hyperbolic-distance `band` cutoff.

`band` defaults to `None` (disabled). The distance-band experiment showed
same-tree (mean 1.10) and cross-tree (mean 1.13) distances overlap almost
completely, so no band separates trees — keep it `None` unless re-investigating.

---

## 2. `src/cone_utils.py` — orchestration layer (new file)

Keeps cone-specific logic out of `lorentz.py` and `callbacks.py`.

Constants: `LEVEL_MAP` (text/image hierarchy levels), `CONE_SCALE_2D = 5.0`.

### `_load_hd_embeddings(dataset_name)`
Loads the 512D embeddings, **preferring** `embeddings_subset_aligned.pkl`
(produced by `build_aligned_subset.py`) and falling back to
`embeddings_subset.pkl` with a console warning. The aligned file is required for
the 512D purple-ring highlights to land on the correct 2D points.

### `_get_tree_id(key)` / `_get_gt_relatives(...)`
Ground-truth parent/child lookup from `meta_data_trees.json`.

### `compute_cone_aperture_2d(anchor_2d, scale=CONE_SCALE_2D)`
Aperture in degrees for a 2D point (wraps the lorentz functions).

### `compute_cone_members_2d(...)` / `compute_coverage(...)`
2D geometric cone membership and the coverage metric (recall of ground-truth
children inside the 2D cone).

### `compute_cone_data(anchor_idx, coords_2d, points, labels_2d, dataset_name)`
The main entry the UI calls. Returns a dict with: `aperture_deg`, `anchor_type`,
`anchor_norm`, `outward_indices`, `inward_indices`, `gt_children`, `gt_parents`,
`coverage`, plus 512D fields (`outward_512d`, `inward_512d`, `precision_512d`,
`recall_512d`).

### `compute_cone_highlights_512d(anchor_idx, dataset_name, scale=1.0, band=None)`
Computes geometrically-correct cone membership in the **original 512D** Lorentz
space and returns indices to ring on the 2D disk.
NOTE: `hd_labels` is unpacked but unused (can be `_`), and the `direction`
parameter is unused (the function always returns both directions) — safe to
clean up.

### `compute_cone_wedge_path(anchor_2d, aperture_deg, disk_radius=..., n_points=50)`
Returns the SVG path string for the **outward** cone wedge on the Poincaré disk.
The wedge opens from the anchor away from the origin with half-angle
aperture/2, and each boundary ray is extended to the disk rim via ray–circle
intersection so the wedge always reaches the boundary at the correct angle.

### `compute_inward_cone_wedge_path(anchor_2d, aperture_deg, disk_radius=..., n_points=50)`
Mirror of the above for the **inward** wedge (parents), opening toward the
origin; rendered with a dashed line to distinguish it.

---

## 3. `src/layout.py` — UI structure (edited)

- Title → "HIVE-C: Diagnosing Hyperbolic Projections via Entailment Cone";
  dark header/sidebar (`#2d3748`).
- Mode buttons regrouped: **Exploration** (Compare/Traverse/Tree/Neighbors)
  and **Analysis** (Cones). Cones button is orange when inactive, green when
  active.
- Dual View changed from a button to a toggle switch.
- `cones-controls` block: Single/Multi mode buttons + Outward/Inward/Both
  direction toggle.
- Cone panel restructured into `cone-tab-bar` + `cone-tab-content`
  (replaced the four old static divs `cone-selected-info`, `cone-parents-section`,
  `cone-children-section`, `cone-coverage-section`).
- New stores: `cone-direction`, `cone-data`, `cone-active-tab`,
  `cone-multi-mode`, `hidden-types`.

---

## 4. `src/callbacks.py` — interaction logic (edited)

- `CONE_COLORS` palette (5 orange tones) at top of `register_callbacks`.
- ColorBrewer Set1 point colours in all scatter functions
  (parent_text red, child_text blue, child_image green, parent_image purple).
- Selected-point marker: transparent fill + gold ring; re-click deselects
  (via `customdata`).
- `_scatter` / `_fig_disk`: draws one cone wedge per selected point (up to 5);
  dynamic `disk_radius`; grid/axes removed; legend click-to-filter.
- `_select`: `max_points = 5 if cone_multi_mode else 1`
  (added `State("cone-multi-mode","data")` + matching parameter).
- `_update_mode`: Cones handling with persistent-orange `cones_inactive` style.
- New callbacks: `_update_cone_tabs`, `_update_cone_panel` (per-point tabs +
  intersection tab), `_update_cone_direction`, `_update_cone_multi_mode`,
  `_toggle_comparison_mode`, `_update_hidden_types`.

> ⚠️ Recurring gotcha: every `State`/`Input` in a callback decorator must match
> the function signature in **count and order**, with **no blank line** between
> decorator and `def`. Most of our crashes were signature mismatches.

---

## 5. `build_aligned_subset.py` — data alignment (new, root)

One-off script. `embeddings_subset.pkl` (196 pts) and `horopca_embeddings.pkl`
(497 pts) came from different runs, so their indices don't line up. This script
replays the seed=42 sampling on the full `embeddings.pkl` and writes
`embeddings_subset_aligned.pkl` in the same order as the 2D file, so 512D
highlights map to the correct 2D points. Run once per dataset:
`python build_aligned_subset.py`.

---

# UMAP integration — where to work

Goal: add **Euclidean UMAP** as a third projection alongside HoroPCA and CO-SNE.
The cleanest approach mirrors exactly how the existing two projections work, so
the dashboard logic barely changes.

## How projections currently flow

```
projection_methods/create_projections.py   ← generates *_embeddings.pkl
        ↓ (writes to each dataset folder)
hierchical_datasets/<DS>/horopca_embeddings.pkl
hierchical_datasets/<DS>/cosne_embeddings.pkl
        ↓ loaded by
callbacks.py :: _update_dataset_stores   (Input: dataset-dropdown, proj)
        ↓ stored in dcc.Store("emb"/"proj")
callbacks.py :: _scatter → _fig_disk      (plots dx,dy from the pkl)
```

Each projection is just a precomputed `<name>_embeddings.pkl` with keys
`embeddings` and `labels`. The dashboard picks the file based on the `proj`
store. **So adding UMAP is mostly: generate one more pkl + wire one more button.**

## Step-by-step

**A. Generate the projection (offline)**
In `projection_methods/create_projections.py`, add a UMAP branch that takes the
same input embeddings the other two use and writes `umap_embeddings.pkl` with
the identical structure (`{"embeddings": ..., "labels": ...}`) and — critically
— the **same point ordering/seed (42)** as HoroPCA/CO-SNE so indices stay
aligned with the ground truth and the 512D subset.

```python
import umap  # pip install umap-learn
reducer = umap.UMAP(n_components=2, metric="euclidean", random_state=42)
coords_2d = reducer.fit_transform(high_dim_embeddings)   # (N, 2)
# save coords_2d + labels into umap_embeddings.pkl, same order as the others
```

Note: UMAP outputs **Euclidean** 2D coords, not Poincaré. Two honest options:
1. Plot them as-is and label the view "Euclidean UMAP (not a Poincaré disk)" —
   simplest, and the contrast vs HoroPCA/CO-SNE is exactly the point.
2. Min-max scale into the unit disk just for display. Don't pretend it's
   hyperbolic — the cone wedge math assumes hyperbolic geometry, so for UMAP the
   wedge may not be meaningful. Easiest is to keep cones disabled or clearly
   caveated on the UMAP view.

**B. Add the button (layout.py)**
Next to the HoroPCA / CO-SNE buttons in `_config_panel`:
```python
html.Button("UMAP", id="proj-umap-btn", style={...same as others...})
```

**C. Wire the button (callbacks.py)**
Find the projection-selection callback (handles `proj-horopca-btn` /
`proj-cosne-btn`, writes the `proj` store). Add `proj-umap-btn` as an Input and
a branch that sets `proj = "umap"`.

**D. Load the file (callbacks.py :: _update_dataset_stores)**
This callback builds the embeddings filename from the `proj` value, e.g.
`f"{dataset_dir}/{proj}_embeddings.pkl"`. If `proj` is `"umap"` it will load
`umap_embeddings.pkl` automatically — just make sure the file exists and the
naming matches.

**E. Coordinate handling (callbacks.py :: _fig_disk)**
HoroPCA/CO-SNE store 3D Lorentz-ish coords and the code computes
`dx = xh/(1+zh)`. UMAP gives plain 2D. Add a guard: if the loaded array is
already 2D, use the columns directly instead of the Poincaré division.

**F. Cones on UMAP — decide and document**
Because UMAP is Euclidean, the entailment-cone wedge geometry doesn't carry the
same meaning. Recommended: still allow selecting points and showing the 512D
purple-ring highlights (those come from the original space and are projection-
independent), but caveat the wedge. This is itself a nice result for the
report — it visually shows why hyperbolic projections preserve cone structure
and Euclidean ones don't.

## Verification checklist
- `umap_embeddings.pkl` has the same number of points and same label order as
  `horopca_embeddings.pkl` (compare `labels`).
- UMAP button switches the plot without errors.
- 512D highlights (purple rings) still appear and match the same points across
  all three projections (they should, since they're computed in 512D).
- Document in the report that UMAP cones are shown with a caveat.

---

## Contacts / notes for the reviewer
- Cones were validated by computing precision/recall against ground-truth tree
  children in 512D: ~1% precision, ~22% recall — a structural property of the
  HyCoCLIP space (trees overlap angularly), confirmed independently by the
  distance-band overlap. This is finding C3 in the report.
- Scale K=5 is for the 2D visual aperture only; it does not change 512D
  precision/recall.
