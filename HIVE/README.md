# HIVE: Hyperbolic Interactive Visualization Explorer

HIVE is an interactive dashboard for visualizing and exploring hierarchical and
hyperbolic data representations. This fork extends the original
[HIVE](https://github.com/GoncaloBFM/HIVE) with two additional projection methods
(**Euclidean UMAP** and **fully hyperbolic TriMap**), an **entailment-cone**
analysis panel, and **Single / Dual / Grid** comparison views.

## Overview

Hierarchical data (taxonomies, parent/child image–text trees) is naturally
embedded in *hyperbolic* space, where tree depth maps to radius and there is
exponentially more room near the boundary. HIVE loads such embeddings, projects
them to the 2D Poincaré disk with several methods, and lets you compare and probe
them interactively — inspecting neighborhoods, lineages, geodesic traversals, and
entailment cones, all colour-coded by node type (parent/child × text/image).

## Key Features

* Interactive visualization of hierarchical & hyperbolic embeddings on the Poincaré disk
* Built-in support for **GRIT** and **ImageNet** subsets
* Four projection methods to compare:
  * **HoroPCA** — hyperbolic PCA
  * **CO-SNE** — hyperbolic t-SNE
  * **UMAP** — Euclidean (shown for contrast)
  * **TriMap** — fully hyperbolic, optimised on the Poincaré ball
* **Single / Dual / Grid** view modes, with cross-projection **brushing** (hover a
  point in one panel to highlight the same item in all panels)
* Exploration modes: **Compare**, **Traverse** (geodesic interpolation),
  **Tree** (lineage), **Neighbors** (hyperbolic kNN)
* **Entailment Cones** analysis: per-point cones, 512D membership rings,
  coverage / precision / recall metrics, and multi-cone intersections
* Modular codebase for plugging in new models, datasets, and projections

## Installation

HIVE uses the [`uv`](https://docs.astral.sh/uv/) package manager.

```bash
# 1. install uv (see https://docs.astral.sh/uv/ for other platforms)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. create and activate a virtual environment
uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. install dependencies
uv sync

# 4. launch the dashboard
python src/main.py
```

The app serves at **http://127.0.0.1:8081**.

> Prefer plain pip? `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` also works. Requires Python ≥ 3.11.

## Using the dashboard

The interface has three panels: a **left config panel**, the **centre disk**, and a
**right detail panel**.

1. **Pick a dataset** (ImageNet or GRIT) from the dropdown.
2. **Choose a projection** — HoroPCA, CO-SNE, UMAP, or TriMap. The disk redraws in
   that projection. Each point is coloured by its type: parent-text (red),
   child-text (blue), child-image (green), parent-image (purple).
3. **Choose a view** with the Single / Dual / Grid selector:
   * **Single** — one large disk for the active projection.
   * **Dual** — two projections side by side. Click two projection buttons to pick which.
   * **Grid** — all four projections at once in a 2×2 grid.
   In Dual/Grid, **hover any point** to brush the same item across every panel.
   Use **← Return to single view** to exit.
4. **Pick an exploration mode** (buttons in the left panel):
   * **Compare** — select up to 5 points to inspect their content side by side.
   * **Traverse** — select 2 points and interpolate along the geodesic between them.
   * **Tree** — select a point to see its full parent→child lineage.
   * **Neighbors** — select a point to highlight its hyperbolic nearest neighbours
     (adjust *k* with the slider).
   * **⬡ Cones** — select 1–5 points to draw entailment cones (see below).

### Entailment cones

In **Cones** mode, each selected point gets a cone wedge on the disk
(outward = descendants, inward = ancestors; toggle Outward / Inward / Both). The
right panel reports, per cone, the aperture, ground-truth parents/children, and
**coverage / precision / recall** of the cone against the dataset's true hierarchy.
Toggle **Ground Geometry** to ring the points whose membership is computed in the
original 512D space (projection-independent), and **GT Children** to ring the true
hierarchical children. With multiple cones selected, the **Intersection** tab shows
pairwise and k-way overlaps in both 2D and 512D.

> Note: UMAP is Euclidean, so its cone wedges are shown **for reference only** and
> don't carry hyperbolic meaning — the 512D membership rings remain valid there.

## Generating projections

The four `*_embeddings.pkl` files per dataset are produced offline by
`projection_methods/create_projections.py`. Generate **all four methods in one run**
so they share the same sampled points and ordering (this keeps point *i* the same
item across every projection, which the brushing and cones rely on):

```bash
python projection_methods/create_projections.py \
    --dataset-path hierchical_datasets/GRIT \
    --methods horopca cosne umap trimap \
    --n-project 700          # 0 = all points; >0 = balanced tree sampling
```

This writes `horopca_embeddings.pkl`, `cosne_embeddings.pkl`, `umap_embeddings.pkl`,
`trimap_embeddings.pkl`, and the 512D reference `embeddings_subset.pkl` into the
dataset folder — all aligned to one point set. Key options: `--seed` (default 42),
`--n-project`, `--children-per-tree`, plus per-method hyperparameters
(`--umap-n-neighbors`, `--trimap-n-iters`, …). Run `--help` for the full list.

## Dataset structure

Custom datasets must be preprocessed into this layout:

```
hierchical_datasets/<dataset_name>/
    trees/
        tree1/
            parent_images/
            parent_texts/
            child_images/
            child_texts/
        ...
    embeddings.pkl            # source high-dim (512D) embeddings
    meta_data_trees.json      # tree metadata (text, image paths, hierarchy)
```

Running `create_projections.py` (above) then generates the per-method
`*_embeddings.pkl` files alongside these. Each projection pkl is a dict with keys
`embeddings` (N×2 disk coords, or N×3 Lorentz) and `labels` (node types).

## Project structure

```
src/
    main.py          # entry point — builds the Dash app, serves on :8081
    layout.py        # three-panel UI, view/mode controls, cone panel, stores
    callbacks.py     # all interaction logic (projection switching, modes, cones, brushing)
    cone_utils.py    # entailment-cone math, 512D membership, coverage/precision/recall
    lorentz.py       # Lorentz/Poincaré geometry primitives
    projection.py    # geodesic traversal engine
    image_utils.py   # image/text rendering for the detail panel
projection_methods/
    create_projections.py    # offline generation of the *_embeddings.pkl files
    HoroPCA/ , CO-SNE/       # vendored projection implementations
```

## Citation

This dashboard builds on the original HIVE by Nijdam, Prinzhorn, de Heus, and
Brouwer (2nd Beyond Euclidean Workshop on hyperbolic and hyperspherical learning).
If you use HIVE in academic work, please cite the original publication — see the
[upstream repository](https://github.com/GoncaloBFM/HIVE) for the BibTeX entry.

## License

Licensed under the MIT License.
