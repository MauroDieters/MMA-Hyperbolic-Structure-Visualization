from dash import dcc, html

def _config_panel() -> html.Div:
    return html.Div(
        [
            html.H4("Configuration", style={"color": "white", "marginBottom": "1rem"}),
            html.Label("Dataset", style={"color": "#e2e8f0"}),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[
                    {"label": "ImageNet", "value": "imagenet"},
                    {"label": "GRIT", "value": "grit"},
                ],
                value="imagenet",  # Default, can be overridden
                clearable=False,
                style={
                    "marginBottom": "1rem",
                },
            ),
            html.Label("Choose projection:", style={"color": "#e2e8f0"}),
            html.Div([
                html.Button(
                    "HoroPCA",
                    id="proj-horopca-btn",
                    style={
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
                    },
                ),
                html.Button(
                    "CO-SNE",
                    id="proj-cosne-btn",
                    style={
                        "backgroundColor": "#6c757d",
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
                    },
                ),
            ], style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.5rem"}),
            # Projection comparison section
            html.Div([
                html.Div([
    html.Span("Single View", style={
        "fontSize": "0.78rem",
        "color": "#a0aec0",
        "marginRight": "0.6rem",
        "fontWeight": "500",
    }),
    html.Button(
        id="compare-projections-btn",
        n_clicks=0,
        children=[
            html.Div(id="toggle-knob", style={
                "width": "18px",
                "height": "18px",
                "backgroundColor": "white",
                "borderRadius": "50%",
                "transition": "transform 0.2s",
                "transform": "translateX(0px)",
            })
        ],
        style={
            "width": "44px",
            "height": "24px",
            "backgroundColor": "#4a5568",
            "borderRadius": "12px",
            "border": "none",
            "cursor": "pointer",
            "padding": "3px",
            "display": "flex",
            "alignItems": "center",
            "transition": "background-color 0.2s",
        },
    ),
    html.Span("Dual View", style={
        "fontSize": "0.78rem",
        "color": "#a0aec0",
        "marginLeft": "0.6rem",
        "fontWeight": "500",
    }),
], style={
    "display": "flex",
    "alignItems": "center",
    "marginBottom": "0.5rem",
}),
            ]),
            dcc.Store(id="proj", data="horopca"),  # Hidden store for compatibility
            # Hyperparameters display
            html.Div(
                id="hyperparams-display",
                style={
                    "marginTop": "0.5rem",
                    "marginBottom": "1rem",
                    "padding": "0.75rem",
                    "backgroundColor": "#3d4f63",
                    "borderRadius": "6px",
                    "border": "1px solid #4a5568",
                },
                children=[
                    html.H6("Hyperparameters", style={
                            "margin": "0 0 0.5rem 0",
                            "color": "#e2e8f0",
                            "fontSize": "0.9rem"
                            }),
                    html.Div(id="hyperparams-table")
                ]
            ),
            html.Br(),
            html.Label("Mode", style={
                    "fontWeight": "600",
                    "marginBottom": "0.3rem",
                    "display": "block",
                    "color": "white",
                    }),
html.Div([
    # ── Exploration modes ──────────────────────────────
    html.P("Exploration", style={
        "fontSize": "0.72rem",
        "color": "#a0aec0",
        "margin": "0 0 0.3rem 0",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "fontWeight": "600",
    }),
    html.Div([
        html.Button("Compare", id="compare-btn", style={
            "backgroundColor": "green", "color": "white",
            "border": "none", "padding": "0.5rem 1rem",
            "borderRadius": "6px", "cursor": "pointer",
            "flex": "1 1 0", "transition": "background-color 0.2s",
        }),
        html.Button("Traverse", id="interpolate-mode-btn", style={
            "backgroundColor": "#007bff", "color": "white",
            "border": "none", "padding": "0.5rem 1rem",
            "borderRadius": "6px", "cursor": "pointer",
            "flex": "1 1 0", "transition": "background-color 0.2s",
        }),
    ], style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.4rem"}),
    html.Div([
        html.Button("Tree", id="tree-mode-btn", style={
            "backgroundColor": "#007bff", "color": "white",
            "border": "none", "padding": "0.5rem 1rem",
            "borderRadius": "6px", "cursor": "pointer",
            "flex": "1 1 0", "transition": "background-color 0.2s",
        }),
        html.Button("Neighbors", id="neighbors-mode-btn", style={
            "backgroundColor": "#007bff", "color": "white",
            "border": "none", "padding": "0.5rem 1rem",
            "borderRadius": "6px", "cursor": "pointer",
            "flex": "1 1 0", "transition": "background-color 0.2s",
        }),
    ], style={"display": "flex", "gap": "0.5rem"}),

    # ── Divider ────────────────────────────────────────
    html.Hr(style={
        "margin": "0.75rem 0",
        "borderColor": "#4a5568",
    }),

    # ── Analysis modes ─────────────────────────────────
    html.P("Analysis", style={
        "fontSize": "0.72rem",
        "color": "#a0aec0",
        "margin": "0 0 0.3rem 0",
        "textTransform": "uppercase",
        "letterSpacing": "0.05em",
        "fontWeight": "600",
    }),
    html.Div([
        html.Button("⬡ Cones", id="cones-mode-btn", style={
            "backgroundColor": "#e67e22", "color": "white",
            "border": "none", "padding": "0.5rem 1rem",
            "borderRadius": "6px", "cursor": "pointer",
            "width": "100%", "transition": "background-color 0.2s",
            "fontWeight": "600",
        }),
    ]),

], style={
    "display": "flex",
    "flexDirection": "column",
    "gap": "0",
    "marginBottom": "1rem",
}),
            html.P(id="mode-instructions", children="Select up to 5 points.",
                style={"color": "#e2e8f0", "fontSize": "0.85rem"}),
            html.Div(
                id="interpolate-controls",
                style={"display": "none"},
                children=[
                    html.Div(
                        [
                            html.Label("Choose traverse path length:",
                                style={"color": "#e2e8f0", "marginBottom": "0.5rem",
                                     "display": "block", "fontWeight": "500"}), html.Div([
                                html.Button(
                                    "−",
                                    id="interpolation-decrease-btn",
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "border": "1px solid #ccc",
                                        "borderRadius": "6px 0 0 6px",
                                        "padding": "0.5rem",
                                        "cursor": "pointer",
                                        "fontSize": "1.2rem",
                                        "fontWeight": "bold",
                                        "width": "40px",
                                        "height": "40px",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "color": "#495057",
                                        "transition": "background-color 0.2s",
                                    },
                                ),
                                dcc.Input(
                                    id="interpolation-slider",
                                    type="number",
                                    min=1,
                                    step=1,
                                    value=5,
                                    debounce=False,
                                    style={
                                        "width": "80px",
                                        "padding": "0.5rem",
                                        "border": "1px solid #ccc",
                                        "borderLeft": "none",
                                        "borderRight": "none",
                                        "fontSize": "0.9rem",
                                        "textAlign": "center",
                                        "height": "40px",
                                        "boxSizing": "border-box",
                                    },
                                ),
                                html.Button(
                                    "+",
                                    id="interpolation-increase-btn",
                                    style={
                                        "backgroundColor": "#f8f9fa",
                                        "border": "1px solid #ccc",
                                        "borderRadius": "0 6px 6px 0",
                                        "padding": "0.5rem",
                                        "cursor": "pointer",
                                        "fontSize": "1.2rem",
                                        "fontWeight": "bold",
                                        "width": "40px",
                                        "height": "40px",
                                        "display": "flex",
                                        "alignItems": "center",
                                        "justifyContent": "center",
                                        "color": "#495057",
                                        "transition": "background-color 0.2s",
                                    },
                                ),
                            ], style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                                "width": "fit-content",
                                "margin": "0 auto",
                            }),
                        ],
                        style={"marginBottom": "1rem"},
                    ),
                    html.Button(
                        "Create Path",
                        id="run-interpolate-btn",
                        disabled=True,
                        style={
                            "backgroundColor": "#007bff",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "width": "100%",
                            "marginBottom": "0.5rem",
                        },
                    ),
                    html.Button(
                        "Clear Path",
                        id="clear-path-btn",
                        style={
                            "backgroundColor": "#dc3545",
                            "color": "white",
                            "border": "none",
                            "padding": "0.5rem 1rem",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "width": "100%",
                        },
                    ),
                ],
            ),
            html.Div(
                id="neighbors-controls",
                style={"display": "none"},
                children=[
                     html.Label("Number of neighbors (k):",
                        style={"color": "#e2e8f0", "marginBottom": "0.5rem",
                          "display": "block"}),
                    dcc.Slider(
                        id="neighbors-slider",
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ],
            ),
            html.Div(
    id="cones-controls",
    style={"display": "none"},
    children=[
        html.Label("Cone Direction:",
           style={"marginBottom": "0.5rem", "display": "block",
                  "fontWeight": "500", "color": "#e2e8f0"}),
                html.Div([
            html.Button(
                "Single",
                id="cone-single-btn",
                style={
                    "backgroundColor": "#e67e22",
                    "color": "white",
                    "border": "none",
                    "padding": "0.3rem 0.8rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "flex": "1 1 0",
                    "fontSize": "0.82rem",
                    "transition": "background-color 0.2s",
                },
            ),
            html.Button(
                "Multi (up to 5)",
                id="cone-multi-btn",
                style={
                    "backgroundColor": "#6c757d",
                    "color": "white",
                    "border": "none",
                    "padding": "0.3rem 0.8rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "flex": "1 1 0",
                    "fontSize": "0.82rem",
                    "transition": "background-color 0.2s",
                },
            ),
        ], style={"display": "flex", "gap": "0.5rem",
                  "marginBottom": "0.75rem"}),
        html.Div([
            html.Button(
                "↓ Outward",
                id="cone-outward-btn",
                style={
                    "backgroundColor": "#e67e22",
                    "color": "white",
                    "border": "none",
                    "padding": "0.4rem 0.8rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "flex": "1 1 0",
                    "fontSize": "0.85rem",
                    "transition": "background-color 0.2s",
                },
            ),
            html.Button(
                "↑ Inward",
                id="cone-inward-btn",
                style={
                    "backgroundColor": "#6c757d",
                    "color": "white",
                    "border": "none",
                    "padding": "0.4rem 0.8rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "flex": "1 1 0",
                    "fontSize": "0.85rem",
                    "transition": "background-color 0.2s",
                },
            ),
            html.Button(
                "Both",
                id="cone-both-btn",
                style={
                    "backgroundColor": "#6c757d",
                    "color": "white",
                    "border": "none",
                    "padding": "0.4rem 0.8rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "flex": "1 1 0",
                    "fontSize": "0.85rem",
                    "transition": "background-color 0.2s",
                },
            ),
        ], style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.5rem"}),
        html.P(
            id="cone-projection-warning",
            children="",
            style={
                "color": "#dc3545",
                "fontSize": "0.8rem",
                "margin": "0.5rem 0 0 0",
                "display": "none"
            }
        ),
    ],
),
        ],
        style={
            "width": "20vw",
            "minWidth": "240px",
            "maxWidth": "300px",
            "padding": "1rem",
            "backgroundColor": "#2d3748",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "flexShrink": 0,
            "overflowY": "auto",
        },
    )

def _centre_panel() -> html.Div:
    return html.Div(
        [
            html.Div(
                id="single-plot-container",
                children=[
                    dcc.Graph(
                        id="scatter-disk",
                        figure=None,  # Will be set by callback
                        style={"width": "100%", "height": "100%"},
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                            "modeBarButtonsToAdd": [],
                            "showTips": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "scatter_plot",
                                "height": 600,
                                "width": 800,
                                "scale": 2
                            },
                            "modeBarButtons": [
                                ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                ["toImage"]
                            ]
                        },
                        
                    ),
                ],
                style={
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
            ),
            html.Div(
                id="comparison-plot-container",
                children=[
                    html.Div([
                        html.H5("HoroPCA", style={"textAlign": "center", "margin": "0 0 1rem 0", "color": "#333"}),
                        dcc.Graph(
                            id="scatter-disk-1",
                            figure=None,
                            style={
                                "width": "100%", 
                                "height": "100%",
                                "aspectRatio": "2 / 3",
                                "maxWidth": "50vh",
                                "maxHeight": "75vh",
                                "minWidth": "300px",
                                "minHeight": "450px"
                            },
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "modeBarButtonsToAdd": [],
                                "showTips": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "horopca_plot",
                                    "height": 600,
                                    "width": 800,
                                    "scale": 2
                                },
                                "modeBarButtons": [
                                    ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                    ["toImage"]
                                ]
                            },
                        ),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "minWidth": "0", "overflow": "visible", "alignItems": "center", "justifyContent": "center"}),
                    html.Div([
                        html.H5("CO-SNE", style={"textAlign": "center", "margin": "0 0 1rem 0", "color": "#333"}),
                        dcc.Graph(
                            id="scatter-disk-2",
                            figure=None,
                            style={
                                "width": "100%", 
                                "height": "100%",
                                "aspectRatio": "2 / 3",
                                "maxWidth": "50vh",
                                "maxHeight": "75vh",
                                "minWidth": "300px",
                                "minHeight": "450px"
                            },
                            config={
                                "displayModeBar": True,
                                "displaylogo": False,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                "modeBarButtonsToAdd": [],
                                "showTips": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": "cosne_plot",
                                    "height": 600,
                                    "width": 800,
                                    "scale": 2
                                },
                                "modeBarButtons": [
                                    ["pan2d", "zoom2d", "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
                                    ["toImage"]
                                ]
                            },
                        ),
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"}),
                ],
                style={
                    "display": "none",
                    "width": "100%",
                    "height": "100%",
                    "margin": "auto",
                    "gap": "2rem",
                    "flexDirection": "row",
                    "overflow": "visible",
                    "alignItems": "center",
                    "justifyContent": "center",
                },
            ),
        ],
        style={
            "flex": 1,
            "width": "60vw",
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "alignItems": "center",
            "minHeight": 0,
            "overflow": "visible",
        },
    )

def _tree_node(title: str, content: html.Div, is_current: bool = False) -> html.Div:
    return html.Div([
        html.H6(
            title,
            style={
                "color": "#007bff" if is_current else "#666",
                "fontWeight": "bold" if is_current else "normal",
                "margin": "0 0 0.5rem 0",
                "padding": "0.5rem",
                "backgroundColor": "#f8f9fa" if is_current else "transparent",
                "borderRadius": "4px",
            }
        ),
        content
    ], style={
        "marginBottom": "1rem",
        "position": "relative",
        "padding": "0.5rem",
        "backgroundColor": "white",
        "borderRadius": "8px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
    })

def _cmp_panel() -> html.Div:
    return html.Div(
        [
            html.Div(id="cmp-header"),
            html.Div(id="cmp-instructions"),
            html.Div(
                [
                    html.H5(
                        "Tree Traversal",
                        style={
                            "marginTop": "1rem",
                            "color": "#007bff",
                            "padding": "0.5rem",
                            "borderBottom": "2px solid #007bff",
                            "marginBottom": "1rem"
                        }
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(id="tree-levels-above"),
                                    html.Div(
                                        style={
                                            "height": "1rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    html.Div(id="tree-selected-level"),
                                    html.Div(
                                        style={
                                            "height": "1rem",
                                            "width": "2px",
                                            "backgroundColor": "#007bff",
                                            "margin": "0 auto",
                                            "position": "relative",
                                        }
                                    ),
                                    html.Div(id="tree-levels-below"),
                                ],
                                id="tree-traversal",
                            ),
                        ],
                    ),
                ],
                id="tree-traversal-section",
                style={"display": "none"},
            ),
            html.Div(
                    id="cone-panel",
                    style={"display": "none"},
                    children=[
                    html.Div(
                    id="cone-tab-bar",
                    style={
                        "display": "flex",
                        "gap": "0.25rem",
                        "marginBottom": "0.75rem",
                        "flexWrap": "wrap",
                        },
            ),
        html.Div(id="cone-tab-content"),
    ]
),
            html.Div(id="cmp")
        ],
        style={
            "width": "18vw",
            "minWidth": "300px",
            "maxWidth": "350px",
            "padding": "1rem",
            "backgroundColor": "white",
            "borderRadius": "8px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
            "flexShrink": 0,
            "overflowY": "auto",
        },
    )


#-------------------------------- The Assembler ----------------------------------#
def make_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                html.H2(
                    "HIVE-C: Diagnosing Hyperbolic Projections via Entailment Cone",
                    style={"color": "white", "margin": 0, "padding": "0.5rem 0"},
                ),
                style={
                    "padding": "0 1rem",
                    "backgroundColor": "#2d3748",
                    "boxShadow": "0 2px 8px rgba(0,0,0,0.1)",
                    "height": "48px",
                    "display": "flex",
                    "alignItems": "center",
                },
            ),
            dcc.Store(id="data-store"),
            dcc.Store(id="labels-store"),
            dcc.Store(id="feature-names-store"),
            dcc.Store(id="target-names-store"),
            dcc.Store(id="images-store"),
            dcc.Store(id="meta-store"),
            dcc.Store(id="points-store"),
            dcc.Store(id="emb"),
            dcc.Store(id="sel", data=[]),
            dcc.Store(id="mode", data="compare"),
            dcc.Store(id="interpolated-point"),
            dcc.Store(id="comparison-mode", data=False),
            dcc.Store(id="cone-direction", data="outward"),
            dcc.Store(id="cone-data"),
            dcc.Store(id="cone-active-tab", data=0),
            dcc.Store(id="cone-multi-mode", data=False),
            html.Div(
                [_config_panel(), _centre_panel(), _cmp_panel()],
                style={
                    "display": "flex",
                    "flex": 1,
                    "minHeight": 0,
                    "padding": "0.5rem",
                    "gap": "0.5rem",
                },
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "height": "100vh",
            "width": "100vw",
            "margin": 0,
            "padding": 0,
            "fontFamily": "Inter, sans-serif",
            "backgroundColor": "#f7f9fc",
            "overflow": "hidden",
        },
    ) 