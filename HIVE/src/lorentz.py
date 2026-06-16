import torch
from torch import Tensor
import math


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Map points from the tangent space at the vertex of hyperboloid, on to the
    hyperboloid. This mapping is done using the exponential map of Lorentz model.

    Args:
        x: Tensor of shape (B, D) giving batch of Euclidean vectors to project
            onto the hyperboloid. These vectors are interpreted as velocity
            vectors in the tangent space at the hyperboloid vertex.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as x, giving space components of the mapped
        vectors on the hyperboloid.
    """
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    _output = torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Inverse of the exponential map: map points from the hyperboloid on to the
    tangent space at the vertex, using the logarithmic map of Lorentz model.

    Args:
        x: Tensor of shape (B, D) giving space components of points
            on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid division by zero.

    Returns:
        Tensor of same shape as x, giving Euclidean vectors in the tangent
        space of the hyperboloid vertex.
    """
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    _distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    _output = _distance0 * x / torch.clamp(rc_xnorm, min=eps)
    return _output


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0):
    """
    Compute pairwise Lorentzian inner product between input vectors.

    Args:
        x: Tensor of shape (B1, D) giving space components of a batch
           of vectors on the hyperboloid.
        y: Tensor of shape (B2, D) giving space components of another
           batch of points on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape (B1, B2) giving pairwise Lorentzian inner product.
    """
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1, keepdim=True))
    xyl = x @ y.T - x_time @ y_time.T
    return xyl


############################# For HIVE-C #################################
#-----------------------------------------------------------------------
def poincare_to_lorentz(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Convert points from Poincaré disk coordinates to Lorentz space components.

    The Poincaré disk and the hyperboloid are two representations of the same
    hyperbolic space. This conversion allows us to go from the 2D visualization
    coordinates to the proper Lorentz model for cone computation.

    Formula:
        x_lorentz = 2x / (1 - ||x||^2)
        x_time    = (1 + ||x||^2) / (1 - ||x||^2)

    Note: We only return the space components (x_lorentz) since the time
    component is always derived from them in our functions.

    Args:
        x: Tensor of shape (B, D) giving Poincaré disk coordinates.
           Must satisfy ||x|| < 1 (inside the unit disk).
        curv: Curvature parameter (default 1.0).
        eps: Small float for numerical stability near the boundary.

    Returns:
        Tensor of same shape as x giving Lorentz space components.
    """
    # Euclidean norm squared of each point
    norm_sq = torch.sum(x**2, dim=-1, keepdim=True)  # shape (B, 1)

    # Clamp to ensure we stay strictly inside the disk
    # Points at or beyond the boundary would cause division by zero
    norm_sq = torch.clamp(norm_sq, max=1.0 - eps)

    # Denominator: (1 - ||x||^2)
    denom = 1.0 - norm_sq  # shape (B, 1)

    # Lorentz space components: 2x / (1 - ||x||^2)
    x_lorentz = (2.0 * x) / denom  # shape (B, D)

    return x_lorentz

#-----------------------------------------------------------------------
def cone_aperture(x: Tensor, curv: float | Tensor = 1.0, 
                  scale: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Compute the entailment cone aperture angle for points in Lorentz space.

    The aperture angle ψ(x) defines how wide the entailment cone is at point x.
    Points closer to the origin (general concepts) have wider cones.
    Points near the boundary (specific concepts) have narrower cones.

    Formula: ψ(x) = arcsin(sinh(K) / sinh(||x||_H * K))
    where ||x||_H = arccosh(x_0) is the hyperbolic norm.

    The K parameter adapts the formula to the actual range of embeddings.
    For compressed 2D projections, scale > 1 spreads the aperture values.

    Args:
        x: Tensor of shape (B, D) giving Lorentz space components.
        curv: Curvature parameter.
        scale: Scaling factor to adapt formula to embedding distribution.
               Default 1.0 follows MERU paper exactly.
               Increase for compressed embeddings (e.g. 3.0-5.0).
        eps: Small float for numerical stability.

    Returns:
        Tensor of shape (B,) giving aperture angles in radians.
    """
    # Compute time component: x0 = sqrt(1/curv + ||x_space||^2)
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))

    # Hyperbolic norm = arccosh(x0)
    hyperbolic_norm = torch.acosh(
        torch.clamp(x_time, min=1.0 + eps)
    )

    # Scaled aperture formula
    sinh_scale = torch.tensor(math.sinh(scale), dtype=x.dtype, device=x.device)
    sinh_norm = torch.sinh(
        torch.clamp(hyperbolic_norm * scale, min=eps)
    )

    ratio = torch.clamp(
        sinh_scale / torch.clamp(sinh_norm, min=eps),
        min=-1.0,
        max=1.0
    )
    aperture = torch.asin(ratio)

    return aperture

#-----------------------------------------------------------------------
def cone_aperture_degrees(x: Tensor, curv: float | Tensor = 1.0,
                          scale: float = 5.0, eps: float = 1e-8) -> Tensor:
    """
    Convenience wrapper returning cone aperture in degrees.
    Uses scale=5.0 as default, calibrated for HyCoCLIP 2D projections.
    
    Args:
        x: Tensor of shape (B, D) giving Lorentz space components.
        curv: Curvature parameter.
        scale: Scaling factor calibrated to embedding distribution.
        eps: Numerical stability.
    
    Returns:
        Tensor of shape (B,) giving aperture angles in degrees.
    """
    return torch.rad2deg(cone_aperture(x, curv, scale, eps))

#-----------------------------------------------------------------------
def lorentz_norm(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Compute the Lorentz norm of points.
    ||x||_L = sqrt(-<x,x>_L) = sqrt(x_time^2 - ||x_space||^2 - 1/curv... )
    
    For points on the hyperboloid: <x,x>_L = -1/curv
    So ||x||_L = 1/sqrt(curv) for all points on hyperboloid.
    
    We instead compute the scaled version useful for cone math:
    ||x||_L = arccosh(x_time * sqrt(curv)) = hyperbolic norm from origin
    
    Args:
        x: Tensor of shape (B, D) - Lorentz space components
        curv: Curvature parameter
        eps: Numerical stability
    
    Returns:
        Tensor of shape (B,) giving Lorentz norms
    """
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1))
    return torch.acosh(torch.clamp(x_time * curv**0.5, min=1.0 + eps))

#-----------------------------------------------------------------------
def is_inside_cone(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 5.0,
    eps: float = 1e-8
) -> Tensor:
    """
    OUTWARD cone — finds children (more specific than x).
    y is a child of x if x is inside the cone of y... 
    
    Actually: y entailed by x means y is in x's cone.
    Condition: -<x,y>_L < ||x||_L * cosh(ψ(x))
    AND y is further from origin than x.
    """
    # Convert Poincaré → Lorentz
    x_lorentz = poincare_to_lorentz(x.unsqueeze(0), curv, eps).squeeze(0)
    candidates_lorentz = poincare_to_lorentz(candidates, curv, eps)

    # Aperture for x
    psi = cone_aperture(x_lorentz.unsqueeze(0), curv, scale, eps).squeeze(0)

    # Lorentz inner products
    x_time = torch.sqrt(1.0 / curv + torch.sum(x_lorentz**2))
    candidates_time = torch.sqrt(
        1.0 / curv + torch.sum(candidates_lorentz**2, dim=-1)
    )
    spatial_dot = candidates_lorentz @ x_lorentz
    lorentz_inner = spatial_dot - x_time * candidates_time

    # Lorentz norm of x
    x_lorentz_norm = torch.acosh(
        torch.clamp(x_time * curv**0.5, min=1.0 + eps)
    )

    lhs = -lorentz_inner
    rhs = x_lorentz_norm * torch.cosh(psi)

    # OUTWARD: candidates that are MORE SPECIFIC than x
    # These have SMALLER lhs values (less inner product overlap)
    # AND are further from origin in Poincaré space
    x_poincare_norm = torch.norm(x)
    candidates_poincare_norm = torch.norm(candidates, dim=-1)
    further_out = candidates_poincare_norm > x_poincare_norm

    # Condition: lhs < rhs (opposite of what we had)
    # AND further from origin
    inside = (lhs < rhs) & further_out

    return inside

#-----------------------------------------------------------------------
def is_inside_cone_inward(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 5.0,
    eps: float = 1e-8
) -> Tensor:
    """
    INWARD cone — finds parents (more general than x).
    y is a parent of x if y is closer to origin AND x is in y's cone.
    Condition: -<x,y>_L >= ||x||_L * cosh(ψ(x))
    AND y is closer to origin than x.
    """
    # Convert Poincaré → Lorentz
    x_lorentz = poincare_to_lorentz(x.unsqueeze(0), curv, eps).squeeze(0)
    candidates_lorentz = poincare_to_lorentz(candidates, curv, eps)

    # Aperture for x
    psi = cone_aperture(x_lorentz.unsqueeze(0), curv, scale, eps).squeeze(0)

    # Lorentz inner products
    x_time = torch.sqrt(1.0 / curv + torch.sum(x_lorentz**2))
    candidates_time = torch.sqrt(
        1.0 / curv + torch.sum(candidates_lorentz**2, dim=-1)
    )
    spatial_dot = candidates_lorentz @ x_lorentz
    lorentz_inner = spatial_dot - x_time * candidates_time

    x_lorentz_norm = torch.acosh(
        torch.clamp(x_time * curv**0.5, min=1.0 + eps)
    )

    lhs = -lorentz_inner
    rhs = x_lorentz_norm * torch.cosh(psi)

    # INWARD: candidates closer to origin
    x_poincare_norm = torch.norm(x)
    candidates_poincare_norm = torch.norm(candidates, dim=-1)
    closer_in = candidates_poincare_norm < x_poincare_norm

    # Condition: lhs >= rhs AND closer to origin
    inside = (lhs >= rhs) & closer_in

    return inside

#-----------------------------------------------------------------------
def is_inside_cone_lorentz(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 1.0,
    eps: float = 1e-8
) -> Tensor:
    """
    OUTWARD cone — finds children (more specific than x).
    Works directly with high-dimensional Lorentz space components.

    y is a child of x if:
    1. -<x,y>_L >= ||x||_L * cosh(ψ(x))  [large inner product = far apart in right direction]
    2. hyperbolic_norm(y) > hyperbolic_norm(x)  [y is further from origin]
    """
    psi = cone_aperture(x.unsqueeze(0), curv, scale, eps).squeeze(0)

    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2))
    candidates_time = torch.sqrt(
        1.0 / curv + torch.sum(candidates**2, dim=-1)
    )
    spatial_dot = candidates @ x
    lorentz_inner = spatial_dot - x_time * candidates_time

    x_lorentz_norm = torch.acosh(
        torch.clamp(x_time * curv**0.5, min=1.0 + eps)
    )

    lhs = -lorentz_inner
    rhs = x_lorentz_norm * torch.cosh(psi)

    x_hyp_norm = x_lorentz_norm
    candidates_hyp_norm = torch.acosh(
        torch.clamp(candidates_time * curv**0.5, min=1.0 + eps)
    )
    further_out = candidates_hyp_norm > x_hyp_norm

    # FIXED: lhs >= rhs for outward (children are further, higher lhs)
    inside = (lhs >= rhs) & further_out

    return inside

#-----------------------------------------------------------------------
def is_inside_cone_inward_lorentz(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 1.0,
    eps: float = 1e-8
) -> Tensor:
    """
    INWARD cone — finds parents (more general than x).
    y is a parent of x if x falls inside y's outward cone.
    Uses each candidate y's own aperture ψ(y).
    """
    # Compute aperture for each candidate y
    psi_candidates = cone_aperture(candidates, curv, scale, eps)  # shape (N,)

    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2))
    candidates_time = torch.sqrt(
        1.0 / curv + torch.sum(candidates**2, dim=-1)
    )

    # Lorentz inner product <y, x>_L for each candidate y
    spatial_dot = candidates @ x
    lorentz_inner = spatial_dot - candidates_time * x_time

    # Lorentz norm of each candidate y
    candidates_lorentz_norm = torch.acosh(
        torch.clamp(candidates_time * curv**0.5, min=1.0 + eps)
    )

    lhs = -lorentz_inner                                    # shape (N,)
    rhs = candidates_lorentz_norm * torch.cosh(psi_candidates)  # shape (N,)

    # x must be further out than y (x is more specific)
    x_hyp_norm = torch.acosh(
        torch.clamp(x_time * curv**0.5, min=1.0 + eps)
    )
    candidates_hyp_norm = torch.acosh(
        torch.clamp(candidates_time * curv**0.5, min=1.0 + eps)
    )
    x_further = x_hyp_norm > candidates_hyp_norm

    # y entails x if x is in y's outward cone
    # i.e. lhs >= rhs AND x is further out than y
    inside = (lhs >= rhs) & x_further

    return inside

#-----------------------------------------------------------------------


##### THIS WAS JUST USED FOR A TEST #####
def cone_members_truncated_lorentz(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 1.0,
    direction: str = "outward",
    band: float = 0.5,
    eps: float = 1e-8,
) -> Tensor:
    """
    Truncated entailment cone in high-dimensional Lorentz space.

    A candidate y is a member of x's cone if BOTH:
      1. y lies in the angular cone of x (the standard MERU condition), and
      2. the hyperbolic distance d(x, y) is within `band` of the
         radial gap between x and y  ->  prunes points that merely share a
         direction but live far away (e.g. other trees).

    direction:
        "outward" -> children  (y further from origin than x)
        "inward"  -> parents   (y closer to origin than x)

    band:
        Maximum allowed hyperbolic distance between x and y for membership.
        Smaller band = stricter = higher precision, lower recall.

    Args:
        x:          Tensor (D,)   anchor, Lorentz space components.
        candidates: Tensor (N, D) all points, Lorentz space components.
        curv:       curvature.
        scale:      aperture scale (1.0 for original 512D space).
        direction:  "outward" or "inward".
        band:       hyperbolic-distance cutoff.
        eps:        numerical stability.

    Returns:
        Tensor (N,) boolean mask of cone members.
    """
    # --- aperture of the anchor -------------------------------------------
    psi = cone_aperture(x.unsqueeze(0), curv, scale, eps).squeeze(0)

    # --- time components ---------------------------------------------------
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2))
    cand_time = torch.sqrt(1.0 / curv + torch.sum(candidates**2, dim=-1))

    # --- Lorentz inner product <x, y>_L -----------------------------------
    spatial_dot = candidates @ x
    lorentz_inner = spatial_dot - x_time * cand_time     # this is <x,y>_L
    lhs = -lorentz_inner                                 # = -<x,y>_L >= 1

    # --- hyperbolic norms (distance from origin) --------------------------
    x_hyp = torch.acosh(torch.clamp(x_time * curv**0.5, min=1.0 + eps))
    cand_hyp = torch.acosh(torch.clamp(cand_time * curv**0.5, min=1.0 + eps))

    rhs = x_hyp * torch.cosh(psi)

    # --- angular cone membership ------------------------------------------
    angular = lhs >= rhs

    # --- radial direction filter ------------------------------------------
    if direction == "outward":
        radial = cand_hyp > x_hyp        # children: further out
    else:
        radial = cand_hyp < x_hyp        # parents: closer in

    # --- hyperbolic distance d(x, y) = arccosh(-<x,y>_L) ------------------
    dist = torch.acosh(torch.clamp(lhs, min=1.0 + eps))
    if band is not None:
        within_band = dist <= band
        return angular & radial & within_band
    return angular & radial