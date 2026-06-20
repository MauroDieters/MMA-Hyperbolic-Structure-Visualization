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
    (Operates on Lorentz-space components, not Poincaré)

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
    x_time = torch.sqrt(1.0 / curv + torch.sum(x**2, dim=-1)) # x0

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
        sinh_scale / torch.clamp(sinh_norm, min=eps), # Avoid zero division
        min=-1.0,
        max=1.0
    ) # arcsin defined in [-1,1]
    aperture = torch.asin(ratio)

    return aperture

########## HELPERS
#-----------------------------------------------------------------------
def cone_aperture_degrees(x: Tensor, curv: float | Tensor = 1.0,
                          scale: float = 5.0, eps: float = 1e-8) -> Tensor:
    """
    Helper function returning cone aperture in degrees.    
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
    Compute the Lorentz norm of points. HYPERBOLIC
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
    return torch.acosh(torch.clamp(x_time * curv**0.5, min=1.0 + eps)) # arccosh(z) is only defined for z ≥ 1

################## FOR POINCARE
#-----------------------------------------------------------------------
def is_inside_cone(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 5.0,
    eps: float = 1e-8
) -> Tensor:
    """
    OUTWARD cone membership in 2D Poincaré, matching the drawn wedge.

    The wedge is anchored at x and opens outward (away from origin) with
    half-angle psi/2 around the radial direction. A candidate is inside if
    it is further from the origin than x AND the angle (measured AT x)
    between the outward radial direction and the direction x->candidate is
    within psi/2.
    """
    x_lorentz = poincare_to_lorentz(x.unsqueeze(0), curv, eps).squeeze(0)
    psi = cone_aperture(x_lorentz.unsqueeze(0), curv, scale, eps).squeeze(0)

    x_norm = torch.norm(x).clamp(min=eps)

    # outward radial direction at the anchor (origin -> anchor)
    radial_dir = x / x_norm # shape (2,)

    # direction from anchor to each candidate
    delta = candidates - x.unsqueeze(0)          # shape (N, 2)
    delta_norm = torch.norm(delta, dim=-1).clamp(min=eps)
    delta_dir = delta / delta_norm.unsqueeze(1)

    cos_angle = (delta_dir @ radial_dir).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle) # angle AT the anchor

    cand_norm = torch.norm(candidates, dim=-1)
    further_out = cand_norm > x_norm
    within_angle = angle < (psi / 2.0)

    return further_out & within_angle

#-----------------------------------------------------------------------
def is_inside_cone_inward(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 5.0,
    eps: float = 1e-8
) -> Tensor:
    """
    INWARD cone membership in 2D Poincaré, matching the drawn inward wedge.
    Opens from x toward the origin with half-angle psi/2.
    """
    x_lorentz = poincare_to_lorentz(x.unsqueeze(0), curv, eps).squeeze(0)
    psi = cone_aperture(x_lorentz.unsqueeze(0), curv, scale, eps).squeeze(0)

    x_norm = torch.norm(x).clamp(min=eps)

    # inward radial direction at the anchor (anchor -> origin)
    radial_dir = -x / x_norm

    delta = candidates - x.unsqueeze(0)
    delta_norm = torch.norm(delta, dim=-1).clamp(min=eps)
    delta_dir = delta / delta_norm.unsqueeze(1)

    cos_angle = (delta_dir @ radial_dir).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)

    cand_norm = torch.norm(candidates, dim=-1)
    closer_in = cand_norm < x_norm
    within_angle = angle < (psi / 2.0)

    return closer_in & within_angle


#-----------------------------------------------------------------------

################# For true dimensions
def cone_members_truncated_lorentz(
    x: Tensor,
    candidates: Tensor,
    curv: float | Tensor = 1.0,
    scale: float = 1.0,
    direction: str = "outward",
    band: float = None,
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
        Smaller band = stricter = higher precision, lower recall. (this was for an experiment we did)

    Membership inequality: −⟨x,y⟩_L  ≥  ‖x‖_H · cosh(ψ)

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

    # 1) angular cone membership ------------------------------------------
    angular = lhs >= rhs / 2.0

    # 2) radial direction filter ------------------------------------------
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