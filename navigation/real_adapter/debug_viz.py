from __future__ import annotations

from pathlib import Path
import numpy as np

from .spec import Stage49ObservationSpec


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _class_to_int_map(class_map: np.ndarray) -> tuple[np.ndarray, list[str]]:
    labels = [str(x) for x in np.unique(class_map.astype(str))]
    lookup = {label: idx for idx, label in enumerate(labels)}
    encoded = np.vectorize(lambda x: lookup[str(x)], otypes=[np.int32])(class_map)
    return encoded, labels


def polar_costmap_to_cartesian(
    costmap: np.ndarray,
    spec: Stage49ObservationSpec | None = None,
    *,
    x_bins: int = 96,
    y_bins: int = 96,
    lateral_extent_m: float = 12.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Resample a polar Stage49 costmap into an x-forward/y-left grid.

    Returns a `(channels, x_bins, y_bins)` grid and a Matplotlib extent in
    `(y_min, y_max, x_min, x_max)` order for `imshow`.
    """
    spec = spec or Stage49ObservationSpec()
    cm = np.asarray(costmap, dtype=np.float32)
    expected_shape = (spec.channels, spec.angle_bins, spec.distance_bins)
    if cm.shape != expected_shape:
        raise ValueError(f"costmap must have shape {expected_shape}, got {cm.shape}")
    if x_bins <= 0 or y_bins <= 0:
        raise ValueError("x_bins and y_bins must be positive")
    if lateral_extent_m <= 0.0:
        raise ValueError("lateral_extent_m must be positive")

    x_centers = (np.arange(x_bins, dtype=np.float32) + 0.5) / x_bins * spec.max_distance_m
    y_centers = np.linspace(
        -float(lateral_extent_m),
        float(lateral_extent_m),
        y_bins,
        endpoint=False,
        dtype=np.float32,
    )
    y_centers += float(lateral_extent_m) / y_bins
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    dist = np.sqrt(x_grid * x_grid + y_grid * y_grid)
    angle = np.arctan2(y_grid, x_grid)
    valid = (dist > 0.0) & (dist <= spec.max_distance_m)

    angle_bins = ((angle + np.pi) / (2.0 * np.pi) * spec.angle_bins).astype(np.int64)
    distance_bins = (dist / spec.max_distance_m * spec.distance_bins).astype(np.int64)
    angle_bins = np.clip(angle_bins, 0, spec.angle_bins - 1)
    distance_bins = np.clip(distance_bins, 0, spec.distance_bins - 1)

    cartesian = np.zeros((spec.channels, x_bins, y_bins), dtype=np.float32)
    for channel in range(spec.channels):
        sampled = cm[channel, angle_bins, distance_bins]
        cartesian[channel, valid] = sampled[valid]
    cartesian[3, ~valid] = 1.0

    extent = (-float(lateral_extent_m), float(lateral_extent_m), 0.0, float(spec.max_distance_m))
    return cartesian, extent


def _cartesian_costmap_rgb(cartesian: np.ndarray) -> np.ndarray:
    cm = np.asarray(cartesian, dtype=np.float32)
    if cm.ndim != 3 or cm.shape[0] < 4:
        raise ValueError("cartesian costmap must have shape (4, x_bins, y_bins)")

    static = np.clip(cm[0], 0.0, 1.0)
    moving = np.clip(cm[1], 0.0, 1.0)
    soft = np.clip(cm[2], 0.0, 1.0)
    unknown = np.clip(cm[3], 0.0, 1.0)

    rgb = np.ones((*static.shape, 3), dtype=np.float32)
    unknown_color = np.asarray([0.80, 0.86, 0.92], dtype=np.float32)
    static_color = np.asarray([0.88, 0.18, 0.12], dtype=np.float32)
    moving_color = np.asarray([0.48, 0.16, 0.86], dtype=np.float32)
    soft_color = np.asarray([0.95, 0.62, 0.12], dtype=np.float32)

    unknown_alpha = 0.45 * unknown[..., None]
    rgb = rgb * (1.0 - unknown_alpha) + unknown_color * unknown_alpha
    for values, color in ((soft, soft_color), (moving, moving_color), (static, static_color)):
        alpha = values[..., None]
        rgb = rgb * (1.0 - alpha) + color * alpha
    return np.clip(rgb, 0.0, 1.0)


def save_adapter_debug_panel(
    output_path: str | Path,
    *,
    rgb: np.ndarray | None,
    depth_m: np.ndarray | None,
    class_map: np.ndarray | None,
    points_robot_xyz: np.ndarray | None,
    point_classes: np.ndarray | None,
    costmap: np.ndarray,
    action_text: str = "",
    spec: Stage49ObservationSpec | None = None,
) -> Path:
    """Save a compact PNG panel for inspecting adapter inputs/outputs."""
    plt = _require_matplotlib()
    spec = spec or Stage49ObservationSpec()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(costmap, dtype=np.float32)
    expected_shape = (spec.channels, spec.angle_bins, spec.distance_bins)
    if cm.shape != expected_shape:
        raise ValueError(f"costmap must have shape {expected_shape}, got {cm.shape}")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    ax = axes.ravel()

    if rgb is not None:
        ax[0].imshow(np.asarray(rgb))
    else:
        ax[0].text(0.5, 0.5, "no RGB", ha="center", va="center")
    ax[0].set_title("RGB")
    ax[0].axis("off")

    if depth_m is not None:
        im = ax[1].imshow(np.asarray(depth_m, dtype=np.float32), cmap="viridis")
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    else:
        ax[1].text(0.5, 0.5, "no depth", ha="center", va="center")
    ax[1].set_title("Depth (m)")
    ax[1].axis("off")

    if class_map is not None:
        encoded, labels = _class_to_int_map(np.asarray(class_map, dtype=object))
        ax[2].imshow(encoded, cmap="tab20", vmin=0, vmax=max(1, len(labels) - 1))
        ax[2].text(0.01, -0.08, ", ".join(labels[:8]), transform=ax[2].transAxes, fontsize=8)
    else:
        ax[2].text(0.5, 0.5, "no classes", ha="center", va="center")
    ax[2].set_title("Class map")
    ax[2].axis("off")

    cartesian, extent = polar_costmap_to_cartesian(cm, spec)
    ax[3].imshow(
        _cartesian_costmap_rgb(cartesian),
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    pts = np.asarray(points_robot_xyz if points_robot_xyz is not None else np.zeros((0, 3)), dtype=np.float32)
    if pts.size:
        ax[3].scatter(pts[:, 1], pts[:, 0], s=8, alpha=0.8)
    ax[3].scatter([0], [0], marker="^", c="red", label="robot")
    ax[3].set_xlim(extent[0], extent[1])
    ax[3].set_ylim(extent[2], extent[3])
    ax[3].set_aspect("equal", adjustable="box")
    ax[3].set_xlabel("robot y left/right (m)")
    ax[3].set_ylabel("robot x forward (m)")
    ax[3].set_title("Cartesian costmap")
    ax[3].grid(True, alpha=0.3)

    channel_titles = ["static", "moving", "soft", "not visible"]
    summary = np.concatenate([cm[i] for i in range(4)], axis=1)
    im = ax[4].imshow(summary.T, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax[4], fraction=0.046, pad=0.04)
    ax[4].set_title("Costmap channels: " + " | ".join(channel_titles))
    ax[4].set_xlabel("angle bin")
    ax[4].set_ylabel("distance bin")

    ax[5].axis("off")
    text = action_text or "No policy action supplied"
    text += f"\n\nCostmap max: {float(cm.max()):.3f}"
    text += f"\nObstacle max: {float(cm[:3].max()):.3f}"
    if point_classes is not None and len(point_classes):
        unique, counts = np.unique(np.asarray(point_classes).astype(str), return_counts=True)
        text += "\nClasses: " + ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
    ax[5].text(0.02, 0.98, text, ha="left", va="top", family="monospace")
    ax[5].set_title("Adapter summary")

    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
