"""The benchmark's one figure, as SVG (no plotting library needed): each cell's score against its sequencing depth,
the cells with the module on in color. A good score separates the colors and does not slope with depth; the
metrics measure both, and the figure shows how."""
import numpy as np
from numpy.typing import NDArray

W, H, PAD = 320, 220, 34


def score_vs_depth(score: NDArray[np.floating], depth: NDArray[np.floating], active: NDArray[np.bool_],
                   title: str) -> str:
    x, y = np.log10(depth), score

    def scale(v: NDArray[np.floating], lo: float, hi: float) -> NDArray[np.floating]:
        span = v.max() - v.min() or 1.0
        return lo + (v - v.min()) / span * (hi - lo)
    px, py = scale(x, PAD, W - 10), scale(y, H - PAD, 22)
    dots = "".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.2" fill="{"#c4352b" if on else "#8a8a90"}" '
                   f'fill-opacity="{0.85 if on else 0.45}"/>' for a, b, on in zip(px, py, active))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">'
            f'<rect width="{W}" height="{H}" fill="#fff"/>'
            f'<text x="{PAD}" y="14" fill="#333">{title}</text>'
            f'<line x1="{PAD}" y1="{H - PAD + 6}" x2="{W - 10}" y2="{H - PAD + 6}" stroke="#999"/>'
            f'<line x1="{PAD - 6}" y1="22" x2="{PAD - 6}" y2="{H - PAD}" stroke="#999"/>'
            f'<text x="{(PAD + W) / 2}" y="{H - 8}" text-anchor="middle" fill="#666">sequencing depth (log)</text>'
            f'<text x="12" y="{H / 2}" text-anchor="middle" fill="#666" transform="rotate(-90 12 {H / 2})">score</text>'
            f'<text x="{W - 10}" y="14" text-anchor="end" fill="#c4352b">● module on</text>'
            f'{dots}</svg>')
