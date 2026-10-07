"""Render separate DenseNet plots from the existing measured CSVs; no model rerun."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIELDS = [
    ("S", "Relative separation S"),
    ("CKA_prev", "Adjacent raw linear CKA"),
    ("d_within_cos", "Within cosine distance"),
    ("d_between_cos", "Between cosine distance"),
    ("PR", "Participation ratio"),
    ("probe_mean", "Held-out probe accuracy"),
]

def render(results):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.2})
    for number in (121, 169, 201):
        slug = f"densenet{number}"
        name = f"DenseNet-{number}"
        geometry = pd.read_csv(results / "geometry" / slug / f"{slug}_geometry.csv")
        entropy = pd.read_csv(results / "lle" / slug / f"{slug}_lle_summary.csv")
        assert set(geometry.model) == {name}
        assert set(entropy.model) == {name}
        geometry = geometry.sort_values("depth")
        output = results / "figures" / slug
        output.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        fig.suptitle(f"{name} — matched-image geometry", fontsize=16)
        for ax, (field, title) in zip(axes.flat, FIELDS):
            ax.plot(geometry.depth, geometry[field], color="#2474a6", linewidth=1.8)
            ax.set(title=title, xlabel="Observation index", ylabel=field)
        fig.savefig(output / f"{slug}_geometry.png", dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for k, values in entropy.groupby("k", sort=True):
            values = values.sort_values("depth")
            ax.plot(values.depth, values.LLE, label=f"k={k}", linewidth=1.6)
        ax.set(title=f"{name} — local label entropy",
               xlabel="Observation index", ylabel="Mean local label entropy (bits)",
               ylim=(0, 1))
        ax.legend(title="Neighborhood size", fontsize=9)
        fig.savefig(output / f"{slug}_lle.png", dpi=160)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    render(parser.parse_args().results)
