from pathlib import Path

import numpy as np
import pandas as pd


def plot_data_distribution(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    *,
    house_alias: str,
    required: list[str],
    zones: list[int],
    results_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    channels = [
        c
        for c in required
        if not (str(c).startswith("zone") and str(c).endswith("_heatcall_fraction"))
    ]
    n_channels = len(channels)
    col_width_in = 1.75
    fig, axes = plt.subplots(
        2,
        n_channels,
        figsize=(col_width_in * n_channels + 1.25, 8),
        squeeze=False,
        gridspec_kw={"wspace": 0.75},
    )
    row_labels = ("Before cleaning", "After cleaning")
    for row, (label, df) in enumerate(zip(row_labels, (df_before, df_after))):
        for col_idx, channel in enumerate(channels):
            ax = axes[row, col_idx]
            ax.boxplot(df[channel].dropna().to_numpy(), vert=True, widths=0.22)
            if row == 0:
                n_nans = int(df_before[channel].isna().sum())
                nan_label = "NaN" if n_nans == 1 else "NaNs"
                ax.set_title(f"{channel}\n({n_nans} {nan_label})", fontsize=8)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.tick_params(axis="y", labelsize=8)
        axes[row, 0].set_ylabel(label)
        if row == 1:
            for z in zones:
                zone_avg_channels = [
                    c for c in channels if str(c).startswith(f"zone{z}_avg_")
                ]
                if not zone_avg_channels:
                    continue
                zone_avg_values = np.concatenate(
                    [df[channel].dropna().to_numpy() for channel in zone_avg_channels]
                )
                if not zone_avg_values.size:
                    continue
                y_min = float(zone_avg_values.min())
                y_max = float(zone_avg_values.max())
                pad = max((y_max - y_min) * 0.05, 0.25)
                shared_ylim = (y_min - pad, y_max + pad)
                zone_avg_set = set(zone_avg_channels)
                for col_idx, channel in enumerate(channels):
                    if channel in zone_avg_set:
                        axes[row, col_idx].set_ylim(shared_ylim)
    fig.suptitle(f"{house_alias.capitalize()}: channel distributions")
    fig.tight_layout()
    results_dir.mkdir(parents=True, exist_ok=True)
    savepath = results_dir / f"{house_alias}_channel_boxplots.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
