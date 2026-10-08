"""Visualization functions for clustering analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


def plot_price_cluster_centroids(price_profiles):
    """Plot price cluster centroids."""
    price_defs = [
        ("Imbalance price", "Imb_",  r"$\lambda^{\mathrm{imb}}$ [€/MWh]"),
        ("aFRR activation price", "EAM_", r"$\lambda^{\mathrm{EAM}}$ [€/MWh]"),
    ]

    clusters = sorted(price_profiles["cluster_price"].unique())
    colors = plt.cm.tab10(range(len(clusters)))
    hours = np.arange(96) * 0.25

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 10),   
        sharex=True,
        constrained_layout=False
    )

    for ax, (_, prefix, ylabel) in zip(axes, price_defs):
        cols = [c for c in price_profiles if c.startswith(prefix)]
        for cl, col in zip(clusters, colors):
            centroid = price_profiles[price_profiles.cluster_price == cl][cols].mean()
            ax.plot(hours, centroid, color=col, lw=2)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Hour of day")

    fig.legend(
        handles=[Line2D([0], [0], color=c, lw=3) for c in colors],
        labels=[f"Cluster {c}" for c in clusters],
        title="Price clusters",
        loc="lower center",
        ncol=len(clusters),
        frameon=False
    )

    fig.subplots_adjust(bottom=0.18, hspace=0.12)
    plt.savefig("fig_price_cluster_centroids.pdf", bbox_inches='tight')
    plt.show()


def plot_consumption_cluster_centroids(cons_profiles_clu):
    """Plot consumption cluster centroids."""
    cons_cols = [c for c in cons_profiles_clu if c.startswith("Cons_")]
    hours = np.arange(len(cons_cols)) * 0.25
    clusters = sorted(cons_profiles_clu["cluster_cons"].unique())
    colors = plt.cm.tab10(range(len(clusters)))

    fig, ax = plt.subplots(
        figsize=(14, 6),          
        constrained_layout=False
    )

    for cl, col in zip(clusters, colors):
        centroid = (
            cons_profiles_clu[cons_profiles_clu.cluster_cons == cl][cons_cols]
            .mean()
        )
        ax.plot(hours, centroid.values, color=col, lw=2)

    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"$C^{\mathrm{base}}$ [MW]")
    ax.set_yticks([])
    ax.grid(alpha=0.3)

    fig.legend(
        handles=[
            Line2D([0], [0], color=col, lw=3, label=f"Cluster {cl}")
            for cl, col in zip(clusters, colors)
        ],
        title="Consumption clusters",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(clusters),
        frameon=False
    )

    fig.subplots_adjust(bottom=0.18)
    plt.tight_layout()
    plt.savefig("fig_consumption_cluster_centroids.pdf", bbox_inches='tight')
    plt.show()


def plot_cluster_pca(X_price, price_profiles, X_cons, cons_profiles_clu):
    """Plot PCA visualization of clusters."""
    from sklearn.decomposition import PCA
    
    pca_p = PCA(n_components=2)
    Pp = pca_p.fit_transform(X_price)

    plt.figure(figsize=(7,5))
    plt.scatter(Pp[:,0], Pp[:,1], c=price_profiles["cluster_price"], cmap="tab10", s=35)
    plt.title("Price clusters (PCA)")
    plt.grid(alpha=0.3)
    plt.show()

    pca_c = PCA(n_components=2)
    Pc = pca_c.fit_transform(X_cons)

    plt.figure(figsize=(7,5))
    plt.scatter(Pc[:,0], Pc[:,1], c=cons_profiles_clu["cluster_cons"], cmap="tab10", s=35)
    plt.title("Consumption clusters (PCA)")
    plt.grid(alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.box(False)
    plt.show()


def plot_similar_days(df, days, value_col, title):
    """Plot similar days overlaid."""
    plt.figure(figsize=(10,5))
    for d in days:
        d0 = pd.Timestamp(d)
        sub = df.loc[d0 : d0 + pd.Timedelta(days=1), value_col]
        if not sub.empty:
            hours = (sub.index - sub.index.normalize()).total_seconds() / 3600
            plt.plot(hours, sub.values, alpha=0.8, label=str(d.date()))
    plt.title(title)
    plt.xlim(0,24)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_forecast_features(spot_vals, temp_vals, wind_vals, day):
    """Plot temperature and wind forecasts."""
    hours = np.arange(96) * 0.25

    plt.figure(figsize=(10,4))
    plt.plot(hours, temp_vals)
    plt.title(f"Temperature forecast – {day}")
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10,4))
    plt.plot(hours, wind_vals)
    plt.title(f"Wind speed forecast – {day}")
    plt.grid(alpha=0.3)
    plt.show()


def plot_raw_days(
    df,
    similar_days,
    value_col,
    title,
    ylabel=None,
):
    """Plot raw profiles for selected days."""
    plt.figure(figsize=(10, 5))

    for d in similar_days:
        day_start = pd.Timestamp(d)
        day_end   = day_start + pd.Timedelta(days=1)

        day_data = df.loc[
            (df.index >= day_start) & (df.index < day_end),
            value_col
        ]

        if not day_data.empty:
            hours = (
                day_data.index - day_data.index.normalize()
            ).total_seconds() / 3600

            plt.plot(
                hours,
                day_data.values,
                alpha=0.8,
                label=str(day_start.date()),
            )

    plt.title(title)
    plt.xlabel("Hour of day")
    plt.ylabel(ylabel if ylabel is not None else value_col)
    plt.xlim(0, 24)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_raw_days_multi(
    panels,
    figsize=(12, 9)
):
    """Plot multiple raw day profiles in a grid."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)

    if n == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):

        df = panel["df"]
        value_col = panel["value_col"]
        similar_days = panel["similar_days"]

        for d in similar_days:
            day_start = pd.Timestamp(d)
            day_end   = day_start + pd.Timedelta(days=1)

            day_data = df.loc[
                (df.index >= day_start) & (df.index < day_end),
                value_col
            ]

            if not day_data.empty:
                hours = (
                    day_data.index - day_data.index.normalize()
                ).total_seconds() / 3600

                ax.plot(hours, day_data.values, alpha=0.7)

        ax.set_title(panel["title"])
        ax.set_ylabel(panel["ylabel"])
        ax.grid(alpha=0.3)
        if panel.get("hide_yticks", False):
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig("fig_raw_days_multi.pdf", bbox_inches="tight")
    plt.show()


def plot_raw_days_per_box(df_per_box, similar_days, value_col='power', title_prefix="Raw profiles per box"):
    """Plot raw profiles per charging box."""
    boxes = df_per_box['chargeBoxId'].unique()
    n_boxes = len(boxes)
    fig, axes = plt.subplots(n_boxes, 1, figsize=(10, 3 * n_boxes), sharex=True)
    if n_boxes == 1:
        axes = [axes]
    for ax, cbid in zip(axes, boxes):
        df_cb = df_per_box[df_per_box['chargeBoxId'] == cbid]
        for d in similar_days:
            day_start, day_end = pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1)
            day_data = df_cb.loc[(df_cb.index >= day_start) & (df_cb.index < day_end), value_col]
            if not day_data.empty:
                hours = (day_data.index - day_data.index.normalize()).total_seconds() / 3600
                ax.plot(hours, day_data.values, label=str(d.date()), alpha=0.7)
        ax.set_title(f"{title_prefix} – {cbid}")
        ax.set_ylabel(value_col)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 3))
    axes[-1].set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    plt.tight_layout()
    plt.show()


def plot_all_days_on_ax(ax, df, value_col, ylabel):
    """Plot all days on a given axis."""
    all_days = sorted(df.index.normalize().unique())

    for d in all_days:
        day_start, day_end = d, d + pd.Timedelta(days=1)
        day_data = df.loc[
            (df.index >= day_start) & (df.index < day_end), value_col
        ]

        if not day_data.empty:
            hours = (
                day_data.index - day_data.index.normalize()
            ).total_seconds() / 3600
            ax.plot(hours, day_data.values, alpha=0.25)

    ax.set_xlim(0, 24)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    print(f"Plotted {len(all_days)} days for {value_col}")
    months = sorted(df.index.normalize().unique().month)
    print("Months included:", months)
