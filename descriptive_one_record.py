"""
descriptive_statistics.py

Generates descriptive statistics, classification summary tables, and visual 
distributions for N=1 YouTube exposure and wellbeing data.
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set clean presentation aesthetic
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Modern palette with translucent neutral styling for 'other' and 'unclassified'
CATEGORY_PALETTE = {
    "Mental Health": "#2A9D8F",   # Soft Calming Teal
    "Sports": "#264653",          # Deep Slate Navy
    "Entertainment": "#E9C46A",   # Warm Gold
    "Gaming": "#F4A261",          # Terracotta Orange
    "Music": "#8AB17D",           # Muted Sage Green
    "News & Politics": "#457B9D", # Steel Blue
    "unclassified": "#90A4AE80",  # Translucent Cool Slate Gray
    "other": "#D1C4E999"          # Translucent Soft Lavender Gray
}

FALLBACK_COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#457B9D", "#8AB17D"]

def get_category_colors(columns):
    """Map categories to consistent custom palette colors."""
    return [CATEGORY_PALETTE.get(col, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]) for i, col in enumerate(columns)]


def generate_descriptive_statistics(
    jsonl_path,
    youtube_csv,
    wellbeing_csv,
    exposure_window_days,
):
    print("--- Step 1: Loading & Linking Datasets ---")

    # 1. Load Zero-Shot classifications
    preds_df = pd.read_json(jsonl_path, lines=True)

    label_col = next(
        (c for c in ["predicted_label", "label"] if c in preds_df.columns), None
    )
    score_col = next(
        (
            c
            for c in ["confidence_score", "confidnce_score", "score"]
            if c in preds_df.columns
        ),
        None,
    )

    preds_subset = preds_df[["video_id", label_col, score_col]].rename(
        columns={label_col: "predicted_label", score_col: "score"}
    )

    # 2. Load YouTube watch history
    yt_df = pd.read_csv(youtube_csv, low_memory=False)
    yt_df["video_id"] = yt_df["titleUrl"].str.extract(r"v=([a-zA-Z0-9_-]{11})")

    # 3. Join predictions onto YouTube metadata
    yt_full = yt_df.merge(preds_subset, on="video_id", how="left")
    
    # Clean label strings
    yt_full["predicted_label"] = (
        yt_full["predicted_label"]
        .fillna("unclassified")
        .astype(str)
        .str.replace('"', "")
        .str.strip()
    )
    
    # Scale down 'other' category count by a factor of 10 for dummy presentation purposes
    is_other = yt_full["predicted_label"].str.lower() == "other"
    if is_other.any():
        yt_other = yt_full[is_other].sample(frac=0.1, random_state=42)
        yt_full = pd.concat([yt_full[~is_other], yt_other]).sort_index()

    yt_full["time_UTC"] = pd.to_datetime(
        yt_full["time_UTC"], utc=True, errors="coerce"
    )

    # 4. Load Digital Wellbeing survey data
    wellbeing_df = pd.read_csv(wellbeing_csv, low_memory=False)
    wellbeing_df["start_UTC"] = pd.to_datetime(
        wellbeing_df["start_UTC"], utc=True, errors="coerce"
    )

    # --- ARTIFICIAL DATE SHIFT (Align last event to 8th Sept 2026) ---
    target_end_date = pd.Timestamp("2026-09-08", tz="UTC")
    latest_event_date = max(yt_full["time_UTC"].max(), wellbeing_df["start_UTC"].max())
    
    if pd.notna(latest_event_date):
        date_offset = target_end_date - latest_event_date.floor("D")
        yt_full["time_UTC"] = yt_full["time_UTC"] + date_offset
        wellbeing_df["start_UTC"] = wellbeing_df["start_UTC"] + date_offset

    target_start_date = pd.Timestamp("2025-09-08", tz="UTC")
    
    # Increased padding to 35 days so the final PHQ-8 text callout box has plenty of room
    plot_start_date = target_start_date.date()
    plot_end_date = (target_end_date + pd.Timedelta(days=35)).date()

    # Dynamic column lookup for PHQ score column ('PHQ-8', 'phq8_score', etc.)
    phq_col = next(
        (c for c in ["PHQ-8", "phq8_score", "phq-8", "PHQ8"] if c in wellbeing_df.columns),
        "PHQ-8",
    )

    print("\n--- Step 2: Descriptive Summary Statistics ---")
    total_videos = len(yt_full)
    unique_videos = yt_full["video_id"].nunique()
    mean_confidence = yt_full["score"].mean()
    median_confidence = yt_full["score"].median()

    print(f"Total Watch Events (with scaled 'other'): {total_videos}")
    print(f"Unique Videos: {unique_videos}")
    print(f"Mean Confidence Score: {mean_confidence:.3f}")
    print(f"Median Confidence Score: {median_confidence:.3f}\n")

    display_cols = ["title", "predicted_label", "score"]
    available_cols = [c for c in display_cols if c in yt_full.columns]
    sample_table = (
        yt_full[available_cols]
        .dropna()
        .sort_values(by="score", ascending=False)
    )

    print("Highest-Confidence Categorized Videos Sample:")
    print(sample_table.head(8).to_string(index=False))

    print("\n--- Step 3: Figure 1 (Category Breakdown) ---")
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    cat_counts = yt_full["predicted_label"].value_counts().sort_values(ascending=True)

    bar_colors = [CATEGORY_PALETTE.get(cat, "#264653") for cat in cat_counts.index]
    bars = ax1.barh(cat_counts.index, cat_counts.values, color=bar_colors, edgecolor="none", height=0.65)
    ax1.set_xlabel("Total Videos Watched", labelpad=8)
    ax1.set_ylabel("Predicted Category", labelpad=8)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6, color="#D0D0D0")

    for bar in bars:
        width = bar.get_width()
        ax1.annotate(
            f"{int(width):,}",
            (width + 0.4, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
            color="#333333"
        )

    plt.tight_layout()
    plt.savefig("fig1_category_distribution.png", dpi=300)
    plt.close()

    print("--- Step 4: Figure 2 (Classifier Certainty Distribution) ---")
    fig, ax2 = plt.subplots(figsize=(8, 4))
    scores = yt_full["score"].dropna()
    ax2.hist(scores, bins=15, color="#457B9D", edgecolor="white", alpha=0.9)
    ax2.axvline(
        mean_confidence,
        color="#D9381E",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_confidence:.2f}",
    )
    ax2.axvline(
        median_confidence,
        color="#E9C46A",
        linestyle=":",
        linewidth=2.2,
        label=f"Median: {median_confidence:.2f}",
    )
    ax2.set_xlabel("Confidence Score", labelpad=8)
    ax2.set_ylabel("Video Frequency", labelpad=8)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")
    ax2.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
    plt.tight_layout()
    plt.savefig("fig2_confidence_scores.png", dpi=300)
    plt.close()

    print("--- Step 5: Figure 3 (Circadian 24-Hour Viewing Pattern) ---")
    yt_valid = yt_full.dropna(subset=["time_UTC"]).copy()
    yt_valid["hour"] = yt_valid["time_UTC"].dt.hour

    hourly_counts = (
        yt_valid.groupby(["hour", "predicted_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(range(24), fill_value=0)
    )

    fig, ax3 = plt.subplots(figsize=(10, 4.5))
    hourly_counts.plot(
        kind="bar", stacked=True, ax=ax3, color=get_category_colors(hourly_counts.columns), width=0.75
    )

    ax3.set_xlabel("Hour of Day (UTC)", labelpad=8)
    ax3.set_ylabel("Total Videos Watched", labelpad=8)
    ax3.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6, color="#D0D0D0")
    ax3.legend(title="Category", bbox_to_anchor=(1.02, 0), loc="lower left", frameon=False)

    plt.tight_layout()
    plt.savefig("fig3_circadian_profile.png", dpi=300)
    plt.close()

    # Pre-process dates for daily longitudinal plots
    yt_valid["date"] = pd.to_datetime(yt_valid["time_UTC"].dt.date)

    # Compute daily total and daily sports counts
    daily_total = yt_valid.groupby("date").size()
    is_sports = yt_valid["predicted_label"].astype(str).str.lower().str.contains("sports")
    daily_sports = yt_valid[is_sports].groupby("date").size()

    survey_dates = pd.to_datetime(
        wellbeing_df["start_UTC"].dropna().dt.date.unique()
    )
    
    full_date_range = pd.date_range(start=plot_start_date, end=plot_end_date, freq="D")
    daily_total = daily_total.reindex(full_date_range, fill_value=0)
    daily_sports = daily_sports.reindex(full_date_range, fill_value=0)

    print("\n--- Step 6: Figure 4 (Total Videos Daily Timeline) ---")
    fig, ax4 = plt.subplots(figsize=(12, 5))
    
    ax4.plot(
        full_date_range, daily_total,
        color="#264653", linewidth=1.3, alpha=1.0, label="Total Videos Watched"
    )

    ax4.set_xlim(plot_start_date, plot_end_date)
    ax4.set_xlabel("Date", labelpad=8)
    ax4.set_ylabel("Daily Videos Watched", labelpad=8)
    ax4.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")
    ax4.legend(loc="upper left", frameon=False)

    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)
    plt.savefig("fig4_daily_engagement_timeline.png", dpi=300)
    plt.close()

    print("\n--- Step 7: Figure 5 (Total Videos Daily Timeline with PHQ-8 Overlay) ---")
    fig, ax5 = plt.subplots(figsize=(12, 5))
    
    ax5.plot(
        full_date_range, daily_total,
        color="#264653", linewidth=1.3, alpha=1.0, label="Total Videos Watched"
    )

    ax5.set_xlim(plot_start_date, plot_end_date)
    ax5.set_xlabel("Date", labelpad=8)
    ax5.set_ylabel("Daily Videos Watched", labelpad=8)
    ax5.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")

    for s_date in survey_dates:
        if plot_start_date <= s_date.date() <= plot_end_date:
            score_str = "Survey"
            if phq_col in wellbeing_df.columns:
                phq_val = wellbeing_df.loc[
                    wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
                ].values
                if len(phq_val) > 0 and pd.notna(phq_val[0]):
                    score_str = f"PHQ-8: {phq_val[0]}"

            ax5.axvline(x=s_date, color="#D9381E", linestyle="--", linewidth=1.8)
            ax5.text(
                s_date,
                ax5.get_ylim()[1] * 0.85,
                f" {score_str}",
                color="#D9381E",
                fontweight="bold",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35", fc="white", ec="#D9381E", lw=1, alpha=0.95
                ),
            )

    ax5.legend(loc="upper left", frameon=False)
    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)
    plt.savefig("fig5_daily_total_phq8.png", dpi=300)
    plt.close()

    print("\n--- Step 8: Figure 6 (Translucent Total & Full Color Sports Videos with PHQ-8 Overlay) ---")
    fig, ax6 = plt.subplots(figsize=(12, 5))
    
    ax6.plot(
        full_date_range, daily_total,
        color="#264653", linewidth=1.3, alpha=0.35, label="Total Videos Watched"
    )
    ax6.plot(
        full_date_range, daily_sports,
        color="#2A9D8F", linewidth=1.5, alpha=1.0, label="Sports Videos"
    )

    ax6.set_xlim(plot_start_date, plot_end_date)
    ax6.set_xlabel("Date", labelpad=8)
    ax6.set_ylabel("Daily Videos Watched", labelpad=8)
    ax6.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")

    for s_date in survey_dates:
        if plot_start_date <= s_date.date() <= plot_end_date:
            score_str = "Survey"
            if phq_col in wellbeing_df.columns:
                phq_val = wellbeing_df.loc[
                    wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
                ].values
                if len(phq_val) > 0 and pd.notna(phq_val[0]):
                    score_str = f"PHQ-8: {phq_val[0]}"

            ax6.axvline(x=s_date, color="#D9381E", linestyle="--", linewidth=1.8)
            ax6.text(
                s_date,
                ax6.get_ylim()[1] * 0.85,
                f" {score_str}",
                color="#D9381E",
                fontweight="bold",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.35", fc="white", ec="#D9381E", lw=1, alpha=0.95
                ),
            )

    ax6.legend(loc="upper left", frameon=False)
    plt.subplots_adjust(left=0.07, right=0.95, top=0.92, bottom=0.12)
    plt.savefig("fig6_longitudinal_sports_exposure.png", dpi=300)
    plt.close()

    print(
        f"--- Step 9: Figure 7 ({exposure_window_days}-Day Preceding Exposure per Assessment) ---"
    )
    exposure_records = []
    exposure_data_plot = []

    for s_date in survey_dates:
        window_start = s_date - pd.Timedelta(days=exposure_window_days)
        in_window = yt_valid[
            (yt_valid["date"] >= window_start) & (yt_valid["date"] <= s_date)
        ]

        phq_str = "N/A"
        if phq_col in wellbeing_df.columns:
            phq_val = wellbeing_df.loc[
                wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
            ].values
            if len(phq_val) > 0 and pd.notna(phq_val[0]):
                phq_str = f"PHQ-8: {phq_val[0]}"

        cat_counts_window = (
            in_window["predicted_label"].value_counts().to_dict()
        )
        record = {
            "survey_date": s_date.date(),
            "PHQ-8": phq_str,
            f"total_in_{exposure_window_days}d": len(in_window),
        }
        for cat, count in cat_counts_window.items():
            record[f"count_{cat}"] = count
        exposure_records.append(record)

        label = f"Survey {s_date.date()}\n({phq_str})"
        for cat, count in in_window["predicted_label"].value_counts().items():
            exposure_data_plot.append(
                {"Survey_Assessment": label, "Category": cat, "Count": count}
            )

    exposure_summary_df = pd.DataFrame(exposure_records).fillna(0)
    print(f"\n{exposure_window_days}-Day Preceding Exposure Summary Table:")
    print(exposure_summary_df.to_string(index=False))

    fig, ax7 = plt.subplots(figsize=(10, 5))
    exp_df = pd.DataFrame(exposure_data_plot)
    if not exp_df.empty:
        exp_pivot = exp_df.pivot(
            index="Survey_Assessment", columns="Category", values="Count"
        ).fillna(0)
        exp_pivot.plot(kind="bar", ax=ax7, color=get_category_colors(exp_pivot.columns), width=0.7)

    ax7.set_ylabel(f"{exposure_window_days}-Day Video Volume", labelpad=8)
    ax7.set_xlabel("")
    ax7.grid(True, axis="y", linestyle=":", alpha=0.6, color="#D0D0D0")
    ax7.tick_params(axis="x", rotation=0)
    ax7.legend(title="Category", bbox_to_anchor=(1.02, 0), loc="lower left", frameon=False)

    plt.tight_layout()
    plt.savefig(f"fig7_{exposure_window_days}day_exposure.png", dpi=300)
    plt.close()

    print("\n--- Step 10: Figure 8 & 9 (4-Wave PHQ-8 vs Sports Exposure Analysis) ---")
    wave_data = []
    for s_date in survey_dates:
        window_start = s_date - pd.Timedelta(days=exposure_window_days)
        in_window = yt_valid[
            (yt_valid["date"] >= window_start) & (yt_valid["date"] <= s_date)
        ]

        total_vids = len(in_window)
        sports_vids = len(
            in_window[
                in_window["predicted_label"].astype(str).str.lower().str.contains("sports")
            ]
        )
        sports_prop = (sports_vids / total_vids) * 100 if total_vids > 0 else 0.0

        phq_val = None
        if phq_col in wellbeing_df.columns:
            vals = wellbeing_df.loc[
                wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
            ].values
            if len(vals) > 0 and pd.notna(vals[0]):
                phq_val = float(vals[0])

        wave_data.append({
            "survey_date": s_date.date(),
            "phq8": phq_val,
            "sports_prop": sports_prop,
            "total_vids": total_vids
        })

    wave_df = pd.DataFrame(wave_data).dropna(subset=["phq8"]).sort_values("survey_date")

    if len(wave_df) >= 2:
        r_val = wave_df["sports_prop"].corr(wave_df["phq8"])
        print(f"Descriptive Pearson r (Sports % vs PHQ-8 across {len(wave_df)} waves): {r_val:.3f}")

        # Figure 8: Dual-Axis Trajectory Plot (Sports Exposure vs PHQ-8)
        fig, ax8_1 = plt.subplots(figsize=(9, 4.5))
        color_phq = "#D9381E"
        color_sports = "#264653"

        ax8_1.set_xlabel("Survey Wave Date", fontweight="bold", labelpad=8)
        ax8_1.set_ylabel("PHQ-8 Depression Score", color=color_phq, fontweight="bold", labelpad=8)
        ax8_1.plot(
            wave_df["survey_date"],
            wave_df["phq8"],
            color=color_phq,
            marker="o",
            markersize=7,
            linewidth=2.5,
            label="PHQ-8 Score",
        )
        ax8_1.tick_params(axis="y", labelcolor=color_phq)
        ax8_1.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")

        ax8_2 = ax8_1.twinx()
        ax8_2.spines["right"].set_visible(True)
        ax8_2.spines["right"].set_color("#CCCCCC")
        ax8_2.set_ylabel(
            f"Sports Exposure (% of Total {exposure_window_days}d Videos)",
            color=color_sports,
            fontweight="bold",
            labelpad=8
        )
        ax8_2.plot(
            wave_df["survey_date"],
            wave_df["sports_prop"],
            color=color_sports,
            marker="s",
            markersize=7,
            linestyle="--",
            linewidth=2.5,
            label="Sports Exposure %",
        )
        ax8_2.tick_params(axis="y", labelcolor=color_sports)

        fig.tight_layout()
        plt.savefig("fig8_sports_dual_axis_trajectory.png", dpi=300)
        plt.close()

        # Figure 9: Scatter Plot with Trendline (Sports % vs PHQ-8)
        fig, ax9 = plt.subplots(figsize=(7, 4.5))
        ax9.scatter(wave_df["sports_prop"], wave_df["phq8"], color="#264653", s=110, zorder=3)

        if len(wave_df) > 1:
            m, b = np.polyfit(wave_df["sports_prop"], wave_df["phq8"], 1)
            x_vals = np.linspace(wave_df["sports_prop"].min(), wave_df["sports_prop"].max(), 100)
            ax9.plot(
                x_vals,
                m * x_vals + b,
                color="#7F8C8D",
                linestyle=":",
                linewidth=2,
                label=f"Trendline (r = {r_val:.2f})",
            )

        for idx, row in wave_df.reset_index(drop=True).iterrows():
            ax9.annotate(
                f" PHQ-8 measurement {idx + 1}",
                (row["sports_prop"], row["phq8"]),
                fontsize=9,
                color="#444444"
            )

        ax9.set_xlabel(f"Preceding {exposure_window_days}-Day Sports Exposure (%)", labelpad=8)
        ax9.set_ylabel("PHQ-8 Score", labelpad=8)
        ax9.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")
        ax9.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
        plt.tight_layout()
        plt.savefig("fig9_sports_vs_phq8_scatter.png", dpi=300)
        plt.close()

    print("\n--- Step 11: Figure 10 & 11 (Late-Night Viewing [9PM-7AM] vs PHQ-8 Analysis) ---")
    late_night_wave_data = []
    for s_date in survey_dates:
        window_start = s_date - pd.Timedelta(days=exposure_window_days)
        in_window = yt_valid[
            (yt_valid["date"] >= window_start) & (yt_valid["date"] <= s_date)
        ]

        total_vids = len(in_window)
        late_vids = len(in_window[(in_window["hour"] >= 21) | (in_window["hour"] < 7)])
        late_prop = (late_vids / total_vids) * 100 if total_vids > 0 else 0.0

        phq_val = None
        if phq_col in wellbeing_df.columns:
            vals = wellbeing_df.loc[
                wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
            ].values
            if len(vals) > 0 and pd.notna(vals[0]):
                phq_val = float(vals[0])

        late_night_wave_data.append({
            "survey_date": s_date.date(),
            "phq8": phq_val,
            "late_prop": late_prop,
            "total_vids": total_vids
        })

    late_df = pd.DataFrame(late_night_wave_data).dropna(subset=["phq8"]).sort_values("survey_date")

    if len(late_df) >= 2:
        r_late = late_df["late_prop"].corr(late_df["phq8"])
        print(f"Descriptive Pearson r (Late Night % vs PHQ-8 across {len(late_df)} waves): {r_late:.3f}")

        # Figure 10: Dual-Axis Trajectory Plot (Late Night % vs PHQ-8)
        fig, ax10_1 = plt.subplots(figsize=(9, 4.5))
        color_phq = "#D9381E"
        color_late = "#6C5CE7"

        ax10_1.set_xlabel("Survey Wave Date", fontweight="bold", labelpad=8)
        ax10_1.set_ylabel("PHQ-8 Depression Score", color=color_phq, fontweight="bold", labelpad=8)
        ax10_1.plot(
            late_df["survey_date"],
            late_df["phq8"],
            color=color_phq,
            marker="o",
            markersize=7,
            linewidth=2.5,
            label="PHQ-8 Score",
        )
        ax10_1.tick_params(axis="y", labelcolor=color_phq)
        ax10_1.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")

        ax10_2 = ax10_1.twinx()
        ax10_2.spines["right"].set_visible(True)
        ax10_2.spines["right"].set_color("#CCCCCC")
        ax10_2.set_ylabel(
            f"Late Night Exposure % (9PM-7AM, {exposure_window_days}d Window)",
            color=color_late,
            fontweight="bold",
            labelpad=8
        )
        ax10_2.plot(
            late_df["survey_date"],
            late_df["late_prop"],
            color=color_late,
            marker="^",
            markersize=7,
            linestyle="--",
            linewidth=2.5,
            label="Late Night Viewing %",
        )
        ax10_2.tick_params(axis="y", labelcolor=color_late)

        fig.tight_layout()
        plt.savefig("fig10_late_night_trajectory.png", dpi=300)
        plt.close()

        # Figure 11: Scatter Plot with Trendline (Late Night % vs PHQ-8)
        fig, ax11 = plt.subplots(figsize=(7, 4.5))
        ax11.scatter(late_df["late_prop"], late_df["phq8"], color="#6C5CE7", s=110, zorder=3)

        if len(late_df) > 1:
            m, b = np.polyfit(late_df["late_prop"], late_df["phq8"], 1)
            x_vals = np.linspace(late_df["late_prop"].min(), late_df["late_prop"].max(), 100)
            ax11.plot(
                x_vals,
                m * x_vals + b,
                color="#7F8C8D",
                linestyle=":",
                linewidth=2,
                label=f"Trendline (r = {r_late:.2f})",
            )

        for idx, row in late_df.reset_index(drop=True).iterrows():
            ax11.annotate(
                f" PHQ-8 measurement {idx + 1}",
                (row["late_prop"], row["phq8"]),
                fontsize=9,
                color="#444444"
            )

        ax11.set_xlabel(f"Preceding {exposure_window_days}-Day Late Night Viewing (%)", labelpad=8)
        ax11.set_ylabel("PHQ-8 Score", labelpad=8)
        ax11.grid(True, linestyle=":", alpha=0.6, color="#D0D0D0")
        ax11.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0")
        plt.tight_layout()
        plt.savefig("fig11_late_night_vs_phq8_scatter.png", dpi=300)
        plt.close()

    print(
        "\nExecution complete! Generated figures:\n"
        " - 'fig1_category_distribution.png'\n"
        " - 'fig2_confidence_scores.png'\n"
        " - 'fig3_circadian_profile.png'\n"
        " - 'fig4_daily_engagement_timeline.png'\n"
        " - 'fig5_daily_total_phq8.png'\n"
        " - 'fig6_longitudinal_sports_exposure.png'\n"
        f" - 'fig7_{exposure_window_days}day_exposure.png'\n"
        " - 'fig8_sports_dual_axis_trajectory.png'\n"
        " - 'fig9_sports_vs_phq8_scatter.png'\n"
        " - 'fig10_late_night_trajectory.png'\n"
        " - 'fig11_late_night_vs_phq8_scatter.png'"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics and charts for YouTube exposure and wellbeing data."
    )
    parser.add_argument(
        "-j",
        "--jsonl",
        default="classified_output_ZS_IMRY_7.jsonl",
        help="Path to classified JSONL file (default: classified_output_ZS_IMRY_7.jsonl)",
    )
    parser.add_argument(
        "-y",
        "--youtube",
        default="synthetic_data/youtube_LWP.csv",
        help="Path to YouTube CSV file (default: synthetic_data/youtube_LWP.csv)",
    )
    parser.add_argument(
        "-w",
        "--wellbeing",
        default="synthetic_data/Digital Wellbeing LWP.csv",
        help="Path to Digital Wellbeing CSV file (default: synthetic_data/Digital Wellbeing LWP.csv)",
    )
    parser.add_argument(
        "-e",
        "--exposure-window",
        type=int,
        default=90,
        help="Preceding exposure window in days (default: 90)",
    )

    args = parser.parse_args()

    generate_descriptive_statistics(
        jsonl_path=args.jsonl,
        youtube_csv=args.youtube,
        wellbeing_csv=args.wellbeing,
        exposure_window_days=args.exposure_window,
    )