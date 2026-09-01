"""
descriptive_statistics.py

Generates descriptive statistics, classification summary tables, and visual 
distributions for N=1 YouTube exposure and wellbeing data.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def generate_descriptive_statistics(
    jsonl_path="classified_output_ZS.jsonl",
    youtube_csv="synthetic_data/youtube_LWP.csv",
    wellbeing_csv="synthetic_data/Digital Wellbeing LWP.csv",
    exposure_window_days=90,
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
    yt_full["predicted_label"] = yt_full["predicted_label"].fillna(
        "unclassified"
    )
    yt_full["time_UTC"] = pd.to_datetime(
        yt_full["time_UTC"], utc=True, errors="coerce"
    )

    # 4. Load Digital Wellbeing survey data
    wellbeing_df = pd.read_csv(wellbeing_csv, low_memory=False)
    wellbeing_df["start_UTC"] = pd.to_datetime(
        wellbeing_df["start_UTC"], utc=True, errors="coerce"
    )

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

    print(f"Total Watch Events: {total_videos}")
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
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cat_counts = (
        yt_full["predicted_label"].value_counts().sort_values(ascending=True)
    )
    bars = ax.barh(cat_counts.index, cat_counts.values, color="#2b5c8f")
    ax.set_title(
        "Figure 1: Viewing Frequency by Predicted Category",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Total Videos Watched")
    ax.set_ylabel("Predicted Category")
    ax.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        ax.annotate(
            f"{int(width)}",
            (width + 0.3, bar.get_y() + bar.get_height() / 2),
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig("fig1_category_distribution.png", dpi=300)
    plt.close()

    print("--- Step 4: Figure 2 (Classifier Certainty Distribution) ---")
    fig, ax = plt.subplots(figsize=(8, 4))
    scores = yt_full["score"].dropna()
    ax.hist(scores, bins=15, color="#2b5c8f", edgecolor="white", alpha=0.85)
    ax.axvline(
        mean_confidence,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_confidence:.2f}",
    )
    ax.axvline(
        median_confidence,
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_confidence:.2f}",
    )
    ax.set_title(
        "Figure 2: Classification Confidence Score Distribution",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Video Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig2_confidence_scores.png", dpi=300)
    plt.close()

    print("\n--- Step 5: Figure 3 (Daily Longitudinal Activity Line Plot) ---")
    yt_valid = yt_full.dropna(subset=["time_UTC"]).copy()
    yt_valid["date"] = pd.to_datetime(yt_valid["time_UTC"].dt.date)

    daily_counts = (
        yt_valid.groupby(["date", "predicted_label"])
        .size()
        .unstack(fill_value=0)
    )

    survey_dates = pd.to_datetime(
        wellbeing_df["start_UTC"].dropna().dt.date.unique()
    )
    all_dates = list(daily_counts.index) + list(survey_dates)
    min_date, max_date = min(all_dates), max(all_dates)
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    daily_counts = daily_counts.reindex(full_date_range, fill_value=0)

    fig, ax3 = plt.subplots(figsize=(12, 5))
    daily_counts.plot(
        kind="line", ax=ax3, colormap="tab10", linewidth=1.8, alpha=0.85
    )
    ax3.set_title(
        "Figure 3: Daily Viewing Exposure & Survey Assessment Dates",
        fontweight="bold",
        pad=12,
    )
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Daily Videos Watched")
    ax3.grid(True, linestyle="--", alpha=0.5)

    for s_date in survey_dates:
        score_str = "Survey"
        if phq_col in wellbeing_df.columns:
            phq_val = wellbeing_df.loc[
                wellbeing_df["start_UTC"].dt.date == s_date.date(), phq_col
            ].values
            if len(phq_val) > 0 and pd.notna(phq_val[0]):
                score_str = f"PHQ-8: {phq_val[0]}"

        ax3.axvline(x=s_date, color="crimson", linestyle="--", linewidth=2)
        ax3.text(
            s_date,
            ax3.get_ylim()[1] * 0.85,
            f" {score_str}",
            color="crimson",
            fontweight="bold",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3", fc="white", ec="crimson", alpha=0.9
            ),
        )

    ax3.legend(title="Category", bbox_to_anchor=(1.02, 0), loc="lower left")
    plt.tight_layout()
    plt.savefig("fig3_longitudinal_exposure.png", dpi=300)
    plt.close()

    print(
        f"--- Step 6: Figure 4 ({exposure_window_days}-Day Preceding Exposure per Assessment) ---"
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
            "total_in_3m": len(in_window),
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
    print("\n3-Month Preceding Exposure Summary Table:")
    print(exposure_summary_df.to_string(index=False))

    fig, ax4 = plt.subplots(figsize=(10, 5))
    exp_df = pd.DataFrame(exposure_data_plot)
    if not exp_df.empty:
        exp_pivot = exp_df.pivot(
            index="Survey_Assessment", columns="Category", values="Count"
        ).fillna(0)
        exp_pivot.plot(kind="bar", ax=ax4, colormap="tab10", alpha=0.85)

    ax4.set_title(
        f"Figure 4: 3-Month ({exposure_window_days}-Day) Preceding Category Exposure Volume",
        fontweight="bold",
        pad=12,
    )
    ax4.set_ylabel("90-Day Video Volume")
    ax4.set_xlabel("")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.tick_params(axis="x", rotation=0)
    ax4.legend(title="Category", bbox_to_anchor=(1.02, 0), loc="lower left")

    plt.tight_layout()
    plt.savefig("fig4_3month_exposure.png", dpi=300)
    plt.close()

    print(
        "\nExecution complete! Generated figures:\n"
        " - 'fig1_category_distribution.png'\n"
        " - 'fig2_confidence_scores.png'\n"
        " - 'fig3_longitudinal_exposure.png'\n"
        " - 'fig4_3month_exposure.png'"
    )


if __name__ == "__main__":
    generate_descriptive_statistics()