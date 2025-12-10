import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import os

output_dir = "export"
save_dir = "anova"


def plot_2_way_anova(df, variable_name):
    
    model_nfix = ols(f'{variable_name} ~ C(image_is_true) * C(condition)', data=df_clean_nfix).fit()
    anova_nfix = sm.stats.anova_lm(model_nfix, typ=2)
    print(f"\nTwo-way ANOVA for {variable_name} (z-scored):")
    print(anova_nfix)

    summary = (
        df.groupby(["image_is_true", "condition"])[variable_name]
          .agg(["mean", "sem"])
          .reset_index()
    )

    # ensure sorting (False → True, familiar → unfamiliar)
    summary["image_is_true"] = summary["image_is_true"].astype(str)
    summary["condition"] = summary["condition"].astype(str)

    image_levels = summary["image_is_true"].unique()
    cond_levels = summary["condition"].unique()

    x = np.arange(len(image_levels))         # positions for True/False
    width = 0.45                             # width of bars

    fig, ax = plt.subplots(figsize=(7,5))
    colors = ["#F8766D", "#00BFC4"]

    # plot bars
    for i, cond in enumerate(cond_levels):
        sub = summary[summary["condition"] == cond]

        # match ordering of x positions
        sub = sub.set_index("image_is_true").loc[image_levels].reset_index()

        ax.bar(
            x + (i - 0.5) * width,
            sub["mean"],
            width,
            yerr=sub["sem"],
            capsize=6,
            color=colors[i],
            label=cond,
            edgecolor="black",
            linewidth=0.8
        )

    # formatting
    ax.set_xticks(x)
    ax.set_xticklabels(image_levels)
    ax.set_xlabel("image_is_true", fontsize=12)
    ax.set_ylabel(variable_name, fontsize=12)
    ax.set_title(f"Interaction Plot ({variable_name})", fontsize=14)

    ax.set_facecolor("#EBEBEB")
    ax.grid(axis="y", color="white", linewidth=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#4D4D4D")

    ax.legend(title="condition")

    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir, dpi=300)
    plt.show()


# load CSV
csv_file = os.path.join(output_dir, "per_trial_with_cond.csv")
df = pd.read_csv(csv_file)

# z-scored
df['n_fixations_z'] = df.groupby('Participant_ID')['n_fixations'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)
df['mean_fix_dur_ms_z'] = df.groupby('Participant_ID')['mean_fix_dur_ms'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)
df['dispersion_r_z'] = df.groupby('Participant_ID')['dispersion_r'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)
df['longest_fix_ms_z'] = df.groupby('Participant_ID')['longest_fix_ms'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=1)
)

# ±3σ outlier
df_clean_nfix = df[(df['n_fixations_z'] >= -3) & (df['n_fixations_z'] <= 3)]
df_clean_mfd  = df[(df['mean_fix_dur_ms_z'] >= -3) & (df['mean_fix_dur_ms_z'] <= 3)]
df_clean_disp = df[(df['dispersion_r_z'] >= -3) & (df['dispersion_r_z'] <= 3)]
df_clean_longfix = df[(df['longest_fix_ms_z'] >= -3) & (df['longest_fix_ms_z'] <= 3)]

plot_2_way_anova(df_clean_nfix, 'n_fixations_z')
plot_2_way_anova(df_clean_mfd, 'mean_fix_dur_ms_z')
plot_2_way_anova(df_clean_disp, 'dispersion_r_z')
plot_2_way_anova(df_clean_longfix, 'longest_fix_ms_z')

# commented out because in this new version, the stats are computed in the plot_2_way_anova() function
"""

# 2way-ANOVA: n_fixations
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 0.026731    1.0  0.027222  0.869140
# C(condition)                     0.845039    1.0  0.860563  0.354852
# C(image_is_true):C(condition)    0.117406    1.0  0.119563  0.729922

model_nfix = ols('n_fixations_z ~ C(image_is_true) * C(condition)', data=df_clean_nfix).fit()
anova_nfix = sm.stats.anova_lm(model_nfix, typ=2)
print("\nTwo-way ANOVA for n_fixations (z-scored):")
print(anova_nfix)

# 2way-ANOVA: mean_fix_dur_ms
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 0.678525    1.0  0.851459  0.357425
# C(condition)                     2.716619    1.0  3.409001  0.066550 .
# C(image_is_true):C(condition)    0.000663    1.0  0.000832  0.977023

model_mfd = ols('mean_fix_dur_ms_z ~ C(image_is_true) * C(condition)', data=df_clean_mfd).fit()
anova_mfd = sm.stats.anova_lm(model_mfd, typ=2)
print("\nTwo-way ANOVA for mean_fix_dur_ms (z-scored):")
print(anova_mfd)

# 2way-ANOVA: dispersion_r ()
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 1.292023    1.0  1.413781  0.236041
# C(condition)                     0.000565    1.0  0.000618  0.980188
# C(image_is_true):C(condition)    3.061990    1.0  3.350548  0.068884 .
model_disp = ols('dispersion_r_z ~ C(image_is_true) * C(condition)', data=df_clean_disp).fit()
anova_disp = sm.stats.anova_lm(model_disp, typ=2)
print("\nTwo-way ANOVA for dispersion_r (z-scored):")
print(anova_disp)

# 2way-ANOVA: longest_fix_ms ()
#                                    sum_sq     df         F    PR(>F)
# C(image_is_true)                 1.048487    1.0  1.071389  0.302060
# C(condition)                     1.792501    1.0  1.831655  0.177678
# C(image_is_true):C(condition)    0.519831    1.0  0.531185  0.467083
Residual                       171.259125  175.0       NaN       NaN.
model_disp = ols('longest_fix_ms_z ~ C(image_is_true) * C(condition)', data=df_clean_disp).fit()
anova_disp = sm.stats.anova_lm(model_disp, typ=2)
print("\nTwo-way ANOVA for longest_fix_ms (z-scored):")
print(anova_disp)

"""
