import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# 1. Import data
df = pd.read_csv("medical_examination.csv")


# 2. Add overweight column
df["overweight"] = (
    df["weight"] / ((df["height"] / 100) ** 2) > 25
).astype(int)


# 3. Normalize cholesterol and glucose
df["cholesterol"] = df["cholesterol"].apply(
    lambda x: 0 if x == 1 else 1
)

df["gluc"] = df["gluc"].apply(
    lambda x: 0 if x == 1 else 1
)


# 4. Draw categorical plot
def draw_cat_plot():

    # 5. Create DataFrame using melt
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=[
            "cholesterol",
            "gluc",
            "smoke",
            "alco",
            "active",
            "overweight"
        ]
    )

    # 6. Group and reformat
    df_cat = (
        df_cat
        .groupby(["cardio", "variable", "value"])
        .size()
        .reset_index(name="total")
    )

    # 7. Create categorical plot
    cat_plot = sns.catplot(
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        data=df_cat,
        kind="bar"
    )

    cat_plot.set_axis_labels("variable", "total")
    cat_plot.set_titles("cardio = {col_name}")

    # 8. Get figure
    fig = cat_plot.fig

    return fig


# 10. Draw heat map
def draw_heat_map():

    # 11. Clean data
    df_heat = df[
        (df["ap_lo"] <= df["ap_hi"])
        & (df["height"] >= df["height"].quantile(0.025))
        & (df["height"] <= df["height"].quantile(0.975))
        & (df["weight"] >= df["weight"].quantile(0.025))
        & (df["weight"] <= df["weight"].quantile(0.975))
    ]

    # 12. Correlation matrix
    corr = df_heat.corr()

    # 13. Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Draw heat map
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".1f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        ax=ax
    )

    return fig