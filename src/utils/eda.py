import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_churn_distribution(df, target_col="Churn"):
    """
    Binary hedefin (0/1, '0'/'1' veya 'No'/'Yes') sınıf dağılımını ve oranlarını gösterir.
    """
    # 1) Sayısal özet
    counts = df[target_col].value_counts(dropna=False)
    ratios = df[target_col].value_counts(normalize=True, dropna=False)

    print("🔹 Sınıf Dağılımı:\n", counts)
    print("\n🔹 Oranlar:\n", ratios)

    # 2) Kategori sırasını al
    cats = list(counts.index)

    # 3) Görsel etiketleme fonksiyonu
    def _label(x):
        s = str(x)
        if s in ("0", "No"):  return "0 = Hayır"
        if s in ("1", "Yes"): return "1 = Evet"
        return s

    xtick_labels = [_label(x) for x in cats]

    # 4) Çizim
    sns.set(style="whitegrid")
    sns.countplot(
        x=target_col,
        data=df,
        order=cats,
        palette=["#1f77b4", "#ff7f0e"]  # ilk bar → mavi, ikinci → turuncu
    )
    plt.title(f"{target_col} Dağılımı")
    plt.xlabel(target_col)
    plt.ylabel("Müşteri Sayısı")
    plt.xticks(ticks=range(len(cats)), labels=xtick_labels)
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(
    df,
    num_cols=None,
    bins=30,
    kde=True,
    figsize=(12, 4)
):
    """
    Her sayısal sütun için yanyana Histogram ve Boxplot çizer.

    Args:
      df        : pandas DataFrame
      num_cols  : list of column names; None ise tüm sayısal sütunlar seçilir
      bins      : histogram için bin sayısı
      kde       : histogram üstüne KDE eğrisi eklesin mi?
      figsize   : her grafiğin figsize parametresi
    """
    # Eğer num_cols verilmemişse, tüm sayısal sütunları al
    if num_cols is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram + KDE
        sns.histplot(
            data=df,
            x=col,
            bins=bins,
            kde=kde,
            ax=axes[0]
        )
        axes[0].set_title(f"{col} – Histogram")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frekans")

        # Boxplot
        sns.boxplot(
            x=df[col],
            ax=axes[1]
        )
        axes[1].set_title(f"{col} – Boxplot")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.show()


def plot_churn_for_binary_codes(
    df,
    code_cols,
    churn_col="Churn",
    hspace=0.4,
    wspace=0.3
):
    """
    code_cols: kodlanmış 0/1 binary sütunların listesi
    churn_col: churn'un 0/1 olduğu sütun
    """
    sns.set(style="whitegrid")
    n = len(code_cols)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5*n),
                             gridspec_kw={'hspace': hspace, 'wspace': wspace})

    for i, col in enumerate(code_cols):
        ax_count, ax_rate = axes[i]

        # 1) COUNTPLOT
        sns.countplot(
            x=col,
            hue=churn_col,
            data=df,
            order=[1, 0],                       # 1 solda, 0 sağda
            palette={0:"#1f77b4", 1:"#ff7f0e"},
            hue_order=[0,1],
            ax=ax_count
        )
        ax_count.set_title(f"{col} – Churn Count")
        ax_count.set_xlabel(col)
        ax_count.set_ylabel("Count")
        # legend’i No/Yes sırasına getir
        handles, _ = ax_count.get_legend_handles_labels()
        ax_count.legend(handles, ["No","Yes"], title="Churn")

        # 2) BARPLOT (Rate)
        rates = (
            df
            .groupby(col)[churn_col]
            .mean()
            .reindex([1,0])          # aynı sıra
            .fillna(0)               # NaN varsa 0 yap
            .reset_index()
        )
        sns.barplot(
            x=col,
            y=churn_col,
            data=rates,
            order=[1, 0],
            palette="Blues_d",
            ax=ax_rate
        )
        ax_rate.set_title(f"{col} – Churn Rate")
        ax_rate.set_xlabel(col)
        ax_rate.set_ylabel("Rate")
        ax_rate.set_ylim(0, rates[churn_col].max() * 1.1)

        # Bar üstü annotasyon
        for p in ax_rate.patches:
            h = p.get_height()
            ax_rate.text(
                p.get_x() + p.get_width() / 2,
                h + 0.01,
                f"{h*100:.1f}%",
                ha="center"
            )

    plt.tight_layout()
    plt.show()


def plot_significant_numeric_scatter(
    df,
    stats_df: pd.DataFrame,
    jitter=0.1,
    palette=( "#1f77b4", "#ff7f0e" )
):
    """
    stats_df: ab_numeric_tests ile üretilen DataFrame
    İçinden significant=True olanları seç, her bir numeric×group için:
      x=numeric, y=group_jit, hue='Churn'
    """
    # 1) Sadece anlamlı kombinasyonlar
    sig = stats_df[stats_df["significant"]]

    if sig.empty:
        print("Anlamlı bir numeric×group kombinasyonu yok.")
        return

    # 2) Jitterlı kopya oluştur
    df2 = df.copy()
    np.random.seed(0)
    for grp in sig["group"].unique():
        df2[f"{grp}_jit"] = df2[grp] + np.random.uniform(-jitter, jitter, len(df2))

    # 3) Figür boyutu
    combos = list(sig[["numeric","group"]].itertuples(index=False, name=None))
    n = len(combos)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(n, 1, figsize=(8, 4*n), squeeze=False)

    for i, (num, grp) in enumerate(combos):
        jit_col = f"{grp}_jit"
        ax = axes[i,0]

        sns.scatterplot(
            data=df2,
            x=num, y=jit_col,
            hue="Churn",
            palette=palette,
            hue_order=[0,1],
            alpha=0.7, s=40,
            ax=ax, legend=False
        )
        ax.set_yticks([0,1])
        ax.set_yticklabels([f"{grp}=0", f"{grp}=1"])
        ax.set_xlabel(num)
        ax.set_ylabel(grp)
        ax.set_title(f"{num} vs {grp} (p={sig.loc[(sig.numeric==num)&(sig.group==grp),'pvalue'].values[0]:.3f})")

    # 4) Ortak legend
    handles = [
        plt.Line2D([0],[0], marker='o', color='w', label='Churn=0',
                   markerfacecolor=palette[0], markersize=8),
        plt.Line2D([0],[0], marker='o', color='w', label='Churn=1',
                   markerfacecolor=palette[1], markersize=8),
    ]
    fig.legend(handles=handles, loc='upper right', title='Churn')
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.show()


def plot_significant_binary_churn(
    df,
    stats_df: pd.DataFrame,
    churn_col: str = "Churn",
    hspace: float = 0.4,
    wspace: float = 0.3
):
    """
    stats_df: ab_binary_tests ile elde edilen DataFrame
    Bu tablodan significant=True olan 'feature'ları alır,
    her biri için yanyana countplot + rate barplot çizer.
    """
    sig = stats_df[stats_df["significant"]]
    if sig.empty:
        print("Anlamlı bir binary değişken yok.")
        return

    code_cols = sig["feature"].tolist()
    n = len(code_cols)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(n, 2, figsize=(12, 5*n),
                             gridspec_kw={'hspace':hspace,'wspace':wspace})

    for i, feat in enumerate(code_cols):
        ax_c, ax_r = axes[i]

        # 1) Countplot
        sns.countplot(
            x=feat, hue=churn_col, data=df,
            order=[1,0], palette={0:"#1f77b4",1:"#ff7f0e"},
            hue_order=[0,1], ax=ax_c
        )
        ax_c.set_title(f"{feat} – Churn Count")
        ax_c.set_xlabel(feat)
        ax_c.set_ylabel("Count")
        hdl,_ = ax_c.get_legend_handles_labels()
        ax_c.legend(hdl, ["No","Yes"], title="Churn")

        # 2) Rate barplot
        rates = (
            df.groupby(feat)[churn_col]
              .mean()
              .reindex([1,0])
              .fillna(0)
              .reset_index()
        )
        sns.barplot(
            x=feat, y=churn_col, data=rates,
            order=[1,0], palette="Blues_d", ax=ax_r
        )
        ax_r.set_title(f"{feat} – Churn Rate")
        ax_r.set_xlabel(feat)
        ax_r.set_ylabel("Rate")
        ax_r.set_ylim(0, rates[churn_col].max()*1.1)

        for p in ax_r.patches:
            h = p.get_height()
            ax_r.text(
                p.get_x()+p.get_width()/2,
                h+0.01,
                f"{h*100:.1f}%", ha="center"
            )

    plt.tight_layout()
    plt.show()


def plot_significant_multigroup(df, stats_df, top_n=5, jitter=0.1, show_strip=True, max_classes=10):
    sig = stats_df[stats_df["significant"]].sort_values("pvalue")
    sig = sig[sig["group"].apply(lambda g: df[g].nunique() <= max_classes)].head(top_n)

    if sig.empty:
        print("Görselleştirilebilir anlamlı kombinasyon bulunamadı.")
        return

    sns.set(style="whitegrid")
    fig, axes = plt.subplots(len(sig), 1, figsize=(8, 4*len(sig)), squeeze=False)

    for i, (_, row) in enumerate(sig.iterrows()):
        num, grp = row["numeric"], row["group"]
        sub = df[[num, grp, "Churn"]].dropna().copy()
        sub[f"{grp}_jit"] = sub[grp].astype("category").cat.codes + np.random.uniform(-jitter, jitter, len(sub))

        ax = axes[i, 0]
        sns.boxplot(x=grp, y=num, data=sub, ax=ax, palette="Pastel1")

        if show_strip:
            sns.stripplot(
                x=f"{grp}_jit", y=num, data=sub,
                hue="Churn", palette=["#1f77b4", "#ff7f0e"],
                hue_order=[0,1], dodge=False, alpha=0.6, size=4,
                ax=ax, legend=False
            )

        ax.set_title(f"{num} by {grp} (p = {row.pvalue:.2e})")
        ax.set_xlabel(grp)
        ax.set_ylabel(num)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Churn=0', markerfacecolor="#1f77b4", markersize=6),
        plt.Line2D([0], [0], marker='o', color='w', label='Churn=1', markerfacecolor="#ff7f0e", markersize=6)
    ]
    fig.legend(handles=handles, loc="upper right", title="Churn")
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.show()
