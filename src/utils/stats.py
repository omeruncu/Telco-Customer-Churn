import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, chi2_contingency, fisher_exact, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

def ab_numeric_tests(df, numeric_cols, group_cols, alpha=0.05):
    """
    Her sayısal × binary grup komb. için:
      • Varsayımları kontrol eder (Shapiro, Levene)
      • Uygun testi seçer (T-Test veya Mann-Whitney)
      • Test istatistikleri, p-value, anlamlılık, notlar
    Returns:
      pd.DataFrame columns=[
        'numeric', 'group', 'test', 'stat', 'pvalue',
        'assumption', 'significant'
      ]
    """
    records = []
    for num in numeric_cols:
        for grp in group_cols:
            sub = df[[num, grp]].dropna()
            classes = sub[grp].unique()
            if len(classes) != 2:
                continue

            x1 = sub.loc[sub[grp]==classes[0], num]
            x2 = sub.loc[sub[grp]==classes[1], num]

            # Varsayım testleri
            p1 = shapiro(x1)[1]
            p2 = shapiro(x2)[1]
            p_levene = levene(x1, x2)[1]

            normal = (p1>alpha and p2>alpha and p_levene>alpha)

            if normal:
                stat, pval = ttest_ind(x1, x2, equal_var=True)
                test_name    = "Independent T-Test"
                assumption   = "Normalite & var. homojen"
            else:
                stat, pval = mannwhitneyu(x1, x2)
                test_name    = "Mann-Whitney U"
                assumption   = "Non-parametrik"

            records.append({
                "numeric":      num,
                "group":        grp,
                "test":         test_name,
                "stat":         stat,
                "pvalue":       pval,
                "assumption":   assumption,
                "significant":  pval < alpha
            })

    return pd.DataFrame(records)


def ab_binary_tests(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Her ikili kategorik değişken için:
      • χ² kontenjan tabloları üzerinden expected frekans kontrolü
      • Fisher’s Exact veya Proportion Z-Test seçimi
      • test istatistiği, p-value, varsayım notu, significant bayrağı
    Dönen DataFrame columns=[
      'feature','test','stat','pvalue','assumption','significant'
    ]
    """
    records = []
    for feat in group_cols:
        sub = df[[feat, target_col]].dropna()
        cats = sorted(sub[feat].unique())
        if len(cats) != 2:
            continue

        # kontenjan tablosu + expected
        ct = pd.crosstab(sub[feat], sub[target_col])
        _, _, _, exp = chi2_contingency(ct)

        if exp.min() < 5:
            test_name, stat, pval = "Fisher’s Exact", *fisher_exact(ct.values)
            assumption = "exp.freq.<5 → Fisher’s Exact"
        else:
            succ = ct.loc[cats, 1].values
            obs  = ct.loc[cats].sum(axis=1).values
            stat, pval = proportions_ztest(succ, obs)
            test_name  = "Proportion Z-Test"
            assumption = "exp.freq.≥5 → Z-Test"

        records.append({
            "feature":      feat,
            "test":         test_name,
            "stat":         stat,
            "pvalue":       pval,
            "assumption":   assumption,
            "significant":  pval < alpha
        })

    return pd.DataFrame(records).sort_values("pvalue")


def ab_multigroup_tests(
    df: pd.DataFrame,
    numeric_cols: list[str],
    group_cols: list[str],
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Her sayısal × çok sınıflı kategorik kombinasyonu için:
      • Shapiro-Wilk (normalite) ve Levene (homojenlik) testi
      • One‐Way ANOVA veya Kruskal‐Wallis seçimi
      • test istatistiği, p‐value, varsayım notu, significant bayrağı
    Dönen DataFrame columns=[
      'numeric','group','test','stat','pvalue','assumption','significant'
    ]
    """
    records = []
    for num in numeric_cols:
        for grp in group_cols:
            sub = df[[grp, num]].dropna()
            cats = sorted(sub[grp].unique())
            if len(cats) < 3:
                continue

            # her grup için seri
            samples = [sub.loc[sub[grp] == c, num] for c in cats]

            # varsayım testleri
            p_norm = [shapiro(s)[1] for s in samples]
            p_levene = levene(*samples)[1]

            if all(p > alpha for p in p_norm) and p_levene > alpha:
                stat, pval = f_oneway(*samples)
                test_name = "One‐Way ANOVA"
                assumption = "Normalite & var. homojen"
            else:
                stat, pval = kruskal(*samples)
                test_name = "Kruskal‐Wallis H Test"
                assumption = "Non‐parametrik"

            records.append({
                "numeric":      num,
                "group":        grp,
                "test":         test_name,
                "stat":         stat,
                "pvalue":       pval,
                "assumption":   assumption,
                "significant":  pval < alpha
            })

    return pd.DataFrame(records).sort_values("pvalue")
