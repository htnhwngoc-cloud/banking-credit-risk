import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

def calc_default_rate(df, group_col):
    stats_df = df.groupby(group_col)['default'].agg(['count', 'sum']).reset_index()
    stats_df.rename(columns={'sum': 'defaults', 'count': 'total'}, inplace=True)
    stats_df['default_rate'] = stats_df['defaults'] / stats_df['total']
    
    ci_low, ci_high = proportion_confint(stats_df['defaults'], stats_df['total'], alpha=0.05, method='wilson')
    stats_df['ci_lower_95'] = ci_low
    stats_df['ci_upper_95'] = ci_high
    
    return stats_df

def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def main():
    out_table = 'reports/analyze/tables'
    out_fig = 'reports/analyze/figures'
    os.makedirs(out_table, exist_ok=True)
    os.makedirs(out_fig, exist_ok=True)
    
    df = pd.read_csv('data/processed/train.csv')
    
    # 1. Descriptive stats tables
    cols_to_check = ['age_group', 'limit_group', 'education', 'marriage', 'pay_sep', 'pay_aug', 'pay_jul', 'pay_jun', 'pay_may', 'pay_apr']
    for col in cols_to_check:
        res = calc_default_rate(df, col)
        res.to_csv(f'{out_table}/default_rate_by_{col}.csv', index=False)
        
    # Binning engineered features
    df['bill_ratio_sep_bin'] = pd.qcut(df['bill_ratio_sep'], q=4, duplicates='drop')
    df['pay_ratio_sep_bin'] = pd.qcut(df['pay_ratio_sep'], q=4, duplicates='drop')
    df['delay_trend_bin'] = pd.qcut(df['delay_trend'], q=4, duplicates='drop')
    
    eng_cols = ['num_delayed_months', 'max_delay', 'num_zero_pay', 'bill_ratio_sep_bin', 'pay_ratio_sep_bin', 'delay_trend_bin']
    eng_res = []
    for col in eng_cols:
        res = calc_default_rate(df, col)
        res['feature'] = col
        res.rename(columns={col: 'group'}, inplace=True)
        res['group'] = res['group'].astype(str)
        eng_res.append(res)
    pd.concat(eng_res, ignore_index=True).to_csv(f'{out_table}/default_rate_by_engineered.csv', index=False)
    
    # Hypothesis testing
    hypotheses = []
    
    # H1: max_delay, num_delayed_months (Chi-square / Mann-Whitney)
    # Using Mann-Whitney U for ordinal/continuous
    stat, pval = stats.mannwhitneyu(df[df['default']==1]['max_delay'], df[df['default']==0]['max_delay'])
    hypotheses.append({'H': 'H1', 'Feature': 'max_delay', 'Test': 'Mann-Whitney U', 'P_value': pval})
    
    stat, pval = stats.mannwhitneyu(df[df['default']==1]['num_delayed_months'], df[df['default']==0]['num_delayed_months'])
    hypotheses.append({'H': 'H1', 'Feature': 'num_delayed_months', 'Test': 'Mann-Whitney U', 'P_value': pval})
    
    # Categorical Chi-Square
    cat_hypotheses = [
        ('H2', 'limit_group'),
        ('H3', 'age_group'),
        ('H4', 'education'),
        ('H5', 'marriage'),
        ('H10', 'num_zero_pay')
    ]
    for h, col in cat_hypotheses:
        c_tab = pd.crosstab(df[col], df['default'])
        chi2, pval, dof, ex = stats.chi2_contingency(c_tab)
        cv = cramers_v(c_tab.values)
        hypotheses.append({'H': h, 'Feature': col, 'Test': 'Chi-square', 'P_value': pval, 'Effect_Size': cv})
        
    # H6: pay_sep vs past months
    for month in ['sep', 'aug', 'jul', 'jun', 'may', 'apr']:
        col = f'pay_{month}'
        c_tab = pd.crosstab(df[col], df['default'])
        chi2, pval, dof, ex = stats.chi2_contingency(c_tab)
        cv = cramers_v(c_tab.values)
        hypotheses.append({'H': 'H6', 'Feature': col, 'Test': 'Chi-square', 'P_value': pval, 'Effect_Size': cv})
        
    # H7, H8, H9: bill_ratio_sep, pay_ratio_sep, delay_trend (Mann-Whitney)
    for h, col in [('H7', 'bill_ratio_sep'), ('H8', 'pay_ratio_sep'), ('H9', 'delay_trend')]:
        stat, pval = stats.mannwhitneyu(df[df['default']==1][col], df[df['default']==0][col])
        hypotheses.append({'H': h, 'Feature': col, 'Test': 'Mann-Whitney U', 'P_value': pval})
        
    ht_df = pd.DataFrame(hypotheses)
    
    # Adjust p-values
    reject, pvals_corrected, _, _ = multipletests(ht_df['P_value'], alpha=0.05, method='fdr_bh')
    ht_df['Adj_P_value'] = pvals_corrected
    ht_df['Conclusion'] = np.where(reject, 'Ủng hộ (Support)', 'Chưa đủ bằng chứng')
    ht_df.to_csv(f'{out_table}/hypothesis_testing.csv', index=False)
    
    # 2. Figures
    # Heatmap pay_sep x limit_group
    heatmap_data = df.groupby(['pay_sep', 'limit_group'])['default'].mean().unstack()
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='Reds')
    plt.title('Default Rate by pay_sep and limit_group')
    plt.savefig(f'{out_fig}/heatmap_pay_sep_limit_group.png')
    plt.close()
    
    # max_delay line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df.groupby('max_delay')['default'].mean().reset_index(), x='max_delay', y='default', marker='o')
    plt.title('Default Rate Trend by max_delay')
    plt.savefig(f'{out_fig}/trend_max_delay.png')
    plt.close()

if __name__ == '__main__':
    main()
