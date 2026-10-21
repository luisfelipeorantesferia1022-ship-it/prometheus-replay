"""Spearman correlations of each trade factor with the option peak.
    python replay/factor_correlations.py results/analysis/trade_factors.csv results/analysis/trade_factor_correlations.csv
"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
d = pd.read_csv(sys.argv[1])
d['mode'] = np.where(d['book'].str.contains('copy'), 'copy', 'real')
d['abs_leverage'] = d['leverage'].abs()
d['iv_over_rv20'] = d['iv_pct'] / d['vol_20d_pct']
d['iv_over_rv1y'] = d['iv_pct'] / d['vol_1y_pct']
d['fc_1d_with_trade'] = np.where(d['opt_type'] == 'CALL', 1, -1) * d['fc_1d_pct']
d['fc_5d_with_trade'] = np.where(d['opt_type'] == 'CALL', 1, -1) * d['fc_5d_pct']
d['abs_tom_pct'] = d['tom_pct'].abs()
d['has_earnings'] = d['earnings_in_window'].notna().astype(int)
d['big_peak'] = (d['peak_bid_pct'] >= 50).astype(int)
book = d[d['seen_in'].str.startswith('sim_min1/lstm_real_0946_r1001')].sort_values('entry_ts').reset_index(drop=True)
book['n'] = [1, 2, 3, 4, 5, 6, 7]
A = book[book['n'].isin([1, 2, 3, 6])]; B = book[book['n'].isin([4, 5, 7])]
feats = {
 'LSTM test (stage 1)': ['s1_dir_accuracy_pct', 's1_hit_rate_pct', 's1_avg_mae_pct', 's1_mae_pctile_all_sp500', 's1_mae_pctile_candidates', 's1_dir_pval', 's1_score', 's1_rank', 's1_avg_snr', 's1_avg_pred_move_pct', 's1_avg_act_move_pct'],
 'Backtest (stage 2)': ['s2_trades', 's2_gain_pct_of_stake', 's2_win_pct', 's2_avg_trailing_pct', 's2_avg_oracle_pct', 's2_max_oracle_pct'],
 'Sims': ['sim1_trades', 'sim1_pnl', 'sim1_win_pct', 'sim1_avg_pnl_pct', 'sim1_avg_peak_pct', 'sim1_max_peak_pct', 'sim1_avg_iv_pct', 'sim2_trades', 'sim2_pnl', 'sim2_win_pct', 'sim2_avg_pnl_pct', 'sim2_avg_peak_pct', 'sim2_max_peak_pct', 'weeks_since_promotion'],
 'Entry morning': ['snr', 'abs_tom_pct', 'fc_1d_with_trade', 'fc_5d_with_trade', 'fc_mae_pct', 'fc_dir_accuracy', 'rank_position'],
 'Option at entry': ['iv_pct', 'iv_over_rv20', 'iv_over_rv1y', 'delta', 'abs_leverage', 'theta_pct_per_day', 'vega_pct_per_vol_pt', 'gamma_pct_per_1pct_move', 'spread_pct_at_entry', 'dte', 'otm_pct', 'premium_pct_of_stock', 'expected_move_to_expiry_pct'],
 'Alpha / beta (before entry)': ['beta_1y', 'beta_60d', 'alpha_1y_with_trade', 'alpha_60d_with_trade', 'r2_1y', 'vol_1y_pct', 'vol_20d_pct', 'idio_vol_1y_pct', 'ret_20d_with_trade', 'spy_vol_20d_pct', 'spy_ret_20d_pct', 'option_beta'],
 'Peak (after entry)': ['peak_bid_pct', 'days_to_peak', 'stock_move_to_peak_with_trade', 'spy_move_to_peak_pct', 'market_part_with_trade', 'stock_specific_part_with_trade', 'option_alpha_to_peak_pct', 'max_fav_move_to_expiry_pct', 'max_adverse_move_to_expiry_pct', 'has_earnings'],
}
real = d[d['mode'] == 'real']
rows = []
for grp, fs in feats.items():
    for f in fs:
        if f not in d.columns:
            rows.append((grp, f, 'MISSING')); continue
        r = {'group': grp, 'feature': f}
        for _, t in book.iterrows():
            r[f"{t['n']}{t['ticker'][:4]}"] = t[f]
        r['A_mean'] = A[f].mean(); r['B_mean'] = B[f].mean()
        r['A_min'] = A[f].min(); r['A_max'] = A[f].max(); r['B_min'] = B[f].min(); r['B_max'] = B[f].max()
        r['separates'] = (r['A_min'] > r['B_max']) or (r['A_max'] < r['B_min'])
        def sp(x):
            x = x[[f, 'peak_bid_pct']].dropna()
            if len(x) < 5 or x[f].nunique() < 2: return (np.nan, np.nan, len(x))
            rho, p = spearmanr(x[f], x['peak_bid_pct']); return (rho, p, len(x))
        if f != 'peak_bid_pct':
            r['rho7'], r['p7'], _ = sp(book)
            r['rho_all'], r['p_all'], r['n_all'] = sp(d)
            r['rho_real'], r['p_real'], r['n_real'] = sp(real)
        rows.append(r)
out = pd.DataFrame([r for r in rows if isinstance(r, dict)])
out.to_csv(sys.argv[2], index=False)
num = out.select_dtypes('number').columns
print(out.round(3).to_string())
