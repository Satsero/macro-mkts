"""
FX market reaction to US elections based on bilateral trade-dependencies | event study -- Python implementation

Institution: Bocconi University (Thesis, 2025)  
Title: "Bilateral Trade Exposure and Cross-Sectional Currency Reactions to U.S. Presidential Elections"  

Overview:
This script implements an event study framework to examine how major global currency pairs 
(CAD, MXN, CNH, EUR, GBP, JPY, AUD, NZD, CHF, KRW, NOK, PHP) react to U.S. presidential 
elections from 1980 to 2024. It classifies currencies into High, Medium, and Low U.S. trade 
dependency groups using a proprietary symmetric bilateral trade index (see core computation below)

Key features:
- Dynamically classifies currencies each cycle based on 4yr avg bilateral trade flow data
- Computes normalized cumulative returns (NCRs) across 5 windows: 2d, 5d, 10d, 20d, 50d
- Applies interaction filters by party outcome (Dem vs Rep) and expectation type (expected vs surprise)
- Measures effects with volatility comparisons, effect sizes, and endpoint-based statistics
- Includes regression analysis (OLS) to estimate contribution of trade dependency, party, and surprise shocks

Data sources:
- FX spot rates from BBG (daily frequency, 12 currencies vs USD)
- Trade data from IMF DOTS, averaged and cleaned outside this script
- Election surprise classification based on polling (pre-2004) and prediction markets (post-2004)

# -----------------------------------------------------------------------------
Core Trade Dependency Index (Symmetric Bilateral):
This index captures the mutual trade exposure between a foreign country (i) and the U.S., accounting for asymmetry in dependence.

Defined as the avg of two ratios:
(a) Country i’s share of total trade with the U.S.
(b) U.S. share of total trade with country i

TradeDependencyIndex_i = 0.5 * [ 
     (Exports_i_to_US + Imports_i_from_US) / TotalTrade_i 
   + (Exports_US_to_i + Imports_US_from_i) / TotalTrade_US
 ]

 - All values USD
 - Calculated as 4yr rolling avg per election year
 - Used to classify currencies into High, Medium, Low dependency buckets
# -----------------------------------------------------------------------------

NOTE:
Run the main functions to visualize average reaction paths by group and to compute statistical outputs. 
Intended for academic/non-commercial use only as part of a Bachelor's thesis. See LICENSE in main;

Dependencies: pandas, numpy, matplotlib, statsmodels, scipy
# ------------
"""

import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# For regression analysis
import statsmodels.formula.api as smf

election_dates = {
    '1980-11-04': ('Republican', 'Expected'),
    '1984-11-06': ('Republican', 'Expected'),
    '1988-11-08': ('Republican', 'Expected'),
    '1992-11-03': ('Democratic', 'Surprise'),
    '1996-11-05': ('Democratic', 'Expected'),
    '2000-11-07': ('Republican', 'Surprise'),
    '2004-11-02': ('Republican', 'Expected'),
    '2008-11-04': ('Democratic', 'Expected'),
    '2012-11-06': ('Democratic', 'Expected'),
    '2016-11-08': ('Republican', 'Surprise'),
    '2020-11-03': ('Democratic', 'Expected'),
    '2024-11-05': ('Republican', 'Surprise')
}

def get_dependency_groups(election_date):
    """
    Returns currency groups based on election year trade dependency.
    Uses the pre-calculated trade dependency rankings.
    """
    election_year = pd.to_datetime(election_date).year
    
    # Trade dependency classifications by election year (computed separately and plugged in lists below)
    trade_dependency_groups = {
        2024: {
            'high': ['CADUSD', 'MXNUSD', 'CNYUSD', 'EURUSD'],
            'medium': ['JPYUSD', 'KRWUSD', 'CHFUSD', 'GBPUSD'],
            'low': ['PHPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD']
        },
        2020: {
            'high': ['CADUSD', 'MXNUSD', 'CNYUSD', 'EURUSD'],
            'medium': ['JPYUSD', 'KRWUSD', 'GBPUSD', 'CHFUSD'],
            'low': ['PHPUSD', 'NZDUSD', 'AUDUSD', 'NOKUSD']
        },
        2016: {
            'high': ['CADUSD', 'MXNUSD', 'CNYUSD', 'EURUSD'],
            'medium': ['JPYUSD', 'KRWUSD', 'GBPUSD', 'PHPUSD'],
            'low': ['NZDUSD', 'CHFUSD', 'AUDUSD', 'NOKUSD']
        },
        2012: {
            'high': ['CADUSD', 'MXNUSD', 'CNYUSD', 'EURUSD'],
            'medium': ['JPYUSD', 'PHPUSD', 'GBPUSD', 'KRWUSD'],
            'low': ['NZDUSD', 'CHFUSD', 'AUDUSD', 'NOKUSD']
        },
        2008: {
            'high': ['CADUSD', 'MXNUSD', 'CNYUSD', 'JPYUSD'],
            'medium': ['EURUSD', 'PHPUSD', 'KRWUSD', 'GBPUSD'],
            'low': ['NZDUSD', 'AUDUSD', 'CHFUSD', 'NOKUSD']
        },
        2004: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'CNYUSD'],
            'medium': ['EURUSD', 'PHPUSD', 'KRWUSD', 'GBPUSD'],
            'low': ['AUDUSD', 'NZDUSD', 'CHFUSD', 'NOKUSD']
        },
        2000: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'PHPUSD'],
            'medium': ['EURUSD', 'KRWUSD', 'CNYUSD', 'GBPUSD'],
            'low': ['AUDUSD', 'NZDUSD', 'CHFUSD', 'NOKUSD']
        },
        1996: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'PHPUSD'],
            'medium': ['KRWUSD', 'EURUSD', 'CNYUSD', 'GBPUSD'],
            'low': ['AUDUSD', 'NZDUSD', 'CHFUSD', 'NOKUSD']
        },
        1992: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'KRWUSD'],
            'medium': ['PHPUSD', 'EURUSD', 'AUDUSD', 'GBPUSD'],
            'low': ['NZDUSD', 'CNYUSD', 'CHFUSD', 'NOKUSD']
        },
        1988: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'KRWUSD'],
            'medium': ['PHPUSD', 'EURUSD', 'AUDUSD', 'GBPUSD'],
            'low': ['NZDUSD', 'CNYUSD', 'CHFUSD', 'NOKUSD']
        },
        1984: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'KRWUSD'],
            'medium': ['PHPUSD', 'EURUSD', 'AUDUSD', 'GBPUSD'],
            'low': ['NZDUSD', 'CNYUSD', 'CHFUSD', 'NOKUSD']
        },
        1980: {
            'high': ['CADUSD', 'MXNUSD', 'JPYUSD', 'KRWUSD'],
            'medium': ['PHPUSD', 'EURUSD', 'AUDUSD', 'GBPUSD'],
            'low': ['NZDUSD', 'CNYUSD', 'CHFUSD', 'NOKUSD']
        }
    }
    
    available_years = list(trade_dependency_groups.keys())
    if election_year not in available_years:
        closest_year = min(available_years, key=lambda x: abs(x - election_year))
        print(f"Warning: No trade data for election year {election_year}, using {closest_year} data instead")
        election_year = closest_year
    
    return (
        trade_dependency_groups[election_year]['high'],
        trade_dependency_groups[election_year]['medium'],
        trade_dependency_groups[election_year]['low']
    )


ANALYSIS_WINDOWS = {
    'Ultra-Short': 2,
    'Short': 5,
    'Medium': 10,
    'Extended': 20,
    'Long': 50
}


def match_columns_for_group(df_cols, group_currencies):
    """
    Return a list of column names from df_cols that match any currency in group_currencies
    via partial substring matching. E.g., 'CADUSD' in 'CADUSD Curncy'.
    """
    valid_cols = []
    for c in df_cols:
        c_clean = c.replace(' ', '').upper()
        for base_name in group_currencies:
            if base_name.upper() in c_clean:
                valid_cols.append(c)
                break
    return valid_cols


def calculate_window_returns_explicit(df, event_date, window):
    """
    calculates_window_returns that:
    1) Explicitly logs when an exact date match isn't found
    2) Returns the actual event date used for reference
    3) Ensures consistent window sizes even with edge effects
    """
    df = df.sort_index()
    
    exact_match = event_date in df.index
    
    if not exact_match:
        nearest_date = df.index[df.index.get_loc(event_date, method='nearest')]
        days_diff = (nearest_date - event_date).days
        actual_event_date = nearest_date
        # Uncomment only to see warnings when dates do not match exactly
        # print(f"Warning: Election date {event_date.strftime('%Y-%m-%d')} not found, using {nearest_date.strftime('%Y-%m-%d')} instead (diff: {days_diff} days)")
    else:
        actual_event_date = event_date
    
    event_date_loc = df.index.get_loc(actual_event_date)
    
    start_loc = max(0, event_date_loc - window)
    end_loc = min(len(df) - 1, event_date_loc + window)
    
    df_window = df.iloc[start_loc:end_loc + 1].copy()
    if df_window.empty:
        return pd.DataFrame(), actual_event_date
    
    event_index_in_window = event_date_loc - start_loc
    
    out_df = pd.DataFrame()
    for col in df_window.columns:
        if 'USD' in col.upper():
            event_price = df_window.iloc[event_index_in_window][col]
            if event_price == 0:
                returns = pd.Series(np.nan, index=df_window.index)
            else:
                returns = (df_window[col].astype(float) / event_price - 1) * 100
            out_df[col] = returns
    
    row_indices = np.arange(start_loc, end_loc + 1)
    offsets = row_indices - event_date_loc
    out_df.index = offsets
    
    # Create full range and reindex to ensure consistent window sizes
    full_range = range(-window, window + 1)
    out_df = out_df.reindex(full_range)
    
    return out_df, actual_event_date


def calculate_group_returns_multi_window_consistent(df, dependency_type, filter_func=None):
    """
    Enhanced version that ensures consistent window sizes and logs date alignment issues
    """
    global election_dates
    
    all_results = {}
    event_dates_used = {}
    
    for w_name, w_size in ANALYSIS_WINDOWS.items():
        window_returns = []
        dates_used = []
        
        for date_str, (party, expectation) in election_dates.items():
            if filter_func is None or filter_func(party=party, expectation=expectation):
                high_group, medium_group, low_group = get_dependency_groups(date_str)
                
                if dependency_type == 'high':
                    group_currencies = high_group
                elif dependency_type == 'medium':
                    group_currencies = medium_group
                else:
                    group_currencies = low_group
                
                cum_df, actual_date = calculate_window_returns_explicit(df, pd.to_datetime(date_str), w_size)
                if cum_df.empty:
                    continue
                
                valid_cols = match_columns_for_group(cum_df.columns, group_currencies)
                if not valid_cols:
                    continue
                
                group_avg = cum_df[valid_cols].mean(axis=1)
                window_returns.append(group_avg)
                dates_used.append((date_str, actual_date))
        
        if window_returns:
            # Check for window completeness
            complete_windows = []
            for w_ret in window_returns:
                # Only include series that have complete windows (-w to +w)
                if -w_size in w_ret.index and w_size in w_ret.index:
                    complete_windows.append(w_ret)
                else:
                    # Uncomment for debugging
                    # print(f"Warning: Incomplete window detected for {w_name}, skipping--")
                    pass
            
            if complete_windows:
                combined = pd.concat(complete_windows, axis=1)
                final_avg = combined.mean(axis=1)
                all_results[w_name] = final_avg
                event_dates_used[w_name] = dates_used
            else:
                all_results[w_name] = pd.Series(dtype=float)
                event_dates_used[w_name] = []
        else:
            all_results[w_name] = pd.Series(dtype=float)
            event_dates_used[w_name] = []
    
    return all_results, event_dates_used


def count_elections(filter_func=None):
    global election_dates
    c = 0
    details = []
    for d, (p, e) in election_dates.items():
        if filter_func is None or filter_func(party=p, expectation=e):
            c += 1
            details.append(f"{d}: {p} ({e})")
    return c, details


def compute_endpoint_stats(series_cum, window_size):
    """
    Calculate stats using only the endpoint values (-window_size and +window_size).
    Also collect vol from all pre/post days for context.
    """
    if series_cum.empty or -window_size not in series_cum.index or window_size not in series_cum.index:
        return {
            'PreEndpoint': np.nan,
            'PostEndpoint': np.nan,
            'Difference': np.nan,
            'PreVol': np.nan,
            'PostVol': np.nan,
            'EffectSize': np.nan,
            'tStat': np.nan,
            'pVal': np.nan
        }
    
    # endpoint values
    pre_endpoint = series_cum.loc[-window_size]
    post_endpoint = series_cum.loc[window_size]
    difference = post_endpoint - pre_endpoint
    
    #  pre/post values for volatility
    pre_vals = series_cum[series_cum.index < 0].dropna()
    post_vals = series_cum[series_cum.index > 0].dropna()
    
    pre_vol = pre_vals.std() if not pre_vals.empty else np.nan
    post_vol = post_vals.std() if not post_vals.empty else np.nan
    
    # effect size using average endpoint values
    pooled_std = np.sqrt((pre_vol**2 + post_vol**2) / 2) if not np.isnan(pre_vol) and not np.isnan(post_vol) else np.nan
    effect_size = (post_endpoint - pre_endpoint) / pooled_std if not np.isnan(pooled_std) and pooled_std > 0 else np.nan
    
    # For t-test use fact that this is across multiple elections. Since it is already aggregated, use an approx
    avg_diff = difference
    stderr = pooled_std / np.sqrt(12)  # Assuming n=12 elections as set initially
    t_stat = avg_diff / stderr if not np.isnan(stderr) and stderr > 0 else np.nan
    
    # Approx p-value (2-tailed) using t-distr with n-1 degrees of freedom
    from scipy.stats import t as t_dist
    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), 11)) if not np.isnan(t_stat) else np.nan
    
    return {
        'PreEndpoint': pre_endpoint,
        'PostEndpoint': post_endpoint,
        'Difference': difference,
        'PreVol': pre_vol,
        'PostVol': post_vol,
        'EffectSize': effect_size,
        'tStat': t_stat,
        'pVal': p_val
    }


def plot_analysis(df, filter_func=None, title_suffix="", ax=None):
    """
    Returns calculated relative to the event day
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    n_elec, details = count_elections(filter_func)
    window_size = ANALYSIS_WINDOWS['Long']
    
    for dependency_type, label, color in [
        ('high', 'High Dependency', 'blue'),
        ('medium', 'Medium Dependency', 'green'),
        ('low', 'Low Dependency', 'red')
    ]:
        ret_dict, _ = calculate_group_returns_multi_window_consistent(df, dependency_type, filter_func)
        series_long = ret_dict.get('Long', pd.Series(dtype=float))
        if not series_long.empty:
            ax.plot(series_long.index, series_long.values, label=label, color=color, linewidth=2)
    
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, fontsize=18)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{title_suffix}\n(n={n_elec} elections)', fontsize=22, pad=15)
    ax.set_xlabel('Offset in Trading Days (0 = Election)', fontsize=18)
    ax.set_ylabel('Returns (%) (Day 0 = Election)', fontsize=18)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=20)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    print(f"\nElections included in {title_suffix}:")
    for d in details:
        print(d)


def print_stats_tables_endpoints(df):
    """
    Print statistics tables using endpoint values rather than avgs
    """
    print("\n=== COMPACT TABLES WITH PRE-/POST-STATS (ENDPOINT VALUES) ===\n")

    scenario_sets = {
        'Overall': [('All', None)],
        'Party': [
            ('Democratic', lambda party, expectation: party == 'Democratic'),
            ('Republican', lambda party, expectation: party == 'Republican')
        ],
        'Expectation': [
            ('Expected', lambda party, expectation: expectation == 'Expected'),
            ('Surprise', lambda party, expectation: expectation == 'Surprise')
        ],
        'Interaction': [
            ('Rep×Expected', lambda party, expectation: party == 'Republican' and expectation == 'Expected'),
            ('Rep×Surprise', lambda party, expectation: party == 'Republican' and expectation == 'Surprise'),
            ('Dem×Expected', lambda party, expectation: party == 'Democratic' and expectation == 'Expected'),
            ('Dem×Surprise', lambda party, expectation: party == 'Democratic' and expectation == 'Surprise')
        ]
    }
    
    group_order = [
        ('High Dependency', 'high'),
        ('Medium Dependency', 'medium'),
        ('Low Dependency', 'low')
    ]
    
    sorted_windows = sorted(ANALYSIS_WINDOWS.items(), key=lambda x: x[1])
    
    for scenario_label, scenario_list in scenario_sets.items():
        print(f"\n=== {scenario_label} ===")
        
        for scenario_name, filt in scenario_list:
            n_e, _ = count_elections(filt)
            print(f"\nScenario: {scenario_name} (n={n_e})")
            if n_e == 0:
                continue
            
            table_rows = []
            
            for g_label, g_list in group_order:
                for w_name, w_size in sorted_windows:
                    ret_dict, _ = calculate_group_returns_multi_window_consistent(df, g_list, filt)
                    series_cum = ret_dict.get(w_name, pd.Series(dtype=float))
                    
                    stats_res = {}
                    if not series_cum.empty:
                        stats_res = compute_endpoint_stats(series_cum, w_size)
                    
                    # apply sign flip only to pre-election mean for display
                    pre_value = stats_res.get('PreEndpoint', np.nan)
                    pre_value_display = -pre_value if not np.isnan(pre_value) else np.nan
                    
                    row = {
                        'Currency Group': g_label,
                        'Window': f"{w_size}d",
                        'Pre-Election (%)': round(pre_value_display, 3),  # sign flipped for display
                        'Post-Election (%)': round(stats_res.get('PostEndpoint', np.nan), 3),
                        'Total Change (%)': round(stats_res.get('Difference', np.nan), 3),
                        'Effect Size': round(stats_res.get('EffectSize', np.nan), 3),
                    }
                    
                    pre_vol = stats_res.get('PreVol', np.nan)
                    post_vol = stats_res.get('PostVol', np.nan)
                    if not np.isnan(pre_vol) and not np.isnan(post_vol):
                        row['Volatility (Pre/Post %)'] = f"{pre_vol:.3f} / {post_vol:.3f}"
                    else:
                        row['Volatility (Pre/Post %)'] = "NaN / NaN"
                    
                    t_stat = stats_res.get('tStat', np.nan)
                    p_val = stats_res.get('pVal', np.nan)
                    row['tStat'] = round(t_stat, 3) if not np.isnan(t_stat) else np.nan
                    row['pVal'] = round(p_val, 3) if not np.isnan(p_val) else np.nan
                    
                    table_rows.append(row)
            
            df_table = pd.DataFrame(table_rows)
            
            display_cols = [
                'Currency Group', 'Window',
                'Pre-Election (%)', 'Post-Election (%)',
                'Total Change (%)', 'Volatility (Pre/Post %)',
                'Effect Size', 'tStat', 'pVal'
            ]
            
            print(df_table[display_cols].to_string(index=False))
            print()


def run_regression_analysis(df, window='Long', spec='full'):
    """
    two alternative OLS regression specs of Normalized Cumulative Returns (NCR):
    
    – spec='compact': only High/Low × Republican and High/Low × Surprise (2-way)
    – spec='full'   : adds High/Low × Republican × Surprise (3-way)
    
    Baseline: Medium dependency × Democrat × Expected
    Robust (HC1) SEs; drops the first year dummy automatically.
    """
    rows = []
    for date_str, (party, expectation) in election_dates.items():
        e_date = pd.to_datetime(date_str)
        h      = ANALYSIS_WINDOWS[window]
        cum, _ = calculate_window_returns_explicit(df, e_date, h)
        if cum.empty or h not in cum.index:
            continue

        high, med, low = get_dependency_groups(date_str)
        for ccy, ncr in cum.loc[h].dropna().items():
            if   ccy in high: dep = 'high'
            elif ccy in med:  dep = 'medium'
            elif ccy in low:  dep = 'low'
            else:             continue

            rows.append({
                'NCR':        ncr,
                'year':       e_date.year,
                'group':      dep,
                'Republican': int(party       == 'Republican'),
                'Surprise':   int(expectation == 'Surprise')
            })

    reg = pd.DataFrame(rows)
    if reg.empty:
        print("No regression data available.")
        return

    # bucket dummies
    reg['High'] = (reg['group']=='high').astype(int)
    reg['Low']  = (reg['group']=='low').astype(int)
    # make year a categorical with drop_first
    reg['year_cat'] = pd.Categorical(reg['year']).remove_unused_categories()

    # build formula
    if spec == 'compact':
        # only 2-way interactions
        formula = (
            'NCR ~ High + Low + Republican + Surprise '
            '+ High:Republican + Low:Republican '
            '+ High:Surprise   + Low:Surprise '
            '+ C(year_cat, Treatment(reference=0))'
        )
    else:
        # full 2- and 3-way
        formula = (
            'NCR ~ High + Low + Republican + Surprise '
            '+ High:Republican + Low:Republican '
            '+ High:Surprise   + Low:Surprise '
            '+ Surprise:Republican '
            '+ High:Surprise:Republican + Low:Surprise:Republican '
            '+ C(year_cat, Treatment(reference=0))'
        )

    model = smf.ols(formula, data=reg).fit(cov_type='HC1')
    print(f"\n--- Regression spec='{spec}' | window='{window}' ---")
    print(model.summary())


def calculate_avg_returns_all(df, filter_func, window='Long'):
    
    all_returns = []
    for date_str, (party, expectation) in election_dates.items():
        if filter_func is None or filter_func(party=party, expectation=expectation):
            event_date = pd.to_datetime(date_str)
            w_size = ANALYSIS_WINDOWS[window]
            cum_df, _ = calculate_window_returns_explicit(df, event_date, w_size)
            if cum_df.empty:
                continue
            valid_cols = [col for col in cum_df.columns if 'USD' in col.upper()]
            if not valid_cols:
                continue
            avg_series = cum_df[valid_cols].mean(axis=1)
            all_returns.append(avg_series)
    if all_returns:
        combined = pd.concat(all_returns, axis=1)
        final_avg = combined.mean(axis=1)
        return final_avg
    else:
        return pd.Series(dtype=float)


def print_spread_analysis_all_windows(df):
    """
    Comp avg NCR for different partisan and expectation scenarios, and print the spreads for each window size defined in ANALYSIS_WINDOWS.
    Both pre-event (at t = -window) and post-event (at t = +window) results shown.
    """
    for window_name, window_size in ANALYSIS_WINDOWS.items():
        print(f"\n=== Spread Analysis ({window_name} Window) ===\n")
        
        # Pre-event at offset = -window_size
        overall_pre = calculate_avg_returns_all(df, filter_func=None, window=window_name)
        dem_pre = calculate_avg_returns_all(df, filter_func=lambda party, expectation: party == 'Democratic', window=window_name)
        rep_pre = calculate_avg_returns_all(df, filter_func=lambda party, expectation: party == 'Republican', window=window_name)
        exp_pre = calculate_avg_returns_all(df, filter_func=lambda party, expectation: expectation == 'Expected', window=window_name)
        surp_pre = calculate_avg_returns_all(df, filter_func=lambda party, expectation: expectation == 'Surprise', window=window_name)
        
        # Post-event at offset = +window_size
        overall_post = calculate_avg_returns_all(df, filter_func=None, window=window_name)
        dem_post = calculate_avg_returns_all(df, filter_func=lambda party, expectation: party == 'Democratic', window=window_name)
        rep_post = calculate_avg_returns_all(df, filter_func=lambda party, expectation: party == 'Republican', window=window_name)
        exp_post = calculate_avg_returns_all(df, filter_func=lambda party, expectation: expectation == 'Expected', window=window_name)
        surp_post = calculate_avg_returns_all(df, filter_func=lambda party, expectation: expectation == 'Surprise', window=window_name)
        
        pre_offset = -window_size
        post_offset = window_size
        
        if pre_offset in overall_pre.index and post_offset in overall_post.index:
            print(f"Pre-Event Analysis (t = {pre_offset}):")
            overall_pre_val = overall_pre.loc[pre_offset]
            dem_pre_val = dem_pre.loc[pre_offset] if pre_offset in dem_pre.index else np.nan
            rep_pre_val = rep_pre.loc[pre_offset] if pre_offset in rep_pre.index else np.nan
            exp_pre_val = exp_pre.loc[pre_offset] if pre_offset in exp_pre.index else np.nan
            surp_pre_val = surp_pre.loc[pre_offset] if pre_offset in surp_pre.index else np.nan

            print(f"  Overall Average NCR: {overall_pre_val:.3f}%")
            print(f"  Democratic Average NCR: {dem_pre_val:.3f}%")
            print(f"  Republican Average NCR: {rep_pre_val:.3f}%")
            print(f"  Spread (Rep - Dem): {rep_pre_val - dem_pre_val:.3f}%\n")
            print(f"  Expected Average NCR: {exp_pre_val:.3f}%")
            print(f"  Surprise Average NCR: {surp_pre_val:.3f}%")
            print(f"  Spread (Surprise - Expected): {surp_pre_val - exp_pre_val:.3f}%\n")
            
            print(f"Post-Event Analysis (t = +{post_offset}):")
            overall_post_val = overall_post.loc[post_offset]
            dem_post_val = dem_post.loc[post_offset] if post_offset in dem_post.index else np.nan
            rep_post_val = rep_post.loc[post_offset] if post_offset in rep_post.index else np.nan
            exp_post_val = exp_post.loc[post_offset] if post_offset in exp_post.index else np.nan
            surp_post_val = surp_post.loc[post_offset] if post_offset in surp_post.index else np.nan

            print(f"  Overall Average NCR: {overall_post_val:.3f}%")
            print(f"  Democratic Average NCR: {dem_post_val:.3f}%")
            print(f"  Republican Average NCR: {rep_post_val:.3f}%")
            print(f"  Spread (Rep - Dem): {rep_post_val - dem_post_val:.3f}%\n")
            print(f"  Expected Average NCR: {exp_post_val:.3f}%")
            print(f"  Surprise Average NCR: {surp_post_val:.3f}%")
            print(f"  Spread (Surprise - Expected): {surp_post_val - exp_post_val:.3f}%\n")
        else:
            print("Insufficient data at the required offsets for spread analysis for this window.")


def check_event_date_alignments(df):
    
    print("\n=== event date alignment check ===\n")
    print("alignemnt checl for exact election dates in dataset:")
    
    alignment_issues = []
    
    for date_str, (party, expectation) in election_dates.items():
        event_date = pd.to_datetime(date_str)
        
        if event_date not in df.index:
            # Find nearest date
            nearest_date = df.index[df.index.get_loc(event_date, method='nearest')]
            days_diff = (nearest_date - event_date).days
            
            alignment_issues.append({
                'Election Date': event_date.strftime('%Y-%m-%d'),
                'Nearest Available Date': nearest_date.strftime('%Y-%m-%d'),
                'Difference (days)': days_diff
            })
    
    if alignment_issues:
        print("\nFound alignment issues where exact election dates arenot in the dataset:")
        for issue in alignment_issues:
            print(f"  {issue['Election Date']} -> {issue['Nearest Available Date']} (Difference: {issue['Difference (days)']} days)")
        
        total_affected = len(alignment_issues)
        max_shift = max([abs(issue['Difference (days)']) for issue in alignment_issues])
        print(f"\nSummary: {total_affected} elections affected, maximum shift is {max_shift} days")
    else:
        print("All election dates are exactly matched in dataset")


def compare_pre_post_calculations(df):
    """
    This function demonstrates the dif between: taking the avg of all pre-election days (-w to -1) and taking just the value at the start of the window (-w)
    helps verify if interpretation issues arise from the avging approach
    """
    print("\n=== COMPARISON OF CALCULATION METHODS ===\n")
    
    w_name = 'Long'
    w_size = ANALYSIS_WINDOWS[w_name]
    dependency_type = 'high'
    
    ret_dict, _ = calculate_group_returns_multi_window_consistent(df, dependency_type, None)
    series_cum = ret_dict.get(w_name, pd.Series(dtype=float))
    
    if not series_cum.empty:
        # method 1 - avg of all pre-election days
        pre_vals = series_cum[series_cum.index < 0].dropna()
        pre_mean_all = pre_vals.mean()
        pre_mean_all_display = -pre_mean_all  # Apply sign flip for display
        
        # 2 - just the endpoint value
        pre_endpoint = series_cum.loc[-w_size] if -w_size in series_cum.index else np.nan
        pre_endpoint_display = -pre_endpoint  # Apply sign flip for display
        
        # 3- Running the computation without the sign flip
        pre_mean_no_flip = pre_vals.mean()
        
        print(f"Window: {w_name} ({w_size}d), Group: {dependency_type.capitalize()}")
        print(f"1. Average of all pre-election days (with sign flip): {pre_mean_all_display:.3f}%")
        print(f"2. Value at start of window (with sign flip): {pre_endpoint_display:.3f}%")
        print(f"3. Average of all pre-election days (no sign flip): {pre_mean_no_flip:.3f}%")
        
        # Compare to actual plot values
        print("\nVisual check: Values in the plot at key points:")
        for day in [-w_size, -w_size//2, -1, 0, 1, w_size//2, w_size]:
            if day in series_cum.index:
                print(f"  Day {day}: {series_cum.loc[day]:.3f}%")
    else:
        print("No data available for comparison")


if __name__ == "__main__":
    print("Reading data--")
    df = pd.read_csv(
        r"#FILEPATH HERE",
        float_precision='high'
    )
    df['Dates'] = pd.to_datetime(df['Dates'])
    df.set_index('Dates', inplace=True)
    df.columns = [col.replace(' Curncy', '') for col in df.columns]
    
    print(f"Data range: {df.index.min()} to {df.index.max()}\n")
    
    check_event_date_alignments(df)
    
    scenarios_for_plot = [
        (None, "All Elections"),
        (lambda party, expectation: party == 'Democratic', "Democratic Winners"),
        (lambda party, expectation: party == 'Republican', "Republican Winners"),
        (lambda party, expectation: expectation == 'Expected', "Expected Outcomes"),
        (lambda party, expectation: expectation == 'Surprise', "Surprise Outcomes")
    ]
    
    for ffunc, title_str in scenarios_for_plot:
        plot_analysis(df, ffunc, title_str)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sub_scenarios = [
        (lambda party, expectation: party == 'Republican' and expectation == 'Expected', "Rep × Expected"),
        (lambda party, expectation: party == 'Republican' and expectation == 'Surprise', "Rep × Surprise"),
        (lambda party, expectation: party == 'Democratic' and expectation == 'Expected', "Dem × Expected"),
        (lambda party, expectation: party == 'Democratic' and expectation == 'Surprise', "Dem × Surprise")
    ]
    for (func, sub_title), ax in zip(sub_scenarios, axes.flatten()):
        plot_analysis(df, filter_func=func, title_suffix=sub_title, ax=ax)
    plt.tight_layout()
    plt.show()
    
    print_stats_tables_endpoints(df)
    
    run_regression_analysis(df, window='Short', spec='compact')
    run_regression_analysis(df, window='Short', spec='full')
    run_regression_analysis(df, window='Medium', spec='compact')
    run_regression_analysis(df, window='Medium', spec='full')
    run_regression_analysis(df, window='Extended', spec='compact')
    run_regression_analysis(df, window='Extended', spec='full')
    run_regression_analysis(df, window='Long', spec='compact')
    run_regression_analysis(df, window='Long', spec='full')
    
    print_spread_analysis_all_windows(df)
    
    compare_pre_post_calculations(df)
