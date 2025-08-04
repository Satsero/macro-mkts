"""
Long-Short momentum strategy -- Python backtest

Description:
This script implements and backtests a dynamic L/S equity mom strategy using the "6 Portfolios Formed on Size and Momentum (2x3)" dataset from the widely-used Kenneth French's library.

Core strategy:
- Each month, the strategy calculates 6-month momentum scores for four portfolios:
  ['SMALL LoPRIOR', 'SMALL HiPRIOR', 'BIG LoPRIOR', 'BIG HiPRIOR']
- It dynamically allocates long positions to the two portfolios with the highest scores 
  and short positions to the two with the lowest scores.
- Dollar-neutral construction with ±50% weights per side.
- Monthly rebalancing and position sizing vary with relative momentum strength.

Transaction cost modelling:
- Size-adjusted spreads and market impact estimates
- Volatility scaling based on rolling market vol (12mnth window)
- Borrow cost assumptions: 200bps pa for big caps, 400bps for small caps

Performance analysis:
- Gross and net return series generated
- Full performance metrics: annual return, vol, Sharpe, drawdown, hit rate, alpha, beta
- CAPM regressions (gross/net)
- Rolling performance and correlation plots
- Simulated arbitrage structure using a market-neutral short position in the strategy

Underlying data source:
- [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html]

NOTE:
This script is intended for academic and research, non-commercial use. Please see LICENSE in main;
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data():
    portfolio_path = r"#path here"
    port_df = pd.read_csv(portfolio_path)
    port_df['Date'] = pd.to_datetime(port_df['Date'], format='%m/%Y')
    port_df.set_index('Date', inplace=True)
    
    market_path = r"C:\Users\orest\Downloads\Monthly - MKT and RF.CSV"
    mkt_df = pd.read_csv(market_path)
    mkt_df['Date'] = pd.to_datetime(mkt_df['Date'], format='%m/%Y')
    mkt_df.set_index('Date', inplace=True)
    
    port_df.sort_index(inplace=True)
    mkt_df.sort_index(inplace=True)
    
    return port_df, mkt_df

def select_strategy_portfolios(port_df):
    return ['SMALL LoPRIOR', 'SMALL HiPRIOR', 'BIG LoPRIOR', 'BIG HiPRIOR']

def calculate_momentum_score(port_df, lookback=6):
    """as expained in the report's strategy methodology section - this function calculate momentum scores for selected portfolios"""
    portfolios = select_strategy_portfolios(port_df)
    
    momentum_scores = pd.DataFrame(index=port_df.index)
    for portfolio in portfolios:
        returns = port_df[portfolio]
        rolling_returns = returns/100
        cum_returns = rolling_returns.rolling(window=lookback).apply(
            lambda x: np.prod(1 + x) - 1
        )
        momentum_scores[f'{portfolio}_Score'] = cum_returns
    
    return momentum_scores

def construct_portfolio(port_df, mkt_df, lookback=6, n_positions=2):
    """construction of the long-short portfolio based on the momentum scores found"""
    momentum_scores = calculate_momentum_score(port_df, lookback)
    
    portfolio_columns = select_strategy_portfolios(port_df)
    weights = pd.DataFrame(0, index=port_df.index, columns=port_df.columns)
    
    for date in port_df.index[lookback:]:
        current_scores = {}
        for col in portfolio_columns:
            score_col = f'{col}_Score'
            if score_col in momentum_scores.columns:
                current_scores[col] = momentum_scores.loc[date, score_col]
        
        if current_scores:
            sorted_portfolios = sorted(current_scores.items(), key=lambda x: x[1])
            
            for port, _ in sorted_portfolios[-n_positions:]:
                weights.loc[date, port] = 1.0/n_positions

            for port, _ in sorted_portfolios[:n_positions]:
                weights.loc[date, port] = -1.0/n_positions
    
    weights = weights[portfolio_columns]
    return weights

def get_size_adjusted_costs(portfolio_name):
    """ the variables here can also be manually changed to adjust size of trading costs type, but below are chosen
    assumptions"""
    base_spread = 0.0010  # base spread
    base_impact = 0.0005   # basemarket impact
    
    if 'SMALL' in portfolio_name:
        spread = base_spread * 2.25
        impact = base_impact * 3.5
    elif 'ME1' in portfolio_name:
        spread = base_spread * 1.75
        impact = base_impact * 2
    elif 'ME2' in portfolio_name:
        spread = base_spread * 1.5
        impact = base_impact * 1.5
    else:
        spread = base_spread
        impact = base_impact
    
    return spread, impact

def calculate_transaction_costs(weights_df, port_df, mkt_df):
    total_costs = pd.Series(0, index=weights_df.index)
    
    # mkt vol scaling
    market_vol = mkt_df['Mkt-RF'].rolling(window=12, min_periods=1).std()
    vol_scalar = market_vol / market_vol.mean()
    
    for portfolio in weights_df.columns:
        spread, impact = get_size_adjusted_costs(portfolio)
        position_changes = weights_df[portfolio].diff().fillna(0).abs()
        scaled_impact = impact * vol_scalar
        portfolio_costs = position_changes * (spread + scaled_impact)
        is_short = weights_df[portfolio] < 0
        borrow_cost_annual = 0.02 if 'BIG' in portfolio else 0.04
        borrow_cost_monthly = borrow_cost_annual/12
        borrowing_costs = np.where(is_short, abs(weights_df[portfolio]) * borrow_cost_monthly, 0)
        
        total_costs += portfolio_costs + borrowing_costs
    
    return total_costs

def calculate_strategy_returns(port_df, weights):
    returns = port_df / 100
    strategy_returns = (weights * returns).sum(axis=1)
    return strategy_returns

def calculate_performance_metrics(df):
    metrics = pd.DataFrame(columns=['Strategy (Gross)', 'Strategy (Net)', 'Market'])
    
    # ann. returns
    metrics.loc['Annual Return (%)'] = [
        ((1 + df['Strategy_Return_Gross']).prod() ** (12/len(df)) - 1) * 100,
        ((1 + df['Strategy_Return_Net']).prod() ** (12/len(df)) - 1) * 100,
        ((1 + df['Market_Return']).prod() ** (12/len(df)) - 1) * 100
    ]
    
    #ann. volatility
    metrics.loc['Annual Volatility (%)'] = [
        df['Strategy_Return_Gross'].std() * np.sqrt(12) * 100,
        df['Strategy_Return_Net'].std() * np.sqrt(12) * 100,
        df['Market_Return'].std() * np.sqrt(12) * 100
    ]
    
    # Sharpe ratio
    rf_rate = df['RF_Rate'].mean()
    metrics.loc['Sharpe Ratio'] = [
        (df['Strategy_Return_Gross'].mean() - rf_rate) / df['Strategy_Return_Gross'].std() * np.sqrt(12),
        (df['Strategy_Return_Net'].mean() - rf_rate) / df['Strategy_Return_Net'].std() * np.sqrt(12),
        (df['Market_Return'].mean() - rf_rate) / df['Market_Return'].std() * np.sqrt(12)
    ]
    
    # max drawdown
    metrics.loc['Maximum Drawdown (%)'] = [
        df['Strategy_DD_Gross'].min() * 100,
        df['Strategy_DD_Net'].min() * 100,
        df['Market_DD'].min() * 100
    ]
    
    # hit rate for the strategy
    metrics.loc['Monthly Hit Rate (%)'] = [
        (df['Strategy_Return_Gross'] > 0).mean() * 100,
        (df['Strategy_Return_Net'] > 0).mean() * 100,
        (df['Market_Return'] > 0).mean() * 100
    ]
    
    # 
    excess_returns_gross = df['Strategy_Return_Gross'] - df['Market_Return']
    excess_returns_net = df['Strategy_Return_Net'] - df['Market_Return']
    
    metrics.loc['Information Ratio'] = [
        excess_returns_gross.mean() / excess_returns_gross.std() * np.sqrt(12),
        excess_returns_net.mean() / excess_returns_net.std() * np.sqrt(12),
        np.nan
    ]
    
    # comaparison of strategy vs market (beta/alpha)
    beta_gross, alpha_gross = np.polyfit(df['Market_Return'], df['Strategy_Return_Gross'], 1)
    beta_net, alpha_net = np.polyfit(df['Market_Return'] - df['RF_Rate'], 
                                df['Strategy_Return_Net'] - df['RF_Rate'], 1)
    
    metrics.loc['Beta'] = [beta_gross, beta_net, 1.0]
    metrics.loc['Alpha (annual %)'] = [alpha_gross * 12 * 100, alpha_net * 12 * 100, np.nan]
    return metrics.round(4)

def test_capm(df, return_column):
    """linear regression function to test CAPM"""
    clean_df = df[[return_column, 'Market_Return', 'RF_Rate']].dropna()
    
    excess_strategy = clean_df[return_column] - clean_df['RF_Rate']
    excess_market = clean_df['Market_Return'] - clean_df['RF_Rate']
    
    try:
        model = stats.linregress(excess_market, excess_strategy)
        
        n = len(clean_df)
        residuals = excess_strategy - (model.intercept + model.slope * excess_market)
        std_err_regression = np.sqrt(np.sum(residuals**2) / (n-2))
        
        x_mean = np.mean(excess_market)
        x_var = np.var(excess_market, ddof=1)
        std_err_alpha = std_err_regression * np.sqrt(1/n + x_mean**2/(n*x_var))
        
        t_stat_alpha = model.intercept / std_err_alpha
        p_value_alpha = 2 * (1 - stats.t.cdf(abs(t_stat_alpha), n-2))
        
        return {
            'Alpha (annual %)': model.intercept * 12 * 100,
            'Beta': model.slope,
            'R-squared': model.rvalue**2,
            'Alpha t-stat': t_stat_alpha,
            'Alpha p-value': p_value_alpha,
            'Observations': n
        }
    except Exception as e:
        print(f"error CAPM analysis for {return_column}: {str(e)}")
        return None

def create_returns_plot(df):

    #log scale plot
    plt.figure(figsize=(15, 8))
    df[['Cum_Strategy_Gross', 'Cum_Strategy_Net', 'Cum_Market']].plot(logy=True)
    plt.title('Cumulative Returns Backtest: Momentum Strategy vs Market (Log Scale)', fontsize=16)
    plt.ylabel('Cumulative Return (Log)', fontsize=16)
    plt.grid(True)
    plt.legend(['Strategy (Gross)', 'Strategy (Net)', 'Market'], fontsize=14)
    plt.show()
    
    # normal scale plot (only net strategy returns here - edit-in gross if needed)
    plt.figure(figsize=(15, 8))
    df[['Cum_Strategy_Net', 'Cum_Market']].plot()
    plt.title('Cumulative Returns Backtest: Momentum Strategy vs Market (Normal Scale)', fontsize=16)
    plt.ylabel('Cumulative Return', fontsize=16)
    plt.grid(True)
    plt.legend(['Strategy (Net)', 'Market'], fontsize=14)
    plt.show()

def create_performance_plots(df):
    rolling_correlation = df['Strategy_Return_Net'].rolling(window=12).corr(df['Market_Return'])
    avg_correlation = rolling_correlation.mean()
    
    fig = plt.figure(figsize=(15, 20))
    
    #rolling returns plot
    ax1 = plt.subplot(3, 1, 1)
    rolling_strategy_gross = df['Strategy_Return_Gross'].rolling(12).mean() * 12 * 100
    rolling_strategy_net = df['Strategy_Return_Net'].rolling(12).mean() * 12 * 100
    rolling_market = df['Market_Return'].rolling(12).mean() * 12 * 100
    rolling_strategy_gross.plot(ax=ax1, label='Strategy (Gross)')
    rolling_strategy_net.plot(ax=ax1, label='Strategy (Net)')
    rolling_market.plot(ax=ax1, label='Market')
    ax1.set_title('Rolling 12-Month Returns (%)')
    ax1.grid(True)
    ax1.legend()
    
    #dawdown plot
    ax2 = plt.subplot(3, 1, 2)
    df[['Strategy_DD_Gross', 'Strategy_DD_Net', 'Market_DD']].plot(ax=ax2)
    ax2.set_title('Drawdowns')
    ax2.grid(True)
    ax2.legend(['Strategy (Gross)', 'Strategy (Net)', 'Market'])
    
    # rolling correl
    ax3 = plt.subplot(3, 1, 3)
    rolling_correlation.plot(ax=ax3, color='blue', label='Rolling Correlation')
    ax3.axhline(y=avg_correlation, color='red', linestyle='--',
                label=f'Average Correlation: {avg_correlation:.3f}')
    ax3.set_title('Rolling 12-Month Correlation with Market')
    ax3.grid(True)
    ax3.legend()
    plt.tight_layout()
    plt.show()


def analyze_and_plot_arbitrage(results_df, short_position=1000000):
    """function to analyze CAPM-based arbitrage strategy w/ simulated 1mil short position"""
    valid_data = results_df[
        (results_df['Strategy_Return_Net'].notna()) &
        (results_df['Market_Return'].notna()) &
        (results_df['RF_Rate'].notna())
    ].copy()
    
    # 75years beta used (individual data check), assumed to hold true on avergae for the range
    beta = -0.0646
    market_position = abs(beta * short_position)
    rf_position = short_position - market_position
    
    print(f"\nPosition breakdown:")
    print(f"Short strategy: ${-short_position:,.2f}")
    print(f"Long market: ${market_position:,.2f}")
    print(f"Risk-free: ${rf_position:,.2f}")
    
    arbitrage_returns = pd.DataFrame(index=valid_data.index)
    arbitrage_returns['Strategy_Return'] = -valid_data['Strategy_Return_Net']
    arbitrage_returns['Market_Return'] = valid_data['Market_Return']
    arbitrage_returns['RF_Return'] = valid_data['RF_Rate']
    arbitrage_returns['Total_Return'] = (
        (-short_position * arbitrage_returns['Strategy_Return'] +
         market_position * arbitrage_returns['Market_Return'] +
         rf_position * arbitrage_returns['RF_Return']
        ) / short_position
    )
    
    #cumulative returns for this arbitrage strategy
    arbitrage_returns['Cum_Return'] = (1 + arbitrage_returns['Total_Return']).cumprod() - 1

    annualized_ret = ((1 + arbitrage_returns['Cum_Return'].iloc[-1]) ** (12/len(arbitrage_returns)) - 1)
    annualized_vol = arbitrage_returns['Total_Return'].std() * np.sqrt(12)
    monthly_mean = arbitrage_returns['Total_Return'].mean()
    monthly_vol = arbitrage_returns['Total_Return'].std()
    
    # max drawdown
    cum_returns = (1 + arbitrage_returns['Total_Return']).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    print("\nDetailed arbitrage strategy metrics:")
    print("-" * 50)
    print(f"Time period analysis:")
    print(f"Start date: {arbitrage_returns.index[0].strftime('%Y-%m-%d')}")
    print(f"End date: {arbitrage_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"Number of months: {len(arbitrage_returns)}")
    print("\nReturn Metrics:")
    print(f"Total cumulative return: {arbitrage_returns['Cum_Return'].iloc[-1]:.2%}")
    print(f"Annualized return: {annualized_ret:.2%}")
    print(f"Average Monthly return: {monthly_mean:.2%}")
    print("\nRisk Metrics:")
    print(f"Annualized vol: {annualized_vol:.2%}")
    print(f"Monthly return std dev: {monthly_vol:.2%}")
    print(f"Max drawdown: {max_drawdown:.2%}")
    print("\nRisk-adjusted metrics:")
    print(f"Annualized Sharpe ratio: {(annualized_ret / annualized_vol):.2f}")
    print(f"% Positive Months: {(arbitrage_returns['Total_Return'] > 0).mean():.2%}")
    
    print("\nCorrelation Analysis:")
    print("Correlation with Strategy: {:.2f}".format(
        arbitrage_returns['Total_Return'].corr(valid_data['Strategy_Return_Net'])))
    print("Correlation with Market: {:.2f}".format(
        arbitrage_returns['Total_Return'].corr(valid_data['Market_Return'])))
    
    # verifying component contribution analysis
    strategy_contrib = (-short_position * arbitrage_returns['Strategy_Return']) / short_position
    market_contrib = (market_position * arbitrage_returns['Market_Return']) / short_position
    rf_contrib = (rf_position * arbitrage_returns['RF_Return']) / short_position
    print("\nAverage monthly component Contributions:")
    print(f"Strategy contribution: {strategy_contrib.mean():.2%}")
    print(f"Market contribution: {market_contrib.mean():.2%}")
    print(f"Risk-free contribution: {rf_contrib.mean():.2%}")
    
    return stats, arbitrage_returns, arbitrage_returns


def main():
    try:
        print("\nMonthly momentum strategy analysis")
        print("=" * 40)
        
        # Get date range
        min_start_date = pd.to_datetime('1950-01-01')
        lookback_months = 12
        start_date = None
        end_date = None
        
        print("\nDate range selection (press Enter for full dataset starting from 1950 - CLEANED DATA FILES)")
        start_input = input("Enter start date (MM/YYYY, minimum 01/1950): ").strip()
        if start_input:
            start_date = pd.to_datetime(start_input, format='%m/%Y')
            if start_date < min_start_date:
                print(f"Start date adjusted to {min_start_date.strftime('%m/%Y')}")
                start_date = min_start_date
        else:
            start_date = min_start_date
        
        end_input = input("Enter end date (MM/YYYY): ").strip()
        if end_input:
            end_date = pd.to_datetime(end_input, format='%m/%Y')
        
        port_df, mkt_df = load_data()
        
        lookback_start = start_date - pd.DateOffset(months=lookback_months)
        
        port_df_with_lookback = port_df[port_df.index >= lookback_start]
        mkt_df_with_lookback = mkt_df[mkt_df.index >= lookback_start]
        
        if end_date:
            port_df_with_lookback = port_df_with_lookback[port_df_with_lookback.index <= end_date]
            mkt_df_with_lookback = mkt_df_with_lookback[mkt_df_with_lookback.index <= end_date]
        
        weights = construct_portfolio(port_df_with_lookback, mkt_df_with_lookback)
        
        strategy_returns = calculate_strategy_returns(port_df_with_lookback, weights)
        transaction_costs = calculate_transaction_costs(weights, port_df_with_lookback, mkt_df_with_lookback)
        market_returns = mkt_df_with_lookback['Mkt-RF']/100 + mkt_df_with_lookback['RF']/100

        results_df = pd.DataFrame({
            'Strategy_Return_Gross': strategy_returns,
            'Strategy_Return_Net': strategy_returns - transaction_costs,
            'Market_Return': market_returns,
            'RF_Rate': mkt_df_with_lookback['RF']/100,
            'Transaction_Costs': transaction_costs
        })

        results_df['Cum_Strategy_Gross'] = (1 + results_df['Strategy_Return_Gross']).cumprod()
        results_df['Cum_Strategy_Net'] = (1 + results_df['Strategy_Return_Net']).cumprod()
        results_df['Cum_Market'] = (1 + results_df['Market_Return']).cumprod()

        results_df['Strategy_DD_Gross'] = results_df['Cum_Strategy_Gross']/results_df['Cum_Strategy_Gross'].expanding().max() - 1
        results_df['Strategy_DD_Net'] = results_df['Cum_Strategy_Net']/results_df['Cum_Strategy_Net'].expanding().max() - 1
        results_df['Market_DD'] = results_df['Cum_Market']/results_df['Cum_Market'].expanding().max() - 1

        print("\nCalculating performance metrics...")
        metrics = calculate_performance_metrics(results_df)
        print("\nPerformance Metrics:")
        print(metrics)
        
        # CAPM analysis
        print("\nCAMP analysis:")
        for returns_type in ['Gross', 'Net']:
            col_name = f'Strategy_Return_{returns_type}'
            results = test_capm(results_df, col_name)
            
            if results:
                print(f"\n{returns_type} Returns Analysis:")
                print(f"Number of observations: {results['Observations']}")
                print(f"Alpha (annual %): {results['Alpha (annual %)']:.4f}")
                print(f"Beta: {results['Beta']:.4f}")
                print(f"R-squared: {results['R-squared']:.4f}")
                print(f"Alpha t-statistic: {results['Alpha t-stat']:.4f}")
                print(f"Alpha p-value: {results['Alpha p-value']:.4f}")
                
                sig_level = ""
                if results['Alpha p-value'] < 0.01:
                    sig_level = "***"
                elif results['Alpha p-value'] < 0.05:
                    sig_level = "**"
                elif results['Alpha p-value'] < 0.1:
                    sig_level = "*"
                else:
                    sig_level = "not significant"
                    
                print(f"Statistical significance: {sig_level}")
                print("-" * 50)

        # Generate plots
        create_returns_plot(results_df)
        create_performance_plots(results_df)
        
        
        #cost analysis
        total_cost_pct = transaction_costs.sum() * 100
        avg_monthly_cost = transaction_costs.mean() * 100
        print(f"\nTransaction Cost Analysis:")
        print(f"Total Transaction Costs: {total_cost_pct:.2f}%")
        print(f"Average Monthly Cost: {avg_monthly_cost:.2f}%")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
