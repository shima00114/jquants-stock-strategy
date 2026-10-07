import os
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter('ignore')

def load_history(name: str) -> pd.DataFrame:
    train_path = f"{name}_train.parquet" if os.path.exists(f"{name}_train.parquet") else f"input/{name}_train.parquet"
    valid_path = f"{name}_valid.parquet" if os.path.exists(f"{name}_valid.parquet") else f"input/{name}_valid.parquet"
    
    train = pd.read_parquet(train_path) if os.path.exists(train_path) else pd.DataFrame()
    valid = pd.read_parquet(valid_path) if os.path.exists(valid_path) else pd.DataFrame()
    
    return pd.concat([train, valid]).sort_index()

def predict() -> pd.DataFrame:
    df = load_history("prices_daily_quotes")
    
    valid_path = "prices_daily_quotes_valid.parquet" if os.path.exists("prices_daily_quotes_valid.parquet") else "input/prices_daily_quotes_valid.parquet"
    valid_index = pd.read_parquet(valid_path).index if os.path.exists(valid_path) else df.index

    df_work = df.reset_index()
    close_col = 'AdjustmentClose' if 'AdjustmentClose' in df_work.columns else 'Close'
    df_work = df_work.sort_values(['Code', 'Date']).reset_index(drop=True)

    # 1. 10-day price return
    window = 10
    past_close = df_work.groupby('Code')[close_col].shift(window)
    df_work['stock_ret'] = ((df_work[close_col] - past_close) / (past_close + 1e-8)).fillna(0)

    # 2. TOPIX return integration & beta adjustment
    topix_df = load_history("topix_return_1day")
    if not topix_df.empty:
        topix_reset = topix_df.reset_index()
        topix_col = [c for c in topix_reset.columns if c != 'Date'][0]
        topix_reset['topix_ret'] = topix_reset[topix_col].rolling(window, min_periods=1).sum()
        df_work = pd.merge(df_work, topix_reset[['Date', 'topix_ret']], on='Date', how='left')
        df_work['topix_ret'] = df_work['topix_ret'].fillna(0)
        
        beta_df = load_history("beta_1day")
        if not beta_df.empty:
            beta_reset = beta_df.reset_index()
            beta_col = [c for c in beta_reset.columns if c not in ['Date', 'Code']][0]
            df_work = pd.merge(df_work, beta_reset[['Date', 'Code', beta_col]], on=['Date', 'Code'], how='left')
            df_work['beta'] = df_work[beta_col].fillna(1.0)
        else:
            df_work['beta'] = 1.0
            
        df_work['residual_ret'] = df_work['stock_ret'] - df_work['beta'] * df_work['topix_ret']
    else:
        df_work['residual_ret'] = df_work['stock_ret']

    # 3. Mean-reversion signal & 40-day moving average smoothing
    df_work['raw_pred'] = -df_work['residual_ret']
    df_work['smoothed'] = df_work.groupby('Code')['raw_pred'].transform(lambda x: x.rolling(40, min_periods=1).mean().fillna(0))
    
    # 4. Cross-sectional rank normalization
    df_work['pred'] = df_work.groupby('Date')['smoothed'].rank(pct=True) - 0.5

    df_work = df_work.set_index(['Date', 'Code'])
    result = df_work['pred'].reindex(valid_index).fillna(0.0)
    
    return result.rename("Return").to_frame()
