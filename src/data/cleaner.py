"""Data cleaning, cancellation filtering, and outlier handling."""
from __future__ import annotations

import pandas as pd
import numpy as np
from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit

def remove_missing_customers(df:pd.DataFrame)->pd.DataFrame:
    initial_rows=len(df)
    cleaned_df=df.dropna(subset=['CustomerID']).copy()
    dropped_rows=initial_rows - len(cleaned_df)

    pct_dropped=(dropped_rows/initial_rows)*100.0 if initial_rows > 0 else 0.0

    # ensure customer id is stored cleanly as integer
    cleaned_df['CustomerID']=cleaned_df['CustomerID'].astype(int)

    logger.info(
        f'missing customer id filter: dropped {dropped_rows:,} rows '
        f'({pct_dropped:.1f}%). remaining{len(cleaned_df):,}'
    )
    return cleaned_df

def remove_cancellations(df:pd.DataFrame)->pd.DataFrame:
    prefix = str(config.cleaning.cancellation_prefix)
    initial_rows=len(df)

    is_cancellation =df['InvoiceNo'].astype(str).str.startswith(prefix)
    cleaned_df=df[~is_cancellation].copy()

    dropped_rows=initial_rows - len(cleaned_df)
    pct_dropped=(dropped_rows /initial_rows)* 100 if initial_rows > 0 else 0.0

    logger.info(f'cancellation filtr( {prefix}) : dropped: {dropped_rows:,} rows'
    f'({pct_dropped:.1f}%). remaining: {len(cleaned_df):,} rows.')

    return cleaned_df

def filter_invalid_transactions(df:pd.DataFrame)-> pd.DataFrame:
    min_qty=config.cleaning.min_quantity
    min_price=config.cleaning.min_unit_price
    initial_rows=len(df)

    valid_mask=(df['Quantity']>=min_qty) &(df['UnitPrice']>=min_price)

    cleaned_df=df[valid_mask].copy()

    dropped_rows=initial_rows-len(cleaned_df)
    pct_dropped=(dropped_rows/initial_rows) *100 if initial_rows >0 else 0.0

    logger.info(
        f' valid transactions filter (qty > = {min_qty},price >= {min_price}):'
        f' dropped {dropped_rows:,} rows ( {pct_dropped:.1f}%). remaining: {len(cleaned_df):,} rows'
        
    )

    return cleaned_df

def handle_outliers(df:pd.DataFrame)-> pd.DataFrame:
    quantile_thres = config.cleaning.outlier_quantile
    cleaned_df=df.copy()

    qty_cap = cleaned_df['Quantity'].quantile(quantile_thres)
    price_cap=cleaned_df['UnitPrice'].quantile(quantile_thres)

    n_capped_qty=(cleaned_df['Quantity']> qty_cap).sum()
    n_capped_price=(cleaned_df['UnitPrice']> price_cap).sum()

    cleaned_df['Quantity']=cleaned_df['Quantity'].clip(upper=qty_cap)
    cleaned_df['UnitPrice']=cleaned_df['UnitPrice'].clip(upper=price_cap)

    logger.info(
        f' outlier capping ({quantile_thres * 100:.1f}th pct):'
        f' capped {n_capped_qty:,} quantities at {qty_cap:.1f},'
        f'{n_capped_price:,} prices at {price_cap:.2f}.'

    )   

    return cleaned_df


@timeit

def clean_data(df: pd.DataFrame)-> pd.DataFrame:

    logger.info(f'starting data cleaning on {len(df):,} raw transactions...')
    cleaned=df.copy()

    cleaned['InvoiceDate']=pd.to_datetime(cleaned['InvoiceDate'])

    if config.cleaning.drop_missing_customer_id:
        cleaned=remove_missing_customers(cleaned)

    cleaned=remove_cancellations(cleaned)

    cleaned=filter_invalid_transactions(cleaned)

    cleaned=handle_outliers(cleaned)

    cleaned['TotalPrice']=cleaned['Quantity']* cleaned['UnitPrice']

    if 'Description' in cleaned.columns:
        cleaned['Description']=cleaned['Description'].astype(str).str.strip()

    if 'Country' in cleaned.columns:
        cleaned['Country']=cleaned['Country'].astype(str).str.strip()


    logger.info(
        f' cleaning complete final dataset: {len(cleaned):,}rows,'
        f' {cleaned["CustomerID"].nunique():,} unique customers'

    )

    return cleaned
