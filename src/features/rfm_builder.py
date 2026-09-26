"""RFM, behavioral, loyalty, and customer lifetime value feature builder."""


from __future__ import annotations

from pathlib import Path
from typing import Optional,Union
import numpy as np
import pandas as pd

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit

@timeit
def build_rfm_features(
    df:pd.DataFrame,save_csv:bool =True,output_path:Optional[Union[Path,str]]=None,
)-> pd.DataFrame:
    logger.info(f' building rfm &  behavioural feaures from {len(df):,} transactions')

    # reference date: 1 day after the latest transaction in the dataset
    ref_date= df['InvoiceDate'].max() + pd.Timedelta(days=1)
    logger.info(f' calculated reference snapshot date: {ref_date.date()}')

    # 1. base customer aggregations
    customer_agg= df.groupby('CustomerID').agg(
        Recency=('InvoiceDate',lambda d: (ref_date-d.max()).days),
        Frequency=('InvoiceNo','nunique'),
        Monetary=('TotalPrice','sum'),
        ProductVariety=('StockCode','nunique'),
        FirstPurchase=('InvoiceDate','min'),
        LastPurchase=('InvoiceDate','max'),
        TotalUnits=('Quantity','sum'),
        TotalTransactions=('InvoiceNo','count'),
        Country=('Country',lambda c: c.mode()[0] if not c.empty else 'Unknown'),



    ).reset_index()

    # tenure /lifespan in days

    customer_agg['LifespanDays']=(customer_agg['LastPurchase']-customer_agg['FirstPurchase']).dt.days

    # average order value
    customer_agg['AvgOrderValue']=(customer_agg['Monetary']/customer_agg['Frequency'])

    # purchase rate 
    customer_agg['PurchaseRate']=(customer_agg['Frequency']/(customer_agg['LifespanDays'] +1))

    # average days between purchase
    customer_agg['AvgDaysBetweenPurchases']=np.where(
        customer_agg['Frequency']>1,
        customer_agg['LifespanDays']/(customer_agg['Frequency']-1),
        customer_agg['LifespanDays'],
    )

    # spend per product

    customer_agg['SpendPerProduct']=customer_agg['Monetary']/np.maximum(customer_agg['ProductVariety'],1)

    # quantity per order
    customer_agg['QtyPerOrder']=customer_agg['TotalUnits']/ np.maximum(customer_agg['Frequency'],1)

    # revenue per day
    customer_agg['RevenuePerDay']=customer_agg['Monetary']/(customer_agg['LifespanDays']+1)

    # unique product percentage

    customer_agg['UniqueProductPct']=customer_agg['ProductVariety']/np.maximum(
        customer_agg['TotalTransactions'],1
    )

    # order value consistency 
    logger.info('computing order value consistency ')
    invoice_totals=df.groupby(['CustomerID','InvoiceNo'])['TotalPrice'].sum().reset_index()
    order_std=(
        invoice_totals.groupby('CustomerID')['TotalPrice'].std().fillna(0.0).rename('OrderValueConsistency')

    )
    customer_agg=customer_agg.merge(order_std,on='CustomerID',how='left')
    customer_agg['OrderValueConsistency']=customer_agg['OrderValueConsistency'].fillna(0.0)

    # customer lifetime value (clv) proxy score
    customer_agg['CLV_score']=customer_agg['AvgOrderValue']*customer_agg['Frequency']*(customer_agg['LifespanDays']+1)/(customer_agg['Recency']+1)

    # RFM Quintile Scoring (1 to 5)
    n_bins = min(config.features.n_rfm_bins, len(customer_agg))
    if n_bins >= 2:
        customer_agg["R_Score"] = pd.qcut(
            customer_agg["Recency"].rank(method="first"), n_bins, labels=list(range(n_bins, 0, -1))
        ).astype(int)
        customer_agg["F_Score"] = pd.qcut(
            customer_agg["Frequency"].rank(method="first"), n_bins, labels=list(range(1, n_bins + 1))
        ).astype(int)
        customer_agg["M_Score"] = pd.qcut(
            customer_agg["Monetary"].rank(method="first"), n_bins, labels=list(range(1, n_bins + 1))
        ).astype(int)
    else:
        customer_agg["R_Score"] = 1
        customer_agg["F_Score"] = 1
        customer_agg["M_Score"] = 1

    customer_agg["RFM_Score"] = (
        customer_agg["R_Score"] + customer_agg["F_Score"] + customer_agg["M_Score"]
    )
    customer_agg["RFM_Segment"] = (
        customer_agg["R_Score"].astype(str)
        + customer_agg["F_Score"].astype(str)
        + customer_agg["M_Score"].astype(str)
    )

    #clean date helper columns
    customer_agg=customer_agg.drop(columns=['FirstPurchase','LastPurchase'])
    
    logger.info(
        f'feature engineering finished: {len(customer_agg):,} customer records'
        f'with {len(customer_agg.columns)} features'
    )

    if save_csv:
        target_path=Path(output_path).resolve() if output_path else config.paths.rfm_features_csv
        target_path.parent.mkdir(parents=True,exist_ok=True)
        customer_agg.to_csv(target_path,index=False)
        logger.info(f'Saved rfm features csv to : {target_path}')

    return customer_agg
