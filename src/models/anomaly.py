"""Customer anomaly and outlier detection models (Isolation Forest, LOF)."""

from __future__ import annotations
from typing import Any,Dict,Optional,Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import config
from src.utils.logger import logger
from src.utils.helpers import save_joblib,load_joblib,timeit

class CustomerAnomalyDetector:

    def __init__(self,contamination:Optional[float]=None,n_estimators: Optional[int]=None,random_state:Optional[int]=None,n_jobs:Optional[int]=None,)->None:
        self.contamination=(
            contamination if contamination is not None else config.anomaly_detection.contamination
        )

        self.n_estimators=(
            n_estimators if n_estimators is  not None else config.anomaly_detection.n_estimators
        )
        self.random_state=(
            random_state if random_state is not None else config.anomaly_detection.random_state
        )
        self.n_jobs=(
            n_jobs if n_jobs is not None else config.anomaly_detection.n_jobs
        )

        self.model=IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.is_fitted=False
    @timeit
    def fit(self,x_scaled:np.ndarray,save_artifact:bool=True,)->'CustomerAnomalyDetector':

        logger.info(
            f'fitting isolation forest (contamination={self.contamination},'
            f'trees={self.n_estimators}) on {len(x_scaled):,} customer records....'

        )

        self.model.fit(x_scaled)
        self.is_fitted=True

        raw_preds=self.model.predict(x_scaled)
        n_anomalies=int((raw_preds ==-1).sum())
        pct=(n_anomalies/len(x_scaled))*100
        logger.info(
            f'isolation forest fitting complete:flagged {n_anomalies:,} anomalies ({pct:.2f}% of base)'


        )

        if save_artifact:
            self.save()

        return self


    
    def predict(self,x_scaled: np.ndarray)->np.ndarray:
        if not self.is_fitted:
            raise RuntimeError('anomaly model is not fitted yet')

        return self.model.predict(x_scaled)

    def decision_function(self,x_scaled:np.ndarray)->np.ndarray:

        if not self.is_fitted:
            raise RuntimeError('anomaly model is not fitted yet')
        return self.model.decision_function(x_scaled)

    def detect(self,x_scaled:np.ndarray)->pd.DataFrame:
        preds=self.predict(x_scaled)
        scores=self.decision_function(x_scaled)

        return pd.DataFrame({
            'is_anomaly':preds==-1,
            'anomaly_score':np.round(scores,4),
        })
    
    def profile_anomalies(self,customer_df:pd.DataFrame,is_anomaly_mask:np.ndarray,)->Dict[str,Any]:
        df_copy=customer_df.copy()
        df_copy['is_anomaly']=is_anomaly_mask

        anomalies=df_copy[df_copy['is_anomaly']]
        normals=df_copy[~df_copy['is_anomaly']]
        summary={
            'total_customers':len(df_copy),
            'anomaly_count': len(anomalies),
            'normal_count':len(normals),
            'anomaly_revenue_share_pct':round(
                (anomalies['Monetary'].sum()/df_copy['Monetary'].sum())*100,2
            ),
            'anomalies':{
                'avg_monetary':round(float(anomalies['Monetary'].mean()),2),
                'avg_frequency':round(float(anomalies['Frequency'].mean()),2),
                'avg_order_value':round(float(anomalies['AvgOrderValue'].mean()),2),
            },
            'normals':{
                'avg_monetary':round(float(normals['Monetary'].mean()),2),
                'avg_frequency':round(float(normals['Frequency'].mean()),2),
                'avg_order_value':round(float(normals['AvgOrderValue'].mean()),2),
            }



        }

        logger.info(
            f'anomaly profile: {len(anomalies):,} anomalies control'
            f'{summary["anomaly_revenue_share_pct"]}% of total revenue'
            f' (avg speed : {summary["anomalies"]["avg_monetary"]:,.2f} vs '
            f"{summary['normals']['avg_monetary']:,.2f}"
        )
        return summary

        
    def save(self,filepath:Optional[Union[Path,str]]=None)->None:
        target_path=(
            Path(filepath).resolve()
            if filepath else config.paths.isolation_forest_model_path
        )

        save_joblib(self.model,target_path)
        logger.info(f' saved isolation forest model to: {target_path}')

    def load(self,filepath: Optional[Union[Path,str]]=None)->'CustomerAnomalyDetector':
        target_path=(Path(filepath).resolve() if filepath else config.paths.isolation_forest_model_path)
        self.model=load_joblib(target_path)
        self.is_fitted=True
        logger.info(f'loaded isolation forest model from :{target_path}')
        return self