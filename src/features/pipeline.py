
"""Feature transformation and scaling pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List,Optional,Tuple,Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler

from src.config import config
from src.features.transformers import PercentileClipper
from src.utils.logger import logger
from src.utils.helpers import save_joblib,load_joblib,save_numpy , timeit


def get_model_features()->List[str]:
    return list(config.features.model_features)

def build_feature_pipeline()->Pipeline:
    lower_q=config.features.clipping_lower_quantile
    upper_q=config.features.clipping_upper_quantile

    pipeline=Pipeline(
        steps=[
            ('log1p',
            FunctionTransformer(np.log1p,feature_names_out='one-to-one',),),
            (
                'percentile_clipper',
                PercentileClipper(
                    lower_quantile=lower_q,
                    upper_quantile=upper_q,
                ),

            ),
            ('standard_scaler',StandardScaler(),),

        ]
    )

    logger.info(
        f' built feature pipeline: [log1p -> PercentileClipper({lower_q},{upper_q})-> StandardScaler]'

    )
    return pipeline


@timeit
def fit_transform_features(
    df:pd.DataFrame,
    save_artifacts: bool =True,
)-> Tuple[np.ndarray,Pipeline]:
    feature_cols=get_model_features()
    missing_cols=[c for c in feature_cols if c not in df.columns]
    if missing_cols:
        err_msg=f'Missing required model features in DataFrame: {missing_cols}'
        logger.error(err_msg)
        raise KeyError(err_msg)

    logger.info(
        f' Extracting {len(feature_cols)} features for {len(df):,} customer: {feature_cols}'
    )
    x=df[feature_cols].copy()

    # built and fit pipeline
    pipeline=build_feature_pipeline()
    x_scaled=pipeline.fit_transform(x)

    logger.info(
        f'feature transaction complete: scaled shape={x_scaled.shape},'
        f' mean={np.mean(x_scaled):.4f}, std={np.std(x_scaled):.4f}'
    )

    if save_artifacts:
        save_joblib(pipeline,config.paths.preprocessor_path)

        save_numpy(x_scaled,config.paths.processed_features_npy)
        feat_txt_path=Path(config.paths.model_features_txt).resolve()
        feat_txt_path.parent.mkdir(parents=True,exist_ok=True)
        with open(feat_txt_path,'w',encoding='utf-8') as f:
            for feat in feature_cols:
                f.write(f'{feat}\n')

        logger.info(f'saved model feature list to {feat_txt_path}')

    return x_scaled,pipeline


def transform_features(df:pd.DataFrame,pipeline:Optional[Pipeline]=None,)-> np.ndarray:

    if pipeline is None:
        pipeline=load_joblib(config.paths.preprocessor_path)

    feature_cols=get_model_features()
    missing_cols=[c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f' input data is missing features: {missing_cols}')

    x=df[feature_cols].copy()
    x_scaled=pipeline.transform(x)
    return x_scaled

    