"""Market Basket Analysis using FP-Growth and Apriori association rules."""
from __future__ import annotations

from pathlib import Path
from typing import Any,Dict,List,Optional,Set,Union
import pandas as pd
from mlxtend.frequent_patterns import association_rules,fpgrowth
from src.config import config
from src.utils.logger import logger
from src.utils.helpers import timeit

def build_product_lookup(df:pd.DataFrame)->Dict[str,str]:
    valid_products=df.dropna(subset=['StockCode','Description']).copy()
    valid_products['StockCode']=valid_products['StockCode'].astype(str).str.strip()
    valid_products['Description']=valid_products['Description'].astype(str).str.strip()

    lookup=(
        valid_products.drop_duplicates(subset=['StockCode']).set_index('StockCode')['Description']
        .to_dict()
    )
    return lookup

def build_basket_matrix(
    df:pd.DataFrame,
    top_n_products:Optional[int]=None
)->pd.DataFrame:
    n_top=(top_n_products if top_n_products is not None else config.market_basket.top_n_products)

    logger.info(f'filtering to top {n_top} most frequent items for basket analysis')
    top_items=df['StockCode'].astype(str).value_counts().head(n_top).index.tolist()
    df_top=df[df['StockCode'].astype(str).isin(top_items)].copy()

    basket=(df_top.groupby(['InvoiceNo','StockCode'])['Quantity'].sum().unstack(fill_value=0))
    basket=basket>0

    # filter out single -item baskets 
    basket=basket[basket.sum(axis=1)>=2]

    sparsity=float((~basket).mean().mean()*100)
    logger.info(
        f'build transaction matrix: {basket.shape[0]:,} multi-item invoices x'
        f'{basket.shape[1]} products (sparsity: {sparsity:.1f})'

    )

    return basket


@timeit
def run_market_basket_analysis(
    df:pd.DataFrame,save_csv:bool=True,output_path:Optional[Union[Path,str]]=None,
)->pd.DataFrame:
    min_sup=config.market_basket.min_support
    min_conf=config.market_basket.min_confidence
    min_lift = config.market_basket.min_lift

    product_lookup= build_product_lookup(df)
    basket=build_basket_matrix(df)

    if basket.empty or basket.shape[0]<5:
        logger.warning('transaction matrix too small to extract association rules')
        return pd.DataFrame()

    logger.info(
        f'mining frequent itemsets with fb-growth (min_support={min_sup})'

    )

    frequent_itemsets=fpgrowth(basket,min_support=min_sup,use_colnames=True)
    logger.info(f'discoverd {len(frequent_itemsets):,} frequent itemsets.')

    if frequent_itemsets.empty:
        logger.warning('no frequent itemset found with currect min_support threshold')
        return pd.DataFrame()

    logger.info(f'generating association rules (min_confidence={min_conf},min_lift={min_lift})')

    rules=association_rules(
        frequent_itemsets,
        metric='confidence',
        min_threshold=min_conf,

    )

    rules=rules[rules['lift'] >= min_lift].copy()

    if rules.empty:
        logger.warning('no rules met the confidence and lift thresholds')
        return rules

    # format antecedents and consequents into readable strings and names
    rules['antecedents_codes'] = rules['antecedents'].apply(lambda s: ", ".join([str(c) for c in s]))
    rules['consequents_codes'] = rules['consequents'].apply(lambda s: ", ".join([str(c) for c in s]))

    rules['antecedents_names'] = rules['antecedents'].apply(
        lambda s: " + ".join([product_lookup.get(str(code), str(code)) for code in s])
    )
    rules['consequents_names'] = rules['consequents'].apply(
        lambda s: " + ".join([product_lookup.get(str(code), str(code)) for code in s])
    )
 
    # round metric columns
    metric_cols=['support','confidence','lift','leverage','conviction']
    for col in metric_cols:
        if col in rules.columns:
            rules[col]=rules[col].round(4)
    rules=rules.sort_values(by='lift',ascending=False).reset_index(drop=True)
    logger.info(f'discoverd {len(rules):,} high-confidence association rules')

    if save_csv:
        target_path=(
            Path(output_path).resolve()
            if output_path
            else config.paths.association_rules_csv
        )

        target_path.parent.mkdir(parents=True,exist_ok=True)

        # export clean columns
        export_cols=[
            'antecedents_codes',
            'consequents_codes',
            'antecedents_names',
            'consequents_names',
            'support',
            'confidence',
            'lift',

        ]

        available_cols=[c for c in export_cols if c in rules.columns]
        rules[available_cols].to_csv(target_path,index=False)
        logger.info(f'saved association rules csv to: {target_path}')

    return rules

def recommend_products(purchased_items: List[str], rules_df: pd.DataFrame, top_k: int = 3) -> List[Dict[str, Any]]:
    if rules_df.empty:
        return []

    cart_set={str(item).strip().upper() for item in purchased_items}
    matched_recs=[]

    for _,row in rules_df.iterrows():
        ant_codes={c.strip().upper() for c in str(row.get('antecedents_codes',"")).split(',')}
        ant_names={n.strip().upper() for n in str (row.get('antecedents_names',"")).split("+")}

        if ant_codes.intersection(cart_set) or ant_names.intersection(cart_set):
            matched_recs.append(
                {
                    'recommended_product':row.get('consequents_names',""),
                    'product_code':row.get('consequents_codes',""),
                    'confidence': float(row.get('confidence',0.0)),
                    'lift':float(row.get('lift',1.0)),

                }
            )

    matched_recs=sorted(matched_recs,key=lambda x: x ['lift'],reverse=True)[:top_k]
    return matched_recs

        