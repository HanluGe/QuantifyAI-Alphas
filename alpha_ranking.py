from math import isnan

import pandas as pd
from alphagen.trade.base import StockPosition, StockStatus
from alphagen_qlib.calculator import QLibStockDataCalculator

from alphagen_qlib.strategy import TopKSwapNStrategy
from alphagen_qlib.utils import load_alpha_pool_by_path, load_data
from alphagen_generic.features import *
import os
import json


def run_alpha_ranking(
    interval="60min",
    start_date="2016-01-04 08:30:00",
    end_date="2022-12-30 15:00:00",
    instrument="all_60min",
    threshold=0.7,
    pool_file_suffix='22528_steps_pool.json',
    file_duffix='new_all_60min_20_0_20241003172838',
    max_future_days=24,
    offset=1
):

    pool_path = f'./result/checkpoints_benchmark/{file_duffix}/{pool_file_suffix}'
    alpha_path = f'./result/output_alpha/benchmark/{instrument}'
    return_path = f'./result/return/benchmark/{instrument}'

    # load data
    data, latest_date = load_data(
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        offset=offset,
        max_future_days=max_future_days,
        interval=interval
    )

    calculator = QLibStockDataCalculator(data=data, target=target)

    exprs, weights = load_alpha_pool_by_path(pool_path)

    if not os.path.exists(alpha_path):
        os.makedirs(alpha_path)
        
    if not os.path.exists(return_path):
        os.makedirs(return_path)

    alpha_dict = {}
    unique_alpha_exprs = []
    final_alpha_tensor = None

    n = len(exprs)
    ranked_alphas = []
    for i in range(n):
        ic_score = calculator.calc_single_IC_ret(exprs[i])
        ranked_alphas.append((ic_score, exprs[i], weights[i], i))
    
    ranked_alphas.sort(key=lambda x: abs(x[0]), reverse=True)

    for rank, (ic_score, expr, weight, index) in enumerate(ranked_alphas):
        is_unique = True
        for unique_alpha_expr in unique_alpha_exprs:
            mutual_ic = calculator.calc_mutual_IC(expr, unique_alpha_expr)
            if abs(mutual_ic) > threshold:
                is_unique = False
                break

        if is_unique:
            unique_alpha_exprs.append(expr)
            alpha_name = f"alpha{len(unique_alpha_exprs)}"
            alpha_dict[alpha_name] = (str(expr), weight, ic_score)
            calculator.store_alpha(expr, weight, alpha_path, alpha_name)
            
    calculator.store_alpha(target, 1, return_path, 'return')
    
    alpha_dict_path = f"{alpha_path}/alpha_dict.json"
    
    with open(alpha_dict_path, 'w') as f:
        json.dump(alpha_dict, f, indent=4)

    print(f"Alpha dictionary saved at {alpha_dict_path}")
    
    return alpha_dict

if __name__ == '__main__':
    # run_alpha_ranking(interval="60min",
    #                   instrument="all_60min",
    #                   file_duffix='new_all_60min_20_0_20241003172838',
    #                   pool_file_suffix='22528_steps_pool.json')
    
    run_alpha_ranking(interval="60min",
                      instrument=['CL', 'NG', 'HO', 'RB'],
                      file_duffix="new_['CL', 'NG', 'HO', 'RB']_80_0_20250621150612",
                      pool_file_suffix='112640_steps_pool.json')


