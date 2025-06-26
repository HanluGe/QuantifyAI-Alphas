import datetime
import json
from typing import List, Tuple
from alphagen.data.expression import *
from alphagen_generic.features import *

from alphagen_qlib.stock_data import StockData


def load_data(instrument: str,
                     start_date: str,
                     end_date: str,
                     max_future_days: int,
                     interval: str,
                     offset: int = 1,
                     **kwargs) -> Tuple[StockData, str]:
    
    # start_date = str(today - datetime.timedelta(days=window_size))
    # end_date = str(today - datetime.timedelta(days=offset))

    return StockData(instrument=instrument,
                     start_time=start_date,
                     end_time=end_date,
                     max_future_days=max_future_days,
                     interval=interval,
                     **kwargs), end_date


def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    exprs = [eval(expr_raw.replace('$open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    weights = raw['weights']
    return exprs, weights


def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
        return load_alpha_pool(raw)
