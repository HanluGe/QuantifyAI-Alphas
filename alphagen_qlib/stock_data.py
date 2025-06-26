import os
from typing import List, Union, Optional, Tuple
import torch
import pandas as pd
import numpy as np
from enum import IntEnum


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
class StockData:
    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        interval: str,
        data_path: str = "data/all",
        max_backtrack_days: int = 5 * 24,
        max_future_days: int = 1 * 24,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device('cpu')
    ):
        self._instrument = instrument if isinstance(instrument, list) else [instrument]
        self._start_time = pd.Timestamp(start_time)
        self._end_time = pd.Timestamp(end_time)
        self._interval = interval
        self.data_path = data_path
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.device = device
        self._features = features if features else list(FeatureType)

        self.data, self._dates, self._stock_ids = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        dfs = []
        for symbol in self._instrument:
            file_path = os.path.join(self.data_path, f"{symbol}.feather")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Feather file for symbol {symbol} not found at {file_path}")

            df = pd.read_feather(file_path)
            df = df.loc[self._start_time - pd.Timedelta(days=30):self._end_time + pd.Timedelta(days=30)]
            df = df.sort_index()
            df = df.rename(columns={'vol': 'volume'})  # 标准化列名

            feature_names = ['open', 'close', 'high', 'low', 'volume']
            df_feat = df[[feature_names[f] for f in self._features]].copy()
            df_feat.columns = [f.name.lower() for f in self._features]
            df_feat['symbol'] = symbol
            dfs.append(df_feat)

        df_all = pd.concat(dfs)
        df_all = df_all.reset_index().set_index(['datetime', 'symbol']).sort_index()

        pivoted = df_all.unstack(level=1)
        #pivoted = pivoted.iloc[self.max_backtrack_days:-self.max_future_days]
        dates = pivoted.index
        #data = data[max_backtrack_days : -max_future_days]
        stock_ids = pivoted.columns.levels[1]
        values = pivoted.values
        values = values.reshape(len(dates), len(self._features), len(stock_ids))
        
        #import pdb; pdb.set_trace()

        return torch.tensor(values, dtype=torch.float32, device=self.device), dates, stock_ids


    @property
    def n_features(self):
        return len(self._features)

    @property
    def n_stocks(self):
        return self.data.shape[-1]

    @property
    def n_days(self):
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(self, data: Union[torch.Tensor, List[torch.Tensor]], columns: Optional[List[str]] = None) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)

        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]

        n_days, n_stocks, n_columns = data.shape
        assert self.n_days == n_days, "day count mismatch"
        assert self.n_stocks == n_stocks, "stock count mismatch"
        assert len(columns) == n_columns, "column count mismatch"

        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]

        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
