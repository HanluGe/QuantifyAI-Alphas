from typing import List, Optional, Tuple
from torch import Tensor
import torch
from alphagen.data.calculator import AlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
#from alphagen_qlib.stock_data import StockData
from alphagen_qlib.stock_data import StockData
import os
import pandas as pd
import json

class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def _calc_alpha(self, expr: Expression) -> Tensor:
        #import pdb;pdb.set_trace()
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    # def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
    #     n = len(exprs)
    #     factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
    #     return sum(factors)  # type: ignore

    def make_ensemble_alpha(self, exprs: List[Expression], weights: List[float]) -> Tensor:
        n = len(exprs)
        if n == 0:
            # 返回一个和 target_value 相同 shape 的 0 tensor
            return torch.zeros_like(self.target_value)
    
        factors: List[Tensor] = [self._calc_alpha(exprs[i]) * weights[i] for i in range(n)]
        return sum(factors)  # type: ignore

    
    def store_alpha(self, expr: Expression, weight: float, alpha_path: str, alpha_name: str) -> torch.Tensor:
        if not os.path.exists(alpha_path):
            os.makedirs(alpha_path)

        alpha_value = self._calc_alpha(expr) #* weight
        file_path = os.path.join(alpha_path, alpha_name+'.csv')
        #alpha_df = pd.DataFrame(alpha_value.cpu().numpy()) 
        if self.data.max_future_days == 0:
            date_index = self.data._dates[self.data.max_backtrack_days:]
        else:
            date_index = self.data._dates[self.data.max_backtrack_days:-self.data.max_future_days]
            
        date_range = date_index[
            (date_index >= self.data._start_time) & (date_index <= self.data._end_time)
        ]
        #import pdb; pdb.set_trace()
        print(expr)
        #print("date_index length:", len(date_range))
        #print(self.data._dates)
        #print("stock_ids length:", len(self.data._stock_ids))
        #print("expected total rows:", len(date_index) * len(self.data._stock_ids))

        alpha_df = pd.DataFrame(
            alpha_value.cpu().numpy(),
            columns=[stock_id for stock_id in self.data._stock_ids],  
            index=[date_time for date_time in date_range]
        )
        #import pdb; pdb.set_trace()
        alpha_df.rename_axis('datetime', inplace=True) 
        alpha_df.to_csv(file_path, index=True)
    
    # def store_alpha(self, expr: Expression, weight: float, alpha_path: str, alpha_name: str) -> torch.Tensor:
    #     """
    #     存储 alpha 表达式评估结果为 csv 文件，格式为 MultiIndex (datetime, symbol)
    #     """
    #     if not os.path.exists(alpha_path):
    #         os.makedirs(alpha_path)

    #     # 计算 alpha 表达式对应的值 (n_days, n_stocks)
    #     alpha_value = self._calc_alpha(expr)  # shape: (n_days, n_stocks)
        
    #     # 获取时间范围和股票列表
    #     date_range = self.data._dates[
    #         (self.data._dates >= self.data._start_time) & (self.data._dates <= self.data._end_time)
    #     ]
    #     stock_ids = self.data._stock_ids

    #     # 检查形状是否匹配
    #     expected_shape = (len(date_range), len(stock_ids))
    #     if alpha_value.shape != expected_shape:
    #         raise ValueError(f"Shape mismatch: alpha_value shape {alpha_value.shape} != expected {expected_shape}")

    #     # 构建 MultiIndex：日期 × 股票
    #     multi_index = pd.MultiIndex.from_product([date_range, stock_ids], names=["datetime", "symbol"])

    #     # 拉平成一维数组，创建 DataFrame
    #     alpha_array = alpha_value.cpu().numpy().reshape(-1)
    #     alpha_df = pd.DataFrame({"alpha": alpha_array}, index=multi_index)

    #     # 保存为 CSV
    #     file_path = os.path.join(alpha_path, alpha_name + ".csv")
    #     alpha_df.to_csv(file_path)

    #     print(f"[Saved] Alpha '{alpha_name}' saved to {file_path}")
    #     return alpha_value


    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        #import pdb; pdb.set_trace()
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:

        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            #print("ensemble_value:", type(ensemble_value), ensemble_value)

            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)
