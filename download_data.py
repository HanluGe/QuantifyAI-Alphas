import os
import shutil
import requests
import pandas as pd
import datetime
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from data_collection.qlib_dump_bin import DumpDataAll
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
        

class BinanceDataManager:
    _all_crypto: List[str]
    _fields = [
        "date", "open", "high", "low", "close", "preclose", "volume"
    ]
    
    def __init__(
        self, 
        interval: tuple, 
        start_date: str, 
        end_date: str, 
        save_path: str, 
        qlib_export_path: str,
        qlib_base_data_path: Optional[str],
        max_workers: int = 10,
    ):
        self._save_path = save_path
        self._export_path = f"{self._save_path}/export"
        os.makedirs(self._save_path, exist_ok=True)
        os.makedirs(self._export_path, exist_ok=True)
        self._qlib_export_path = os.path.expanduser(qlib_export_path)
        self._qlib_path = qlib_base_data_path
        if self._qlib_path is not None:
            self._qlib_path = os.path.expanduser(self._qlib_path)
        self._interval = interval
        self._start_date = start_date
        self._end_date = end_date
        self.url = 'https://api.binance.us/api/v3/'
        self._max_workers = max_workers

    @property
    def _crypto_list_path(self) -> str:
        return f"{self._save_path}/crypto_list.txt"

    def _load_all_crypto(self) -> None:
        """Load all cryptocurrencies and save them to a text file."""
        url = self.url + "exchangeInfo"
        response = requests.get(url)
        symbols_info = response.json().get("symbols", [])

        # Extract the symbol names and save them to the list
        self._all_crypto = [symbol['symbol'] for symbol in symbols_info if symbol['status'] == 'TRADING']

        # Save the crypto list to a text file
        with open(self._crypto_list_path, "w") as f:
            for crypto in self._all_crypto:
                f.write(crypto + '\n')
                
    def _parallel_foreach(
        self,
        callable,
        input: List[dict],
        max_workers: Optional[int] = None
    ) -> list:
        if max_workers is None:
            max_workers = self._max_workers
        with tqdm(total=len(input)) as pbar:
            results = []
            with ProcessPoolExecutor(max_workers) as executor:
                futures = [executor.submit(callable, **elem) for elem in input]
                for f in as_completed(futures):
                    results.append(f.result())
                    pbar.update(n=1)
            return results

    def _get_unix_timestamp(self, date_str: str) -> int:
        """Convert date string to Unix timestamp in milliseconds."""
        dt = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return int(dt.timestamp() * 1000)

    # def _download_klines_job(self, code:str) -> pd.DataFrame:
    #     """Fetch Kline data from Binance API."""
    #     kline_url = self.url+"klines"
    #     params = {
    #         'symbol': code,
    #         'interval': self._interval[0],
    #         'startTime': self._get_unix_timestamp(self._start_date),
    #         'endTime': self._get_unix_timestamp(self._end_date),
    #         'limit': 100000  # Maximum number of data points per request
    #     }
        
    #     response = requests.get(kline_url, params=params)
    #     data = response.json()
    #     df = self._result_to_data_frame(data)
    #     df['date'] = pd.to_datetime(df['timestamp'])
    #     df['preclose'] = df['close'].shift(1)
    #     df['pctChg'] = ((df['close'] - df['preclose']) / df['preclose']) * 100
    #     df['amount'] = df['volume'] * df['close']
        
    #     df = df[self._fields]
    #     df.set_index('date', inplace=True)
    #     df.to_pickle(f"{self._save_path}/k_data/{code}.pkl")
    #     #import pdb; pdb.set_trace()
    #     return response.json()
    
    def _download_klines_job(self, code: str) -> pd.DataFrame:
        """Fetch Kline data from Binance API with pagination to handle data limits."""
        
        kline_url = self.url + "klines"
        limit = 1000  # Maximum number of data points per request
        all_data = []  
        
        start_time = self._get_unix_timestamp(self._start_date)
        end_time = self._get_unix_timestamp(self._end_date)
        while start_time<end_time:
            params = {
                'symbol': code,
                'interval': self._interval[0],
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
            }
            
            response = requests.get(kline_url, params=params)
            data = response.json()
            
            if not data or len(data) == 0:
                break

            try:
                start_time = data[-1][0]+1
                all_data.extend(data)
            except:
                start_time = end_time 
        #import pdb;pdb.set_trace()
        if all_data:
            all_df = self._result_to_data_frame(all_data)
            all_df['date'] = pd.to_datetime(all_df['timestamp'])
            all_df['preclose'] = all_df['close'].shift(1)
            all_df['pctChg'] = ((all_df['close'] - all_df['preclose']) / all_df['preclose']) * 100
            all_df['amount'] = all_df['volume'] * all_df['close']
            
            all_df = all_df[self._fields]
            all_df.set_index('date', inplace=True)
            
            all_df.to_pickle(f"{self._save_path}/k_data/{code}.pkl")
        
        return all_data

    def _download_klines_data(self) -> None:
        print("Download crypto data")
        os.makedirs(f"{self._save_path}/k_data", exist_ok=True)
        #import pdb;pdb.set_trace()
        for code in tqdm(self._all_crypto):
            all_data = self._download_klines_job(code)
        # self._parallel_foreach(
        #     self._download_klines_job,
        #     [dict(code=code)
        #      for code in self._all_crypto]
        # )
        
    def _save_csv_job(self, path: Path) -> None:
        code = path.stem
        code = f"{code[:2].upper()}{code[-6:]}"
        df: pd.DataFrame = pd.read_pickle(path)
        df.rename(columns={"foreAdjustFactor": "factor"}, inplace=True)
        df["code"] = code
        out = Path(self._export_path) / f"{code}.csv"
        df.to_csv(out)
        
    def _save_csv(self) -> None:
        print("Export to csv")
        children = list(Path(f"{self._save_path}/k_data").iterdir())
        self._parallel_foreach(
            self._save_csv_job,
            [dict(path=path) for path in children]
        )
        
    @classmethod
    def _result_to_data_frame(cls, data: list) -> pd.DataFrame:
        # 创建 DataFrame，设置列名
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
            "ignore"
        ])
        # 转换时间戳为日期时间格式
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # 将数据类型转换为 float
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    
    def _dump_qlib_data(self) -> None:
        DumpDataAll(
            csv_path=self._export_path,
            qlib_dir=self._qlib_export_path,
            freq = self._interval[1],
            max_workers=self._max_workers,
            exclude_fields="date,code",
            symbol_field_name="code",
            interval=self._interval[1]
        ).dump()
        #self.generate_hourly_calendar()
        shutil.copy(f"{self._qlib_export_path}/calendars/{self._interval[1]}.txt",
            f"{self._qlib_export_path}/calendars/{self._interval[1]}_future.txt")
        self._fix_constituents()
        
        
    def _fix_constituents(self) -> None:
        today = str(datetime.date.today())
        path = f"{self._qlib_export_path}/instruments"

        for p in Path(path).iterdir():
            if p.stem == "all":
                continue
            df = pd.read_csv(p, sep='\t', header=None)
            df.sort_values([2, 1, 0], ascending=[False, False, True], inplace=True)     # type: ignore
            latest_data = df[2].max()
            df[2] = df[2].replace(latest_data, today)
            #import pdb;pdb.set_trace()
            df.to_csv(p, header=False, index=False, sep='\t')

    def fetch_and_save_data(self):
        """Fetch Kline data and save it to a CSV file."""
        self._load_all_crypto()
        self._download_klines_data()
        self._save_csv()
        self._dump_qlib_data()

    def _read_all_text(self, path: str) -> str:
        """Read all text from a file."""
        with open(path, "r") as f:
            return f.read()

def collect_data(interval, start_date,end_date,save_path):
    # Create an instance of the BinanceDataManager
    binance_dm = BinanceDataManager(
        interval=interval, 
        start_date=start_date, 
        end_date=end_date, 
        save_path=save_path,
        qlib_export_path=".qlib/qlib_data/crypto_data_rolling",
        qlib_base_data_path=".qlib/qlib_data/crypto_data",
    )
        
    # Fetch and save Kline data
    binance_dm.fetch_and_save_data()
    

if __name__ == "__main__":
    collect_data(
        interval = ("1h","60min"),  # 60 minute interval
        start_date = "2022-01-01 00:00:00",
        end_date = "2023-12-31 00:00:00",
        save_path = ".qlib/cpt_60min")
    
    # collect_data(
    #     interval = ("5m","5min"),  # 60 minute interval
    #     start_date = "2022-01-01 00:00:00",
    #     end_date = "2023-12-31 00:00:00",
    #     save_path = ".qlib/cpt_5min")
