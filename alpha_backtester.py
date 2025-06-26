import os
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np


def load_alpha(alpha_path: str, alpha_name: str) -> torch.Tensor:
    file_path = os.path.join(alpha_path, alpha_name + '.csv')
    try:
        alpha_df = pd.read_csv(file_path)
        alpha_df = alpha_df.set_index("datetime")
        alpha_tensor = torch.tensor(alpha_df.values, dtype=torch.float32)
        return alpha_tensor
    except Exception as e:
        print(f"Error loading {alpha_name}: {e}")
        return None  # Return None if loading fails


def ic_and_icir_calculation(alpha_tensor: torch.Tensor, target: torch.Tensor) -> dict:
    if alpha_tensor.shape != target.shape:
        raise ValueError("Alpha and target tensors must have the same shape (time x stocks)")

    ic_values = []
    #import pdb;pdb.set_trace()

    for t in range(alpha_tensor.shape[0]):
        alpha = alpha_tensor[t, :]
        target_values = target[t, :]

        if torch.std(alpha) == 0 or torch.std(target_values) == 0:
            continue

        stacked_data = torch.stack((alpha, target_values))
        
        cov = torch.cov(stacked_data)

        ic_value = cov[0, 1] / (torch.std(alpha) * torch.std(target_values))
        ic_values.append(ic_value.item())

    if len(ic_values) == 0:
        #raise ValueError("No valid IC values could be calculated")
        return {
        "IC": None,
        "ICIR": None
    }

    average_ic = torch.mean(torch.tensor(ic_values)).item()
    std_ic = torch.std(torch.tensor(ic_values)).item()

    if std_ic == 0:
        raise ValueError("Standard deviation of IC values is zero, ICIR cannot be calculated")

    icir_value = average_ic / std_ic

    return {
        "IC": average_ic,
        "ICIR": icir_value
    }


def hierarchical_backtest(alpha_tensor: torch.Tensor, target: torch.Tensor, ic: int, num_layers: int = 5) -> dict:
    if alpha_tensor.shape != target.shape:
        raise ValueError("Alpha and target returns must have the same shape (time x stock)")
    
    long_short_returns = [] 
    turnover = []
    previous_long_indices = set()
    
    for t in range(alpha_tensor.shape[0]):
        alpha = alpha_tensor[t, :]
        target_values = target[t, :]

        if ic is None:
            return {
                        "long_short_return": 0,  
                        "long_short_sharpe": 0,
                        "long_short_max_drawdown": 0,
                        "long_avg_turnover": 0
                    }
            
        if ic>0:
            sorted_indices = torch.argsort(alpha,descending=True)
        else:
            sorted_indices = torch.argsort(alpha)
            
        layer_size = alpha.shape[0] // num_layers
        
        long_indices = sorted_indices[:int(layer_size)] 
        long_returns = torch.mean(target_values[long_indices]) 

        short_indices = sorted_indices[-int(layer_size):]  
        short_returns = torch.mean(target_values[short_indices]) 

        long_short_return = long_returns - short_returns
        long_short_returns.append(long_short_return)
        
        current_long_indices = set(long_indices.tolist())
        if previous_long_indices: 
            turnover_value = len(previous_long_indices.symmetric_difference(current_long_indices)) / len(previous_long_indices)
            turnover.append(turnover_value)
        
        previous_long_indices = current_long_indices

    long_short_returns = torch.stack(long_short_returns)

    sharpe_ratio = long_short_sharpe(long_short_returns)
    max_down = max_drawdown(long_short_returns)
    avg_turnover = torch.mean(torch.tensor(turnover)) if turnover else torch.tensor(0.0)
    
    return {
        "long_short_return": torch.mean(long_short_returns).item(),  
        "long_short_sharpe": sharpe_ratio,
        "long_short_max_drawdown": max_down,
        "long_avg_turnover": avg_turnover.item()
    }


def long_short_sharpe(returns: torch.Tensor) -> float:
    mean_return = torch.mean(returns)
    return_std = torch.std(returns)
    
    if return_std == 0:
        return 0.0  

    sharpe_ratio = mean_return / return_std
    return sharpe_ratio.item()


def max_drawdown(returns: torch.Tensor) -> float:
    cumulative_returns = torch.cumprod(1+returns, dim=0) -1
    running_max = np.maximum.accumulate(cumulative_returns)
    max_drawdown_rate = ((cumulative_returns - running_max).nan_to_num(0)/running_max).min().item()
    #import pdb;pdb.set_trace()
    return round(max_drawdown_rate, 2)


def plot_heatmap(alpha_performance: dict, output_path: str, interval: str):
    heatmap_data = pd.DataFrame(alpha_performance).T

    sorted_index = sorted(heatmap_data.index, key=lambda x: int(x[5:]))  
    heatmap_data = heatmap_data.loc[sorted_index]

    norm_data = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_data, annot=heatmap_data, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5)
    
    plt.title('Alpha Performance (IC and ICIR Scores)')
    plt.xlabel('Metrics')
    plt.ylabel('Alphas')
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, f'alpha_heatmap{interval}.png'))
    plt.show()
    

def run_alpha_backtester(
    interval="5min",
):
    alpha_path = 'result/output_alpha/'+interval  # Path where alphas are stored
    output_path = 'result/output_result/'+interval  # Path to save the heatmap
    return_path = 'result/return/'+interval+'/return.csv'

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
        
    # Load target returns
    try:
        target_df = pd.read_csv(return_path)
        target_df = target_df.set_index("datetime")
        target = torch.tensor(target_df.values, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading return data: {e}")
        return
    
    alpha_files = [f for f in os.listdir(alpha_path) if f.endswith(".csv")]

    alpha_performance = {}
    for alpha_file in alpha_files:
        alpha_name = alpha_file.split('.')[0]
        alpha_value = load_alpha(alpha_path, alpha_name)
        if alpha_value is None:
            continue  # Skip if loading failed
        alpha_performance[alpha_name] = ic_and_icir_calculation(alpha_value, target)
        alpha_performance[alpha_name].update(hierarchical_backtest(alpha_value, target, alpha_performance[alpha_name]['IC'], num_layers=5))

    plot_heatmap(alpha_performance, output_path,interval)

if __name__ == '__main__':
    run_alpha_backtester(interval="60min")
