
# AI-Driven Alpha Generation for Cryptocurrencies

This repository extends the work from [RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen) to enable the application of AI-driven alpha generation methods in the cryptocurrency market.

## Overview

The original `alphagen` project focuses on using reinforcement learning and machine learning techniques to generate alphas for traditional financial markets. This repository adapts those methodologies for the unique challenges and opportunities of cryptocurrency trading.

Key modifications include:
- Support for cryptocurrency-specific data formats and APIs (e.g., Binance, Coinbase).
- Customized features tailored for high-frequency and volatile market conditions in crypto.
- New pre-processing and model enhancements optimized for crypto market behaviors.

## Features

- **Alpha Generation**: Automatically generate alphas using AI models tailored for cryptocurrencies.
- **Data Pipeline**: Includes data extraction, cleaning, and feature engineering for crypto datasets.
- **Reinforcement Learning**: Uses reinforcement learning to optimize alpha signals for crypto trading.
- **Backtesting Support**: Evaluate generated alphas using historical crypto data.
- **Customizable**: Easily adapt to different cryptocurrency exchanges and data sources.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/crypto-alphagen.git
   cd crypto-alphagen
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have access to the original `alphagen` repository:
   ```bash
   git clone https://github.com/RL-MLDM/alphagen.git
   ```

4. Follow the instructions in the original `alphagen` repository for setting up the base environment.

## Usage

1. Preprocess cryptocurrency data:
   ```bash
   python preprocess.py --exchange binance --symbols BTC,ETH --start 2020-01-01 --end 2023-01-01
   ```

2. Train alpha models:
   ```bash
   python train.py --config config/crypto_alpha.json
   ```

3. Generate alpha signals:
   ```bash
   python generate_alpha.py --model trained_model.pth --data crypto_data.csv
   ```

4. Backtest generated alphas:
   ```bash
   python backtest.py --alpha generated_alpha.csv --capital 10000
   ```

## Contributions

This project builds upon the foundational work of [RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen). Significant effort has been made to adapt and enhance the original code for cryptocurrency markets.

If you find this project useful or have suggestions for further improvement, feel free to submit a pull request or open an issue.

## License

This project follows the licensing terms of the original [RL-MLDM/alphagen](https://github.com/RL-MLDM/alphagen) repository. Please refer to their [LICENSE](https://github.com/RL-MLDM/alphagen/blob/main/LICENSE) for details.

## Acknowledgments

Special thanks to the authors of the original [alphagen](https://github.com/RL-MLDM/alphagen) project for providing a robust framework for alpha generation.

---
