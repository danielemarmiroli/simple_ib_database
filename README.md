# Interactive Brokers Data Management System

A comprehensive system for retrieving, storing, and exploring market data from Interactive Brokers.

## Overview

This project provides tools to download historical market data from Interactive Brokers (IB) at various timeframes and efficiently store it in a structured format. It also includes utilities for discovering available contracts across different exchanges and a web interface for exploring market data.

## Features

- **Historical Data Retrieval**: Download stock data at multiple timeframes (5sec, 1min, 1hour, 1day)
- **Efficient Storage**: Store market data in Parquet format, organized by timeframe
- **Contract Discovery**: Search and explore available contracts on various exchanges
- **Data Visualization**: Interactive OHLC charts with volume indicators
- **Web Interface**: Browse and manage contracts through a multi-page Streamlit application

![Diagram](/docs/img/find_contract.png)
![Diagram](/docs/img/data_download.png)
![Diagram](/docs/img/inspect_data.png)


## Components

- **Web Application** (`src/app.py`)
  - Multi-page Streamlit interface with "Contracts" and "Market Data" pages
  - Contract search, filtering, and management functionality
  - Historical data downloading with chunking for large requests
  - Interactive data visualization with Plotly charts
  - Connection management for Interactive Brokers TWS/Gateway

- **Core Modules** (in `src/` directory)
  - `ib_data_handlers.py`: Contains two main classes:
    - `IBDataDownloader`: For downloading and managing market data
    - `ExchangeContractLister`: For discovering available contracts
  - `market_data_downloader.py`: Script for downloading historical data in chunks
  - `match_contracts.py`: Script for searching contracts by pattern/symbol

- **Data Storage**
  - `market_data/`: Organized by timeframe (5sec, 1min, 1hour, 1day)
  - `contract_data/`: For storing saved contract lists
  - `logs_dir/`: For application logs

## Requirements

- Interactive Brokers TWS or IB Gateway
- Python 3.7+
- Required libraries: ib_insync, pandas, numpy, pytz, streamlit, plotly

## Setup

1. Install Interactive Brokers TWS or IB Gateway
2. Enable API access in TWS/Gateway settings
3. Install required Python libraries:
   ```
   pip install ib_insync pandas numpy pytz streamlit plotly nest-asyncio
   ```

## Usage

### Running the Web Application

The easiest way to use the system is through the web interface:

```bash
streamlit run src/app.py
```

This launches a multi-page application that allows you to:
1. Search and save contract definitions
2. Download historical market data at various timeframes
3. Visualize and explore the downloaded data

### Programmatic Usage

#### Downloading Market Data

```python
from src.ib_data_handlers import IBDataDownloader

# Initialize the downloader
downloader = IBDataDownloader(
    host='127.0.0.1',  # Use 'host.docker.internal' in Docker
    port=4001,         # 7497 for TWS Paper, 7496 for TWS Live, 4002 for Gateway Paper, 4001 for Gateway Live
    client_id=1,
    data_directory='./market_data'
)

# Connect to IB
downloader.connect()

# Download 1-minute data for the last 5 days for AAPL
downloader.download_and_save(
    symbol='AAPL',
    frequency='1min',
    days=5
)

# Download for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
results = downloader.download_multiple_symbols(
    symbols=symbols,
    frequency='1hour',
    days=30
)

# Disconnect when done
downloader.disconnect()
```

#### Exploring Contracts

```python
from src.ib_data_handlers import ExchangeContractLister

# Initialize the contract lister
lister = ExchangeContractLister(
    host='127.0.0.1',
    port=4001,
    client_id=2,
    output_dir='./contract_data'
)

# Connect to IB
lister.connect()

# Get stocks from NASDAQ
nasdaq_stocks = lister.list_exchange_stocks('NASDAQ')

# Search with specific patterns
contracts = lister.search_contracts_by_exchange(
    exchange='NYSE', 
    sec_type='STK',
    patterns=['IBM*', 'MSFT*']
)

# Disconnect when done
lister.disconnect()
```

#### Using the Market Data Downloader Script

For efficient downloading of large historical datasets:

```bash
python src/market_data_downloader.py --symbol AAPL --time-value 30 --time-unit days --data-type 1hour --host 127.0.0.1 --port 4001
```

The script automatically chunks large requests to avoid IB API limitations.

## Notes

- Ensure TWS/Gateway is running before connecting
- Be aware of IB's market data subscription requirements
- Throttle requests to avoid overwhelming the IB server
- For large data downloads, the system automatically splits requests into chunks
