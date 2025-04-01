import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pytz
import time
import logging
import asyncio
import nest_asyncio
# Fix asyncio event loop issue
nest_asyncio.apply()
# Set the event loop policy to prevent "no current event loop" error
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from ib_insync import IB, Contract, BarData, util
from ib_insync import ContractDetails


class ExchangeContractLister:
    """
    A class to discover available contracts on a given exchange using Interactive Brokers.
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, output_dir='./contract_data'):
        """
        Initialize the ExchangeContractLister class.
        
        Parameters:
        -----------
        host : str
            The hostname or IP address of the IB Gateway/TWS
        port : int
            The port number of the IB Gateway/TWS
        client_id : int
            The client ID to identify this connection
        output_dir : str
            Directory to save output files
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _setup_logger(self):
        """Set up logging"""
        logger = logging.getLogger('ExchangeContractLister')
        logger.setLevel(logging.INFO)
        
        # Create console handler if not already added
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(ch)
        
        return logger
    
    def connect(self):
        """Connect to Interactive Brokers"""
        try:
            # Set a larger timeout for all requests
            self.ib.RequestTimeout = 30  # 30 seconds timeout
            
            # Connect with read-only API to avoid permission issues
            self.ib.connect(
                self.host, 
                self.port, 
                clientId=self.client_id,
                readonly=True  # Use read-only mode for data queries
            )
            
            # Wait a moment for the connection to stabilize
            time.sleep(1)
            
            self.logger.info(f"Connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IB")
    
    def get_exchange_list(self):
        """
        Get a list of available exchanges.
        
        Returns:
        --------
        list
            List of exchange codes
        """
        # This is a static list of common exchanges available in IB
        # IB doesn't provide a direct API to get all exchanges
        exchanges = [
            'SMART', 'AMEX', 'NYSE', 'CBOE', 'PHLX', 'ISE', 'CHX', 'ARCA', 'ISLAND',
            'DRCTEDGE', 'BEX', 'BATS', 'EDGEA', 'CBOT', 'CME', 'NYMEX', 'ICEUS',
            'NYSELIFFE', 'NASDAQ', 'TSE', 'VENTURE', 'VSE', 'SBF', 'FTA',
            'IBIS', 'LSE', 'LSEETF', 'MILAN', 'EUREX', 'STOCKH', 'HKFE',
            'SGX', 'KOSPI', 'OSE', 'HKSE', 'ASX', 'CHIX', 'SEHK', 'JSE',
            'MEXDER', 'BVME', 'DME', 'MONEP'
        ]
        return exchanges
    
    def match_symbol_by_pattern(self, exchange, pattern, sec_type='STK', max_results=100, timeout=30):
        """
        Match symbols on an exchange by pattern. Adapted for Streamlit compatibility.
        
        Parameters:
        -----------
        exchange : str
            The exchange code
        pattern : str
            Pattern to match symbols (can use wildcards like *)
        sec_type : str
            Security type ('STK', 'FUT', 'OPT', etc.)
        max_results : int
            Maximum number of results to return
        timeout : int
            Timeout in seconds for the matching symbols request
            
        Returns:
        --------
        list
            List of contract details matching the pattern
        """
        if not self.ib.isConnected():
            self.connect()
            
        # Store original timeout
        original_timeout = self.ib.RequestTimeout
        
        try:
            self.logger.info(f"Searching for {sec_type} contracts on {exchange} matching '{pattern}'")
            
            # Set longer timeout for this specific request
            self.ib.RequestTimeout = timeout
            
            # Create a dedicated run function that will execute in the current event loop
            def run_in_loop():
                # Create a new event loop specifically for this task
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Try direct contract creation approach - often more reliable in Streamlit
                    if not pattern.endswith('*'):
                        # If not a wildcard pattern, try direct contract
                        contract = Contract(symbol=pattern, secType=sec_type, exchange=exchange)
                        details = self.ib.reqContractDetails(contract)
                        
                        if details:
                            # Convert contract details to symbol match format
                            results = []
                            for detail in details:
                                class SymbolMatch:
                                    pass
                                    
                                match = SymbolMatch()
                                match.symbol = detail.contract.symbol
                                match.description = getattr(detail, 'longName', '')
                                match.derivativeSecTypes = [detail.contract.secType]
                                match.exchange = detail.contract.exchange
                                results.append(match)
                                
                            return results
                    
                    # If we get here, either it was a wildcard or direct contract didn't work
                    # Try searchContracts which is well-supported in Streamlit
                    search_term = pattern.rstrip('*')
                    try:
                        contracts = self.ib.searchContracts(search_term, exchange)
                        
                        # Filter by security type if specified
                        if sec_type != 'STK':
                            contracts = [c for c in contracts if c.secType == sec_type]
                        
                        # Convert to format similar to matching symbols for consistency
                        results = []
                        for contract in contracts:
                            class SymbolMatch:
                                pass
                                
                            match = SymbolMatch()
                            match.symbol = contract.symbol
                            match.description = getattr(contract, 'description', '')
                            match.derivativeSecTypes = [contract.secType]
                            match.exchange = contract.exchange if hasattr(contract, 'exchange') else exchange
                            results.append(match)
                            
                        return results
                            
                    except Exception as search_err:
                        self.logger.warning(f"searchContracts failed: {search_err}")
                    
                    # Last resort: try reqMatchingSymbols but use run_until_complete to prevent timeout issues
                    try:
                        # Create a coroutine and run it to completion in our controlled event loop
                        coro = self.ib.reqMatchingSymbolsAsync(pattern)
                        matching_symbols = loop.run_until_complete(coro)
                        
                        # Filter by exchange and security type
                        filtered = []
                        for symbol in matching_symbols:
                            if hasattr(symbol, 'derivativeSecTypes') and symbol.derivativeSecTypes:
                                if sec_type in symbol.derivativeSecTypes or 'STK' in symbol.derivativeSecTypes:
                                    if exchange == 'SMART' or symbol.exchange == exchange:
                                        filtered.append(symbol)
                        
                        return filtered
                        
                    except Exception as req_err:
                        self.logger.error(f"reqMatchingSymbolsAsync failed: {req_err}")
                        return []
                        
                finally:
                    # Clean up the event loop
                    loop.close()
            
            # Execute our function
            matching_symbols = run_in_loop()
            
            # Reset the timeout
            self.ib.RequestTimeout = original_timeout
            
            self.logger.info(f"Found {len(matching_symbols)} matches")
            
            # Limit results
            return matching_symbols[:max_results]
            
        except Exception as e:
            # Reset the timeout in case of error
            self.ib.RequestTimeout = original_timeout
            self.logger.error(f"Error matching symbols on {exchange}: {e}")
            return []
    
    def search_contracts_by_exchange(self, exchange, sec_type='STK', patterns=None):
        """
        Search for contracts on a specific exchange.
        
        Parameters:
        -----------
        exchange : str
            The exchange code
        sec_type : str
            Security type ('STK', 'FUT', 'OPT', etc.)
            
        Returns:
        --------
        list
            List of contract details
        """
        # For exchanges like stock exchanges, we can search by common letters
        if patterns is None:
            patterns = ['A*', 'B*', 'C*', 'D*', 'E*', 'F*', 'G*', 'H*', 'I*', 'J*',
                       'K*', 'L*', 'M*', 'N*', 'O*', 'P*', 'Q*', 'R*', 'S*', 'T*',
                       'U*', 'V*', 'W*', 'X*', 'Y*', 'Z*']
        elif isinstance(patterns, str):
            patterns = [patterns]
        
        all_results = []
        for pattern in patterns:
            self.logger.info(f"Searching with pattern {pattern}")
            matches = self.match_symbol_by_pattern(exchange, pattern, sec_type)
            all_results.extend(matches)
            # Add a delay to avoid rate limits
            time.sleep(1)
        
        return all_results
    
    def get_contract_details(self, contract):
        """
        Get contract details for a given contract.
        
        Parameters:
        -----------
        contract : Contract
            The IB contract object
            
        Returns:
        --------
        list
            List of contract details
        """
        if not self.ib.isConnected():
            self.connect()
            
        try:
            details = self.ib.reqContractDetails(contract)
            return details
        except Exception as e:
            self.logger.error(f"Error getting details for {contract.symbol}: {e}")
            return []
    
    def list_exchange_stocks(self, exchange, save_csv=True):
        """
        List all stocks on a given exchange.
        
        Parameters:
        -----------
        exchange : str
            The exchange code
        save_csv : bool
            Whether to save results to CSV
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with stock information
        """
        if not self.ib.isConnected():
            self.connect()
        
        self.logger.info(f"Listing stocks on exchange: {exchange}")
        
        # Method 1: Use reqMatchingSymbols with patterns
        matching_symbols = self.search_contracts_by_exchange(exchange, 'STK')
        
        # Process results
        stock_info = []
        for symbol_match in matching_symbols:
            try:
                # Create a contract for each match
                contract = Contract(
                    symbol=symbol_match.symbol,
                    secType='STK',
                    exchange=exchange,
                    currency='USD'  # Assuming USD, can be modified
                )
                
                # Get full contract details
                details = self.get_contract_details(contract)
                
                # Process details (if any)
                for detail in details:
                    info = {
                        'symbol': detail.contract.symbol,
                        'conId': detail.contract.conId,
                        'secType': detail.contract.secType,
                        'exchange': detail.contract.exchange,
                        'primaryExchange': detail.contract.primaryExchange,
                        'currency': detail.contract.currency,
                        'longName': getattr(detail, 'longName', ''),
                        'industry': getattr(detail, 'industry', ''),
                        'category': getattr(detail, 'category', ''),
                        'subcategory': getattr(detail, 'subcategory', '')
                    }
                    stock_info.append(info)
                
                # Add a small delay to avoid overwhelming the IB server
                time.sleep(0.2)
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol_match.symbol}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(stock_info)
        
        # Remove duplicates
        if not df.empty:
            df = df.drop_duplicates(subset=['conId'])
        
        # Save to CSV if requested
        if save_csv and not df.empty:
            filename = f"{self.output_dir}/{exchange}_stocks_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(filename, index=False)
            self.logger.info(f"Saved {len(df)} stocks to {filename}")
        
        return df




class IBDataDownloader:
    """
    A class to download market data from Interactive Brokers at various frequencies
    and time horizons, and save it locally in parquet format.
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, data_directory='./data'):
        """
        Initialize the IBDataDownloader class.
        
        Parameters:
        -----------
        host : str
            The hostname or IP address of the IB Gateway/TWS
        port : int
            The port number of the IB Gateway/TWS
        client_id : int
            The client ID to identify this connection
        data_directory : str
            The directory where parquet files will be saved.
            Files will be organized into subdirectories by frequency.
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.data_directory = data_directory
        self.ib = IB()
        self.logger = self._setup_logger()
        
        # Create main data directory if it doesn't exist
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
            
    def _setup_logger(self):
        """Set up logging"""
        logger = logging.getLogger('IBDataDownloader')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger
            
    def connect(self):
        """Connect to Interactive Brokers"""
        
        # this is necessary to run in a notebook
        # util.startLoop()
        
        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.logger.info(f"Connected to IB on {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to IB: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        if self.ib.isConnected():
            self.ib.disconnect()
            self.logger.info("Disconnected from IB")
    
    def create_contract(self, symbol, sec_type='STK', exchange='SMART', currency='USD', **kwargs):
        """
        Create an IB contract object.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the instrument
        sec_type : str
            The security type (STK, FUT, OPT, IND, etc.)
        exchange : str
            The exchange where the instrument is traded
        currency : str
            The currency of the instrument
        **kwargs : dict
            Additional parameters for the contract (e.g., expiry for futures)
            
        Returns:
        --------
        Contract
            An IB contract object
        """
        contract = Contract(symbol=symbol, secType=sec_type, 
                           exchange=exchange, currency=currency, **kwargs)
        return contract
    
    def _get_duration_str(self, days):
        """
        Convert number of days to IB duration string.
        
        Parameters:
        -----------
        days : int
            Number of days for the historical data request
            
        Returns:
        --------
        str
            IB duration string
        """
        if days <= 1:
            return f"{days} D"
        elif days <= 7:
            return f"{days} D"
        elif days <= 30:
            return f"{days} D"
        elif days <= 365:
            return f"{days} D"
        else:
            years = days // 365
            return f"{years} Y"
    
    def _get_bar_size(self, frequency):
        """
        Convert frequency to IB bar size string.
        
        Parameters:
        -----------
        frequency : str
            Data frequency (e.g., '1min', '1hour', '1day')
            
        Returns:
        --------
        str
            IB bar size string
        """
        freq_map = {
            '1min': '1 min',
            '5min': '5 mins',
            '15min': '15 mins',
            '30min': '30 mins',
            '1hour': '1 hour',
            '4hour': '4 hours',
            '1day': '1 day',
            '1week': '1 week',
        }
        
        if frequency in freq_map:
            return freq_map[frequency]
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
    
    def _get_filename(self, symbol, sec_type, frequency):
        """
        Generate a filename for the parquet file.
        Organizes files into subdirectories based on frequency.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the instrument
        sec_type : str
            The security type
        frequency : str
            Data frequency
            
        Returns:
        --------
        str
            Parquet filename with directory path
        """
        # Create frequency directory if it doesn't exist
        freq_dir = f"{self.data_directory}/{frequency}"
        if not os.path.exists(freq_dir):
            os.makedirs(freq_dir)
            
        return f"{freq_dir}/{symbol}_{sec_type}.parquet"
    
    def download_historical_data(self, contract, frequency, days, end_datetime=None, what_to_show='TRADES'):
        """
        Download historical data from Interactive Brokers.
        
        Parameters:
        -----------
        contract : Contract
            The IB contract object
        frequency : str
            Data frequency (e.g., '1min', '1hour', '1day')
        days : int
            Number of days for the historical data request
        end_datetime : datetime, optional
            End datetime for the request (default: now)
        what_to_show : str
            Type of data to request (TRADES, MIDPOINT, BID, ASK, etc.)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with historical data
        """
        if not self.ib.isConnected():
            self.connect()
            
        bar_size = self._get_bar_size(frequency)
        duration_str = self._get_duration_str(days)
        
        if end_datetime is None:
            end_datetime = datetime.now()
            
        end_datetime_str = end_datetime.strftime('%Y%m%d %H:%M:%S')
        
        self.logger.info(f"Requesting {days} days of {frequency} data for {contract.symbol}")
        
        try:
            bars = self.ib.reqHistoricalData(
                contract=contract,
                endDateTime=end_datetime_str,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=True,
                formatDate=1
            )
            
            if bars:
                df = util.df(bars)
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'])
                self.logger.info(f"Downloaded {len(df)} bars")
                return df
            else:
                self.logger.warning("No data received")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Failed to download data: {e}")
            return pd.DataFrame()
    
    def save_to_parquet(self, df, symbol, sec_type, frequency, remove_duplicates=True):
        """
        Save data to parquet file and handle duplicates.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with data to save
        symbol : str
            The symbol of the instrument
        sec_type : str
            The security type
        frequency : str
            Data frequency
        remove_duplicates : bool
            Whether to remove duplicates from the data
            
        Returns:
        --------
        bool
            True if the operation was successful, False otherwise
        """
        if df.empty:
            self.logger.warning("No data to save")
            return False
            
        filename = self._get_filename(symbol, sec_type, frequency)
        
        try:
            # If file exists, append new data and remove duplicates
            if os.path.exists(filename):
                existing_df = pd.read_parquet(filename)
                self.logger.info(f"Read {len(existing_df)} rows from existing file")
                
                # Combine existing and new data
                combined_df = pd.concat([existing_df, df])
                
                if remove_duplicates:
                    # Remove duplicates based on date
                    combined_df = combined_df.drop_duplicates(subset=['date'])
                    
                # Sort by date
                combined_df = combined_df.sort_values('date')
                
                # Save to parquet
                combined_df.to_parquet(filename, index=False)
                self.logger.info(f"Saved {len(combined_df)} rows to {filename}")
            else:
                # If file doesn't exist, just save the new data
                df.to_parquet(filename, index=False)
                self.logger.info(f"Saved {len(df)} rows to {filename}")
                
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return False
    
    def download_and_save(self, symbol, sec_type='STK', exchange='SMART', currency='USD', 
                          frequency='1hour', days=5, end_datetime=None, what_to_show='TRADES', 
                          remove_duplicates=True, **kwargs):
        """
        Download historical data and save it to parquet file.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the instrument
        sec_type : str
            The security type
        exchange : str
            The exchange where the instrument is traded
        currency : str
            The currency of the instrument
        frequency : str
            Data frequency (e.g., '1min', '1hour', '1day')
        days : int
            Number of days for the historical data request
        end_datetime : datetime, optional
            End datetime for the request (default: now)
        what_to_show : str
            Type of data to request (TRADES, MIDPOINT, BID, ASK, etc.)
        remove_duplicates : bool
            Whether to remove duplicates from the data
        **kwargs : dict
            Additional parameters for the contract
            
        Returns:
        --------
        bool
            True if the operation was successful, False otherwise
        """
        contract = self.create_contract(symbol, sec_type, exchange, currency, **kwargs)
        
        df = self.download_historical_data(
            contract=contract,
            frequency=frequency,
            days=days,
            end_datetime=end_datetime,
            what_to_show=what_to_show
        )
        
        if not df.empty:
            return self.save_to_parquet(df, symbol, sec_type, frequency, remove_duplicates)
        else:
            return False
    
    def download_multiple_symbols(self, symbols, sec_type='STK', exchange='SMART', currency='USD',
                                 frequency='1hour', days=5, end_datetime=None, what_to_show='TRADES',
                                 remove_duplicates=True, **kwargs):
        """
        Download historical data for multiple symbols.
        
        Parameters:
        -----------
        symbols : list
            List of symbols to download
        sec_type : str
            The security type
        exchange : str
            The exchange where the instrument is traded
        currency : str
            The currency of the instrument
        frequency : str
            Data frequency (e.g., '1min', '1hour', '1day')
        days : int
            Number of days for the historical data request
        end_datetime : datetime, optional
            End datetime for the request (default: now)
        what_to_show : str
            Type of data to request (TRADES, MIDPOINT, BID, ASK, etc.)
        remove_duplicates : bool
            Whether to remove duplicates from the data
        **kwargs : dict
            Additional parameters for the contract
            
        Returns:
        --------
        dict
            Dictionary with symbols as keys and success status as values
        """
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"Processing {symbol}")
            success = self.download_and_save(
                symbol=symbol,
                sec_type=sec_type,
                exchange=exchange,
                currency=currency,
                frequency=frequency,
                days=days,
                end_datetime=end_datetime,
                what_to_show=what_to_show,
                remove_duplicates=remove_duplicates,
                **kwargs
            )
            
            results[symbol] = success
            
            # Add a small delay to avoid overwhelming the IB server
            time.sleep(1)
            
        return results
    
    def read_parquet_data(self, symbol, sec_type, frequency):
        """
        Read data from parquet file.
        
        Parameters:
        -----------
        symbol : str
            The symbol of the instrument
        sec_type : str
            The security type
        frequency : str
            Data frequency
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with data from parquet file
        """
        filename = self._get_filename(symbol, sec_type, frequency)
        
        if os.path.exists(filename):
            df = pd.read_parquet(filename)
            return df
        else:
            self.logger.warning(f"File {filename} does not exist")
            return pd.DataFrame()
