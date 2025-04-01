from ib_insync import IB, Contract, ContractDetails, util
import pandas as pd
import time
import logging
from datetime import datetime
import os
import argparse
import json

class ExchangeContractLister:
    """
    A class to discover available contracts on a given exchange using Interactive Brokers.
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1, output_dir='./contract_data', log_dir='./logs_dir'):
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
        self.log_dir = log_dir
        self.logger = self._setup_logger()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _setup_logger(self):
        """Set up logging to a file"""
        logger = logging.getLogger('ExchangeContractLister')
        logger.setLevel(logging.INFO)
        
        # Make sure log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create log file path
        log_file = os.path.join(self.log_dir, f'exchange_contract_lister_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Only add handler if not already added
        if not logger.handlers:
            # Create file handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(fh)
            
            # Log the start of a new session
            logger.info(f"Starting new logging session in {log_file}")
        
        return logger
    
    def connect(self):
        """Connect to Interactive Brokers"""
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
            'AMEX', 'NYSE', 'CBOE', 'PHLX', 'ISE', 'CHX', 'ARCA', 'ISLAND',
            'DRCTEDGE', 'BEX', 'BATS', 'EDGEA', 'CBOT', 'CME', 'NYMEX', 'ICEUS',
            'NYSELIFFE', 'NASDAQ', 'TSE', 'VENTURE', 'VSE', 'SBF', 'FTA',
            'IBIS', 'LSE', 'LSEETF', 'MILAN', 'EUREX', 'STOCKH', 'HKFE',
            'SGX', 'KOSPI', 'OSE', 'HKSE', 'ASX', 'CHIX', 'SEHK', 'JSE',
            'MEXDER', 'BVME', 'DME', 'MONEP'
        ]
        return exchanges
    
    def match_symbol_by_pattern(self, exchange, pattern, sec_type='STK', max_results=100):
        """
        Match symbols on an exchange by pattern.
        
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
            
        Returns:
        --------
        list
            List of contract details matching the pattern
        """
        if not self.ib.isConnected():
            self.connect()
            
        try:
            contract = Contract(symbol=pattern, secType=sec_type, exchange=exchange)
            self.logger.info(f"Searching for {sec_type} contracts on {exchange} matching '{pattern}'")
            
            # This is the key method to get matching symbols
            matching_symbols = self.ib.reqMatchingSymbols(pattern)
            matching_symbols = [c for c in matching_symbols if c.contract.secType==sec_type]
            matching_symbols = [c for c in matching_symbols if c.contract.primaryExchange==exchange]
            
            self.logger.info(f"Found {len(matching_symbols)} matches")
            
            # Limit results
            return matching_symbols[:max_results]
            
        except Exception as e:
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

# Example usage
if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Match contracts on Interactive Brokers by pattern')
    parser.add_argument('--pattern', type=str, default='A*', help='Pattern to match (e.g., "A*" for symbols starting with A)')
    parser.add_argument('--exchange', type=str, default='SMART', help='Exchange to search on (e.g., SMART, NASDAQ)')
    parser.add_argument('--sec-type', type=str, default='STK', help='Security type (e.g., STK, FUT, OPT)')
    parser.add_argument('--max-results', type=int, default=100, help='Maximum number of results to return')
    parser.add_argument('--host', type=str, default='host.docker.internal', help='IB Gateway/TWS host')
    parser.add_argument('--port', type=int, default=4002, help='IB Gateway/TWS port')
    parser.add_argument('--client-id', type=int, default=12, help='Client ID for IB connection')
    parser.add_argument('--output-dir', type=str, default='./exchange_data', help='Directory to save results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the lister with command-line arguments
    lister = ExchangeContractLister(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        output_dir=args.output_dir
    )
    
    # Connect to IB
    if lister.connect():
        try:
            # Get exchange and pattern from arguments
            exchange = args.exchange
            pattern = args.pattern
            
            # Get matches
            matches = lister.match_symbol_by_pattern(
                exchange=exchange, 
                pattern=pattern, 
                sec_type=args.sec_type, 
                max_results=args.max_results
            )
            
            # Convert matches to JSON-serializable format
            json_matches = []
            for match in matches:
                contract = match.contract
                match_data = {
                    'symbol': contract.symbol,
                    'conId': contract.conId if hasattr(contract, 'conId') else None,
                    'secType': contract.secType,
                    'primaryExchange': contract.primaryExchange,
                    'currency': contract.currency if hasattr(contract, 'currency') else None,
                    'description': contract.description if hasattr(contract, 'description') else None,
                    'derivativeSecTypes': match.derivativeSecTypes if hasattr(match, 'derivativeSecTypes') else []
                }
                json_matches.append(match_data)
            
            # Output as JSON
            print(json.dumps(json_matches, indent=2))
            
        finally:
            # Disconnect when done
            lister.disconnect()