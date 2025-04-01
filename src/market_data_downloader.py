#!/usr/bin/env python3

import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import logging
import sys
import json
import math
from ib_insync import IB, Contract, util

# Global constants
MAX_DATA_ROWS_PER_REQUEST = 1000

def setup_logger():
    """Set up logging to file and console"""
    logger = logging.getLogger('market_data_downloader')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Create file handler
    log_dir = './market_data/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'market_data_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger

def get_duration_str(time_value, time_unit):
    """Convert time value and unit to IB duration string"""
    if time_unit == 'hours':
        if time_value <= 24:
            return f"{time_value} H"
        else:
            days = time_value // 24
            return f"{days} D"
    elif time_unit == 'days':
        return f"{time_value} D"
    elif time_unit == 'years':
        return f"{time_value} Y"
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")

def get_bar_size(data_type):
    """Convert data type to IB bar size string"""
    bar_size_map = {
        '5sec': '5 secs',
        '1min': '1 min',
        '1hour': '1 hour',
        '1day': '1 day',
    }
    
    if data_type in bar_size_map:
        return bar_size_map[data_type]
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def estimate_bar_count(time_value, time_unit, data_type):
    """Estimate the number of data bars based on duration and bar size"""
    # Convert to total hours for consistent calculation
    if time_unit == 'hours':
        total_hours = time_value
    elif time_unit == 'days':
        total_hours = time_value * 24
    elif time_unit == 'years':
        total_hours = time_value * 365 * 24
    else:
        raise ValueError(f"Unknown time unit: {time_unit}")
    
    # Estimate number of bars based on data frequency
    if data_type == '5sec':
        # Assume 8 market hours per day (conservative estimate)
        market_hours_per_day = 8
        # For '5sec' data, there are 720 bars per market hour (60*60/5)
        bars_per_hour = 720
        # Adjust total hours to market hours
        market_hours = (total_hours / 24) * market_hours_per_day
        return math.ceil(market_hours * bars_per_hour)
    elif data_type == '1min':
        # Assume 8 market hours per day (conservative estimate)
        market_hours_per_day = 8
        # For '1min' data, there are 60 bars per market hour
        bars_per_hour = 60
        # Adjust total hours to market hours
        market_hours = (total_hours / 24) * market_hours_per_day
        return math.ceil(market_hours * bars_per_hour)
    elif data_type == '1hour':
        # For '1hour' data, there are about 8 bars per market day
        market_hours_per_day = 8
        # Adjust total hours to market hours
        market_hours = (total_hours / 24) * market_hours_per_day
        return math.ceil(market_hours)
    elif data_type == '1day':
        # For '1day' data, there are about 252 trading days per year (approximation)
        trading_days_per_year = 252
        if time_unit == 'years':
            return math.ceil(time_value * trading_days_per_year)
        elif time_unit == 'days':
            return time_value
        else:  # hours
            return math.ceil(total_hours / 24)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def split_duration(time_value, time_unit, data_type):
    """Split a long duration into smaller chunks that won't exceed MAX_DATA_ROWS_PER_REQUEST"""
    estimated_bars = estimate_bar_count(time_value, time_unit, data_type)
    
    if estimated_bars <= MAX_DATA_ROWS_PER_REQUEST:
        # No need to split
        return [(time_value, time_unit)]
    
    # Calculate how many chunks we need
    num_chunks = math.ceil(estimated_bars / MAX_DATA_ROWS_PER_REQUEST)
    
    # Convert to most appropriate unit for splitting
    if time_unit == 'years':
        # Convert to days for more granular splitting
        days = time_value * 365
        chunk_size = math.ceil(days / num_chunks)
        return [(chunk_size, 'days') for _ in range(num_chunks)]
    elif time_unit == 'days':
        chunk_size = math.ceil(time_value / num_chunks)
        return [(chunk_size, 'days') for _ in range(num_chunks)]
    elif time_unit == 'hours':
        if time_value >= 24:
            # Convert to days if it's at least 24 hours
            days = time_value // 24
            remaining_hours = time_value % 24
            chunks = []
            
            days_per_chunk = math.ceil(days / num_chunks)
            for _ in range(num_chunks - 1):
                chunks.append((days_per_chunk, 'days'))
            
            # Last chunk includes remaining hours
            if remaining_hours > 0:
                chunks.append((days_per_chunk + (remaining_hours / 24), 'days'))
            else:
                chunks.append((days_per_chunk, 'days'))
                
            return chunks
        else:
            # Keep as hours
            chunk_size = math.ceil(time_value / num_chunks)
            return [(chunk_size, 'hours') for _ in range(num_chunks)]
    
    # Should not reach here
    raise ValueError(f"Unsupported time unit: {time_unit}")

def download_market_data(symbol, sec_type, exchange, currency, time_value, time_unit, data_type, 
                        host, port, client_id, output_dir='./market_data'):
    """Download market data for a specific contract and save as parquet file"""
    logger = logging.getLogger('market_data_downloader')
    
    # Create appropriate subdirectory
    data_dir = os.path.join(output_dir, data_type)
    os.makedirs(data_dir, exist_ok=True)
    
    # Connect to IB
    ib = IB()
    try:
        logger.info(f"Connecting to IB at {host}:{port} with client ID {client_id}")
        ib.connect(host, port, clientId=client_id)
        
        # Create contract
        logger.info(f"Creating contract for {symbol} ({sec_type} on {exchange})")
        contract = Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency=currency)
        
        # Qualify contract
        qualified_contracts = ib.qualifyContracts(contract)
        if not qualified_contracts:
            logger.error(f"Failed to qualify contract: {symbol}")
            ib.disconnect()
            return False

        contract = qualified_contracts[0]
        
        # Split the duration into chunks if necessary
        chunks = split_duration(time_value, time_unit, data_type)
        logger.info(f"Split request into {len(chunks)} chunks for {symbol}")
        
        # Initialize empty DataFrame to store all data
        all_data_frames = []
        successful_chunks = 0
        bar_size = get_bar_size(data_type)
        
        # Process each chunk
        for i, (chunk_time_value, chunk_time_unit) in enumerate(chunks):
            duration_str = get_duration_str(chunk_time_value, chunk_time_unit)
            
            # Calculate end time for this chunk (except for the last chunk which uses current time)
            if i < len(chunks) - 1:
                # If it's not the last chunk, calculate end time based on previous chunks
                if chunk_time_unit == 'days':
                    end_time = datetime.now() - timedelta(days=sum([cv for cv, cu in chunks[:i]]))
                elif chunk_time_unit == 'hours':
                    end_time = datetime.now() - timedelta(hours=sum([cv for cv, cu in chunks[:i]]))
                else:  # years
                    end_time = datetime.now() - timedelta(days=365 * sum([cv for cv, cu in chunks[:i]]))
                end_time_str = end_time.strftime('%Y%m%d %H:%M:%S')
            else:
                # Use current time for the last chunk
                end_time_str = ''  # Empty string means current time in IB API
            
            try:
                # Download data for this chunk
                chunk_info = f"Requesting chunk {i+1}/{len(chunks)}: {duration_str} of {bar_size} data for {symbol} ending at {end_time_str if end_time_str else 'now'}"
                logger.info(chunk_info)
                # Print to stdout for UI to capture
                print(chunk_info)
                sys.stdout.flush()
                
                bars = ib.reqHistoricalData(
                    contract=contract,
                    endDateTime=end_time_str,
                    durationStr=duration_str,
                    barSizeSetting=bar_size,
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                
                if not bars:
                    logger.warning(f"No data received for {symbol} in chunk {i+1}/{len(chunks)}")
                    # Continue with next chunk
                    continue
                    
                # Convert to pandas dataframe
                chunk_df = util.df(bars)
                
                # Add additional columns
                chunk_df['symbol'] = symbol
                chunk_df['secType'] = sec_type
                chunk_df['exchange'] = exchange
                chunk_df['currency'] = currency
                chunk_df['conId'] = contract.conId

                # Ensure that column order and headers do not change for compatibility with Athena
                expected_headers = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 
                                    'barCount','symbol', 'secType', 'exchange', 'currency', 'conId']
                actual_headers = chunk_df.columns.tolist()
                if all([h in actual_headers for h in expected_headers]):
                    chunk_df = chunk_df[expected_headers].sort_values('date')
                    all_data_frames.append(chunk_df)
                    successful_chunks += 1
                    success_msg = f"Successfully downloaded {len(chunk_df)} bars for chunk {i+1}/{len(chunks)}"
                    logger.info(success_msg)
                    # Print to stdout for UI to capture
                    print(success_msg)
                    sys.stdout.flush()
                else:
                    missing_cols = [h for h in expected_headers if h not in actual_headers]
                    logger.warning(f"Data schema error in chunk {i+1}/{len(chunks)} for {symbol}: missing columns {missing_cols}")
                    # Continue with next chunk
                
                # Add a small delay between requests to avoid overwhelming IB API
                time.sleep(3)
                
            except Exception as e:
                logger.warning(f"Error downloading chunk {i+1}/{len(chunks)} for {symbol}: {e}")
                # Continue with next chunk
        
        # Check if we got any data at all
        if not all_data_frames:
            logger.error(f"Failed to download any data for {symbol} after trying {len(chunks)} chunks")
            ib.disconnect()
            return False
            
        # Combine all chunks
        df = pd.concat(all_data_frames)
        
        # Remove duplicates that might occur between chunks
        df = df.drop_duplicates(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Save as parquet file
        filename = f"{symbol}_{sec_type}.parquet"
        file_path = os.path.join(data_dir, filename)
        
        # If file exists, merge with existing data to avoid duplicates
        if os.path.exists(file_path):
            logger.info(f"Merging with existing data in {file_path}")
            existing_df = pd.read_parquet(file_path)
            
            # Combine and drop duplicates
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df.drop_duplicates(subset=['date'])
            df = combined_df
        
        # Save to parquet
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(df)} bars to {file_path} from {successful_chunks}/{len(chunks)} successful chunks")
        
        # Disconnect
        ib.disconnect()
        return True
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {e}")
        if ib.isConnected():
            ib.disconnect()
        return False

def main():
    """Main function for CLI use"""
    parser = argparse.ArgumentParser(description='Download market data from Interactive Brokers')
    
    # Contract parameters
    parser.add_argument('--symbol', type=str, required=True, help='Symbol of the contract')
    parser.add_argument('--sec-type', type=str, default='STK', help='Security type (e.g., STK, FUT, OPT)')
    parser.add_argument('--exchange', type=str, default='SMART', help='Exchange to use')
    parser.add_argument('--currency', type=str, default='USD', help='Currency of the contract')
    
    # Data parameters
    parser.add_argument('--time-value', type=int, required=True, help='Time value (e.g., 1, 5, 10)')
    parser.add_argument('--time-unit', type=str, required=True, choices=['hours', 'days', 'years'], 
                       help='Time unit (hours, days, or years)')
    parser.add_argument('--data-type', type=str, required=True, choices=['5sec', '1min', '1hour', '1day'],
                       help='Type of data to download')
    
    # Connection parameters
    parser.add_argument('--host', type=str, default='127.0.0.1', help='IB Gateway/TWS host')
    parser.add_argument('--port', type=int, default=4001, help='IB Gateway/TWS port')
    parser.add_argument('--client-id', type=int, default=1, help='Client ID for IB connection')
    parser.add_argument('--output-dir', type=str, default='./market_data', help='Output directory')
    
    # Add parameter for verbose output (used by UI to get chunk information)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output with chunk progress information')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    
    # Get number of chunks before download to print info
    chunks = split_duration(args.time_value, args.time_unit, args.data_type)
    
    # Print chunk information if in verbose mode (for UI progress tracking)
    if args.verbose:
        chunk_info = {
            "total_chunks": len(chunks),
            "symbol": args.symbol
        }
        print(f"CHUNK_INFO: {json.dumps(chunk_info)}")
        sys.stdout.flush()  # Ensure output is immediately available to parent process
    
    # Download data
    success = download_market_data(
        symbol=args.symbol,
        sec_type=args.sec_type,
        exchange=args.exchange,
        currency=args.currency,
        time_value=args.time_value,
        time_unit=args.time_unit,
        data_type=args.data_type,
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        output_dir=args.output_dir
    )
    
    if success:
        logger.info(f"Successfully downloaded data for {args.symbol}")
        return 0
    else:
        logger.error(f"Failed to download data for {args.symbol}")
        return 1

if __name__ == "__main__":
    sys.exit(main())