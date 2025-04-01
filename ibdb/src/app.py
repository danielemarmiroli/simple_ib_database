import streamlit as st
import pandas as pd
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import the custom contract lister class for exchange listing functionality
from ib_data_handlers import ExchangeContractLister

# Setup page config
st.set_page_config(
    page_title="IB Data Management System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Contracts", "Market Data"])

# Separator in sidebar
st.sidebar.markdown("---")

# Function to load saved contracts from CSV
def load_saved_contracts():
    # Ensure the contract_data directory exists
    contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
    if not os.path.exists(contract_data_dir):
        os.makedirs(contract_data_dir)
    
    csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
    
    if os.path.exists(csv_path):
        try:
            # Read CSV into DataFrame and convert to list of dictionaries
            df = pd.read_csv(csv_path)
            
            # Convert string representations of lists back to actual lists
            if 'derivativeSecTypes' in df.columns:
                df['derivativeSecTypes'] = df['derivativeSecTypes'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else []
                )
                
            return df.to_dict('records')  # Convert DataFrame to list of dictionaries
        except Exception as e:
            print(f"Error loading saved contracts: {e}")
            return []
    return []

# Function to save contracts to CSV
def save_contracts(contracts):
    # Ensure the contract_data directory exists
    contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
    if not os.path.exists(contract_data_dir):
        os.makedirs(contract_data_dir)
    
    csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
    
    try:
        # Convert to DataFrame for easy CSV saving
        df = pd.DataFrame(contracts)
        
        # If the DataFrame is empty, just save an empty file and return
        if df.empty:
            df.to_csv(csv_path, index=False)
            return
        
        # Convert list columns to strings for CSV storage
        if 'derivativeSecTypes' in df.columns:
            df['derivativeSecTypes'] = df['derivativeSecTypes'].apply(json.dumps)
        
        # Drop duplicates based on symbol and conId if available
        if 'conId' in df.columns:
            df = df.drop_duplicates(subset=['symbol', 'conId'])
        else:
            df = df.drop_duplicates(subset=['symbol'])
        
        # Sort by symbol alphabetically
        df = df.sort_values(by='symbol')
        
        # Reset index after sorting
        df = df.reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error saving contracts: {e}")

# Initialize session state variables if they don't exist
if 'connection_verified' not in st.session_state:
    st.session_state.connection_verified = False
if 'connection_params' not in st.session_state:
    st.session_state.connection_params = {
        'host': 'host.docker.internal',
        'port': 4002,
        'client_id': 1,
        'output_dir': './contract_data'
    }
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'saved_contracts' not in st.session_state:
    st.session_state.saved_contracts = load_saved_contracts()
if 'exchanges' not in st.session_state:
    st.session_state.exchanges = []

# Sidebar for connection settings
st.sidebar.header("Connection Settings")

# Connection settings (use values from session state)
host = st.sidebar.text_input("Host", value=st.session_state.connection_params['host'])
port = st.sidebar.number_input("Port", value=st.session_state.connection_params['port'], min_value=1, max_value=65535)
client_id = st.sidebar.number_input("Client ID", value=st.session_state.connection_params['client_id'], min_value=1, max_value=999)
output_dir = st.sidebar.text_input("Output Directory", value=st.session_state.connection_params['output_dir'])

# Function to verify connection
def verify_connection(host, port, client_id, output_dir):
    """Verify connection to IB and get available exchanges"""
    try:
        # Initialize a temporary lister to verify connection
        temp_lister = ExchangeContractLister(
            host=host,
            port=port,
            client_id=client_id,
            output_dir=output_dir
        )
        
        # Try to connect
        if temp_lister.connect():
            # Get exchanges if connection successful
            exchanges = temp_lister.get_exchange_list()
            # Disconnect immediately since we don't need to keep connection open
            temp_lister.disconnect()
            return True, exchanges, "Connected to Interactive Brokers!"
        else:
            return False, [], "Failed to connect to Interactive Brokers!"
    except Exception as e:
        return False, [], f"Error connecting to IB: {e}"

# Verify Connection button
if st.sidebar.button("Verify Connection"):
    # Try to connect to verify settings
    is_connected, exchanges, message = verify_connection(host, port, client_id, output_dir)
    
    # Update session state
    st.session_state.connection_verified = is_connected
    if is_connected:
        st.session_state.exchanges = exchanges
        st.session_state.connection_params = {
            'host': host,
            'port': port,
            'client_id': client_id,
            'output_dir': output_dir
        }
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# Main content based on selected page
if page == "Contracts":
    st.title("Interactive Brokers Contract Explorer")
    
    if st.session_state.connection_verified:
        # Contract search section
        st.header("Contract Search")
        
        col1, col2, col3 = st.columns([2, 2, 1])
    
        with col1:
            # Exchange selection
            selected_exchange = st.selectbox(
                "Select Exchange",
                options=st.session_state.exchanges
            )
        
        with col2:
            # Pattern input
            patterns = st.text_input(
                "Search Pattern (use * as wildcard, comma-separated for multiple)",
                value="A*"
            )
        
        with col3:
            # Security type selection
            sec_type = st.selectbox(
                "Security Type",
                options=["STK", "FUT", "OPT", "IND", "BOND", "CASH"]
            )
    
        # Function to call match_contracts.py script
        def call_match_contracts_script(pattern, exchange, sec_type, host, port, client_id, output_dir, max_results=100):
            """Call the match_contracts.py script with the given parameters and return the JSON results"""
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'match_contracts.py')
            
            cmd = [
                sys.executable,  # Python executable
                script_path,
                '--pattern', pattern,
                '--exchange', exchange,
                '--sec-type', sec_type,
                '--host', host,
                '--port', str(port),
                '--client-id', str(client_id),
                '--output-dir', output_dir,
                '--max-results', str(max_results)
            ]
            
            try:
                # Run the script and capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                # Parse JSON output
                return json.loads(result.stdout)
            except subprocess.CalledProcessError as e:
                st.error(f"Error calling match_contracts.py: {e}")
                st.error(f"Error output: {e.stderr}")
                return []
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON output: {e}")
                st.error(f"Raw output: {result.stdout}")
                return []
        
        # Search button
        if st.button("Search Contracts"):
            with st.spinner("Searching for contracts..."):
                try:
                    # Parse patterns
                    pattern_list = [p.strip() for p in patterns.split(",")]
                    
                    # Store search results
                    all_results = []
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    
                    # Search for each pattern
                    for i, pattern in enumerate(pattern_list):
                        st.info(f"Searching with pattern: {pattern}")
                        
                        # Call match_contracts.py script with connection params from session state
                        matches = call_match_contracts_script(
                            pattern=pattern,
                            exchange=selected_exchange,
                            sec_type=sec_type,
                            host=st.session_state.connection_params['host'],
                            port=st.session_state.connection_params['port'],
                            client_id=st.session_state.connection_params['client_id'],
                            output_dir=st.session_state.connection_params['output_dir'],
                            max_results=100
                        )
                        
                        # Matches are already in the correct format as returned by the script
                        all_results.extend(matches)
                        
                        # Update progress bar
                        progress_bar.progress((i + 1) / len(pattern_list))
                        
                        # Add a small delay to avoid rate limits
                        time.sleep(0.5)
                    
                    # Store in session state
                    st.session_state.search_results = all_results
                    
                    # Done
                    st.success(f"Found {len(all_results)} contracts matching your criteria.")
                    
                except Exception as e:
                    st.error(f"Error searching for contracts: {e}")
    
        # Display search results
        if st.session_state.search_results:
            st.header("Search Results")
            
            # Create a DataFrame from the results
            df = pd.DataFrame(st.session_state.search_results)
            
            # Display the results in a table
            st.dataframe(df)
            
            # Create a list of options for the multiselect
            options = []
            for result in st.session_state.search_results:
                option_text = f"{result['symbol']} ({result['secType']} on {result['primaryExchange']})"
                options.append(option_text)
            
            # Add select all checkbox
            col1, col2 = st.columns([1, 3])
            with col1:
                select_all = st.checkbox("Select All", key="select_all_checkbox")
            
            # Multi-select for contract selection
            with col2:
                if select_all:
                    selected_options = options.copy()  # Select all options
                    # Show the multiselect with all options selected (for display purposes)
                    st.multiselect(
                        "Selected contracts",
                        options=options,
                        default=options,
                        disabled=True
                    )
                else:
                    # Regular multiselect when "Select All" is not checked
                    selected_options = st.multiselect(
                        "Select contracts to save",
                        options=options
                    )
            
            # Add selected contracts to saved list
            if st.button("Save Selected Contracts"):
                if selected_options:
                    # Find the selected contracts in the search results
                    new_saved_contracts = []
                    for option in selected_options:
                        # Extract symbol from the option text
                        symbol = option.split(" ")[0]
                        
                        # Find the corresponding contract in search results
                        for result in st.session_state.search_results:
                            if result['symbol'] == symbol:
                                # Create a copy of the result to avoid modifying the original
                                saved_result = result.copy()
                                # Add timestamp to the contract
                                saved_result['added_on'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                new_saved_contracts.append(saved_result)
                                break
                    
                    # Add to session state
                    st.session_state.saved_contracts.extend(new_saved_contracts)
                    
                    # Save to file
                    save_contracts(st.session_state.saved_contracts)
                    
                    # Reload the saved contracts to ensure the list is up to date
                    st.session_state.saved_contracts = load_saved_contracts()
                    
                    # Inform user where the file was saved
                    contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
                    csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
                    st.success(f"Saved {len(new_saved_contracts)} contracts to {csv_path}")
        
        # Saved contracts section
        st.header("Saved Contracts")
        
        if st.session_state.saved_contracts:
            # Create a DataFrame from saved contracts
            saved_df = pd.DataFrame(st.session_state.saved_contracts)
            
            # Display saved contracts
            st.dataframe(saved_df)
            
            # Add buttons for managing saved contracts
            col1, col2 = st.columns(2)
            
            # Option to remove contracts with confirmation
            with col1:
                # Initialize confirmation state if not exists
                if 'clear_contracts_confirm' not in st.session_state:
                    st.session_state.clear_contracts_confirm = False
                if 'delete_confirmation_text' not in st.session_state:
                    st.session_state.delete_confirmation_text = ""
                
                # Button to show confirmation popup
                if not st.session_state.clear_contracts_confirm:
                    if st.button("Clear All Saved Contracts"):
                        st.session_state.clear_contracts_confirm = True
                
                # Show confirmation popup if button was clicked
                if st.session_state.clear_contracts_confirm:
                    st.warning("⚠️ This will permanently delete all saved contracts!")
                    st.text_input(
                        "Type 'delete' to confirm deletion:", 
                        key="delete_confirmation_text"
                    )
                    
                    col1a, col1b = st.columns(2)
                    with col1a:
                        if st.button("Proceed with Deletion"):
                            if st.session_state.delete_confirmation_text.lower() == "delete":
                                # Perform deletion
                                st.session_state.saved_contracts = []
                                save_contracts([])
                                contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
                                csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
                                st.success(f"Cleared all saved contracts from {csv_path}")
                                # Only reset confirmation flag - don't modify the text input state directly
                                st.session_state.clear_contracts_confirm = False
                            else:
                                st.error("Confirmation text does not match 'delete'. Operation cancelled.")
                    with col1b:
                        if st.button("Cancel"):
                            # Only reset confirmation flag - don't modify the text input state directly
                            st.session_state.clear_contracts_confirm = False
            
            # Option to download the CSV file directly
            with col2:
                # Create a download button for the saved contracts CSV
                contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
                csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
                
                if os.path.exists(csv_path):
                    with open(csv_path, 'r') as f:
                        csv_data = f.read()
                    st.download_button(
                        label="Download Saved Contracts (CSV)",
                        data=csv_data,
                        file_name="saved_contracts.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No saved contracts yet. Search and select contracts to save them.")

    else:
        st.info("Please verify your connection to Interactive Brokers using the sidebar button to start exploring contracts.")

elif page == "Market Data":
    st.title("Interactive Brokers Market Data")
    
    if st.session_state.connection_verified:
        # Initialize session state for market data if not already present
        if 'market_data_status' not in st.session_state:
            st.session_state.market_data_status = None
        if 'market_data_running' not in st.session_state:
            st.session_state.market_data_running = False
        
        # Data download section
        st.header("Download Historical Data")
        st.write("Select time range and data type to download historical data for your saved contracts.")
        
        # Time range and data type selectors
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_value = st.number_input("Time Value", min_value=1, max_value=180, value=1)
        
        with col2:
            time_unit = st.selectbox(
                "Time Unit",
                options=["hours", "days", "years"],
                index=1  # Default to days
            )
        
        with col3:
            data_type = st.selectbox(
                "Data Type",
                options=["5sec", "1min", "1hour", "1day"],
                index=2  # Default to 1hour
            )
        
        # Function to update market data for selected contracts
        def update_market_data(time_value, time_unit, data_type):
            import subprocess
            import os
            import pandas as pd
            import time
            
            # Path to saved contracts
            contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
            csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
            
            # Check if the file exists
            if not os.path.exists(csv_path):
                st.error("No saved contracts found. Please save contracts from the Contracts page first.")
                return False
            
            try:
                # Read saved contracts
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    st.error("No contracts found in the saved contracts file.")
                    return False
                
                # Filter for selected symbols if any are selected
                if hasattr(st.session_state, 'selected_symbols') and len(st.session_state.selected_symbols) > 0:
                    df = df[df['symbol'].isin(st.session_state.selected_symbols)]
                
                if df.empty:
                    st.error("No symbols selected for data download.")
                    return False
                
                # Get connection parameters from session state
                host = st.session_state.connection_params['host']
                port = st.session_state.connection_params['port']
                client_id = st.session_state.connection_params['client_id']
                output_dir = os.path.join(os.getcwd(), 'market_data')
                
                # Make sure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each contract
                total_contracts = len(df)
                successful = 0
                failed = 0
                
                for i, (_, row) in enumerate(df.iterrows()):
                    # Update status
                    status_text.info(f"Processing {i+1}/{total_contracts}: {row['symbol']}")
                    
                    # Prepare parameters for the script
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'market_data_downloader.py')
                    
                    # Get exchange - either primaryExchange or exchange depending on what's available
                    exchange = row.get('primaryExchange', row.get('exchange', 'SMART'))
                    
                    # Default to STK if secType is not available
                    sec_type = row.get('secType', 'STK')
                    
                    # Default to USD if currency is not available
                    currency = row.get('currency', 'USD')
                    
                    # Construct command
                    cmd = [
                        sys.executable,  # Python executable
                        script_path,
                        '--symbol', row['symbol'],
                        '--sec-type', sec_type,
                        '--exchange', exchange,
                        '--currency', currency,
                        '--time-value', str(time_value),
                        '--time-unit', time_unit,
                        '--data-type', data_type,
                        '--host', host,
                        '--port', str(port),
                        '--client-id', str(client_id),
                        '--output-dir', output_dir,
                        '--verbose'  # Enable verbose output for chunk progress
                    ]
                    
                    try:
                        # Create a container for chunk information
                        chunk_info = None
                        total_chunks = 1  # Default to 1 if no chunk info available
                        
                        # Start the process
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            universal_newlines=True
                        )
                        
                        # Create secondary progress bar for chunks
                        chunk_status = st.empty()
                        chunk_progress_bar = st.progress(0)
                        
                        # Process output in real-time
                        for line in process.stdout:
                            # Check if line contains chunk info
                            if "CHUNK_INFO:" in line:
                                try:
                                    # Extract JSON chunk info
                                    chunk_info_str = line.split("CHUNK_INFO:")[1].strip()
                                    chunk_info = json.loads(chunk_info_str)
                                    total_chunks = chunk_info.get("total_chunks", 1)
                                    chunk_status.info(f"Symbol {row['symbol']} will be processed in {total_chunks} chunks")
                                except Exception as e:
                                    st.warning(f"Error parsing chunk info: {e}")
                            
                            # If line contains chunk progress information
                            elif "Requesting chunk" in line and "of" in line:
                                try:
                                    # Extract current chunk from log line
                                    # Example format: "Requesting chunk 2/5: ..."
                                    chunk_str = line.split("Requesting chunk")[1].split(":")[0].strip()
                                    current_chunk = int(chunk_str.split("/")[0])
                                    total_chunks = int(chunk_str.split("/")[1])
                                    
                                    # Update chunk progress bar
                                    chunk_progress_bar.progress(current_chunk / total_chunks)
                                    chunk_status.info(f"Processing chunk {current_chunk}/{total_chunks} for {row['symbol']}")
                                except Exception as e:
                                    # If can't parse chunk info, just display the line
                                    pass
                            
                            # Display chunk completion information
                            elif "Successfully downloaded" in line and "bars for chunk" in line:
                                try:
                                    # Display information about each chunk's data
                                    chunk_status.success(line.strip())
                                except Exception:
                                    pass
                        
                        # Wait for process to complete
                        process.wait()
                        
                        # Check return code
                        if process.returncode == 0:
                            successful += 1
                            # Clear chunk progress display
                            chunk_progress_bar.empty()
                            chunk_status.empty()
                        else:
                            failed += 1
                            stderr_output = process.stderr.read()
                            st.error(f"Error downloading data for {row['symbol']}: {stderr_output}")
                    
                    except Exception as e:
                        failed += 1
                        st.error(f"Exception while downloading data for {row['symbol']}: {e}")
                    
                    # Update main progress bar for symbols
                    progress_bar.progress((i + 1) / total_contracts)
                    
                    # Add a small delay to avoid overwhelming the IB server
                    time.sleep(1)
                
                # Final status
                if failed == 0:
                    status_text.success(f"Successfully downloaded data for all {successful} contracts!")
                else:
                    status_text.warning(f"Downloaded data for {successful} contracts. Failed for {failed} contracts.")
                
                return True
                
            except Exception as e:
                st.error(f"Error updating market data: {e}")
                return False
        
        # Display saved contracts count and allow symbol selection
        contract_data_dir = os.path.join(os.getcwd(), 'contract_data')
        csv_path = os.path.join(contract_data_dir, 'saved_contracts.csv')
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    st.info(f"You have {len(df)} saved contracts available.")
                    
                    # Symbol selector section
                    st.subheader("Select Symbols to Update")
                    
                    # Initialize selected_symbols in session state if not present
                    if 'selected_symbols' not in st.session_state:
                        st.session_state.selected_symbols = list(df['symbol'])
                    
                    # Buttons for select all and reset
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        if st.button("Select All Symbols"):
                            st.session_state.selected_symbols = list(df['symbol'])
                            st.rerun()
                    
                    with col2:
                        if st.button("Reset Selection"):
                            st.session_state.selected_symbols = []
                            st.rerun()
                            
                    # Empty column to balance layout
                    with col3:
                        pass
                    
                    # Multi-select for symbols
                    st.session_state.selected_symbols = st.multiselect(
                        "Select symbols to update",
                        options=list(df['symbol']),
                        default=st.session_state.selected_symbols
                    )
                    
                    st.info(f"Selected {len(st.session_state.selected_symbols)} out of {len(df)} symbols for data update.")
                else:
                    st.warning("No contracts found in the saved contracts file.")
            except Exception as e:
                st.warning(f"Could not read saved contracts file: {e}")
        else:
            st.warning("No saved contracts found. Please save contracts from the Contracts page first.")
        
        # Update button
        if st.button("Update Market Data", key="update_market_data_button"):
            with st.spinner("Downloading market data..."):
                st.session_state.market_data_running = True
                success = update_market_data(time_value, time_unit, data_type)
                st.session_state.market_data_running = False
                st.session_state.market_data_status = success
        
        # Data exploration section
        st.header("Explore Market Data")
        
        # Check for available data
        market_data_dir = os.path.join(os.getcwd(), 'market_data')
        has_data = False
        
        data_types = ["5sec", "1min", "1hour", "1day"]
        # default calendar lookback window for timeseries visualization
        lookback = dict(zip(data_types, [0.5, 1, 5, 21]))
        data_counts = {}
        
        for dt in data_types:
            data_dir = os.path.join(market_data_dir, dt)
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
                data_counts[dt] = len(files)
                if len(files) > 0:
                    has_data = True
        
        if has_data:
            st.write("Available market data:")
            
            for dt, count in data_counts.items():
                if count > 0:
                    st.write(f"- {dt}: {count} symbol(s)")
            
            # Data visualization section
            st.header("Data Visualization")
            
            # Collect all available symbols across all timeframes
            available_symbols = set()
            symbol_to_timeframes = {}
            
            for dt in data_types:
                data_dir = os.path.join(market_data_dir, dt)
                if os.path.exists(data_dir):
                    for file in os.listdir(data_dir):
                        if file.endswith('.parquet'):
                            symbol = file.split('_')[0]
                            available_symbols.add(symbol)
                            if symbol not in symbol_to_timeframes:
                                symbol_to_timeframes[symbol] = []
                            symbol_to_timeframes[symbol].append(dt)
            
            # Convert to sorted list
            available_symbols = sorted(list(available_symbols))
            
            if available_symbols:
                # Symbol selection
                selected_symbol = st.selectbox(
                    "Select Symbol",
                    options=available_symbols
                )
                
                # Show available timeframes for the selected symbol
                available_timeframes = symbol_to_timeframes.get(selected_symbol, [])
                
                if available_timeframes:
                    # Timeframe selection
                    selected_timeframe = st.selectbox(
                        "Select Timeframe",
                        options=available_timeframes
                    )
                    
                    # Path to the data file
                    data_file = os.path.join(market_data_dir, selected_timeframe, f"{selected_symbol}_STK.parquet")
                    
                    if os.path.exists(data_file):
                        # Load the data
                        try:
                            df = pd.read_parquet(data_file)
                            
                            # Make sure we have the required columns
                            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                            
                            if all(col in df.columns for col in required_columns):
                                # Convert date column to datetime if needed
                                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                                    df['date'] = pd.to_datetime(df['date'])
                                
                                # Sort by date
                                df = df.sort_values('date')
                                
                                # Date range selection
                                min_date = df['date'].min().date()
                                max_date = df['date'].max().date()
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    start_date = st.date_input(
                                        "Start Date",
                                        value=max_date - timedelta(days=lookback[selected_timeframe]),  # Default to last 30 days
                                        min_value=min_date,
                                        max_value=max_date
                                    )
                                
                                with col2:
                                    end_date = st.date_input(
                                        "End Date",
                                        value=max_date,
                                        min_value=min_date,
                                        max_value=max_date
                                    )
                                
                                # Filter data based on selected date range
                                mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
                                filtered_df = df[mask]
                                
                                if not filtered_df.empty:
                                    # Create plot
                                    st.subheader(f"{selected_symbol} OHLC Chart ({selected_timeframe})")
                                    
                                    # Create figure with secondary y-axis for volume
                                    fig = make_subplots(
                                        rows=2, 
                                        cols=1, 
                                        shared_xaxes=True,
                                        vertical_spacing=0.03,
                                        subplot_titles=('OHLC', 'Volume'),
                                        row_heights=[0.7, 0.3]
                                    )
                                    
                                    # Add OHLC trace
                                    fig.add_trace(
                                        go.Candlestick(
                                            x=filtered_df['date'],
                                            open=filtered_df['open'],
                                            high=filtered_df['high'],
                                            low=filtered_df['low'],
                                            close=filtered_df['close'],
                                            name="OHLC"
                                        ),
                                        row=1, col=1
                                    )
                                    
                                    # Colors for volume bars (green for up days, red for down days)
                                    colors = np.where(filtered_df['close'] >= filtered_df['open'], 'green', 'red')
                                    
                                    # Add volume trace
                                    fig.add_trace(
                                        go.Bar(
                                            x=filtered_df['date'],
                                            y=filtered_df['volume'],
                                            marker_color=colors,
                                            name="Volume"
                                        ),
                                        row=2, col=1
                                    )
                                    
                                    # Update layout
                                    fig.update_layout(
                                        title=f"{selected_symbol} ({min(filtered_df['date']).date()} to {max(filtered_df['date']).date()})",
                                        xaxis_rangeslider_visible=False,
                                        height=600,
                                        width=900,
                                        showlegend=False,
                                        yaxis=dict(title="Price"),
                                        yaxis2=dict(title="Volume"),
                                        xaxis2_title="Date"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Add data table with collapsed view
                                    with st.expander("View Data Table"):
                                        st.dataframe(
                                            filtered_df[['date', 'open', 'high', 'low', 'close', 'volume']]
                                            .sort_values('date', ascending=False)
                                        )
                                        
                                        # Add download button for the data
                                        csv = filtered_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV",
                                            data=csv,
                                            file_name=f"{selected_symbol}_{selected_timeframe}_{start_date}_to_{end_date}.csv",
                                            mime="text/csv"
                                        )
                                else:
                                    st.warning(f"No data available for the selected date range ({start_date} to {end_date}).")
                            else:
                                missing_cols = [col for col in required_columns if col not in df.columns]
                                st.error(f"Data file is missing required columns: {', '.join(missing_cols)}")
                        except Exception as e:
                            st.error(f"Error loading data: {e}")
                    else:
                        st.error(f"Data file not found: {data_file}")
                else:
                    st.warning(f"No data available for {selected_symbol}")
            else:
                st.info("No symbols with data found. Use the form above to download market data.")
        else:
            st.info("No market data available yet. Use the form above to download data for your saved contracts.")
    
    # Message for unverified connection
    else:
        st.warning("Please verify your connection to Interactive Brokers using the sidebar button to access market data features.")

# Footer
st.markdown("---")
st.markdown("### Usage Guide")
if page == "Contracts":
    st.markdown("""
    1. **Verify Connection**: Enter your connection details in the sidebar and click "Verify Connection"
    2. **Search for Contracts**: Select an exchange, enter search patterns (use * as wildcard), and click "Search Contracts"
    3. **Save Contracts**: Select contracts from the search results and click "Save Selected Contracts"
    4. **Review Saved Contracts**: View your saved contracts in the "Saved Contracts" section
    """)
elif page == "Market Data":
    st.markdown("""
    1. **Verify Connection**: Enter your connection details in the sidebar and click "Verify Connection"
    2. **Market Data Features**: Coming soon!
    """)

# Add some additional information in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("""
**IB Contract Explorer** is a tool to search and save Interactive Brokers contracts.

This application requires:
- Interactive Brokers Trader Workstation (TWS) or IB Gateway running
- API connections enabled
- Appropriate market data subscriptions
""")