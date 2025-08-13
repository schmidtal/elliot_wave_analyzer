
def get_usa():
    symbols = [
    'MMM', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL',
    'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AME', 'AMGN', 'APH', 'ADI', 'AON', 'APO', 'AAPL',
    'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'T', 'ATO', 'ADSK', 'ADP', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX',
    'BBY', 'BIIB', 'BX', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'BRO', 'BLDR', 'BXP', 'CHRW', 'CDNS', 'CPT',
    'COF', 'CAH', 'CCL', 'CARR', 'CAT', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB',
    'CHD', 'CI', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO',
    'CPRT', 'GLW', 'CTVA', 'CSGP', 'COST', 'CRWD', 'CCI', 'CSX', 'CVS', 'DHR', 'DRI', 'DE', 'DELL', 'DAL',
    'DVN', 'DXCM', 'FANG', 'DLR', 'DG', 'D', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'ETN', 'EBAY', 'ECL', 'EIX',
    'EW', 'EA', 'ELV', 'EMR', 'ETR', 'EOG', 'EQT', 'EFX', 'EQR', 'EL', 'ES', 'EXC', 'EXE',
    'EXR', 'XOM', 'FICO', 'FAST', 'FDX', 'FIS', 'FITB', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX',
    'FCX', 'GRMN', 'GE', 'GEHC', 'GEV', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GDDY', 'GS', 'HAL', 'HIG',
    'HCA', 'DOC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM', 'HBAN', 'IBM',
     'ITW', 'INCY', 'IR', 'INTC', 'ICE', 'IFF', 'IP', 'INTU', 'ISRG', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL',
     'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KDP', 'KEY', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LHX', 'LRCX',
    'LVS', 'LDOS', 'LEN', 'LLY', 'LIN', 'LYV', 'LMT', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MAR', 'MMC', 'MAS',
    'MA', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MCHP', 'MU', 'MSFT', 'MRNA', 'TAP', 'MDLZ',
    'MNST', 'MS', 'MOS', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NEE', 'NKE', 'NI', 'NSC', 'NTRS',
    'NOC', 'NRG', 'NUE', 'NVDA', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PLTR', 'PANW',
    'PAYX', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD',
    'PRU', 'PEG', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL',
    'ROST', 'RCL', 'SPGI', 'CRM', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX',
    'STT', 'STLD', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TER', 'TSLA', 'TXN',
     'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TRV', 'TRMB', 'TFC', 'TSN', 'USB', 'UBER', 'UDR', 'UNP', 'UAL', 'UPS',
     'UNH', 'VLO', 'VTR', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'WMT',
    'DIS', 'WBD', 'WM', 'WEC', 'WFC', 'WELL', 'WDC', 'WY', 'WSM', 'WMB', 'WDAY', 'XEL', 'XYL', 'YUM', 'ZBH', 'ZTS'
      ]
    iliqid_symbols = ['MKTX', 'NWS',  'NVR', 'TPL', 'ERIE', 'AZO', 'MTD', 'FICO', 'GWW', 'BKNG', 'TYL', 'TDG', 'TDY', 'FDS']
    return symbols

def get_usa_1h():
    elliott_good_1h = [
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'AMZN',   # Amazon
    'NVDA',   # NVIDIA
    'TSLA',   # Tesla
    'META',   # Meta (Facebook)
    'GOOG',   # Alphabet Class C
    'NFLX',   # Netflix
    'ADBE',   # Adobe
    'AMD',    # AMD
    'INTC',   # Intel
    'QCOM',   # Qualcomm
    'CSCO',   # Cisco
    'CRM',    # Salesforce
    'PYPL',   # PayPal
    'NOW',    # ServiceNow
    'ZM',     # Zoom Video (wenn in der ursprünglichen Liste, sonst ersetzen)
    'UBER',   # Uber
    'SNPS',   # Synopsys
    'LRCX',   # Lam Research
    'MU',     # Micron Technology
    'MRNA',   # Moderna
    'DOCU',   # DocuSign (wenn vorhanden)
    ]
    return elliott_good_1h

def get_symbols_sector(sector):
    Industrials = ['MMM', 'ALLE', 'AME', 'BA', 'BLDR', 'CHRW', 'CARR', 'CAT', 'CTAS', 'CPRT', 'CSX', 'DE', 'DAL', 'DOV', 'ETN', 'EMR', 'EFX', 'FAST', 'FDX', 'GE', 'GEV', 'GD', 'GPN', 'HON',
                   'HWM', 'ITW', 'IR', 'JBHT',
                   'JCI', 'LHX', 'LMT', 'MAS', 'NSC', 'NOC', 'ODFL', 'OTIS', 'PCAR', 'PNR', 'PWR', 'RTX', 'RSG', 'ROK', 'LUV', 'SWK', 'TXT', 'TT', 'UNP', 'UAL', 'UPS', 'VLTO', 'WM', 'XYL']
    
    Healthcare = ['ABT', 'ABBV', 'A', 'ALGN', 'AMGN', 'BAX', 'BDX', 'BIIB', 'BSX', 'BMY', 'CAH', 'COR', 'CNC', 'CI', 'COO', 'CVS', 'DHR', 'DXCM', 'EW', 'ELV', 'GEHC', 'GILD', 'HCA', 'HOLX',
                  'HUM', 'INCY', 'ISRG', 'IQV', 'JNJ', 'LLY', 'MCK', 'MDT', 'MRK', 'MRNA', 'PFE', 'DGX', 'REGN', 'RMD', 'RVTY', 'SOLV', 'SYK', 'TMO', 'UNH', 'VRTX', 'VTRS', 'ZBH', 'ZTS']

    Technology = ['ACN', 'ADBE', 'AMD', 'AKAM', 'APH', 'ADI', 'AAPL', 'AMAT', 'ANET', 'ADSK', 'ADP', 'CDNS', 'CDW', 'CSCO', 'CTSH', 'GLW', 'CRWD', 'DELL', 'FICO', 'FIS', 'FI', 'FTNT', 'FTV', 'GRMN', 'GDDY',
                  'HPE', 'HPQ', 'IBM', 'INTC', 'INTU', 'JBL', 'JNPR', 'KLAC', 'LRCX', 'LDOS',
                  'MCHP', 'MU', 'MSFT', 'NTAP', 'NVDA', 'NXPI', 'ON', 'ORCL', 'PLTR', 'PANW', 'PAYX', 'QCOM', 'CRM', 'STX', 'NOW', 'SWKS', 'SMCI', 'SNPS', 'TEL', 'TER', 'TXN', 'TRMB', 'UBER', 'WDC', 'WDAY']

    Financial_Services = ['AFL', 'ALL', 'AXP', 'AIG', 'AON', 'APO', 'ACGL', 'AJG', 'BAC', 'BX', 'BK', 'BRO', 'COF', 'SCHW', 'CB', 'C', 'CFG', 'CME', 'FITB', 'GS', 'HIG', 'HBAN', 'ICE', 'JPM', 'KEY', 'KKR',
                          'MTB', 'MMC', 'MA', 'MS', 'NDAQ', 'NTRS', 'PYPL', 'PNC', 'PFG', 'PGR', 'PRU', 'RJF', 'RF', 'SPGI', 'STT', 'SYF', 'TROW', 'TRV', 'TFC', 'USB', 'V', 'WRB', 'WFC']

    Basic_Materials = ['APD', 'CF', 'CTVA', 'DOW', 'DD', 'ECL', 'FCX', 'IFF', 'LIN', 'LYB', 'MOS', 'NEM', 'NUE', 'PPG', 'SHW', 'STLD', 'VMC']

    Consumer_Cyclical = ['ABNB', 'AMZN', 'AMCR', 'APTV', 'BALL', 'BBY', 'BKNG', 'CCL', 'CMG', 'DRI', 'DASH', 'DHI', 'EBAY', 'F', 'GM', 'GPC', 'HLT', 'HD', 'IP', 'LVS', 'LEN', 'LOW', 'LULU', 'MAR', 'MCD',
                         'NKE', 'ORLY', 'PHM', 'RL', 'ROL', 'ROST', 'RCL', 'SW', 'SBUX', 'TPR', 'TSLA', 'TJX', 'TSCO', 'WSM', 'YUM']

    Real_Estate = ['ARE', 'AMT', 'BXP', 'CPT', 'CBRE', 'CSGP', 'CCI', 'DLR', 'EQR', 'EXR', 'DOC', 'HST', 'INVH', 'IRM', 'KIM', 'PLD', 'O', 'REG', 'SPG', 'UDR', 'VTR', 'VICI', 'WELL', 'WY']

    Utilities = ['LNT', 'AEE', 'AEP', 'AWK', 'ATO', 'CNP', 'CMS', 'ED', 'CEG', 'D', 'DTE', 'DUK', 'EIX', 'ETR', 'ES', 'EXC', 'FE', 'NEE', 'NI', 'NRG', 'PCG', 'PNW', 'PPL', 'PEG', 'SRE', 'SO', 'VST', 'WEC',
                 'XEL']

    Communication_Services = ['GOOGL', 'GOOG', 'T', 'CHTR', 'CMCSA', 'EA', 'FOXA', 'FOX', 'LYV', 'META', 'NFLX', 'NWSA', 'OMC', 'TMUS', 'TTWO', 'TKO', 'VZ', 'DIS', 'WBD']

    Consumer_Defensive = ['MO', 'ADM', 'BG', 'CHD', 'CLX', 'KO', 'CL', 'CAG', 'STZ', 'COST', 'DG', 'EL', 'GIS', 'HSY', 'HRL', 'K', 'KVUE', 'KDP', 'KMB', 'KHC', 'KR', 'MKC',
                          'TAP', 'MDLZ', 'MNST', 'PEP', 'PM', 'PG', 'SJM', 'SYY', 'TGT', 'TSN', 'WMT']

    Energy = ['BKR', 'CVX', 'COP', 'DVN', 'FANG', 'EOG', 'EQT', 'EXE', 'XOM', 'HAL', 'HES', 'KMI', 'MPC', 'OXY', 'OKE', 'PSX', 'SLB', 'TRGP', 'VLO', 'WMB']

    sectors = ['Industrials', 'Healthcare','Technology', 'Financial_Services', 'Basic_Materials', 'Consumer_Cyclical', 'Real_Estate', 'Utilities', 'Communication_Services', 'Consumer_Defensive',
               'Energy']
    if sector == 'Industrials' :
        return Industrials
    elif sector == 'Healthcare' :
        return Healthcare
    elif sector == 'Technology' :
        return Technology
    else :
        return []
    

def get_forex():
    forex_symbols = [
    "EURUSD=X", 
    "USDJPY=X", 
    "GBPUSD=X",
    "USDCHF=X",# für Elliott nicht geeignet. ZB von CHF interveniert zu oft
    "AUDUSD=X", 
    "USDCAD=X", 
    "NZDUSD=X",# kehrt oft im letzten 3tel um noch zu machen
    "EURJPY=X", 
    "EURGBP=X", 
    "EURCHF=X",# nicht geeignet
    "EURCAD=X",
    "EURAUD=X", 
    "EURNZD=X", 
    "GBPJPY=X", 
    "GBPCHF=X", 
    "GBPCAD=X", 
    "GBPAUD=X", 
    "GBPNZD=X",
    "AUDJPY=X", 
    "AUDCHF=X", 
    "AUDCAD=X",
    "AUDNZD=X",
    "CADJPY=X",
    "CADCHF=X",
    #"CHFJPY=X",
    "NZDJPY=X",
     # "NZDCHF=X",
    "NZDCAD=X", #Neuseländischer Doller sollte noch separat betrachtet werden
    "GC=F",   # Gold Futures
    #"SI=F",   # Silver Futures
    #"PL=F",   # Platinum Futures
    #"HG=F",   # Copper Futures
    #"PA=F" ,   # Palladium Futures
    "CL=F",   # Crude Oil WTI
    #"BZ=F",   # Brent Crude Oil
    #"NG=F",   # Natural Gas
    #"HO=F",   # Heating Oil
    #"RB=F",   # RBOB Gasoline

    # Landwirtschaft
    #"ZC=F",   # Corn
    #"ZS=F",   # Soybeans
    #"ZW=F",   # Wheat
    #"KC=F",   # Coffee
    #"SB=F",   # Sugar
    "CC=F"   # Cocoa
    #"CT=F",   # Cotton
    #"OJ=F",   # Orange Juice
    #"LE=F",   # Live Cattle
    #"HE=F",   # Lean Hogs
    #"GF=F"    # Feeder Cattle
 ]
    return forex_symbols


def get_indices():
    indices = [
    "^GSPC",      # S&P 500 (USA)
    "^DJI",       # Dow Jones Industrial Average (USA)
    "^NDX",       # Nasdaq 100 (USA)
    "000001.SS",  # SSE Composite (China)
    "000300.SS",  # CSI 300 (China)
    "^N225",      # Nikkei 225 (Japan)
    "^GDAXI",     # DAX (Deutschland)
    "^FTSE",      # FTSE 100 (Großbritannien)
    "^FCHI",      # CAC 40 (Frankreich)
    "^NSEI",      # Nifty 50 (Indien)
    "^BSESN",     # BSE Sensex (Indien)
    "^BVSP",      # Bovespa (Brasilien)
    "^GSPTSE",    # S&P/TSX Composite (Kanada)
    "^AXJO",      # ASX 200 (Australien)
    "^KS11"      # KOSPI (Südkorea)
    ]
    return indices


    
def get_krypto():
    crypto_symbols = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "BNB-USD",  # Binance Coin
    "SOL-USD",  # Solana
    "XRP-USD",  # Ripple
    "ADA-USD",  # Cardano
    "DOGE-USD", # Dogecoin
    "AVAX-USD", # Avalanche
    "DOT-USD",  # Polkadot
    "TRX-USD",  # TRON
    "LINK-USD", # Chainlink
    "LTC-USD",  # Litecoin
    "BCH-USD",  # Bitcoin Cash
    "XLM-USD",  # Stellar
    "ATOM-USD", # Cosmos
    "ETC-USD",  # Ethereum Classic
    "NEAR-USD", # NEAR Protocol
    "HBAR-USD", # Hedera
    "ICP-USD"   # Internet Computer 
    ]
    return crypto_symbols

def get_nasdaq():
    symbols = ['^NDX',
               'ADBE', 'AMD', 'ABNB', 'GOOG', 'AMZN', 'AEP', 'AMGN', 'ADI', 'ANSS', 'AAPL', 'AMAT', 'APP', 'ARM', 'ASML', 'AZN', 'TEAM', 'ADSK', 'ADP', 'AXON', 'BKR', 'BIIB',
                'CDNS', 'CDW', 'CHTR', 'CTAS', 'CSCO', 'CCEP', 'CTSH', 'CMCSA', 'CEG', 'CPRT', 'CSGP', 'COST', 'CRWD', 'CSX', 'DDOG', 'DXCM', 'FANG', 'DASH', 'EA', 'EXC',
               'FAST', 'FTNT', 'GEHC', 'GILD', 'GFS', 'HON', 'IDXX', 'INTC', 'INTU', 'ISRG', 'KDP', 'KLAC', 'KHC', 'LRCX', 'LIN', 'LULU', 'MAR', 'MRVL', 'MELI', 'META', 'MCHP', 'MU', 'MSFT',
               'MSTR', 'MDLZ', 'MDB', 'MNST', 'NFLX', 'NVDA', 'NXPI', 'ORLY', 'ODFL', 'ON', 'PCAR', 'PLTR',
               'PANW', 'PAYX', 'PYPL', 'PDD', 'PEP', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SNPS', 'TTWO', 'TMUS', 'TSLA', 'TXN', 'TTD', 'VRTX', 'WBD', 'WDAY', 'XEL', 'ZS']
    return symbols

def get_dow_jones():
    symbols = ['MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'NVDA', 'PG', 'CRM',
               'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT']
    return symbols

def get_dax():
    dax_tickers = [
    "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "CON.DE", "1COV.DE", "DHER.DE",
    "DBK.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "FME.DE", "HEI.DE", "HEN3.DE", "IFX.DE", "LIN.DE",
    "MRK.DE", "MBG.DE", "MTX.DE", "MUV2.DE", "PUM.DE", "QIA.DE", "RWE.DE", "SAP.DE", "SRT3.DE",
    "SIE.DE", "SHL.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "HNR1.DE", "DWNI.DE", "ENR.DE", "BNR.DE",
    "RHM.DE", "NEM.DE", "LEG.DE", "PAH3.DE"]
    return dax_tickers

def get_mdax():
    symbols = ['BOSS.DE', 'LEG.DE', 'GIL.DE', 'GBF.DE', 'DWNI.DE', 'AOX.DE', 'HOT.DE', 'AIR.DE', 'KU2.DE','HLE.DE', 'LXS.DE', 'HNR1.DE',
               'BNR.DE', 'KRN.DE', 'FIE.DE', 'KGX.DE', 'ARL.DE', 'GXI.DE', 'EVK.DE', 'FPE3.DE', '1COV.DE', 'MEO.DE', 'G1A.DE',
               'LEO.DE', 'DEQ.DE', 'JUN3.DE', 'EVD.DE', 'DUE.DE', 'FRA.DE', 'MTX.DE']
    return symbols


def get_cac40( ):
    cac40 = [
    "AI.PA", "AIR.PA", "ALO.PA", "OR.PA", "MC.PA", "ML.PA", "BN.PA", "CAP.PA", "CA.PA", "ACA.PA",
    "BNP.PA", "ENGI.PA", "EN.PA", "KER.PA", "RI.PA", "SAF.PA", "SAN.PA", "SGO.PA", "SU.PA", "GLE.PA",
    "STLAP.PA", "SW.PA", "TTE.PA", "VIE.PA", "VIV.PA", "DG.PA", "PUB.PA", "FR.PA", "HO.PA",
    "EL.PA", "RMS.PA", "URW.PA", "WLN.PA", "ALV.PA", "CS.PA", "RNO.PA"]

    return cac40

def get_neatherlands():
    netherlands_tickers = [
    "ASML.AS", "ADYEN.AS", "PHIA.AS", "AKZA.AS", "MT.AS", "KPN.AS", "RAND.AS", "IMCD.AS",
    "NN.AS", "AGN.AS", "TKWY.AS", "WKL.AS", "BESI.AS", "HEIA.AS", "FUR.AS", "VPK.AS", "PRX.AS"]

    return netherlands_tickers


def get_london():
    lse_tickers = [
    "AZN.L", "HSBA.L", "SHEL.L", "ULVR.L", "GSK.L", "BATS.L", "BP.L", "DGE.L", "RIO.L", "GLEN.L",
    "REL.L", "AAL.L", "LSEG.L", "NG.L", "BHP.L", "BARC.L", "STAN.L", "LLOY.L", "VOD.L"]
    return lse_tickers

def get_schweiz():
    switzerland_tickers = [
    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "CFR.SW", "SGSN.SW", "ADEN.SW",
    "SREN.SW", "GIVN.SW", "UHR.SW", "ALC.SW", "LOGN.SW", "DOKA.SW", "BAER.SW"]
    return switzerland_tickers

 
def get_staff():
    staff = [ "CL=F", "^GSPC", "^NDX", "^DJI", "GC=F", "EURUSD=X", "EURJPY=X", "EURCHF=X", "EURGBP=X"]
    return staff

def get_europa():
    europians = get_dax()+ get_cac40() + get_neatherlands() + get_london() + get_schweiz()+get_staff()
    return europians

