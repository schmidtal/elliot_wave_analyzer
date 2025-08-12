from dataclasses import dataclass, replace
import copy

@dataclass(frozen=True)
class CoreValues:
    TIINGO_API_TOKEN = '9cb64b1ee6217683e3db3a1995f53a0745f0c156'
    MIN_FIBO_RETRACE = 0.50
    MAX_FIBO_RETRACE = 0.8
    CORRECTION_WAVE_1: float = 0.22
    CORRECTION_WAVE_2: float = 0.3
    INFLUENCE_OF_INDEX: float = 0.3
    CORRECTION_SECOND_RUN: float = 0.25
    RATIO_3_WAVES: float = 0.2
    OVERLAP_FACTOR: float = 0.005
    MIND_CRV: float = 1.0
    MIND_MOVEMENT: float = 0.007
    TARGET_FIBO_FACTOR: float = 1.0
    IMPULS_RATIO_CORRECTION: float = 1
    LOOKBACK: int = 6
    MAX_TRADE_DURATION: int = 160
    TOL_CORR_CORR_LEN: float = 0.2 # Toleranze Correspondierene Correcture in der Länge
    MIND_DURATION = 6


    BT_STOP_LOSS: float = 0.0
    

    
    MIN_PRICE_RATIO: float = 0.33
    MAX_PRICE_RATIO: float = 0.8
    MIN_EXTREMES_FOR_IMPULSE: int = 4
    MAX_CORRECTION_SIMILARITY: float = 0.3

# Original-Konfiguration (NICHT ändern!)
DEFAULT_CoreValues = CoreValues()

# Funktion zum Erzeugen einer veränderbaren Kopie
def get_config_copy() -> CoreValues:
    return copy.deepcopy(DEFAULT_CoreValues)
