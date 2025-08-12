import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from mplfinance.original_flavor import candlestick_ohlc
from CoreValues import CoreValues
import warnings
import math
from typing import Tuple


@dataclass
class Extreme:
    is_valid: bool
    idx: int = None
    price: float = None
    kind: str = None  # Maximum oder Minimum


@dataclass
class PatternResult:
    is_valid: bool
    extreme_2: Extreme = None
    extreme_1: Extreme = None
    present_extreme: Extreme = None
    strategy: str = None
    intermediate_extremes_wave_1: list[Extreme] = None
    is_wave1_impuls: bool = False
    intermediate_extremes_wave_2 : list[Extreme] = None
    is_wave2_impuls: bool = True
    entry: Extreme = None
    retracement: float = -1
    angle_ratio: float = -1
    avarage_volume_wave_1 : float = 0
    avarage_volume_wave_2 : float = 0
    tunnel_width_wave_1: float = 0
    tunnel_width_wave_2: float = 0
    r_privious_ratio: float = 0



class TickerData:
    def __init__(self, symbol: str, period: str = "2y", interval: str = "1d"):
        self.symbol = symbol
        self.df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust = False )

        if self.df.empty:
            return

        # Falls MultiIndex (z. B. bei mehreren Ticker), extrahiere den ersten
        if isinstance(self.df.columns, pd.MultiIndex):
            if isinstance(symbol, list):
                symbol = symbol[0]
            self.df = self.df.xs(symbol, axis=1, level=1)

        self.df.reset_index(inplace=True)

        # Zeitspalte prüfen/umbenennen
        if 'Datetime' in self.df.columns:
            self.df.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'Date' not in self.df.columns:
            raise ValueError("❌ Keine gültige Zeitspalte (Date/Datetime) gefunden.")

        # Erwartete Spalten
        base_cols = {'Date', 'Open', 'High', 'Low', 'Close'}
        optional_cols = {'Volume'}
        available_cols = set(self.df.columns)

        # Fehlt Volume? -> keine Warnung oder Fehler, einfach ignorieren
        used_cols = list(base_cols | (optional_cols & available_cols))

        self.df = self.df[used_cols]
        if 'Volume' not in self.df.columns:
            self.df['Volume'] = pd.NA  # oder 0, falls gewünscht

        self.df['Index'] = range(len(self.df))

    @classmethod
    def from_csv(cls, filepath: str, symbol: str = "unknown"):
        df = pd.read_csv(filepath)

        required_cols = {'Open', 'High', 'Low', 'Close'}
        optional_cols = {'Volume'}
        all_possible = required_cols | optional_cols

        available_cols = set(df.columns)
        if not required_cols.issubset(available_cols):
            raise ValueError(f"❌ CSV fehlt Spalten: {required_cols - available_cols}")

        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in optional_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = pd.NA  # oder 0, je nach Bedarf

        df.dropna(subset=required_cols, inplace=True)

        # Datum verarbeiten
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            raise ValueError("❌ CSV enthält keine Zeitspalte (Date oder Datetime)")

        df.dropna(subset=['Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['Index'] = df.index

        obj = cls.__new__(cls)
        obj.symbol = symbol
        obj.df = df
        return obj



    
class PatternAnalyzer:
    def __init__(self, df, end_idx = None ):
        self.df = df if isinstance(df, pd.DataFrame) else df.df
        self.end_idx = end_idx if end_idx is not None else len(self.df)

        self.df = self.df.iloc[:self.end_idx]

        self.x_factor = 0.0
        self.result = self.find_last_suitable_pattern()
        if self.result.is_valid :
            self.result.tunnel_width_wave_1, e1, e2= self.get_total_perpendicular_deviation(self.result.extreme_2,
                                                                                            self.result.extreme_1,
                                                                                            self.result.strategy
                                                                                              )
            self.result.tunnel_width_wave_2, extreme_pos, extreme_neg = self.get_total_perpendicular_deviation(self.result.extreme_1,
                                                                                                               self.result.present_extreme,
                                                                                                               'long' if self.result.strategy == 'short' else 'short'
                                                                                                            )
            
            #if self.result.strategy == 'long':
                #self.result.entry = extreme_neg
            #else:
                #self.result.entry = extreme_pos
            self.result.avarage_volume_wave_1 = self.get_strategy_adjusted_average_volume(self.result.extreme_2, self.result.extreme_1, self.result.strategy )
            self.result.avarage_volume_wave_2 = self.get_strategy_adjusted_average_volume(self.result.extreme_1, self.result.present_extreme, 'long' if self.result.strategy == 'short' else 'short' )
        

    def geht(self):
        if not self.result.is_valid  :
            return False
        
        #if  self.result.r_privious_ratio > 0.5 :
        #   return False
            

        
        if abs(self.result.extreme_1.price - self.result.extreme_2.price)/ self.result.extreme_2.price < CoreValues.MIND_MOVEMENT:
            return False
        
        if (CoreValues.MIN_FIBO_RETRACE < self.result.retracement < CoreValues.MAX_FIBO_RETRACE ):
            fibo_retracement = 1
        else:
            return False
            fibo_retracement = 0

        if (self.result.angle_ratio > 1.0) :
            angle_condition = 1
        else:
            return False
            angle_condition = 0
        
        
        if self.result.avarage_volume_wave_1 == 0 or ( self.result.avarage_volume_wave_1 > 1*self.result.avarage_volume_wave_2 ):
            volume_condition = 1
        else:
            volume_condition = 0


        candels_wave_1 = self.result.extreme_1.idx - self.result.extreme_2.idx
        candels_wave_2 = self.result.present_extreme.idx - self.result.extreme_1.idx

        condition_duration = 1
        if True if candels_wave_2 == 0 else candels_wave_1/candels_wave_2 <= 0.2:
            condition_duration = 0 

        impuls_factor_wave_2 = self.calculate_impulsfaktor(self.result.extreme_1.idx,
                                                            self.result.present_extreme.idx, 
                                                            'short' if self.result.strategy== 'long' else 'long')
        
        if 0.8 > impuls_factor_wave_2 > 0.3:
            impuls_condition_wave_2 = 1
        else:
            impuls_condition_wave_2 = 0

        impuls_factor = self.calculate_impulsfaktor(self.result.extreme_2.idx, self.result.extreme_1.idx, self.result.strategy)
        if impuls_factor > 0.5:
            impuls_condition = 1
        else:
            impuls_condition = 0



        if (self.result.tunnel_width_wave_1 < 0.6*abs(self.result.extreme_2.price - self.result.extreme_1.price)):
            tunnel_wave_1_condition = 1
        else:
            tunnel_wave_1_condition = 0

        if  (self.result.tunnel_width_wave_2 > 0.4*abs(self.result.extreme_1.price - self.result.present_extreme.price)):
            tunnel_wave_2_condition = 1
        else:
            tunnel_wave_2_condition = 0

        if ( self.result.tunnel_width_wave_1 < 0.9*self.result.tunnel_width_wave_2 ):
            tunnels_condition = 1
        else:
            tunnels_condition = 0


        flag = (
            #0.1*(1.0 if angle_degrees_condition else 0.0) +
            #0.1*condition_duration +
            0.1*impuls_condition_wave_2 +
            0.1*impuls_condition +
            0.2*volume_condition +           
            0.2*fibo_retracement +
            0.2*angle_condition +
            0.1*tunnel_wave_1_condition# +
            #0.1*tunnel_wave_2_condition #+
            #0.1*tunnels_condition
            ) >= 0.9
        if flag:
            #print("start", self.result.extreme_2.idx, self.result.extreme_1.idx )
            self.result.intermediate_extremes_wave_1 = self.find_inner_structure( self.result.extreme_2.idx, self.result.extreme_1.idx, self.result.strategy )
            self.result.intermediate_extremes_wave_2 = self.find_inner_structure( self.result.extreme_1.idx, self.result.present_extreme.idx,
                                                                                 'long' if self.result.strategy == 'short' else 'short')
            self.result.entry = self.result.intermediate_extremes_wave_2[-2]
            #print("end", self.result.extreme_2.idx, self.result.extreme_1.idx )
            if len(self.result.intermediate_extremes_wave_1) == 4 or len(self.result.intermediate_extremes_wave_1) > 6 or len(self.result.intermediate_extremes_wave_2)==2:
                #print("bin hier")
                flag = False
            elif len(self.result.intermediate_extremes_wave_1) == 6:
                tol = CoreValues.OVERLAP_FACTOR*self.result.extreme_2.price
                k = ( tol + self.result.intermediate_extremes_wave_1[1].price > self.result.intermediate_extremes_wave_1[4].price if self.result.strategy == 'long' else 
                        self.result.intermediate_extremes_wave_1[1].price < self.result.intermediate_extremes_wave_1[4].price + tol )
                if k:
                    flag = False
        return flag

    def compute_relative_angle(self, start: Extreme, end: Extreme ):
        dx = end.idx - start.idx
        if dx == 0:
            return 90.0 if end.price > start.price else -90.0
        percent_move = (end.price - start.price) / start.price
        slope = percent_move / dx  # % pro Kerze
        angle_deg = math.degrees(math.atan(slope))
        return angle_deg


    def calculate_impulsfaktor(self, idx_start: int, idx_end: int, strategy: str = "long") -> float:
        """
        Berechnet den Impulsfaktor (Anteil richtungskonformer Kerzen) für eine Welle – 
        ignoriert erste/letzte Kerze, wenn sie nicht zur Strategie passen.
        """
        if idx_end <= idx_start:
            return 0.0

        df_view = self.df.iloc[idx_start:idx_end].copy()

        if df_view.empty:
            return 0.0

        # Strategieabhängige Filterung der ersten/letzten Zeile
        if strategy == "long":
            if df_view.iloc[0]['Close'] <= df_view.iloc[0]['Open']:
                df_view = df_view.iloc[1:]
            if not df_view.empty and df_view.iloc[-1]['Close'] <= df_view.iloc[-1]['Open']:
                df_view = df_view.iloc[:-1]
        elif strategy == "short":
            if df_view.iloc[0]['Close'] >= df_view.iloc[0]['Open']:
                df_view = df_view.iloc[1:]
            if not df_view.empty and df_view.iloc[-1]['Close'] >= df_view.iloc[-1]['Open']:
                df_view = df_view.iloc[:-1]
        else:
            raise ValueError(f"❌ Unbekannte Strategie: {strategy}")

        if df_view.empty:
            return 0.0

        close = df_view["Close"].values
        open_ = df_view["Open"].values
        valid_mask = ~np.isnan(close) & ~np.isnan(open_)
        close = close[valid_mask]
        open_ = open_[valid_mask]

        if len(close) == 0:
            return 0.0

        if strategy == "long":
            impuls_kerzen = np.sum(close > open_)
        else:  # short
            impuls_kerzen = np.sum(close < open_)

        return impuls_kerzen / len(close)


    def _find_same_price_in_past(self, price, start_idx=None):
        df = self.df

        if start_idx is None:
            start_idx = len(df)
        if start_idx < 2:
            return False, None

        # Slice ohne copy
        lows = df['Low'].values[:start_idx]
        highs = df['High'].values[:start_idx]

        lows_prev = np.roll(lows, 1)
        highs_prev = np.roll(highs, 1)

        # Setze ersten Eintrag (der keine gültige "prev"-Kerze hat) auf NaN oder Inf, damit Vergleich fehlschlägt
        lows_prev[0] = np.nan
        highs_prev[0] = np.nan

        # Bedingungen als NumPy-Arrays
        cond1 = (lows <= price) & (price <= highs)
        low_min = np.fmin(lows, lows_prev)
        high_max = np.fmax(highs, highs_prev)
        cond2 = (low_min <= price) & (price <= high_max)

        combined = cond1 | cond2
        matches = np.where(combined)[0]

        if matches.size == 0:
            return False, None

        return True, int(matches[-1])


    def _calculate_intersection(self, price_a, idx_a, price_b, idx_b, target_price):
        return (-target_price + price_a) * (idx_b - idx_a) / (price_b - price_a)

    def get_last_extrema_indices(self):
        lookback = CoreValues.LOOKBACK
        if len(self.df) < lookback:
            return -1, 0

        highs = self.df['High'].values[-lookback:]
        lows = self.df['Low'].values[-lookback:]

        high_idx = np.argmax(highs)
        low_idx = np.argmin(lows)

        # Rückgabe soll auf gesamten df bezogen sein → korrigieren
        offset = len(self.df) - lookback
        return high_idx + offset, low_idx + offset


    def find_present_extrema(self):
        idx_max, idx_min = self.get_last_extrema_indices( )
        if idx_max == -1:
            return Extreme(False)

        found_max, idx_same_max = self._find_same_price_in_past(self.df.loc[idx_max, 'High'], idx_max)
        found_min, idx_same_min = self._find_same_price_in_past(self.df.loc[idx_min, 'Low'], idx_min)

        if not (found_max and found_min) or idx_same_max == idx_same_min:
            if found_max and found_min:
                idx_max = idx_same_max
                idx_min = idx_same_min
                found_max, idx_same_max = self._find_same_price_in_past(self.df.loc[idx_max, 'High'], idx_same_max)
                found_min, idx_same_min = self._find_same_price_in_past(self.df.loc[idx_min, 'Low'], idx_same_max)
                if not (found_max and found_min) or idx_same_max == idx_same_min:
                    if idx_same_max and idx_same_min:
                        print("toDo: nach zweitem Durchlauf gleicher Index:", idx_same_max, idx_max, idx_min)
                    return Extreme(False)
            else :
                return Extreme(False)
        
        if idx_same_max > idx_same_min:
            return Extreme(True, idx_min, self.df.loc[idx_min, 'Low'], 'min')
        else:
            return Extreme(True, idx_max, self.df.loc[idx_max, 'High'], 'max')

    def find_last_suitable_pattern(self):
        if len(self.df) < 20:
            return PatternResult(False)

        present_extreme = self.find_present_extrema()
        if not present_extreme.is_valid:
            return PatternResult(False)

        highs = self.df['High'].values
        lows = self.df['Low'].values

        idx_present = present_extreme.idx
        current_price = present_extreme.price
        price_type = present_extreme.kind

        found1, idx_same_price_as_current_extreme = self._find_same_price_in_past(current_price, idx_present)
        if not found1 or idx_same_price_as_current_extreme >= idx_present:
            return PatternResult(False)

        segment1 = slice(idx_same_price_as_current_extreme, idx_present + 1)
        rel_idx_high = np.argmax(highs[segment1])
        rel_idx_low = np.argmin(lows[segment1])
        idx_high = idx_same_price_as_current_extreme + rel_idx_high
        idx_low = idx_same_price_as_current_extreme + rel_idx_low

        dist_high = idx_present - idx_high
        dist_low = idx_present - idx_low

        if dist_high >= dist_low:
            idx_extreme_1 = idx_high
            price_extreme_1 = highs[idx_extreme_1]
            direction = 'max'
        else:
            idx_extreme_1 = idx_low
            price_extreme_1 = lows[idx_extreme_1]
            direction = 'min'

        if idx_extreme_1 <= idx_same_price_as_current_extreme:
            return PatternResult(False)

        found2, idx_same_price_as_extreme_1 = self._find_same_price_in_past(price_extreme_1, idx_extreme_1)
        if not found2 or idx_same_price_as_extreme_1 >= idx_extreme_1:
            return PatternResult(False)


        segment2 = slice(idx_same_price_as_extreme_1, idx_extreme_1)
        rel_idx_high_2 = np.argmax(highs[segment2])
        rel_idx_low_2 = np.argmin(lows[segment2])
        idx_high_2 = idx_same_price_as_extreme_1 + rel_idx_high_2
        idx_low_2 = idx_same_price_as_extreme_1 + rel_idx_low_2

        if direction == 'min':
            idx_extreme_2 = idx_high_2
            price_extreme_2 = highs[idx_extreme_2]
            direction2 = 'max'
        else:
            idx_extreme_2 = idx_low_2
            price_extreme_2 = lows[idx_extreme_2]
            direction2 = 'min'
        
        if idx_present - idx_extreme_2 < CoreValues.MIND_DURATION:
            return PatternResult(False)

        if idx_extreme_1 < idx_extreme_2:
            print("hier darf ich nicht landen, x_factor negativ")
            return PatternResult(False)

        self.x_factor = abs(price_extreme_1 - price_extreme_2) / (idx_extreme_1 - idx_extreme_2)

        if idx_extreme_2 >= idx_same_price_as_current_extreme:
            return PatternResult(False)

        idx_cut = self._calculate_intersection(
            price_extreme_1, idx_extreme_1, price_extreme_2, idx_extreme_2, current_price
        )
        r_angle_ratio = (idx_present - idx_extreme_1) / idx_cut if idx_cut != 0 else float('inf')
        r_privious_ratio = (idx_extreme_2 - idx_same_price_as_extreme_1)/(idx_extreme_1 - idx_extreme_2)

        current_extrema_price = lows[idx_present] if direction == 'max' else highs[idx_present]
        price_diff_total = abs(price_extreme_1 - price_extreme_2)
        price_diff_rest = abs(current_extrema_price - price_extreme_1)

        if price_diff_total == 0:
            return PatternResult(False)

        r_retracement = price_diff_rest / price_diff_total

        extreme_1 = Extreme(True, idx_extreme_1, price_extreme_1, direction)
        extreme_2 = Extreme(True, idx_extreme_2, price_extreme_2, direction2)
        strategy = 'short' if direction2 == 'max' else 'long'

        r_pattern = PatternResult(True, extreme_2, extreme_1, present_extreme, strategy)
        r_pattern.retracement = r_retracement
        r_pattern.angle_ratio = r_angle_ratio
        r_pattern.r_privious_ratio = r_privious_ratio
        return r_pattern


    def get_total_perpendicular_deviation(
        self,
        extreme_a: Extreme,
        extreme_b: Extreme,
        strategy: str = "long",  # ➕ neue Strategieoption
    ) -> Tuple[float, Extreme, Extreme]:
        if not (extreme_a.is_valid and extreme_b.is_valid):
            return 0.0, Extreme(False), Extreme(False)

        x1, y1 = extreme_a.idx, extreme_a.price
        x2, y2 = extreme_b.idx, extreme_b.price

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            return 0.0, Extreme(False), Extreme(False)

        scale = abs(dy / dx)
        x1_scaled = x1 * scale
        x2_scaled = x2 * scale

        denominator = math.hypot(x2_scaled - x1_scaled, y2 - y1)
        if denominator == 0:
            return 0.0, Extreme(False), Extreme(False)

        idx_start, idx_end = min(x1, x2), max(x1, x2)
        df_slice = self.df.loc[idx_start:idx_end].copy()

        if df_slice.empty:
            return 0.0, Extreme(False), Extreme(False)

        # Strategie-basiertes Filtern der ersten und letzten Kerze
        if strategy == "long":
            # Entferne erste Zeile, wenn nicht bullisch
            if df_slice.iloc[0]["Close"] <= df_slice.iloc[0]["Open"]:
                df_slice = df_slice.iloc[1:]

            # Entferne letzte Zeile, wenn nicht bullisch
            if not df_slice.empty and df_slice.iloc[-1]["Close"] <= df_slice.iloc[-1]["Open"]:
                df_slice = df_slice.iloc[:-1]

        elif strategy == "short":
            if df_slice.iloc[0]["Close"] >= df_slice.iloc[0]["Open"]:
                df_slice = df_slice.iloc[1:]

            if not df_slice.empty and df_slice.iloc[-1]["Close"] >= df_slice.iloc[-1]["Open"]:
                df_slice = df_slice.iloc[:-1]

        if df_slice.empty:
            return 0.0, Extreme(False), Extreme(False)

        x0 = df_slice["Index"].to_numpy() * scale
        highs = df_slice["High"].to_numpy()
        lows = df_slice["Low"].to_numpy()

        numerator_high = ((y2 - y1) * x0 - (x2_scaled - x1_scaled) * highs + x2_scaled * y1 - y2 * x1_scaled)
        numerator_low = ((y2 - y1) * x0 - (x2_scaled - x1_scaled) * lows + x2_scaled * y1 - y2 * x1_scaled)

        distance_high = numerator_high / denominator
        distance_low = numerator_low / denominator

        max_pos_val_high = np.max(distance_high)
        max_pos_idx_high = np.argmax(distance_high)

        max_pos_val_low = np.max(distance_low)
        max_pos_idx_low = np.argmax(distance_low)

        max_neg_val_high = np.min(distance_high)
        max_neg_idx_high = np.argmin(distance_high)

        max_neg_val_low = np.min(distance_low)
        max_neg_idx_low = np.argmin(distance_low)

        # Positiver Extrempunkt
        if max_pos_val_high >= max_pos_val_low:
            row = df_slice.iloc[max_pos_idx_high]
            extreme_pos = Extreme(True, idx=int(row["Index"]), price=row["High"], kind="max")
        else:
            row = df_slice.iloc[max_pos_idx_low]
            extreme_pos = Extreme(True, idx=int(row["Index"]), price=row["Low"], kind="min")

        # Negativer Extrempunkt
        if max_neg_val_high <= max_neg_val_low:
            row = df_slice.iloc[max_neg_idx_high]
            extreme_neg = Extreme(True, idx=int(row["Index"]), price=row["High"], kind="max")
        else:
            row = df_slice.iloc[max_neg_idx_low]
            extreme_neg = Extreme(True, idx=int(row["Index"]), price=row["Low"], kind="min")

        total_deviation = abs(
            max_pos_val_high if max_pos_val_high >= max_pos_val_low else max_pos_val_low
        ) + abs(
            max_neg_val_high if max_neg_val_high <= max_neg_val_low else max_neg_val_low
        )

        return total_deviation, extreme_pos, extreme_neg


    def get_strategy_adjusted_average_volume(self, extreme_a: Extreme, extreme_b: Extreme, strategy: str) -> float:

        """
        Schnellste mögliche Berechnung des durchschnittlichen Volumens zwischen zwei Extremen,
        angepasst an die Strategie ('long' → bullische Kerzen, 'short' → bärische Kerzen),
        ohne erste/letzte Kerze wenn sie nicht zur Strategie passen.
        """
        if not (extreme_a.is_valid and extreme_b.is_valid):
            return 0.0

        idx_start = min(extreme_a.idx, extreme_b.idx)
        idx_end = max(extreme_a.idx, extreme_b.idx)

        df = self.df

        # Schneller Zugriff auf relevante Spalten als NumPy-Arrays (kein Copy nötig!)
        volume = df['Volume'].values[idx_start:idx_end + 1]
        close = df['Close'].values[idx_start:idx_end + 1]
        open_ = df['Open'].values[idx_start:idx_end + 1]

        # Nur gültige (nicht NaN) Zeilen berücksichtigen
        valid_mask = (
            ~np.isnan(volume) &
            ~np.isnan(close) &
            ~np.isnan(open_)
        )

        if not np.any(valid_mask):
            return 0.0

        volume = volume[valid_mask]
        close = close[valid_mask]
        open_ = open_[valid_mask]

        n = len(volume)
        if n == 0:
            return 0.0

        # Strategiebedingungen
        if strategy == 'long':
            condition = close > open_
        elif strategy == 'short':
            condition = close < open_
        else:
            return 0.0

        # Erste/letzte Kerze entfernen, wenn sie nicht passen
        if n >= 1 and not condition[0]:
            condition[0] = False
        if n >= 2 and not condition[-1]:
            condition[-1] = False

        # Gefiltertes Volumen
        filtered_volume = volume[condition]

        if filtered_volume.size == 0:
            return 0.0

        return filtered_volume.mean()
    

    def choose_best_extreme_score_based(self, potential_extremes, extreme, strategy):
        best_score = -float('inf')
        best_extreme = potential_extremes[-1]

        for candidate in potential_extremes:
            # Bewegungslänge (absolut)
            len_candidate = self.movement_length_extreme(candidate, extreme)

            # Preisvorteil je nach Strategie
            if strategy == 'long':
                price_diff = candidate.price - extreme.price # je höher, desto besser
            else:
                price_diff = extreme.price - candidate.price  # je niedriger, desto besser

            # Score berechnen mit z. B. 70% Länge, 30% Preis
            score = 0.3 * len_candidate + 0.7 * price_diff

            if score > best_score:
                best_score = score
                best_extreme = candidate

        return best_extreme


    def is_bullish(self, open_price, close_price) -> bool:
        return close_price > open_price

    def movement_length(self, start_move_price, start_move_idx, end_move_price, end_move_idx ):          
        return ( (start_move_price - end_move_price) ** 2 + ((end_move_idx - start_move_idx )*self.x_factor*CoreValues.INFLUENCE_OF_INDEX)**2 )**0.5

    def movement_length_extreme(self, start_extreme: Extreme, end_extreme: Extreme ) -> float:          
        return ( (start_extreme.price - end_extreme.price) ** 2 + ((end_extreme.idx - start_extreme.idx )*self.x_factor*CoreValues.INFLUENCE_OF_INDEX)**2 )**0.5
    


    def find_inner_structure(self, idx_start: int, idx_end: int, strategy: str ) -> list[Extreme]:
        if strategy == 'long' :
            movement_total = self.movement_length(self.df.loc[idx_end, 'High'], idx_end, self.df.loc[idx_start, 'Low'], idx_start )
        else:
            movement_total = self.movement_length(self.df.loc[idx_end, 'Low'], idx_end, self.df.loc[idx_start, 'High'], idx_start ) 

        extremes: list[Extreme] = []
        i = idx_start
        open_arr = self.df['Open'].values
        close_arr = self.df['Close'].values
        high_arr = self.df['High'].values
        low_arr = self.df['Low'].values
        bullish_arr = open_arr < close_arr

        while i < idx_end:
            #print(i, idx_end)
            open_price = open_arr[i]
            close_price = close_arr[i]
            high = high_arr[i]
            low = low_arr[i]
            bullish = bullish_arr[i]

            if strategy == 'long':
                if not bullish and i == idx_start:
                    i += 1
                    continue
                temp_max = high

                if bullish and i + 1 < idx_end: # nicht letzter und bullish
                    #window = self.df.iloc[i + 1:idx_end]
                    #temp_min_idx = window['Low'].idxmin()
                    temp_min_idx = np.argmax(low_arr[i+1:idx_end])+i+1
                    temp_min = low_arr[temp_min_idx]
                elif i + 1 >= idx_end: # letzter toDo wenn nicht bullish
                    temp_min_idx = idx_end
                    temp_min = low_arr[temp_min_idx]
                    if not bullish_arr[i+1]:
                        temp_max = temp_min
                else:
                    #window = self.df.iloc[i:idx_end]
                    #temp_min_idx = window['Low'].idxmin()
                    temp_min_idx = np.argmin(low_arr[i:idx_end])+i
                    temp_min = low_arr[temp_min_idx]

                movement_ratio = self.movement_length(temp_max, i, temp_min, temp_min_idx)/movement_total
                temp_extreme_min = Extreme(True, temp_min_idx, temp_min, 'min')

                if temp_max > temp_min and movement_ratio >= CoreValues.CORRECTION_WAVE_1:
                    temp_min_open = open_arr[temp_min_idx]
                    temp_min_close = close_arr[temp_min_idx]
                    temp_min_bullish = self.is_bullish(temp_min_open, temp_min_close)
                    
                    if i < temp_min_idx - 1:
                        #window_temp = self.df.iloc[i:(temp_min_idx if temp_min_bullish else temp_min_idx+1)]
                        #suitable_max_idx = window_temp['High'].idxmax()
                        k = (temp_min_idx if temp_min_bullish else temp_min_idx+1)
                        suitable_max_idx = np.argmax(high_arr[i:k]) + i
                        max_extreme_candidat = Extreme(True, suitable_max_idx, high_arr[suitable_max_idx], 'max')
                        potential_max_extremes = []
                        for count in range(i,suitable_max_idx-1, 1):
                            
                            if ( self.movement_length(high_arr[count], count, temp_min, temp_min_idx )
                                    - self.movement_length(high_arr[suitable_max_idx], suitable_max_idx, temp_min, temp_min_idx )) >= 0:
                                potential_max_extremes.append( Extreme(True, count, high_arr[count],'max') )
                        potential_max_extremes.append(max_extreme_candidat)
                        max_extreme_candidat = self.choose_best_extreme_score_based(potential_max_extremes, temp_extreme_min, strategy)
                    else:
                        max_extreme_candidat = Extreme(True, i, high_arr[i],'max')

                    extremes.append(max_extreme_candidat)
                    extremes.append(temp_extreme_min)

                    i = temp_min_idx if temp_min_bullish else temp_min_idx + 1
                else:
                    i += 1

            else:  # short
                if bullish and i == idx_start:
                    i += 1
                    continue

                temp_min = low

                if not bullish and i + 1 < idx_end: #nicht bullish und nicht letzte
                    #window = self.df.iloc[i + 1:idx_end]
                    #temp_max_idx = window['High'].idxmax()
                    temp_max_idx = np.argmax( high_arr[i+1:idx_end]) + i + 1
                    temp_max = high_arr[temp_max_idx]
                elif i + 1 >= idx_end:
                    temp_max_idx = idx_end
                    temp_max = high_arr[temp_max_idx]
                    if bullish_arr[i+1]:
                        temp_min = temp_max
                else:
                    #window = self.df.iloc[i:idx_end]
                    #temp_max_idx = window['High'].idxmax()
                    temp_max_idx = np.argmax(high_arr[i:idx_end]) + i
                    temp_max = high_arr[temp_max_idx]

                movement_ratio = self.movement_length(temp_min, i, temp_max, temp_max_idx)/movement_total
                temp_extreme_max = Extreme(True, temp_max_idx, temp_max, 'max')

                if temp_min < temp_max and movement_ratio > CoreValues.CORRECTION_WAVE_1:
                    temp_max_open = open_arr[temp_max_idx]
                    temp_max_close = close_arr[temp_max_idx]
                    temp_max_bullish = self.is_bullish(temp_max_open, temp_max_close)

                    if i < temp_max_idx - 1:
                        #window_temp = self.df.iloc[i:(temp_max_idx if not temp_max_bullish else temp_max_idx + 1)]
                        #suitable_min_idx = window_temp['Low'].idxmin()
                        k = (temp_max_idx if not temp_max_bullish else temp_max_idx + 1)
                        suitable_min_idx = np.argmax(low_arr[i:k]) + i
                        min_extreme_candidat = Extreme(True, suitable_min_idx, low_arr[suitable_min_idx], 'min')
                        potential_min_extremes = []
                        
                        for count in range(i,suitable_min_idx-1, 1):
                            if ( self.movement_length(low_arr[count], count, temp_max, temp_max_idx )
                                    - self.movement_length(low_arr[suitable_min_idx], suitable_min_idx, temp_max, temp_max_idx )) >= 0:
                                potential_min_extremes.append( Extreme(True, count, low_arr[count],'min') )
                        potential_min_extremes.append(min_extreme_candidat)
                        min_extreme_candidat = self.choose_best_extreme_score_based(potential_min_extremes, temp_extreme_max, strategy)
                    else:
                        min_extreme_candidat = Extreme( True, i, low_arr[i], 'min')


                    extremes.append(min_extreme_candidat)
                    extremes.append( temp_extreme_max )

                    i = temp_max_idx if not temp_max_bullish else temp_max_idx + 1
                else:
                    i += 1
                    
        extremes.sort(key=lambda e: e.idx)
        extremes.insert(0, self.result.extreme_2)
        extremes.append(self.result.extreme_1)

        return extremes