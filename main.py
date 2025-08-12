import symbols
from score_based_analyzer import ScoreBased
from modifid_elliott import TickerData, PatternAnalyzer, PatternResult
import pandas as pd
import os
import plots
import modifid_elliott
import time
import datetime
import platform
#from playsound import playsound

def beep():
    """Spielt einen einfachen Ton je nach Betriebssystem."""
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  # Frequenz, Dauer
    else:
        # Mac/Linux: system bell
        print('\a')  # Terminal-Beep – funktioniert nicht immer
        try:
            os.system('play -nq -t alsa synth 0.5 sine 1000')  # Optional: für Linux mit SoX
        except:
            pass

if __name__ == "__main__":
    period = "30d"
    interval = "15m"

    while True:
        print(f"Starte Analyse: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        tickers = symbols.get_usa()  # hole Liste der Ticker

        for ticker in tickers:
            ticker_data = TickerData(ticker, period, interval)
            analyzer = modifid_elliott.PatternAnalyzer(ticker_data.df)

            if analyzer.geht():
                print("⚠️  Take a look:", ticker)
                #plots.plot_modified(analyzer.result, ticker_data)
                beep()  # Spiele Ton ab

        print("✅ Durchlauf abgeschlossen. Warte 15 Minuten...\n")
        time.sleep(15 * 60)  # 15 Minuten warten
