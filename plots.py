import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib
from matplotlib.widgets import Button
import os
from prove import Extreme, PatternResult, PatternAnalyzer

matplotlib.use('Qt5Agg')  # oder 'Qt5Agg', falls du Qt installiert hast


def plot_pattern(result, data):
    df = data.df
    if not result or not result.is_valid:
        print("❌ Kein gültiges PatternResult.")
        return

    ohlc = df[['Index', 'Open', 'High', 'Low', 'Close']].values
    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)



    # Button-Position festlegen
    ax_save = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn_save = Button(ax_save, 'Speichern', color='lightgray', hovercolor='lightblue')

    def on_save(event):
        filename = f"{data.symbol}_pattern.csv"
        path = os.path.join("saved_patterns", filename)
        os.makedirs("saved_patterns", exist_ok=True)
        try:
            data.df.to_csv(path, index=False)
            print(f"✅ Daten gespeichert unter: {path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")

    btn_save.on_clicked(on_save)


    def mark(extreme: Extreme, color, label):
        if not extreme or not extreme.is_valid:
            return
        x = df.loc[extreme.idx, 'Index']
        y = df.loc[extreme.idx, 'High'] if extreme.kind_of_extreme == 'max' else df.loc[extreme.idx, 'Low']
        ax.plot(x, y, marker='o', color=color, label=label)

    def mark_horizontal_line_from_extreme(extreme: Extreme, color='gray', label=None):
        if not extreme or not extreme.is_valid:
            return

        x_start = df.loc[extreme.idx, 'Index']
        x_end = df['Index'].iloc[-1]
        y = df.loc[extreme.idx, 'High'] if extreme.kind_of_extreme == 'max' else df.loc[extreme.idx, 'Low']

        ax.plot([x_start, x_end], [y, y], linestyle='--', color=color, label=label)


    # Haupt-Extreme markieren
    mark(result.extreme_2, 'blue', 'Extreme 2')
    mark(result.extreme_1, 'blue', 'Extreme 1')
    mark(result.present_extreme, 'blue', 'Present Extreme')
    if result.entry.is_valid :
        mark_horizontal_line_from_extreme( result.entry, 'navy', 'Entry')

    # Linie zwischen den Haupt-Extremen zeichnen
    xs = []
    ys = []
    for extreme in [result.extreme_2, result.extreme_1, result.present_extreme]:
        if extreme and extreme.is_valid:
            xs.append(df.loc[extreme.idx, 'Index'])
            y = df.loc[extreme.idx, 'High'] if extreme.kind_of_extreme == 'max' else df.loc[extreme.idx, 'Low']
            ys.append(y)
    if len(xs) == 3 and len(ys) == 3:
        ax.plot(xs, ys, color='purple', linestyle='-', linewidth=2, label='Linie')

    # ✅ Zwischenextreme aus is_path_without_overlap markieren
    if hasattr(result, 'intermediate_extremes_wave_1') and result.intermediate_extremes_wave_1:
        x_w1 = []
        y_w1 = []
        for i, ext in enumerate(result.intermediate_extremes_wave_1):
            if ext.is_valid:
                x = df.loc[ext.idx, 'Index']
                x_w1.append(x)
                y = df.loc[ext.idx, 'High'] if ext.kind_of_extreme == 'max' else df.loc[ext.idx, 'Low']
                y_w1.append(y)
                ax.plot(x, y, marker='p', color='black', markersize=6, alpha=0.7, label='Zwischenextreme' if i == 0 else "")
        ax.plot(x_w1, y_w1, color = 'blue', linestyle = '-', linewidth = 1 )

    if hasattr(result, 'intermediate_extremes_wave_2') and result.intermediate_extremes_wave_2:
        x_w1 = []
        y_w1 = []
        for i, ext in enumerate(result.intermediate_extremes_wave_2):
            if ext.is_valid:
                x = df.loc[ext.idx, 'Index']
                x_w1.append(x)
                y = df.loc[ext.idx, 'High'] if ext.kind_of_extreme == 'max' else df.loc[ext.idx, 'Low']
                y_w1.append(y)
                ax.plot(x, y, marker='p', color='darkorange', markersize=6, alpha=0.7, label='Zwischenextreme' if i == 0 else "")
        ax.plot(x_w1, y_w1, color = 'red', linestyle = '-', linewidth = 1 )

        
    ax.set_title(f"{data.symbol}")
    ax.set_xlabel("Kerzen-Index")
    ax.legend()
    plt.grid(True)

    # Event-Handler für Entf-Taste
    def on_key(event):
        if event.key == 'delete':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_mouse(event):
        if event.button == 2:  # Rechtsklick
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_mouse)

    # Vollbild aktivieren
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except AttributeError:
        try:
            manager.window.state('zoomed')  # Für TkAgg (z. B. Windows)
        except Exception:
            pass

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()


def plot_modified(result, data):
    df = data.df
    if not result or not result.is_valid:
        print("❌ Kein gültiges PatternResult.")
        return

    ohlc = df[['Index', 'Open', 'High', 'Low', 'Close']].values
    fig, ax = plt.subplots(figsize=(12, 6))
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)



    # Button-Position festlegen
    ax_save = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn_save = Button(ax_save, 'Speichern', color='lightgray', hovercolor='lightblue')

    def on_save(event):
        filename = f"{data.symbol}_pattern.csv"
        path = os.path.join("saved_patterns", filename)
        os.makedirs("saved_patterns", exist_ok=True)
        try:
            data.df.to_csv(path, index=False)
            print(f"✅ Daten gespeichert unter: {path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern: {e}")

    btn_save.on_clicked(on_save)


    def mark(extreme: Extreme, color, label):
        if not extreme or not extreme.is_valid:
            return
        x = df.loc[extreme.idx, 'Index']
        y = df.loc[extreme.idx, 'High'] if extreme.kind == 'max' else df.loc[extreme.idx, 'Low']
        ax.plot(x, y, marker='o', color=color, label=label)

    def mark_horizontal_line_from_extreme(extreme: Extreme, color='gray', label=None):
        if not extreme or not extreme.is_valid:
            return

        x_start = df.loc[extreme.idx, 'Index']
        x_end = df['Index'].iloc[-1]
        y = df.loc[extreme.idx, 'High'] if extreme.kind == 'max' else df.loc[extreme.idx, 'Low']

        ax.plot([x_start, x_end], [y, y], linestyle='--', color=color, label=label)


    # Haupt-Extreme markieren
    mark(result.extreme_2, 'blue', 'Extreme 2')
    mark(result.extreme_1, 'blue', 'Extreme 1')
    mark(result.present_extreme, 'blue', 'Present Extreme')

    # Linie zwischen den Haupt-Extremen zeichnen
    xs = []
    ys = []
    for extreme in [result.extreme_2, result.extreme_1, result.present_extreme]:
        if extreme and extreme.is_valid:
            xs.append(df.loc[extreme.idx, 'Index'])
            y = df.loc[extreme.idx, 'High'] if extreme.kind== 'max' else df.loc[extreme.idx, 'Low']
            ys.append(y)
    if len(xs) == 3 and len(ys) == 3:
        ax.plot(xs, ys, color='purple', linestyle='-', linewidth=2, label='Linie')

    # ✅ Zwischenextreme aus is_path_without_overlap markieren
    if hasattr(result, 'intermediate_extremes_wave_1') and result.intermediate_extremes_wave_1:
        x_w1 = []
        y_w1 = []
        for i, ext in enumerate(result.intermediate_extremes_wave_1):
            if ext.is_valid:
                x = df.loc[ext.idx, 'Index']
                x_w1.append(x)
                y = df.loc[ext.idx, 'High'] if ext.kind== 'max' else df.loc[ext.idx, 'Low']
                y_w1.append(y)
                ax.plot(x, y, marker='p', color='black', markersize=6, alpha=0.7, label='Zwischenextreme' if i == 0 else "")
        ax.plot(x_w1, y_w1, color = 'blue', linestyle = '-', linewidth = 1 )

    if hasattr(result, 'intermediate_extremes_wave_2') and result.intermediate_extremes_wave_2:
        x_w1 = []
        y_w1 = []
        for i, ext in enumerate(result.intermediate_extremes_wave_2):
            if ext.is_valid:
                x = df.loc[ext.idx, 'Index']
                x_w1.append(x)
                y = df.loc[ext.idx, 'High'] if ext.kind == 'max' else df.loc[ext.idx, 'Low']
                y_w1.append(y)
                ax.plot(x, y, marker='p', color='darkorange', markersize=6, alpha=0.7, label='Zwischenextreme' if i == 0 else "")
        ax.plot(x_w1, y_w1, color = 'red', linestyle = '-', linewidth = 1 )

        
    ax.set_title(f"{data.symbol}")
    ax.set_xlabel("Kerzen-Index")
    ax.legend()
    plt.grid(True)

    # Event-Handler für Entf-Taste
    def on_key(event):
        if event.key == 'delete':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    def on_mouse(event):
        if event.button == 2:  # Rechtsklick
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_mouse)

    # Vollbild aktivieren
    manager = plt.get_current_fig_manager()
    try:
        manager.full_screen_toggle()
    except AttributeError:
        try:
            manager.window.state('zoomed')  # Für TkAgg (z. B. Windows)
        except Exception:
            pass

    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    plt.show()




