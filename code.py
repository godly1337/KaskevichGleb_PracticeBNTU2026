import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# путь к файлу с данными (можно указать вторым способом - аргументом при запуске)
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "imu_data.csv"

# если названия столбцов в csv не совпадают с автоопределением, вписать сюда вручную
# пример: COLUMN_OVERRIDE = {"time": "Time_s", "ax": "AccX", "ay": "AccY", "az": "AccZ",
#                             "gx": "GyroX", "gy": "GyroY", "gz": "GyroZ"}
COLUMN_OVERRIDE = {}

OUTPUT_DIR = "output_plots"

ACC_THRESHOLD_STD = 5.0
GYRO_THRESHOLD_STD = 5.0
MIN_ABS_ACC_DERIV = 0.5
MIN_ABS_GYRO = 0.3
MIN_EVENT_DISTANCE = 40


def generate_synthetic_data(n=3000, fs=100.0):
    # тестовые данные на случай если csv еще не скачан:
    # покой -> старт -> прямая -> поворот -> прямая -> резкая остановка
    t = np.arange(n) / fs
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.ones(n) * 9.81
    gx = np.zeros(n)
    gy = np.zeros(n)
    gz = np.zeros(n)

    rng = np.random.default_rng(42)
    noise = 0.02

    idx_start = int(0.10 * n)
    idx_turn_begin = int(0.45 * n)
    idx_turn_end = int(0.55 * n)
    idx_stop = int(0.85 * n)

    ax[idx_start:idx_start + 15] += np.linspace(0, 2.5, 15)
    ax[idx_start + 15:idx_stop] += 0.3

    turn_len = idx_turn_end - idx_turn_begin
    gz[idx_turn_begin:idx_turn_end] += np.sin(np.linspace(0, np.pi, turn_len)) * 1.8
    ay[idx_turn_begin:idx_turn_end] += np.sin(np.linspace(0, np.pi, turn_len)) * 1.2

    ax[idx_stop:idx_stop + 15] -= np.linspace(0, 3.0, 15)

    ax += rng.normal(0, noise, n)
    ay += rng.normal(0, noise, n)
    az += rng.normal(0, noise, n)
    gx += rng.normal(0, noise, n)
    gy += rng.normal(0, noise, n)
    gz += rng.normal(0, noise, n)

    return pd.DataFrame({"time": t, "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz})


def detect_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}

    def find(*keywords):
        for kw in keywords:
            for lower, original in cols.items():
                if kw in lower:
                    return original
        return None

    return {
        "time": find("time", "timestamp", "sec", "t_"),
        "ax": find("acc_x", "accx", "ax", "accel_x"),
        "ay": find("acc_y", "accy", "ay", "accel_y"),
        "az": find("acc_z", "accz", "az", "accel_z"),
        "gx": find("gyro_x", "gyrox", "gx", "omega_x", "angvel_x"),
        "gy": find("gyro_y", "gyroy", "gy", "omega_y", "angvel_y"),
        "gz": find("gyro_z", "gyroz", "gz", "omega_z", "angvel_z"),
    }


def load_data(path):
    if not os.path.exists(path):
        print(f"файл '{path}' не найден, использую тестовые данные для проверки скрипта")
        print("скачать датасет: https://github.com/ansfl/MAGF-ID")
        return generate_synthetic_data()

    print(f"загружаю данные из {path}")
    df_raw = pd.read_csv(path)
    mapping = detect_columns(df_raw)
    mapping.update(COLUMN_OVERRIDE)

    missing = [k for k, v in mapping.items() if v is None and k != "time"]
    if missing:
        print(f"не найдены столбцы: {missing}")
        print(f"доступные столбцы: {list(df_raw.columns)}")
        print("пропишите нужные имена в COLUMN_OVERRIDE")
        raise SystemExit(1)

    df = pd.DataFrame()
    if mapping["time"] is not None:
        df["time"] = df_raw[mapping["time"]].astype(float)
        if df["time"].iloc[-1] > 10_000:
            df["time"] = df["time"] / 1000.0
    else:
        df["time"] = np.arange(len(df_raw)) / 100.0

    for key in ["ax", "ay", "az", "gx", "gy", "gz"]:
        df[key] = df_raw[mapping[key]].astype(float)

    return df


def smooth(x, window=15):
    window = max(3, min(window, len(x)))
    kernel = np.ones(window) / window
    x_padded = np.pad(x, (window // 2, window // 2), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")[: len(x)]


def detect_events(df, fs):
    # знак события (старт/остановка) определяем по ax, а не по модулю |a|,
    # т.к. при торможении ax уходит в минус, но его модуль может расти
    acc_mag = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    gyro_mag = np.sqrt(df["gx"] ** 2 + df["gy"] ** 2 + df["gz"] ** 2)

    acc_mag_s = smooth(acc_mag.values)
    gyro_mag_s = smooth(gyro_mag.values)
    ax_s = smooth(df["ax"].values)

    acc_deriv = np.gradient(ax_s, 1.0 / fs)

    n_calib = max(int(0.05 * len(df)), 10)
    acc_deriv_std = np.std(acc_deriv[:n_calib]) + 1e-6
    gyro_std = np.std(gyro_mag_s[:n_calib]) + 1e-6
    gyro_mean = np.mean(gyro_mag_s[:n_calib])

    acc_height = max(ACC_THRESHOLD_STD * acc_deriv_std, MIN_ABS_ACC_DERIV)
    acc_peaks, _ = find_peaks(np.abs(acc_deriv), height=acc_height, distance=MIN_EVENT_DISTANCE)

    starts, stops = [], []
    for p in acc_peaks:
        if acc_deriv[p] > 0:
            starts.append(p)
        else:
            stops.append(p)

    gyro_height = gyro_mean + max(GYRO_THRESHOLD_STD * gyro_std, MIN_ABS_GYRO)
    turn_peaks, _ = find_peaks(gyro_mag_s, height=gyro_height, distance=MIN_EVENT_DISTANCE)

    return {
        "acc_mag": acc_mag_s,
        "gyro_mag": gyro_mag_s,
        "starts": starts,
        "stops": stops,
        "turns": turn_peaks,
    }


def mark_events(ax_plot, t, events):
    for idx in events["starts"]:
        ax_plot.axvline(t[idx], color="green", linestyle="--", alpha=0.8)
        ax_plot.text(t[idx], ax_plot.get_ylim()[1] * 0.9, "старт", rotation=90, color="green", fontsize=8, va="top")
    for idx in events["stops"]:
        ax_plot.axvline(t[idx], color="red", linestyle="--", alpha=0.8)
        ax_plot.text(t[idx], ax_plot.get_ylim()[1] * 0.9, "остановка", rotation=90, color="red", fontsize=8, va="top")
    for idx in events["turns"]:
        ax_plot.axvline(t[idx], color="orange", linestyle=":", alpha=0.8)
        ax_plot.text(t[idx], ax_plot.get_ylim()[1] * 0.75, "поворот", rotation=90, color="darkorange", fontsize=8, va="top")


def plot_accelerometer(df, events, out_dir):
    t = df["time"].values
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    axes[0].plot(t, df["ax"], color="tab:blue")
    axes[0].set_ylabel("ax, м/с²")
    axes[0].set_title("Ускорение по оси X")

    axes[1].plot(t, df["ay"], color="tab:blue")
    axes[1].set_ylabel("ay, м/с²")
    axes[1].set_title("Ускорение по оси Y")

    axes[2].plot(t, df["az"], color="tab:blue")
    axes[2].set_ylabel("az, м/с²")
    axes[2].set_title("Ускорение по оси Z")

    axes[3].plot(t, events["acc_mag"], color="black")
    axes[3].set_ylabel("|a|, м/с²")
    axes[3].set_title("Модуль ускорения с отмеченными событиями")
    axes[3].set_xlabel("Время, с")

    for a in axes:
        mark_events(a, t, events)
        a.grid(alpha=0.3)

    fig.suptitle("Данные акселерометра", fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "accelerometer.png")
    fig.savefig(path, dpi=150)
    print(f"сохранено: {path}")


def plot_gyroscope(df, events, out_dir):
    t = df["time"].values
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    axes[0].plot(t, df["gx"], color="tab:purple")
    axes[0].set_ylabel("gx, рад/с")
    axes[0].set_title("Угловая скорость по оси X")

    axes[1].plot(t, df["gy"], color="tab:purple")
    axes[1].set_ylabel("gy, рад/с")
    axes[1].set_title("Угловая скорость по оси Y")

    axes[2].plot(t, df["gz"], color="tab:purple")
    axes[2].set_ylabel("gz, рад/с")
    axes[2].set_title("Угловая скорость по оси Z")

    axes[3].plot(t, events["gyro_mag"], color="black")
    axes[3].set_ylabel("|ω|, рад/с")
    axes[3].set_title("Модуль угловой скорости с отмеченными событиями")
    axes[3].set_xlabel("Время, с")

    for a in axes:
        mark_events(a, t, events)
        a.grid(alpha=0.3)

    fig.suptitle("Данные гироскопа", fontsize=14, fontweight="bold")
    path = os.path.join(out_dir, "gyroscope.png")
    fig.savefig(path, dpi=150)
    print(f"сохранено: {path}")


def plot_combined_overview(df, events, out_dir):
    t = df["time"].values
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(t, events["acc_mag"], color="tab:blue", label="ускорение")
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("|a|, м/с²", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(t, events["gyro_mag"], color="tab:purple", label="угловая скорость")
    ax2.set_ylabel("|ω|, рад/с", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    mark_events(ax1, t, events)
    ax1.set_title("Характер движения робота во времени")
    ax1.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, "overview.png")
    fig.savefig(path, dpi=150)
    print(f"сохранено: {path}")


def print_summary(df, events):
    t = df["time"].values
    print("\n--- сводка ---")
    print(f"записей: {len(df)}, длительность: {t[-1]:.2f} с")

    print(f"\nстартов: {len(events['starts'])}")
    for idx in events["starts"]:
        print(f"  t = {t[idx]:.2f} с")

    print(f"\nостановок: {len(events['stops'])}")
    for idx in events["stops"]:
        print(f"  t = {t[idx]:.2f} с")

    print(f"\nповоротов: {len(events['turns'])}")
    for idx in events["turns"]:
        print(f"  t = {t[idx]:.2f} с")
    print()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data(CSV_PATH)

    dt = np.median(np.diff(df["time"].values))
    fs = 1.0 / dt if dt > 0 else 100.0
    print(f"частота дискретизации: {fs:.1f} Гц")

    events = detect_events(df, fs)

    plot_accelerometer(df, events, OUTPUT_DIR)
    plot_gyroscope(df, events, OUTPUT_DIR)
    plot_combined_overview(df, events, OUTPUT_DIR)

    print_summary(df, events)

    plt.show()


if __name__ == "__main__":
    main()