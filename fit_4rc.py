"""
Fit a 4-RC model to time-series data in a CSV file with columns: time, signalA, signalB.

Model (sum of 4 exponentials + offset, standard form for a 4-RC ladder step/impulse response):

    y(t) = y_inf + A1*exp(-t/tau1) + A2*exp(-t/tau2) + A3*exp(-t/tau3) + A4*exp(-t/tau4)

Equivalent RC interpretation: tau_i = R_i * C_i, A_i is the amplitude associated with branch i.
No bounds are applied -- amplitudes, tau's, or y_inf can come out negative. This is a bare
least-squares fit, not a physically-constrained identification.

Usage:
    python fit_4rc.py data.csv
    python fit_4rc.py data.csv --time-col time --colA signalA --colB signalB --plot
"""

import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def model(t, y_inf, A1, tau1, A2, tau2, A3, tau3, A4, tau4):
    return (
        y_inf
        + A1 * np.exp(-t / tau1)
        + A2 * np.exp(-t / tau2)
        + A3 * np.exp(-t / tau3)
        + A4 * np.exp(-t / tau4)
    )


def make_initial_guess(t, y):
    """Rough, generic starting point: spread tau's log-uniformly over the time span,
    split the signal's total swing across the 4 amplitudes evenly."""
    t_span = max(t.max() - t.min(), 1e-6)
    taus = np.geomspace(t_span / 50, t_span * 2, 4)
    y_inf0 = y[-1]
    swing = y[0] - y[-1]
    amps = [swing / 4] * 4
    p0 = [y_inf0]
    for a, tau in zip(amps, taus):
        p0 += [a, tau]
    return p0


def fit_signal(t, y, label):
    p0 = make_initial_guess(t, y)
    try:
        popt, pcov = curve_fit(model, t, y, p0=p0, maxfev=50000)
    except RuntimeError as e:
        print(f"[{label}] Fit did not converge: {e}")
        return None, None

    y_fit = model(t, *popt)
    resid = y - y_fit
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * len(popt)

    print(f"\n--- Fit results for {label} ---")
    print(f"y_inf = {popt[0]:.6g}  (+/- {perr[0]:.2g})")
    for i in range(4):
        A = popt[1 + 2 * i]
        tau = popt[2 + 2 * i]
        Ae = perr[1 + 2 * i]
        taue = perr[2 + 2 * i]
        print(f"A{i+1}   = {A:.6g}  (+/- {Ae:.2g})    tau{i+1} = {tau:.6g}  (+/- {taue:.2g})")
    print(f"R^2 = {r2:.6f}")

    return popt, y_fit


def main():
    parser = argparse.ArgumentParser(description="Fit a 4-RC (4-exponential) model to two signal columns.")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--time-col", default="time", help="Name of time column (default: time)")
    parser.add_argument("--colA", default="signalA", help="Name of first signal column (default: signalA)")
    parser.add_argument("--colB", default="signalB", help="Name of second signal column (default: signalB)")
    parser.add_argument("--plot", action="store_true", help="Show a plot of data vs fit")
    parser.add_argument("--out", default=None, help="Optional path to save fitted curves as CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    t = df[args.time_col].to_numpy(dtype=float)
    yA = df[args.colA].to_numpy(dtype=float)
    yB = df[args.colB].to_numpy(dtype=float)

    poptA, yA_fit = fit_signal(t, yA, args.colA)
    poptB, yB_fit = fit_signal(t, yB, args.colB)

    if args.out:
        out_df = pd.DataFrame({args.time_col: t})
        if yA_fit is not None:
            out_df[f"{args.colA}_data"] = yA
            out_df[f"{args.colA}_fit"] = yA_fit
        if yB_fit is not None:
            out_df[f"{args.colB}_data"] = yB
            out_df[f"{args.colB}_fit"] = yB_fit
        out_df.to_csv(args.out, index=False)
        print(f"\nSaved fitted curves to {args.out}")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        if yA_fit is not None:
            axes[0].plot(t, yA, "o", ms=3, label="data")
            axes[0].plot(t, yA_fit, "-", label="4-RC fit")
            axes[0].set_title(args.colA)
            axes[0].legend()
        if yB_fit is not None:
            axes[1].plot(t, yB, "o", ms=3, label="data")
            axes[1].plot(t, yB_fit, "-", label="4-RC fit")
            axes[1].set_title(args.colB)
            axes[1].legend()
        axes[1].set_xlabel(args.time_col)
        plt.tight_layout()
        plt.savefig("fit_4rc_plot.png", dpi=150)
        print("Saved plot to fit_4rc_plot.png")


if __name__ == "__main__":
    main()
