"""Scan small delta_beta values and plot reconstructed h versus delta_beta."""

import numpy as np
import matplotlib.pyplot as pl
from fiveVec_resampler_utils import check_PBH_signal


def main():
    delta_beta_values = np.logspace(-11, -8, 50)
    results = np.array(
        [check_PBH_signal(err=1e4, f0_setting=0.2, gap_fraction=0, delta_beta=i) for i in delta_beta_values]
    )
    fig, ax = pl.subplots()
    ax.plot(delta_beta_values, 1 + results[:, 1], "o")
    ax.set_xscale("log")
    ax.set_xlabel("delta_beta")
    ax.set_ylabel("h_reconstructed")
    ax.set_title("h_reconstructed vs delta_beta")
    fig.savefig("plots/h_reconstructed_vs_delta_beta.png")


if __name__ == "__main__":
    main()
