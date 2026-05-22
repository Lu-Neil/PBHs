STANDARD_PLOT_LIMITS = {
    "trajectory": {
        "xlim": (0.038279401582062736, 57244275.42051861),
        "ylim": (11.000000035863126, 208.99999999829382),
    },
    "residual_time": {
        "xlim": (0.038279401582062736, 57244275.42051861),
        "ylim": (6.135717657192917e-14, 172.41515377732665),
    },
    "residual_frequency": {
        "xlim": (11.024299546274076, 208.48971027966388),
        "ylim": (6.135717657192917e-14, 172.41515377732665),
    },
}


def apply_standard_plot_limits(trajectory_ax, residual_time_ax, residual_frequency_ax):
    trajectory_ax.set_xlim(STANDARD_PLOT_LIMITS["trajectory"]["xlim"])
    trajectory_ax.set_ylim(STANDARD_PLOT_LIMITS["trajectory"]["ylim"])

    residual_time_ax.set_xlim(STANDARD_PLOT_LIMITS["residual_time"]["xlim"])
    residual_time_ax.set_ylim(STANDARD_PLOT_LIMITS["residual_time"]["ylim"])

    residual_frequency_ax.set_xlim(STANDARD_PLOT_LIMITS["residual_frequency"]["xlim"])
    residual_frequency_ax.set_ylim(STANDARD_PLOT_LIMITS["residual_frequency"]["ylim"])
