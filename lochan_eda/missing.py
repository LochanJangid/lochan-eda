import matplotlib.pyplot as plt

class Missing:
    def __init__(self, profiler_df=None):
        self.profiler_df = profiler_df

    def plot(self):
        my_data = self.profiler_df
        if my_data.empty:
            return None

        missingness = (my_data.isna().mean() * 100).sort_values(ascending=False)
        fig, ax = plt.subplots(
            figsize=(10, max(4, len(missingness) * 0.4)),
            constrained_layout=True,
        )
        if missingness.empty:
            ax.text(0.5, 0.5, "No Missing Value Found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

            return fig

        missingness.sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Missing Value by Columns", fontsize=14, fontweight="bold")
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("Column")

        for i, value in enumerate(missingness.sort_values()):
            ax.text(value, i, f"{value:.1f}%", va="center")

        fig.savefig("missing_plot.png", dpi=150, bbox_inches="tight")
        print("Plots is saved into `missing_plot.png`")

        return fig