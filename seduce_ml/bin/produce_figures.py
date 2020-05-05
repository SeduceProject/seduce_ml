import sys
import pandas
import matplotlib.pyplot as plt
import matplotlib
import os


def fix_hist_step_vertical_line_at_end(ax):
    """This function fixes the vertical line that appears when several CDFs are plotted on the same
    plot. The code comes from:

     https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib/39729964

     """
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, matplotlib.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


def main():
    APPROACHES = ["neural", "knearest", "gaussian"]
    SERVERS = [f"ecotype_{i}" for i in range(37, 49)]
    NSTEPS = 4
    CDF_X_LIMIT = (0, 6)

    output_folder = "output"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for nstep in range(0, NSTEPS+1):
        print(f"========== N={nstep} =============")

        csv_file_path = f"csv/data_{nstep}_steps.csv"
        df = pandas.read_csv(csv_file_path, parse_dates=["timestamp"], index_col=0)

        # Produce tables
        tables_df = {}
        for approach in APPROACHES:
            approach_df = pandas.DataFrame()
            for server in SERVERS:
                approach_df[server] = (df[f"{approach}_{server}_expected"] - df[f"{approach}_{server}_predicted"]).abs()
            tables_df[approach] = approach_df.mean().mean()

        # print(tables_df)

        # Produce figures
        for server in SERVERS:
            server_df = pandas.DataFrame()
            server_df["timestamp"] = df["timestamp"]
            server_df["expected"] = df[f"neural_{server}_expected"]
            for approach in APPROACHES:
                server_df[approach] = df[f"{approach}_{server}_predicted"]

            # gca stands for 'get current axis'
            ax = plt.gca()

            server_df.plot(kind='line', x='timestamp', y='expected', ax=ax, linewidth=1)
            for approach in APPROACHES:
                server_df.plot(kind='line', x='timestamp', y=f"{approach}", ax=ax, alpha=0.7, linewidth=0.5)

            plt.xlabel("Date")
            plt.ylabel("Temperature (°C)")

            plt.savefig(f"{output_folder}/{server}_n={nstep}.pdf")
            plt.close()

        # Produce statistical distributions
        tables_df = {}
        for approach in APPROACHES:
            approach_df = pandas.DataFrame()
            done = {}
            for server in SERVERS:
                if approach not in done:
                    done[approach] = "ok"
                    tables_df[approach] = (df[f"{approach}_{server}_expected"] - df[f"{approach}_{server}_predicted"]).abs()
                else:
                    tables_df[approach].append((df[f"{approach}_{server}_expected"] - df[f"{approach}_{server}_predicted"]).abs())

        # gca stands for 'get current axis'
        ax = plt.gca()
        plt.xlim(CDF_X_LIMIT)
        plt.xlabel("Error (°C)")
        plt.ylabel("Cumulative percentage")

        for approach in APPROACHES:
            result = tables_df[approach].hist(cumulative=True, density=1, bins=100, histtype=u'step', label=approach)

        leg = ax.legend(loc=7)
        for l in leg.legendHandles:
            l.set_linewidth(1)
            l.set_height(1)

        fix_hist_step_vertical_line_at_end(ax)

        plt.savefig(f"{output_folder}/cdf_n={nstep}.pdf")
        plt.close()

        print(tables_df)

        print(f"=======================")

    return True


if __name__ == "__main__":
    main()
    sys.exit(0)
