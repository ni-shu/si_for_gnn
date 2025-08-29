# %%
import itertools
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from matplotlib import is_interactive
from scipy.stats import norm
from sicore import pvalues_hist, pvalues_qqplot, uniformity_test

sys.path.append(str(Path(__file__).resolve().parent.parent))

eeg_mode_list = ["eye", "full", "exp", "linear", "ave"]


def get_absolute_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)


def calculate_confidence_interval(n, p_hat, alpha):
    z_alpha_2 = norm.ppf(1 - alpha / 2)
    c = z_alpha_2**2
    D = 4 * c * n * p_hat * (1 - p_hat) + c**2

    lower_bound = min((2 * n * p_hat + c - np.sqrt(D)) / (2 * (n + c)), p_hat)
    upper_bound = max((2 * n * p_hat + c + np.sqrt(D)) / (2 * (n + c)), p_hat)

    return lower_bound, upper_bound

class Results:

    def __init__(
        self,
        num_nodes_list=[28],
        num_features_list=[50],
        corr_list=[False],
        signal_list=[0.0],
        lower_thres_list=[0.3],
        xai_list=["cam"],
        eeg_subids=[0],
        eeg_exp_list=["tpr"],
        eeg_gamma_list=[0.0],
        eeg_mode_list=[False],
        nongauss_list=[False],
        nongauss_type_list=["skewnorm"],
        nongauss_wd_list=[0.01],
        covariance_estimation_list=[False],
        alpha_list=[0.05],
    ):
        self.num_nodes_list = num_nodes_list
        self.num_features_list = num_features_list
        self.signal_list = signal_list
        self.eeg_subids = eeg_subids
        self.eeg_exp_list = eeg_exp_list
        self.lower_thres_list = lower_thres_list
        self.xai_list = xai_list
        self.eeg_gamma_list = eeg_gamma_list
        self.eeg_mode_list = eeg_mode_list
        self.corr_list = corr_list
        self.nongauss_list = nongauss_list
        self.nongauss_type_list = nongauss_type_list
        self.nongauss_wd_list = nongauss_wd_list
        self.covariance_estimation_list = covariance_estimation_list
        self.alpha_list = alpha_list

    def _load_result_pickle(self, file_name):
        file_path = os.path.join("../results", file_name)
        file_path = get_absolute_path(file_path)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            raise FileNotFoundError
        with open(file_path, "rb") as f:
            results = pickle.load(f)
        selective_p_values = results["proposed"]
        naive_p_values = results["naive"]
        oc_p_values = results["oc"]
        bonferroni_p_values = results["bonferroni"]
        num_iters = results["num_iters"]
        seed_list = results["seed"]
        return (
            selective_p_values,
            naive_p_values,
            oc_p_values,
            bonferroni_p_values,
            num_iters,
            seed_list,
        )

    def _plot(self, plot_flag=False, plot_num=None):
        test_class_1_only = False

        fpr_results = []
        for (
            xai_type,
            num_nodes,
            num_features,
            corr,
            signal,
            lower_thres,
            subid,
            eeg_exp_type,
            eeg_gamma,
            eeg_mode,
            nongauss,
            nongauss_type,
            nongauss_wd,
            covariance_estimation,
            alpha,
        ) in tqdm.tqdm(
            itertools.product(
                self.xai_list,
                self.num_nodes_list,
                self.num_features_list,
                self.corr_list,
                self.signal_list,
                self.lower_thres_list,
                self.eeg_subids,
                self.eeg_exp_list,
                self.eeg_gamma_list,
                self.eeg_mode_list,
                self.nongauss_list,
                self.nongauss_type_list,
                self.nongauss_wd_list,
                self.covariance_estimation_list,
                self.alpha_list,
            )
        ):
            upper_thres = 1.0 - lower_thres
            eeg_ave_num = 0

            def get_file_name(subid):
                if eeg_mode == False:
                    eeg_tag = ""
                else:
                    eeg_tag = f"_eeg_{eeg_mode}{eeg_ave_num if eeg_mode=='ave' else eeg_gamma}_sub{subid:02}_{eeg_exp_type}"
                if nongauss:
                    nongauss_tag = f"_nongauss_{nongauss_type}_wd{nongauss_wd}"
                else:
                    nongauss_tag = ""

                file_name = (
                    f"sig{signal}"
                    f"_n{num_nodes}"
                    f"_f{num_features}"
                    f"{'_corr' if corr else ''}"
                    f"_l{lower_thres}"
                    f"_u{upper_thres}"
                    f"{'_c1' if test_class_1_only else ''}"
                    f"_{xai_type}"
                    f"{nongauss_tag}"
                    f"{'_covest' if covariance_estimation else ''}"
                    f"{eeg_tag}.pkl"
                )
                return file_name

            if subid != "all":
                file_name = get_file_name(subid)
                (
                    selective_p_values,
                    naive_p_values,
                    oc_p_values,
                    bonferroni_p_values,
                    num_iters,
                    seed_list,
                ) = self._load_result_pickle(file_name)
            else:
                selective_p_values = []
                naive_p_values = []
                oc_p_values = []
                bonferroni_p_values = []
                for subid_ in self.eeg_subids:
                    file_name = get_file_name(subid_)
                    (
                        _selective_p_values,
                        _naive_p_values,
                        _oc_p_values,
                        _bonferroni_p_values,
                        num_iters,
                        seed_list,
                    ) = self._load_result_pickle(file_name)

                    selective_p_values += _selective_p_values
                    naive_p_values += _naive_p_values
                    oc_p_values += _oc_p_values
                    bonferroni_p_values += _bonferroni_p_values

            assert None not in selective_p_values

            # None values are removed
            selective_p_values = [p for p in selective_p_values if p is not None]
            naive_p_values = [p for p in naive_p_values if p is not None]
            oc_p_values = [p for p in oc_p_values if p is not None]
            bonferroni_p_values = [p for p in bonferroni_p_values if p is not None]

            selective_fpr = sum([1 for p in selective_p_values if p < alpha]) / len(
                selective_p_values
            )
            naive_fpr = sum([1 for p in naive_p_values if p < alpha]) / len(
                naive_p_values
            )
            oc_fpr = sum([1 for p in oc_p_values if p < alpha]) / len(oc_p_values)
            bonferroni_fpr = sum([1 for p in bonferroni_p_values if p < alpha]) / len(
                bonferroni_p_values
            )

            selctive_ci = calculate_confidence_interval(
                len(selective_p_values), selective_fpr, alpha
            )
            naive_ci = calculate_confidence_interval(
                len(naive_p_values), naive_fpr, alpha
            )
            oc_ci = calculate_confidence_interval(len(oc_p_values), oc_fpr, alpha)
            bonferroni_ci = calculate_confidence_interval(
                len(bonferroni_p_values), bonferroni_fpr, alpha
            )

            fpr_results.append(
                {
                    "alpha": alpha,
                    "num_features": num_features,
                    "num_nodes": num_nodes,
                    "signal": signal,
                    "xai_type": xai_type,
                    "lower_thres": lower_thres,
                    "upper_thres": upper_thres,
                    "subid": subid,
                    "exp_type": eeg_exp_type,
                    "eeg_ave_num": eeg_ave_num,
                    "eeg_gamma": eeg_gamma,
                    "eeg_mode": eeg_mode,
                    "selective_pr": selective_fpr,
                    "naive_pr": naive_fpr,
                    "oc_pr": oc_fpr,
                    "non_gauss": nongauss,
                    "nongauss_type": nongauss_type,
                    "nongauss_wd": nongauss_wd,
                    "bonferroni_pr": bonferroni_fpr,
                    "selective_ci_lower": selctive_ci[0],
                    "selective_ci_upper": selctive_ci[1],
                    "naive_ci_lower": naive_ci[0],
                    "naive_ci_upper": naive_ci[1],
                    "oc_ci_lower": oc_ci[0],
                    "oc_ci_upper": oc_ci[1],
                    "bonferroni_ci_lower": bonferroni_ci[0],
                    "bonferroni_ci_upper": bonferroni_ci[1],
                }
            )
        self.p_df = pd.DataFrame(fpr_results)
        return self.p_df

    def plot_curve(
        self,
        x_axis="eeg_gamma",
        z_axis="eeg_mode",
        test_type_list=["selective"],
        x_evenly_spaced=False,
        ylim=[-0.01, 1.01],
        alpha_line=True,
        legend_loc="best",
        sort_by_line_height=False,
        legend_order=[],
        use_color_dict=False,
        legend_ncol=1,
        legend_frameon=True,
        plot_legend=True,
        figsize=(4, 3),
    ):
        assert (
            (z_axis == "alpha") if len(self.alpha_list) > 1 else True
        ), "z_axis must be alpha if len(alpha_list) > 1"

        self._plot()
        fpr_df = self.p_df

        x_col = x_axis
        z_col = z_axis

        eeg_subid = "all"

        line_label_dict = {
            "cam": "CAM",
            # eeg_mode
            "linear": "linear",
            "full": "full",
            "eye": "eye",
            # nongauss_type
            "skewnorm": "skewnorm",
            "exponnorm": "exponnorm",
            "gennormsteep": "gennormsteep",
            "gennormflat": "gennormflat",
            "t": "t",
            # alpha
            0.05: r"$\alpha=0.05$",
            0.01: r"$\alpha=0.01$",
            0.10: r"$\alpha=0.10$",
        }
        xlabel_dict = {
            "lower_thres": "Lower Threshold",
            "eeg_ave_num": "Averaging Number $N$",
            "eeg_gamma": "$\gamma$",
            "num_features": "Number of Features",
            "num_nodes": "Number of Nodes",
            "signal": "Signal",
            "nongauss_wd": "1-Wasserstein Distance",
        }
        test_label_dict = {
            "selective": "proposed",
            "naive": "naive",
            "oc": "w/o-pp",
            "bonferroni": "Bonferroni",
        }
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        test_color_dict = {key: colors[i] for i, key in enumerate(test_label_dict)}
        plt.figure(figsize=figsize)
        handles = []
        labels = []

        if z_col in fpr_df.columns:
            z_type_list = fpr_df[z_col].unique()
        else:
            z_type_list = [None]

        x_labels = np.sort(fpr_df[x_col].unique())
        if x_evenly_spaced:
            x_indices = range(len(x_labels))
        else:
            x_indices = x_labels
        xaxis_length = x_indices[-1] - x_indices[0]
        xlim = [x_indices[0] - xaxis_length * 0.1, x_indices[-1] + xaxis_length * 0.1]

        if alpha_line:
            for alpha in self.alpha_list:
                alphaline = plt.hlines(
                    alpha, xlim[0], xlim[1], linestyles="dashed", colors="gray"
                )

        for test_type in test_type_list:
            for z_type in z_type_list:
                if z_col in fpr_df.columns:
                    fpr_df_sub = fpr_df[fpr_df[z_col] == z_type]
                else:
                    fpr_df_sub = fpr_df

                assert (
                    f"{test_type}_pr" in fpr_df_sub.columns
                ), f"{test_type}_pr is not in the columns"
                assert len(fpr_df_sub[x_col].unique()) == len(fpr_df_sub[x_col])

                if use_color_dict:
                    (line,) = plt.plot(
                        x_indices,
                        fpr_df_sub[f"{test_type}_pr"],
                        color=test_color_dict[test_type],
                        marker="o",
                        markersize=1,
                    )
                else:
                    (line,) = plt.plot(
                        x_indices,
                        fpr_df_sub[f"{test_type}_pr"],
                        marker="o",
                        markersize=1,
                    )

                yerr_lower = list(
                    fpr_df_sub[f"{test_type}_pr"] - fpr_df_sub[f"{test_type}_ci_lower"]
                )
                yerr_upper = list(
                    fpr_df_sub[f"{test_type}_ci_upper"] - fpr_df_sub[f"{test_type}_pr"]
                )

                plt.errorbar(
                    x_indices,
                    fpr_df_sub[f"{test_type}_pr"],
                    yerr=[yerr_lower, yerr_upper],
                    color=line.get_color(),
                    fmt="o",
                    capsize=4,
                    markersize=0,
                )

                handles.append(line)

                label_text = ""
                if len(test_type_list) > 1:
                    label_text += f"{test_label_dict[test_type]}"
                if z_type in line_label_dict:
                    if len(label_text) > 0:
                        label_text += f": {line_label_dict[z_type]}"
                    else:
                        label_text += f"{line_label_dict[z_type]}"
                labels.append(label_text)

        if plot_legend:
            if sort_by_line_height:
                sorted_indices = np.argsort([line.get_ydata()[-1] for line in handles])
                sorted_handles = [handles[i] for i in sorted_indices]
                sorted_labels = [labels[i] for i in sorted_indices]
            elif legend_order != []: 
                lablel_order = [test_label_dict[test_type] for test_type in legend_order]
                ordered_pairs = sorted(
                    zip(handles, labels),
                    key=lambda x: (
                        lablel_order.index(x[1]) if x[1] in lablel_order else float("inf")
                    ),
                )
                handles, labels = zip(*ordered_pairs)
                sorted_handles = list(handles)[::-1]
                sorted_labels = list(labels)[::-1]
            else:
                sorted_indices = range(len(handles))[::-1]
                sorted_handles = [handles[i] for i in sorted_indices]
                sorted_labels = [labels[i] for i in sorted_indices]
            if alpha_line:
                sorted_handles = [alphaline] + sorted_handles
                if len(self.alpha_list) == 1:
                    sorted_labels = [r"$\alpha$=" + f"{self.alpha_list[0]}"] + sorted_labels
            plt.legend(
                sorted_handles[::-1],
                sorted_labels[::-1],
                loc=legend_loc,
                ncol=legend_ncol,
                frameon=legend_frameon,
            )

        plt.xticks(x_indices, x_labels)
        plt.xlim(xlim)
        plt.ylim(ylim[0], ylim[1])
        if x_col in xlabel_dict:
            plt.xlabel(xlabel_dict[x_col])
        else:
            plt.xlabel(x_col)

        if self.eeg_mode_list[0] is False:
            if self.signal_list[0] == 0.0:
                plt.ylabel("Type I Error Rate")
                assert len(self.signal_list) == 1
            else:
                plt.ylabel("Power")
        else:
            assert len(self.eeg_exp_list) == 1
            if self.eeg_exp_list[0] == "fpr":
                plt.ylabel("Type I Error Rate")
            elif self.eeg_exp_list[0] == "tpr":
                plt.ylabel("Power")

        if is_interactive():
            plt.show()
        else:
            assert len(self.corr_list) == 1
            assert len(self.eeg_exp_list) == 1
            cov_est_tag = (
                "_covest" if z_axis == "alpha" else ""
            ) 
            if self.eeg_mode_list[0] is False:
                save_path = get_absolute_path(
                    f"../figures/{x_axis}{cov_est_tag}_{'corr' if self.corr_list[0] else 'iid'}.pdf"
                )
            else:
                save_path = get_absolute_path(
                    f"../figures/{x_axis}{cov_est_tag}_{self.eeg_exp_list[0]}.pdf"
                )
            plt.savefig(save_path)


if __name__ == "__main__":
    legend_order = ["selective", "naive", "bonferroni", "oc"]
    figsize = (4, 3)
    legend_frameon = False

    # FPR: num_nodes
    for corr_flag in [False, True]:
        Results(
            num_features_list=[5],
            num_nodes_list=[32, 64, 128, 256],
            lower_thres_list=[0.3],
            corr_list=[corr_flag],
        ).plot_curve(
            x_axis="num_nodes",
            z_axis="",
            test_type_list=[
                "naive",
                "oc",
                "bonferroni",
                "selective",
            ],
            legend_order=legend_order,
            x_evenly_spaced=True,
            legend_loc="upper left",
            use_color_dict=True,
            plot_legend=True,
            legend_frameon=legend_frameon,
            figsize=figsize,
        )

    # FPR: num_features
    for corr_flag in [False, True]:
        Results(
            num_features_list=[5, 10, 15, 20],
            num_nodes_list=[256],
            lower_thres_list=[0.3],
            corr_list=[corr_flag],
        ).plot_curve(
            x_axis="num_features",
            z_axis="",
            test_type_list=[
                "naive",
                "oc",
                "bonferroni",
                "selective",
            ],
            legend_loc="upper left",
            legend_order=legend_order,
            use_color_dict=True,
            plot_legend=True,
            legend_frameon=legend_frameon,
            figsize=figsize,
        )

    # TPR: signal
    for corr_flag in [False, True]:
        Results(
            num_features_list=[5],
            num_nodes_list=[256],
            lower_thres_list=[0.3],
            corr_list=[corr_flag],
            signal_list=[1.0, 1.5, 2.0, 2.5],
        ).plot_curve(
            x_axis="signal",
            z_axis="",
            test_type_list=[
                "oc",
                "bonferroni",
                "selective",
            ],
            alpha_line=False,
            legend_loc="upper left",
            legend_order=["selective", "bonferroni", "oc"],
            use_color_dict=True,
            plot_legend=True,
            legend_frameon=legend_frameon,
            figsize=figsize,
        )

    # FPR: covariance_estimation
    Results(
        num_features_list=[5],
        num_nodes_list=[32, 64, 128, 256],
        lower_thres_list=[0.3],
        corr_list=[False],
        covariance_estimation_list=[True],
        alpha_list=[0.05, 0.01, 0.10],
    ).plot_curve(
        x_axis="num_nodes",
        z_axis="alpha",
        test_type_list=["selective"],
        x_evenly_spaced=True,
        ylim=[0, 0.2],
    )

    # FPR: nongauss
    Results(
        num_features_list=[5],
        num_nodes_list=[256],
        lower_thres_list=[0.3],
        corr_list=[False],
        nongauss_list=[True],
        nongauss_type_list=[
            "skewnorm",
            "exponnorm",
            "gennormsteep",
            "gennormflat",
            "t",
        ],
        nongauss_wd_list=[0.01, 0.05, 0.1, 0.15],
    ).plot_curve(
        x_axis="nongauss_wd", z_axis="nongauss_type", test_type_list=["selective"]
    )
