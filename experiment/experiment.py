import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MAX_JOBS"] = "1"

import torch

torch.set_num_threads(1)

import argparse
import pickle
import sys
import copy
from typing import Union


sys.path.append("..")
from abc import ABCMeta, abstractmethod

import numpy as np
from tqdm import tqdm
import onnx
import torch
from joblib import Parallel, delayed


from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from source.si4gnn import (
    SI4ONNX,
    SI4TSGNN,
    ThresholdIndexError,
    NegativeCaseTestError,
)
from data_utils import synth_data
from train import gnn_model
import sicore


eeg_gen_mode_list = ["eye", "full", "exp", "linear"]


def get_absolute_path(relative_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, relative_path)


def create_exp_matrix(n):
    # Σ_{i,j} = 0.5^{|i - j|}
    exp_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            exp_matrix[i, j] = 0.5 ** (abs(i - j) / 2)
    return exp_matrix


def create_linear_decay_matrix(n):
    #  Σ_{i,j} = 1 - |i - j| / (n - 1)
    linear_decay_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            linear_decay_matrix[i, j] = 1 - abs(i - j) / (n - 1)
    return linear_decay_matrix


class PararellExperiment(metaclass=ABCMeta):

    def __init__(
        self,
        num_iter: int,
        num_results: int,
        num_worker: int,
        inspect_iteration: Union[int, None] = None,
    ):
        self.num_iter = num_iter
        self.num_results = num_results
        self.num_worker = num_worker
        self.inspect_iteration = inspect_iteration

    @abstractmethod
    def iter_experiment(self, data) -> tuple:
        """Run each iteration of the experiment

        Args:
            data (tuple): Tuple of data for each iteration

        Returns:
            tuple: Tuple of results of each iteration
        """
        pass

    def _iter_experiment(self, i, data) -> tuple:
        try:
            return self.iter_experiment(data)
        except Exception as e:
            print(e)
            print(f"Error occured in experiment (iter = {i}, {data=})")
            sys.exit(1)

    def experiment(self, dataset: list):
        """Execute all iterations of the experiment

        Args:
            dataset (list): List of args for iter_experiment method,
            the size of which must be equal to num_iter

        Returns:
            tupple: tupple of results from iter_experiment method
        """

        assert (
            len(dataset) == self.num_iter
        ), "The size of the dataset must be equal to num_iter"

        if self.inspect_iteration is not None:
            result = self.iter_experiment(dataset[self.inspect_iteration])
            return tuple((r,) for r in result)
        else:
            results = Parallel(n_jobs=self.num_worker)(
                delayed(self._iter_experiment)(i, d)
                for i, d in tqdm(enumerate(dataset), total=self.num_iter)
            )
            print("Experiment finished")
            return zip(*results)

    @abstractmethod
    def run_experiment(self):
        pass


class GNNExperiment(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        signal: float,
        num_nodes: int,
        num_features: int,
        corr: bool,
        lower_thres: float = 0.5,
        upper_thres: float = 0.5,
        inspect_iteration: Union[int, None] = None,
        test_class_1_only: bool = False,
        xai_type: str = "cam",
        eeg_mode: bool = False,
        eeg_subid: int = 0,
        eeg_exp_type: str = "fpr",
        eeg_gamma: float = 1.0,
        nongauss: bool = False,
        nongauss_type: str = "skewnorm",
        nongauss_wd: float = 0.1,
        covariance_estimation: bool = False,
    ):
        super().__init__(num_results, num_results, num_worker, inspect_iteration)
        assert not eeg_mode

        self.nongauss = nongauss
        if nongauss:
            assert not eeg_mode
            assert signal == 0.0
            self.nongauss_rv = sicore.generate_non_gaussian_rv(
                nongauss_type, nongauss_wd
            )

        self.signal = signal
        self.eeg_mode = eeg_mode
        self.eeg_subid = eeg_subid
        self.covariance_estimation = covariance_estimation

        if self.covariance_estimation:
            assert not eeg_mode
            assert not corr
        if corr:
            self.var = None
            self.var_based_on_A = True
        else:
            self.var = 1.0
            self.var_based_on_A = False


        self.eeg_mu = None
        self.eeg_gamma = None
        self.eeg_A = None
        self.eeg_cov = None

        # load onnx model
        self.layer_size = 3
        self.num_features = num_features
        if xai_type == "stcam":
            assert num_features == 50
            onnx_num_features = 5
        else:
            onnx_num_features = num_features

        self.num_nodes = num_nodes
        self.expected_degree = 3

        if lower_thres == upper_thres:
            self.thres = lower_thres
        else:
            self.thres = [lower_thres, upper_thres]
        self.apply_norm = True
        self.test_class_1_only = test_class_1_only

        if self.eeg_mode in eeg_gen_mode_list:
            model_path = (
                f"../model/cam_eeg_l{self.layer_size}_f{onnx_num_features}.onnx"
            )
        elif not self.eeg_mode:
            model_path = (
                f"../model/cam_synth_l{self.layer_size}_f{onnx_num_features}.onnx"
            )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.onnx_model_path = os.path.join(current_dir, model_path)
        self.onnx_model = onnx.load(self.onnx_model_path)

        model = gnn_model.create_model(onnx_num_features, self.layer_size)
        cam_model = gnn_model.create_cam_model(model)
        cam_model_state_path = os.path.join(
            current_dir, model_path.replace(".onnx", ".pth")
        )
        cam_model.load_state_dict(torch.load(cam_model_state_path))
        cam_model.eval()
        self.torch_model = cam_model

        self.xai_type = xai_type
        self.time_window = None



    def iter_experiment(self, seed) -> tuple:

        torch.set_num_threads(1)

        if self.xai_type == "cam":
            self.si_model = SI4ONNX(
                self.onnx_model,
                thr=self.thres,
                num_features=self.num_features,
                apply_norm=self.apply_norm,
                test_class_1_only=self.test_class_1_only,
                time_window=self.time_window,
            )

        elif self.xai_type == "stcam":
            self.si_model = SI4TSGNN(
                self.onnx_model,
                thr=self.thres,
                num_features=self.num_features,
                apply_norm=self.apply_norm,
                test_class_1_only=self.test_class_1_only,
                time_window=self.time_window,
                chunk_length=5,
                chunk_slide_width=5,
            )

        rng = np.random.default_rng(seed)
        max_iter = 100000
        for i in range(1, max_iter + 1):
            sub_seed = rng.integers(0, 2**32 - 1)

            eeg_mu = None
            eeg_cov = None

            try:
                dataset = synth_data.GraphDataset(
                    num_data=1,
                    num_features=self.num_features,
                    num_nodes=self.num_nodes,
                    anomaly_p=1.0,
                    edge_prob=self.expected_degree / (self.num_nodes - 1),
                    bias=self.signal,
                    rng=np.random.default_rng(sub_seed),
                    var=self.var,
                    eeg_mode=self.eeg_mode,
                    eeg_A=self.eeg_A,
                    eeg_mu=eeg_mu,
                    eeg_cov=eeg_cov,
                    var_based_on_A=self.var_based_on_A,
                )
            except np.linalg.LinAlgError as e:
                continue

            (A, X), l = dataset[0] 
            if self.nongauss:
                X = torch.tensor(
                    self.nongauss_rv.rvs(size=X.size(), random_state=sub_seed),
                    dtype=torch.float32,
                )

            A = A.unsqueeze(dim=0)
            X = X.unsqueeze(dim=0)
            inputs = (A, X)

            if self.covariance_estimation:
                var = torch.var(X, unbiased=True).item()
            else:
                if self.eeg_mode in eeg_gen_mode_list:
                    var = eeg_cov.copy()
                else:
                    if self.var_based_on_A:
                        var = dataset.cov
                    else:
                        var = self.var if type(self.var) is float else self.var.copy()

            try:
                p_result = self.si_model.inference(
                    inputs, var=var, inference_mode="parametric", step=1e-10
                )
                selective_p_value = p_result.p_value
                naive_p_value = p_result.naive_p_value()

                D = len(set(self.thres)) + 1 if isinstance(self.thres, list) else 2
                if self.xai_type == "cam":
                    log_num_comparisons = np.log(D) * self.num_nodes
                elif self.xai_type == "stcam":
                    assert self.num_features == 50
                    chunk_length = 5
                    num_chunks = self.num_features // chunk_length
                    log_num_comparisons = np.log(D) * self.num_nodes * num_chunks

                bonferroni_p_value = p_result.bonferroni_p_value(log_num_comparisons)
                oc_p_value = 0
                oc_p_value = self.si_model.inference(
                    inputs, var=var, inference_mode="over_conditioning", step=1e-10
                ).p_value


            except (ThresholdIndexError, NegativeCaseTestError):
                selective_p_value = None
                naive_p_value = None
                oc_p_value = None
                bonferroni_p_value = None

            if selective_p_value is not None:
                break

        return (
            selective_p_value,
            naive_p_value,
            bonferroni_p_value,
            oc_p_value,
            i,
            sub_seed,
        )

    def run_experiment(self):
        dataset = list(range(self.num_results))

        self.results = {}
        (
            selective_p_values,
            naive_p_values,
            bonferroni_p_values,
            oc_p_values,
            num_iters,
            sub_seed,
        ) = self.experiment(dataset)

        self.results["proposed"] = selective_p_values
        self.results["naive"] = naive_p_values
        self.results["oc"] = oc_p_values
        self.results["bonferroni"] = bonferroni_p_values
        self.results["num_iters"] = num_iters
        self.results["seed"] = sub_seed


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{value}'")


def eeg_mode_type(value):
    if not value:
        return value
    if value.lower() in {"false", "0", "no"}:
        return False
    elif value.lower() in eeg_gen_mode_list:
        return str(value)
    else:
        raise argparse.ArgumentTypeError(f"{eeg_gen_mode_list} expected, got '{value}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=55)
    parser.add_argument("--signal", type=float, default=0.0)
    parser.add_argument("--num_nodes", type=int, default=32)
    parser.add_argument("--num_features", type=int, default=1)
    parser.add_argument("--lower_thres", type=float, default=0.5)
    parser.add_argument("--upper_thres", type=float, default=0.5)
    parser.add_argument("--inspect_iteration", type=int, default=None)
    parser.add_argument("--test_class_1_only", type=str_to_bool, default=False)
    parser.add_argument("--xai_type", type=str, default="cam")
    parser.add_argument("--eeg_mode", type=eeg_mode_type, default=False)
    parser.add_argument(
        "--eeg_subid", type=int, default=0, help="EEG mode specific parameter"
    )
    parser.add_argument(
        "--eeg_exp_type", type=str, default="fpr", help="EEG mode specific parameter"
    )
    parser.add_argument("--eeg_gamma", type=float, default=1.0)
    parser.add_argument("--corr", type=str_to_bool, default=False)
    parser.add_argument("--nongauss", type=str_to_bool, default=False)
    parser.add_argument("--nongauss_type", type=str, default="skewnorm")
    parser.add_argument("--nongauss_wd", type=float, default=0.1)
    parser.add_argument("--covariance_estimation", type=str_to_bool, default=False)
    args = parser.parse_args()

    experiment = GNNExperiment(
        num_results=args.num_results,
        num_worker=args.num_worker,
        signal=args.signal,
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        corr=args.corr,
        lower_thres=args.lower_thres,
        upper_thres=args.upper_thres,
        inspect_iteration=args.inspect_iteration,
        test_class_1_only=args.test_class_1_only,
        xai_type=args.xai_type,
        eeg_mode=args.eeg_mode,
        eeg_subid=args.eeg_subid,
        eeg_exp_type=args.eeg_exp_type,
        eeg_gamma=args.eeg_gamma,
        nongauss=args.nongauss,
        nongauss_type=args.nongauss_type,
        nongauss_wd=args.nongauss_wd,
        covariance_estimation=args.covariance_estimation,
    )
    experiment.run_experiment()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(current_dir, "../results")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if not args.eeg_mode:
        eeg_tag = ""
    else:
        eeg_tag = f"_eeg_{args.eeg_mode}{args.eeg_gamma}_sub{args.eeg_subid:02}_{args.eeg_exp_type}"

    if args.nongauss:
        nongauss_tag = f"_nongauss_{args.nongauss_type}_wd{args.nongauss_wd}"
    else:
        nongauss_tag = ""

    file_name = (
        f"sig{args.signal}"
        f"_n{args.num_nodes}"
        f"_f{args.num_features}"
        f"{'_corr' if args.corr else ''}"
        f"_l{args.lower_thres}"
        f"_u{args.upper_thres}"
        f"{'_c1' if args.test_class_1_only else ''}"
        f"_{args.xai_type}"
        f"{nongauss_tag}"
        f"{'_covest' if args.covariance_estimation else ''}"
        f"{eeg_tag}.pkl"
    )

    if args.inspect_iteration is not None:
        file_name = f"debug{args.inspect_iteration}_" + file_name

    file_path = os.path.join(result_path, file_name)

    print(file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
