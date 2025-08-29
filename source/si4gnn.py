import sys
import os
from pathlib import Path
import numpy as np
import onnx
from sicore import SelectiveInferenceNorm

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from si4onnx import si
from si4onnx.utils import NormilzedThresholding
from si4onnx.layers import truncated_interval
from data_utils import synth_data


class ThresholdIndexError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NegativeCaseTestError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def is_int_or_float(value):
    """Check if it is a scalar value.

    Check if a value (including a numpy object) has integer or floating data type. Note
    that infinity values have floating data type.
    """
    type_ = type(value)
    return np.issubdtype(type_, np.integer) or np.issubdtype(type_, np.floating)


class SI4ONNX(si.SI4ONNX):
    def __init__(
        self,
        model,
        thr,
        num_features,
        apply_norm=True,
        test_class_1_only=False,
        time_window=None,
    ):
        super().__init__(model)
        self.model = model
        self.thresholding = NormilzedThresholding(thr, apply_norm=apply_norm)
        self.num_features = num_features
        self.test_class_1_only = test_class_1_only
        self.time_window = time_window
        if not isinstance(thr, list) or len(thr) == 1:
            self.is_double_thresholing = False
        else:
            assert len(thr) == 2, "the number of thresholds must be 1 or 2"
            assert (
                thr[0] <= thr[1]
            ), "the first threshold must be smaller than the second one"
            self.is_double_thresholing = True

    def construct_hypothesis(self, inputs):

        self.interval_log = {}

        self.input_A = inputs[0]  # constant input
        input_X = inputs[1]
        self.shape = input_X.shape  # B, N, T

        assert torch.all(torch.diagonal(self.input_A, dim1=1, dim2=2) != 0)
        assert torch.allclose(
            torch.sum(self.input_A, dim=2),
            torch.ones(1, 1, self.input_A.size(1)),
            atol=1e-6,
        )

        cam_output, logits = self.si_model.forward(inputs)
        if self.test_class_1_only and (logits[0][0] > logits[0][1]):
            raise NegativeCaseTestError("The model predicts class 0")

        anomaly_index = self.thresholding.forward(cam_output[:, :, 1].flatten())
        self.anomaly_index_obs = anomaly_index

        if not self.is_double_thresholing:
            eta = (
                (
                    anomaly_index / torch.sum(anomaly_index)
                    - (1 - anomaly_index) / torch.sum(1 - anomaly_index)
                )
                .double()
                .repeat_interleave(self.num_features)
            )
        else:
            normal_index_ = 1 - anomaly_index[0]
            anomaly_index_ = anomaly_index[1]
            eta = (
                (
                    anomaly_index_ / torch.sum(anomaly_index_)
                    - normal_index_ / torch.sum(normal_index_)
                )
                .double()
                .repeat_interleave(self.num_features)
            )
        if self.time_window is not None:
            for i in range(0, len(eta), self.num_features):
                eta[i : i + self.time_window[0]] = 0
                eta[i + self.time_window[1] : i + self.num_features] = 0

        var = self.var
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
        if is_int_or_float(var):
            self.sigma_eta = var * eta
        elif len(var.shape) == 1:
            vars = var
            self.sigma_eta = vars * eta
        elif len(var.shape) == 2:
            cov = var
            self.sigma_eta = cov @ eta
        eta_sigma_eta = eta @ self.sigma_eta
        assert eta_sigma_eta.dim() == 0
        eta = eta / eta_sigma_eta

        input_vec = input_X.reshape(-1).double()
        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )

        if np.isnan(self.si_calculator.stat):
            raise ThresholdIndexError(
                f"All values in cam is less/greater than threshold:\n {cam_output[:,:,1]=}"
            )


    def is_equal_anomaly_index(self, anomaly_index):
        is_equal = torch.all(torch.eq(self.anomaly_index_obs, anomaly_index))
        return is_equal

    def model_selector(self, model):
        if self.test_class_1_only:
            anomaly_index, is_class1 = model
            return self.is_equal_anomaly_index(anomaly_index) and is_class1
        else:
            anomaly_index = model
            return self.is_equal_anomaly_index(anomaly_index)

    def algorithm(self, a, b, z):
        x = a + b * z 
        B, N, T = self.shape

        input_A_a, input_A_b = (
            self.input_A,
            None,
        )

        input_X_x = x.reshape(self.shape)
        input_X_a = a.reshape(self.shape)
        input_X_b = b.reshape(self.shape)
        INF = torch.tensor(float("inf"), dtype=torch.float64)
        l = -INF
        u = INF

        input_x = [self.input_A, input_X_x]
        input_a = [input_A_a, input_X_a]
        input_b = [input_A_b, input_X_b]
        input_l = [-torch.tensor(float("inf")), l]
        input_u = [torch.tensor(float("inf")), u]

        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x, input_a, input_b, input_l, input_u, z
        )

        assert l[1] <= l[0] and u[0] <= u[1]
        l = torch.max(*l)
        u = torch.min(*u)
        if float(z) in self.interval_log:
            raise Exception("z is already in the interval_log")
        self.interval_log[float(z)] = [float(l), float(u)]

        output_cam_a, output_logits_a = output_a
        output_cam_b, output_logits_b = output_b

        if self.test_class_1_only:
            output_logits_a = output_logits_a.flatten()
            output_logits_b = output_logits_b.flatten()
            logits = output_logits_a + output_logits_b * z
            delta = lambda x: (
                x[0] - x[1] if logits[0] <= logits[1] else x[1] - x[0]
            ).view(1)
            l_logits, u_logits = truncated_interval(
                delta(output_logits_a), delta(output_logits_b)
            )
            l = torch.max(l, l_logits)
            u = torch.min(u, u_logits)
            assert l <= z
            assert z <= u

        output_cam_a = output_cam_a[:, :, 1].flatten()
        output_cam_b = output_cam_b[:, :, 1].flatten()
        anomaly_index, l, u = self.thresholding.forward_si(
            output_cam_a, output_cam_b, l, u, z
        )

        self.interval_log[float(z)].append(l)
        self.interval_log[float(z)].append(u)

        assert l <= z
        assert z <= u
        if self.test_class_1_only:
            return (anomaly_index, torch.argmax(logits) == 1), [l, u]
        else:
            return anomaly_index, [l, u]


class SI4TSGNN(si.SI4ONNX):

    def __init__(
        self,
        model,
        thr,
        num_features,
        apply_norm=True,
        test_class_1_only=False,
        time_window=None,
        chunk_length=0,
        chunk_slide_width=0,
    ):
        super().__init__(model)
        self.model = model
        self.thresholding = NormilzedThresholding(thr, apply_norm=apply_norm)
        self.num_features = num_features
        self.test_class_1_only = test_class_1_only
        self.time_window = time_window
        self.chunk_length = chunk_length
        self.chunk_slide_width = chunk_slide_width
        self.num_chnks = num_features // chunk_slide_width
        if not isinstance(thr, list) or len(thr) == 1:
            self.is_double_thresholing = False
        else:
            assert len(thr) == 2, "the number of thresholds must be 1 or 2"
            assert (
                thr[0] <= thr[1]
            ), "the first threshold must be smaller than the second one"
            self.is_double_thresholing = True

    def construct_hypothesis(self, inputs):

        self.interval_log = {}

        self.input_A = inputs[0]  # constant input
        assert torch.all(torch.diagonal(self.input_A, dim1=1, dim2=2) != 0)
        assert torch.allclose(
            torch.sum(self.input_A, dim=2),
            torch.ones(1, 1, self.input_A.size(1)),
            atol=1e-6,
        )

        if self.chunk_length > 0:
            inputs = synth_data.chunk_time_series(
                inputs[0], inputs[1], self.chunk_length, self.chunk_slide_width
            )
        else:
            inputs = inputs[0], inputs[1]

        self.input_A = inputs[0]  # constant input

        input_X = inputs[1]
        self.shape = input_X.shape  # B, N, T

        cam_output, logits = self.si_model.forward(inputs)
        if self.test_class_1_only and (logits[0][0] > logits[0][1]):
            raise NegativeCaseTestError("The model predicts class 0")

        anomaly_index = self.thresholding.forward(cam_output[:, :, 1].flatten())
        self.anomaly_index_obs = anomaly_index

        if not self.is_double_thresholing:
            eta = (
                (
                    anomaly_index / torch.sum(anomaly_index)
                    - (1 - anomaly_index) / torch.sum(1 - anomaly_index)
                )
                .double()
                .repeat_interleave(self.chunk_length)
            )
        else:
            normal_index_ = 1 - anomaly_index[0]  # 第1閾値の未満のインデックス
            anomaly_index_ = anomaly_index[1]  # 第2閾値を超えたインデックス
            eta = (
                (
                    anomaly_index_ / torch.sum(anomaly_index_)
                    - normal_index_ / torch.sum(normal_index_)
                )
                .double()
                .repeat_interleave(self.chunk_length)
            )
        if self.time_window is not None:
            for i in range(0, len(eta), self.chunk_length):
                eta[i : i + self.time_window[0]] = 0
                eta[i + self.time_window[1] : i + self.chunk_length] = 0

        var = self.var
        if isinstance(var, np.ndarray):
            var = torch.from_numpy(var)
        if is_int_or_float(var):
            self.sigma_eta = var * eta
        elif len(var.shape) == 1:
            vars = var
            self.sigma_eta = vars * eta
        elif len(var.shape) == 2:
            cov = var
            self.sigma_eta = cov @ eta
        eta_sigma_eta = eta @ self.sigma_eta
        assert eta_sigma_eta.dim() == 0
        eta = eta / eta_sigma_eta

        input_vec = input_X.reshape(-1).double()
        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )
        
        if np.isnan(self.si_calculator.stat):
            raise ThresholdIndexError(
                f"All values in cam is less/greater than threshold:\n {cam_output[:,:,1]=}"
            )


    def is_equal_anomaly_index(self, anomaly_index):
        is_equal = torch.all(torch.eq(self.anomaly_index_obs, anomaly_index))
        return is_equal

    def model_selector(self, model):
        if self.test_class_1_only:
            anomaly_index, is_class1 = model
            return self.is_equal_anomaly_index(anomaly_index) and is_class1
        else:
            anomaly_index = model
            return self.is_equal_anomaly_index(anomaly_index)

    def algorithm(self, a, b, z):
        x = a + b * z 
        B, N, T = self.shape

        input_A_a, input_A_b = (
            self.input_A,
            None,
        )

        input_X_x = x.reshape(self.shape)
        input_X_a = a.reshape(self.shape)
        input_X_b = b.reshape(self.shape)
        INF = torch.tensor(float("inf"), dtype=torch.float64)
        l = -INF
        u = INF

        input_x = [self.input_A, input_X_x]
        input_a = [input_A_a, input_X_a]
        input_b = [input_A_b, input_X_b]
        input_l = [-torch.tensor(float("inf")), l]
        input_u = [torch.tensor(float("inf")), u]

        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x, input_a, input_b, input_l, input_u, z
        )

        assert l[1] <= l[0] and u[0] <= u[1]
        l = torch.max(*l)
        u = torch.min(*u)
        if float(z) in self.interval_log:
            raise Exception("z is already in the interval_log")
        self.interval_log[float(z)] = [float(l), float(u)]

        output_cam_a, output_logits_a = output_a
        output_cam_b, output_logits_b = output_b

        if self.test_class_1_only:
            output_logits_a = output_logits_a.flatten()
            output_logits_b = output_logits_b.flatten()
            logits = output_logits_a + output_logits_b * z
            delta = lambda x: (
                x[0] - x[1] if logits[0] <= logits[1] else x[1] - x[0]
            ).view(1)
            l_logits, u_logits = truncated_interval(
                delta(output_logits_a), delta(output_logits_b)
            )
            l = torch.max(l, l_logits)
            u = torch.min(u, u_logits)
            assert l <= z
            assert z <= u

        output_cam_a = output_cam_a[:, :, 1].flatten()
        output_cam_b = output_cam_b[:, :, 1].flatten()
        anomaly_index, l, u = self.thresholding.forward_si(
            output_cam_a, output_cam_b, l, u, z
        )

        self.interval_log[float(z)].append(l)
        self.interval_log[float(z)].append(u)

        assert l <= z
        assert z <= u
        if self.test_class_1_only:
            return (anomaly_index, torch.argmax(logits) == 1), [l, u]
        else:
            return anomaly_index, [l, u]
