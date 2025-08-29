import warnings
import torch
import numpy as np
from .utils import thresholding
from sicore import SelectiveInferenceNorm
from abc import ABC, abstractmethod


class Hypothesis(ABC):
    @abstractmethod
    def construct_hypothesis(self, X, si_model, *kwargs):
        pass

    @abstractmethod
    def model_selector(self, salient_vector, **kwargs):
        pass

    @abstractmethod
    def algorithm(self, a, b, z, **kwargs):
        pass


class BackgroundMeanDiff(Hypothesis):
    def __init__(
        self,
        threshold: float,
        i_idx: int = 0,
        o_idx: int = 0,
        io_diff: bool = False,
        use_sigmoid: bool = False,
        use_abs: bool = False,
        use_norm: bool = False,
        memoization: bool = True,
        **kwargs,
    ):
        """Hypothesis for the mean difference between the salient region and the background region in the input data.

        Args:
            threshold (float): Threshold value for `salient_region`.
            i_idx (int, optional): The index of the input to use for the test statistic.
                This option is for models with multiple inputs.
                Defaults to 0.
            o_idx (int, optional): The index of the output to use for the calculation of the `salient_region`.
                This option is for models with multiple outputs.
                Defaults to 0.
            io_diff (bool, optional): Whether to calculate the salient region by the `output` or the `output - input`.
                If True, the salient region is calculated by the `output - input`.
                Otherwise, the salient region is calculated only by the output.
            use_sigmoid (bool, optional): Whether to use sigmoid for the model output.
                If sigmoid is used for `output[o_idx]`, set this to True.
            use_abs (bool, optional): Whether to take the absolute value of the `saliency_map`.
                If True, abs(output) or abs(output - input) based on `io_diff`
            use_norm (bool, optional): Whether to apply min-max normalization to the `saliency_map`.
                If True, the `output` (if `io_diff` is False) or the `output - input` (if `io_diff` is True) is normalized.
                If `use_abs` is True, it is applied after taking the absolute value.
            memoization (bool, optional): Whether to use memoization.
                If True, the memoization is enabled
        """
        if use_sigmoid and io_diff:
            warnings.warn(
                "use_sigmoid and io_diff are both True. \
                This may cause unexpected results or calculation errors."
            )

        self.thr = torch.tensor(threshold, dtype=torch.float64)
        self.io_diff = io_diff
        self.memoization = memoization
        self.use_abs = use_abs
        self.use_norm = use_norm
        self.use_sigmoid = use_sigmoid
        self.i_idx = i_idx
        self.o_idx = o_idx

    def construct_hypothesis(
        self, X: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ):
        """
        Args:
            X (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]): The input data.
        """
        if isinstance(X, (list)):
            list_X = X
        if isinstance(X, (tuple)):
            list_X = list(X)
        else:
            list_X = [X]

        self.num_inputs = len(list_X)
        self.shape = list_X[self.i_idx].shape
        input_x = list_X[self.i_idx]
        self.saved_inputs = list_X

        output = self.si_model.forward(X)
        self.output = output

        if not isinstance(output, (tuple, list)):
            output_x = output
        else:
            output_x = output[self.o_idx]

        if self.io_diff:
            saliency_map = output_x - input_x
        else:
            saliency_map = output_x

        if self.use_abs:
            saliency_map = torch.abs(saliency_map)

        if self.use_norm:
            score_max = torch.max(saliency_map)
            score_min = torch.min(saliency_map)
            saliency_map = (saliency_map - score_min) / (score_max - score_min)

        saliency_map = saliency_map.squeeze()
        salient_region = saliency_map > self.thr
        self.salient_region = salient_region

        salient_vector = salient_region.reshape(-1).int()
        self.salient_vector = salient_vector

        input_vec = input_x.reshape(-1).double()
        eta = (
            salient_vector / torch.sum(salient_vector)
            - (1 - salient_vector) / torch.sum(1 - salient_vector)
        ).double()

        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )

        assert not np.isnan(self.si_calculator.stat)

    def model_selector(self, salient_vector: torch.Tensor) -> bool:
        """
        Args:
            salient_vector (torch.Tensor): should be a tensor of int
        Returns:
            bool: True if the input `salient_vector` is the same as the `self.salient_vector` in the `construct_hypothesis` method.
        """
        return torch.all(torch.eq(self.salient_vector, salient_vector))

    def algorithm(
        self, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            a (torch.Tensor): The input tensor a.
            b (torch.Tensor): The input tensor b.
            z (torch.Tensor): The input tensor z.
        Returns:
            salient_vector (torch.Tensor): The flattened salient region.
            [l, u] (list): The over-conditioning interval that contains z.
        """
        x = a + b * z

        INF = torch.tensor(float("inf"), dtype=torch.float64)
        input_x = self.saved_inputs
        input_a = [None] * self.num_inputs
        input_b = [None] * self.num_inputs
        input_l = [-INF] * self.num_inputs
        input_u = [INF] * self.num_inputs
        input_x[self.i_idx] = x.reshape(self.shape).double()
        input_a[self.i_idx] = a.reshape(self.shape).double()
        input_b[self.i_idx] = b.reshape(self.shape).double()
        input_l[self.i_idx] = -INF
        input_u[self.i_idx] = INF

        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x if self.num_inputs > 1 else input_x[self.i_idx],
            input_a if self.num_inputs > 1 else input_a[self.i_idx],
            input_b if self.num_inputs > 1 else input_b[self.i_idx],
            input_l if self.num_inputs > 1 else input_l[self.i_idx],
            input_u if self.num_inputs > 1 else input_u[self.i_idx],
            z,
            memoization=self.memoization,
        )

        if not isinstance(output_x, (list, tuple)):
            output_x = [output_x]
            output_a = [output_a]
            output_b = [output_b]
        else:
            l = l[self.o_idx]
            u = u[self.o_idx]

        if self.io_diff:
            saliency_x = output_x[self.o_idx] - input_x[self.i_idx]
            saliency_a = output_a[self.o_idx] - input_a[self.i_idx]
            saliency_b = output_b[self.o_idx] - input_b[self.i_idx]
        else:
            saliency_x = output_x[self.o_idx]
            saliency_a = output_a[self.o_idx]
            saliency_b = output_b[self.o_idx]

        salient_vector, l, u = thresholding(
            self.thr,
            saliency_x,
            saliency_a,
            saliency_b,
            l,
            u,
            z,
            use_sigmoid=self.use_sigmoid,
            use_abs=self.use_abs,
            use_norm=self.use_norm,
        )

        return salient_vector, [l, u]


class NeighborhoodMeanDiff(Hypothesis):
    def __init__(
        self,
        threshold: float,
        neighborhood_range: int = 1,
        i_idx: int = 0,
        o_idx: int = 0,
        io_diff: bool = False,
        use_sigmoid: bool = False,
        use_abs: bool = False,
        use_norm: bool = False,
        memoization: bool = True,
        **kwargs,
    ):
        """
        Args:
            threshold (float): Threshold value for `salient_region`.
            neighborhood_range (int): The range of the neighborhood region.
                If the `salient_region` is True at (i, j),
                the neighborhood region is True at (i - neighborhood_range, j - neighborhood_range)
                to (i + neighborhood_range, j + neighborhood_range).
                Defaults to 1.
            i_idx (int): The index of the input to use for the test statistic.
                This option is for models with multiple inputs.
                Defaults to 0.
            o_idx (int): The index of the output to use for the calculation of the `salient_region`.
                This option is for models with multiple outputs.
                Defaults to 0.
            io_diff (bool, optional): Whether to calculate the salient region by the `output` or the `output - input`.
                If True, the salient region is calculated by the `output - input`.
                Otherwise, the salient region is calculated only by the output.
            use_sigmoid (bool): Whether to use sigmoid for the model output.
                If sigmoid is used for `output[o_idx]`, set this to True.
            use_abs (bool): Whether to take the absolute value of the `saliency_map`.
                If True, abs(output) or abs(output - input) based on `io_diff`
            use_norm (bool, optional): Whether to apply min-max normalization to the `saliency_map`.
                If True, the `output` (if `io_diff` is False) or the `output - input` (if `io_diff` is True) is normalized.
                If `use_abs` is True, it is applied after taking the absolute value.
            memoization (bool): Whether to use memoization.
                If True, the memoization is enabled
        """
        if use_sigmoid and io_diff:
            warnings.warn(
                "use_sigmoid and io_diff are both True. \
                This may cause unexpected results or calculation errors."
            )

        self.thr = torch.tensor(threshold, dtype=torch.float64)
        self.io_diff = io_diff
        self.i_idx = i_idx
        self.o_idx = o_idx
        self.memoization = memoization
        self.neighborhood_range = neighborhood_range
        self.use_abs = use_abs
        self.use_norm = use_norm
        self.use_sigmoid = use_sigmoid

    def construct_hypothesis(
        self, X: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ):
        """
        Args:
            X (torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]): The input data.
        """
        if isinstance(X, (list)):
            list_X = X
        if isinstance(X, (tuple)):
            list_X = list(X)
        else:
            list_X = [X]

        self.num_inputs = len(list_X)
        self.shape = list_X[self.i_idx].shape
        input_x = list_X[self.i_idx]
        self.saved_inputs = list_X

        output = self.si_model.forward(X)
        self.output = output
        if not isinstance(output, (tuple, list)):
            output_x = output
        else:
            output_x = output[self.o_idx]

        if self.io_diff:
            saliency_map = output_x - input_x
        else:
            saliency_map = output_x

        if self.use_abs:
            saliency_map = torch.abs(saliency_map)

        if self.use_norm:
            score_max = torch.max(saliency_map)
            score_min = torch.min(saliency_map)
            saliency_map = (saliency_map - score_min) / (score_max - score_min)

        saliency_map = saliency_map.squeeze()
        salient_region = saliency_map > self.thr
        self.salient_region = salient_region
        salient_vector = salient_region.reshape(-1).int()
        self.salient_vector = salient_vector

        # select the area around the selected area by neighborhood_range pixels
        neighborhood_region = torch.zeros_like(salient_region)
        height = salient_region.shape[0]
        width = salient_region.shape[1]
        for i in range(height):
            for j in range(width):
                if salient_region[i, j] == 1:
                    i_start = max(0, i - self.neighborhood_range)
                    i_end = min(height, i + self.neighborhood_range + 1)
                    j_start = max(0, j - self.neighborhood_range)
                    j_end = min(width, j + self.neighborhood_range + 1)
                    neighborhood_region[i_start:i_end, j_start:j_end] = 1
        neighborhood_region = neighborhood_region.logical_xor(salient_region)
        neighborhood_index = neighborhood_region.reshape(-1) > 0
        self.neighborhood_region = neighborhood_region
        self.neighborhood_index = neighborhood_index

        input_vec = input_x.reshape(-1).double()
        eta = (
            salient_vector / torch.sum(salient_vector)
            - (neighborhood_index) / torch.sum(neighborhood_index)
        ).double()

        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )

        assert not np.isnan(self.si_calculator.stat)

    def model_selector(self, salient_vector: torch.Tensor) -> bool:
        """
        Args:
            salient_vector (torch.Tensor): should be a tensor of int
        Returns:
            bool: True if the input `salient_vector` is the same as the `self.salient_vector` in the `construct_hypothesis` method.
        """
        return torch.all(torch.eq(self.salient_vector, salient_vector))

    def algorithm(
        self, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            a (torch.Tensor): The input tensor a.
            b (torch.Tensor): The input tensor b.
            z (torch.Tensor): The input tensor z.
        Returns:
            salient_vector (torch.Tensor): The flattened salient region.
            [l, u] (list): The over-conditioning interval that contains z.
        """
        x = a + b * z

        INF = torch.tensor(float("inf"), dtype=torch.float64)
        input_x = self.saved_inputs
        input_a = [None] * self.num_inputs
        input_b = [None] * self.num_inputs
        input_l = [-INF] * self.num_inputs
        input_u = [INF] * self.num_inputs
        input_x[self.i_idx] = x.reshape(self.shape).double()
        input_a[self.i_idx] = a.reshape(self.shape).double()
        input_b[self.i_idx] = b.reshape(self.shape).double()
        input_l[self.i_idx] = -INF
        input_u[self.i_idx] = INF

        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x if self.num_inputs > 1 else input_x[self.i_idx],
            input_a if self.num_inputs > 1 else input_a[self.i_idx],
            input_b if self.num_inputs > 1 else input_b[self.i_idx],
            input_l if self.num_inputs > 1 else input_l[self.i_idx],
            input_u if self.num_inputs > 1 else input_u[self.i_idx],
            z,
            memoization=self.memoization,
        )

        if not isinstance(output_x, (list, tuple)):
            output_x = [output_x]
            output_a = [output_a]
            output_b = [output_b]
        else:
            l = l[self.o_idx]
            u = u[self.o_idx]

        if self.io_diff:
            saliency_x = output_x[self.o_idx] - input_x[self.i_idx]
            saliency_a = output_a[self.o_idx] - input_a[self.i_idx]
            saliency_b = output_b[self.o_idx] - input_b[self.i_idx]
        else:
            saliency_x = output_x[self.o_idx]
            saliency_a = output_a[self.o_idx]
            saliency_b = output_b[self.o_idx]

        salient_vector, l, u = thresholding(
            self.thr,
            saliency_x,
            saliency_a,
            saliency_b,
            l,
            u,
            z,
            use_sigmoid=self.use_sigmoid,
            use_abs=self.use_abs,
            use_norm=self.use_norm,
        )

        return salient_vector, [l, u]


class ReferenceMeanDiff(Hypothesis):
    def __init__(
        self,
        threshold: float,
        reference_data: torch.Tensor,
        i_idx: int = 0,
        o_idx: int = 0,
        io_diff: bool = False,
        use_sigmoid: bool = False,
        use_abs: bool = False,
        use_norm: bool = False,
        memoization: bool = True,
        **kwargs,
    ):
        """
        Args:
            threshold (float): Threshold value for `salient_region`.
            reference_data (torch.Tensor): The reference data for comparison with the input data.
            i_idx (int, optional): The index of the input to use for the test statistic.
                This option is for models with multiple inputs.
                Defaults to 0.
            o_idx (int, optional): The index of the output to use for the calculation of the `salient_region`.
                This option is for models with multiple outputs.
                Defaults to 0.
            io_diff (bool, optional): Whether to calculate the salient region by the `output` or the `output - input`.
                If True, the salient region is calculated by the `output - input`.
                Otherwise, the salient region is calculated only by the output.
            use_sigmoid (bool, optional): Whether to use sigmoid for the model output.
                If sigmoid is used for `output[o_idx]`, set this to True.
            use_abs (bool, optional): Whether to take the absolute value of the `saliency_map`.
                If True, abs(output) or abs(output - input) based on `io_diff`
            use_norm (bool, optional): Whether to apply min-max normalization to the `saliency_map`.
                If True, the `output` (if `io_diff` is False) or the `output - input` (if `io_diff` is True) is normalized.
                If `use_abs` is True, it is applied after taking the absolute value.
            memoization (bool, optional): Whether to use memoization.
                If True, the memoization is enabled
        """
        if use_sigmoid and io_diff:
            warnings.warn(
                "use_sigmoid and io_diff are both True. \
                This may cause unexpected results or calculation errors."
            )

        self.thr = torch.tensor(threshold, dtype=torch.float64)
        self.reference_data = reference_data
        self.io_diff = io_diff
        self.i_idx = i_idx
        self.o_idx = o_idx
        self.use_sigmoid = use_sigmoid
        self.use_abs = use_abs
        self.use_norm = use_norm
        self.memoization = memoization

    def construct_hypothesis(self, X):
        if isinstance(X, (list)):
            list_X = X
        if isinstance(X, (tuple)):
            list_X = list(X)
        else:
            list_X = [X]

        self.num_inputs = len(list_X)
        self.shape = list_X[self.i_idx].shape
        input_x = list_X[self.i_idx]
        self.saved_inputs = list_X

        output = self.si_model.forward(X)
        self.output = output

        if not isinstance(output, (tuple, list)):
            output_x = output
        else:
            output_x = output[self.o_idx]

        if self.io_diff:
            saliency_map = output_x - input_x
        else:
            saliency_map = output_x

        if self.use_abs:
            saliency_map = torch.abs(saliency_map)

        if self.use_norm:
            score_max = torch.max(saliency_map)
            score_min = torch.min(saliency_map)
            saliency_map = (saliency_map - score_min) / (score_max - score_min)

        salient_region = saliency_map > self.thr
        self.salient_region = salient_region

        salient_vector = salient_region.reshape(-1).int()
        self.salient_vector = salient_vector

        input_vec = torch.cat(
            [input_x.reshape(-1), self.reference_data.reshape(-1)]
        ).double()

        eta = torch.cat(
            [
                salient_vector / torch.sum(salient_vector),
                -salient_vector / torch.sum(salient_vector),
            ]
        ).double()

        self.si_calculator = SelectiveInferenceNorm(
            input_vec, self.var, eta, use_torch=True
        )

        assert not np.isnan(self.si_calculator.stat)

    def model_selector(self, salient_vector):
        """
        Args:
            salient_vector (torch.Tensor): should be a tensor of int
        Returns:
            bool: True if the salient_vector is the same as the salient_vector in the construct_hypothesis method
        """
        return torch.all(torch.eq(self.salient_vector, salient_vector))

    def algorithm(self, a, b, z):
        a = a[: len(a) // 2]
        b = b[: len(b) // 2]
        x = a + b * z

        INF = torch.tensor(float("inf"), dtype=torch.float64)
        input_x = self.saved_inputs
        input_a = [None] * self.num_inputs
        input_b = [None] * self.num_inputs
        input_l = [-INF] * self.num_inputs
        input_u = [INF] * self.num_inputs
        input_x[self.i_idx] = x.reshape(self.shape).double()
        input_a[self.i_idx] = a.reshape(self.shape).double()
        input_b[self.i_idx] = b.reshape(self.shape).double()
        input_l[self.i_idx] = -INF
        input_u[self.i_idx] = INF

        output_x, output_a, output_b, l, u = self.si_model.forward_si(
            input_x if self.num_inputs > 1 else input_x[self.i_idx],
            input_a if self.num_inputs > 1 else input_a[self.i_idx],
            input_b if self.num_inputs > 1 else input_b[self.i_idx],
            input_l if self.num_inputs > 1 else input_l[self.i_idx],
            input_u if self.num_inputs > 1 else input_u[self.i_idx],
            z,
            memoization=self.memoization,
        )

        if not isinstance(output_x, (list, tuple)):
            output_x = [output_x]
            output_a = [output_a]
            output_b = [output_b]
        else:
            l = l[self.o_idx]
            u = u[self.o_idx]

        if self.io_diff:
            saliency_x = output_x[self.o_idx] - input_x[self.i_idx]
            saliency_a = output_a[self.o_idx] - input_a[self.i_idx]
            saliency_b = output_b[self.o_idx] - input_b[self.i_idx]
        else:
            saliency_x = output_x[self.o_idx]
            saliency_a = output_a[self.o_idx]
            saliency_b = output_b[self.o_idx]

        salient_vector, l, u = thresholding(
            self.thr,
            saliency_x,
            saliency_a,
            saliency_b,
            l,
            u,
            z,
            use_sigmoid=self.use_sigmoid,
            use_abs=self.use_abs,
            use_norm=self.use_norm,
        )

        return salient_vector, [l, u]
