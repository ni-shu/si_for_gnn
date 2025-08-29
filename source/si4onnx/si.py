from abc import ABC, abstractmethod
from typing import Any, Literal

from sicore import SelectiveInferenceNorm, SelectiveInferenceResult

from . import nn


class NoHypothesisError(Exception):
    """If the hypothesis is not obtained from observartion, please raise this error"""

    pass


class SI4ONNX(ABC):
    def __init__(self, model):
        self.model = model
        self.si_model = nn.NN(model)
        self.si_calculator: (
            SelectiveInferenceNorm  # | SelectiveInferenceChiSquared = None
        )
        self.outputs = None

    @abstractmethod
    def construct_hypothesis(self, output):
        """Abstruct method for construct hypothesis from the observed output of NN.

        Args:
            output(tf.Tensor): The observed output of NN

        Returns:
            void :

        Raises
            NoHypothesisError: When hypothesis is not obtained from the output, raise this error.
        """
        pass

    @abstractmethod
    def algorithm(self, a, b, z) -> tuple[Any, tuple[float, float]]:
        """

        Args:
            a: A vector of nuisance parameter
            b: A vector of the direction of test statistic
            z: A test statistic

        Returns:
            Tuple(Any,Tuple(float,float)):First Elements is outputs obtained in the value of z. Second Element is a obtained truncated interval
        """
        pass

    @abstractmethod
    def model_selector(self, outputs) -> bool:
        """Abstruct method for compare whether same model are obtained from outputs and observed outputs(self.outputs)

        Args:
            outputs: outputs of NN

        Returns:
            bool: If same models are obtained from outputs and observed outputs(self.outputs), Return value should be true. If not, return value should be false.
        """
        pass

    def inference(self, inputs, var, **kwargs) -> SelectiveInferenceResult:
        self.var = var
        self.construct_hypothesis(inputs)
        result = self.si_calculator.inference(
            algorithm=self.algorithm,
            model_selector=self.model_selector,
            **kwargs,
        )
        return result


class LoadSI(ABC):
    def __init__(self, model, hypothesis, seed=None):
        self.si_model = nn.NN(model, seed)
        self.hypothesis = hypothesis
        self.si_calculator = None

    def construct_hypothesis(self, X, *kwargs):
        self.hypothesis.si_model = self.si_model
        self.hypothesis.construct_hypothesis(X, *kwargs)
        self.si_calculator = self.hypothesis.si_calculator
        self.output = self.hypothesis.output
        self.salient_region = self.hypothesis.salient_region

    def algorithm(self, a, b, z, **kwargs):
        return self.hypothesis.algorithm(a, b, z, **kwargs)

    def model_selector(self, anomaly_index, **kwargs):
        return self.hypothesis.model_selector(anomaly_index, **kwargs)

    def inference(self, inputs, var, **kwargs) -> SelectiveInferenceResult:
        self.hypothesis.var = var
        self.construct_hypothesis(inputs)
        result = self.si_calculator.inference(
            algorithm=self.algorithm,
            model_selector=self.model_selector,
            **kwargs,
        )
        return result
