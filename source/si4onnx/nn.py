import torch
import numpy as np
from onnx import numpy_helper
from .layers import *
from typing import Optional, Union


class NN:
    def __init__(self, model, seed=None):
        super(NN, self).__init__()
        self.model = model
        self.seed = seed
        self.output_name_set = set(output.name for output in self.model.graph.output)
        self.is_memoization_initialized = True

        # Available layers
        self.layers = {
            "Relu": Relu,
            "Sigmoid": Sigmoid,
            "Conv": Conv,
            "Gemm": Gemm,
            "MaxPool": MaxPool,
            "AveragePool": AveragePool,
            "GlobalAveragePool": GlobalAveragePool,
            "ConvTranspose": ConvTranspose,
            "Transpose": Transpose,
            "Shape": Shape,
            "Slice": Slice,
            "Exp": Exp,
            "Flatten": Flatten,
            "ConstantOfShape": ConstantOfShape,
            "EyeLike": EyeLike,
            "Reciprocal": Reciprocal,
            "ReduceMean": ReduceMean,
            # "Max": Max,
            # "Min": Min,
        }
        self.multi_input_layers = {
            "Reshape": Reshape,
            "Resize": Resize,
            "Concat": Concat,
            "Add": Add,
            "Sub": Sub,
            "Split": Split,
            "BatchNormalization": BatchNormalization,
            "Mul": Mul,
            "MatMul": MatMul,
            "Div": Div,
            "ReduceSum": ReduceSum,
            "Equal": Equal,
            "Greater": Greater,
            "Squeeze": Squeeze,
        }
        self.non_input_layers = {
            "Constant": Constant,
        }
        self.random_layers = {"RandomNormalLike": RandomNormalLike}

    @staticmethod
    def calculate_output_x(a, b, z):
        if isinstance(a, torch.Tensor) and b is not None:
            return a + b * z
        else:
            return a  # constant variable is equal to a

    def forward(self, input):
        """
        Args:
            input (torch.Tensor or List[torch.Tensor]): input tensor or tensor list
        Returns:
            output (torch.Tensor or List[torch.Tensor]): output tensor or tensor list
        """
        self.rng = np.random.default_rng(self.seed)

        node_output = dict()
        for i, input_node in enumerate(self.model.graph.input):
            if len(self.model.graph.input) == 1:
                node_output[input_node.name] = input.detach().to(torch.float64)
            else:
                node_output[input_node.name] = input[i].detach().to(torch.float64)

        for tensor in self.model.graph.initializer:
            arr = numpy_helper.to_array(tensor)
            if tensor.data_type == 7:
                arr = torch.tensor(arr, dtype=torch.int64)
            else:
                arr = torch.tensor(arr, dtype=torch.float64)
            node_output[tensor.name] = arr

        with torch.no_grad():
            for node in self.model.graph.node:
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                op_type = node.op_type
                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    x = node_output[node.input[0]]
                    outputs = layer.forward(x)
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    outputs = layer.forward()
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    outputs = layer.forward()
                elif op_type in self.random_layers:
                    layer = self.random_layers[op_type](inputs, node)
                    outputs = layer.forward(node.input[0], self.rng)
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(outputs, torch.Tensor) or op_type == "Constant":
                    node_output[node.output[0]] = outputs
                else:
                    for i, output_name in enumerate(node.output):
                        node_output[output_name] = outputs[i]

        self.output_obs = node_output
        outputs = [node_output[output.name] for output in self.model.graph.output]

        self.is_memoization_initialized = True

        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def forward_si(self, input, a, b, l, u, z, memoization=True):
        """
        Args:
            input (torch.Tensor or List[torch.Tensor]): input tensor or tensor list
            a (torch.Tensor or List[torch.Tensor]): a tensor or tensor list
            b (torch.Tensor or List[torch.Tensor]): b tensor or tensor list
            l (torch.Tensor or List[torch.Tensor]): l tensor or tensor list
            u (torch.Tensor or List[torch.Tensor]): u tensor or tensor list
        Returns:
            x (torch.Tensor or List[torch.Tensor]): output tensor or tensor list
            a (torch.Tensor or List[torch.Tensor]): output a tensor or tensor list
            b (torch.Tensor or List[torch.Tensor]): output b tensor or tensor list
            l (torch.Tensor or List[torch.Tensor]): output l tensor or tensor list
            u (torch.Tensor or List[torch.Tensor]): output u tensor or tensor list
        """

        node_output = dict()
        node_output_si = dict()
        start_node = None
        for i, input_node in enumerate(self.model.graph.input):
            if len(self.model.graph.input) == 1:
                node_output[input_node.name] = input.detach().to(torch.float64)
                if start_node is None:
                    start_node = input_node.name
                node_output_si[input_node.name] = (
                    a.detach().to(torch.float64),
                    b.detach().to(torch.float64),
                    l.detach().to(torch.float64),
                    u.detach().to(torch.float64),
                )
            else:
                node_output[input_node.name] = input[i].detach().to(torch.float64)
                if start_node is None:
                    start_node = input_node.name
                if a[i] is not None and b[i] is not None:
                    node_output_si[input_node.name] = (
                        a[i].detach().to(torch.float64),
                        b[i].detach().to(torch.float64),
                        l[i].detach().to(torch.float64),
                        u[i].detach().to(torch.float64),
                    )
                else:
                    node_output_si[input_node.name] = (
                        input[i].detach().to(torch.float64),
                        None,
                        torch.tensor(-float("inf"), dtype=torch.float64),
                        torch.tensor(float("inf"), dtype=torch.float64),
                    )

        if not self.is_memoization_initialized:
            node_output = self.output
            node_output_si = self.output_si
        else:
            for tensor in self.model.graph.initializer:
                arr = numpy_helper.to_array(tensor)
                if tensor.data_type == 7:
                    arr = torch.tensor(arr, dtype=torch.int64)
                else:
                    arr = torch.tensor(arr, dtype=torch.float64)

                node_output[tensor.name] = arr
                node_output_si[tensor.name] = (
                    arr,
                    None,
                    torch.tensor(-float("inf"), dtype=torch.float64),
                    torch.tensor(float("inf"), dtype=torch.float64),
                )

        # Find the start node
        output_layers_cnt = 0
        start_node_index = 0
        graph_size = len(self.model.graph.node)
        if memoization and not self.is_memoization_initialized:
            for node_index, node in enumerate(reversed(self.model.graph.node)):
                if node.output[0] in self.output_name_set:
                    output_layers_cnt += 1
                if output_layers_cnt < len(self.model.graph.output):
                    continue
                _a, _b, _l, _u = self.output_si[node.output[0]]
                if _b is not None and (not isinstance(_a, list)) and _l < z < _u:
                    start_node = node.name
                    start_node_index = graph_size - node_index
                    if output_layers_cnt == len(self.model.graph.output):
                        break
        # print("Skipped layers num:", start_node_index)  # Debug

        with torch.no_grad():
            for node in self.model.graph.node[start_node_index:]:
                op_type = node.op_type
                inputs = [
                    node_output[input_name]
                    for input_name in node.input
                    if input_name != ""
                ]
                # print(op_type) # Debug
                if op_type in self.layers:
                    layer = self.layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si(z, *node_output_si[node.input[0]])
                elif op_type in self.multi_input_layers:
                    layer = self.multi_input_layers[op_type](inputs, node, node_output)
                    a, b, l, u = layer.forward_si(node, node_output, node_output_si)
                elif op_type in self.non_input_layers:
                    layer = self.non_input_layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si()
                elif op_type in self.random_layers:
                    layer = self.random_layers[op_type](inputs, node)
                    a, b, l, u = layer.forward_si(z, self.output_obs[node.output[0]])
                else:
                    raise NotImplementedError(f"Layer {op_type} is not supported.")

                if isinstance(a, torch.Tensor) or op_type == "Constant":
                    assert l < u
                    node_output[node.output[0]] = self.calculate_output_x(a, b, z)
                    node_output_si[node.output[0]] = (a, b, l, u)
                else:
                    for i, output_name in enumerate(node.output):
                        assert l[i] < u[i]
                        node_output[output_name] = self.calculate_output_x(
                            a[i], b[i], z
                        )
                        node_output_si[output_name] = (a[i], b[i], l[i], u[i])

                # print(f"l: {l:.8f}, u: {u:.8f}") # Debug

        self.output = node_output
        self.output_si = node_output_si
        self.is_memoization_initialized = False

        x, output_a, output_b, l, u = zip(
            *[
                [
                    self.calculate_output_x(
                        node_output_si[output.name][0],
                        node_output_si[output.name][1],
                        z,
                    ),
                    node_output_si[output.name][0],
                    node_output_si[output.name][1],
                    node_output_si[output.name][2],
                    node_output_si[output.name][3],
                ]
                for output in self.model.graph.output
            ]
        )

        if len(x) == 1:
            return x[0], output_a[0], output_b[0], l[0], u[0]
        else:
            return x, output_a, output_b, l, u
