import numpy as np
import torch
import torch.nn as nn


@torch.jit.script
def truncated_interval(a, b):
    """
    Args:
        a: tensor of the a vector
        b: tensor of the b vector
    Returns:
        l: lower bound of the truncated interval
        u: upper bound of the truncated interval
    """
    nominator = torch.neg(a)
    denominator = b
    plus_index = torch.greater(denominator, 0)
    minus_index = torch.less(denominator, 0)

    if torch.any(minus_index):
        l = torch.max(torch.div(nominator[minus_index], denominator[minus_index]))
    else:
        l = torch.tensor(-float("inf"), dtype=torch.float64)
    if torch.any(plus_index):
        u = torch.min(torch.div(nominator[plus_index], denominator[plus_index]))
    else:
        u = torch.tensor(float("inf"), dtype=torch.float64)
    return l, u


class Layer:
    def __init__(self, inputs, node):
        """
        Args:
            inputs (List): input tensor list
            node: the node has the layer information
        """
        self.attribute = {}
        for attr in node.attribute:
            self.attribute[attr.name] = attr

    def forward(self):
        pass

    def forward_si(self):
        pass


class Relu(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = torch.nn.functional.relu(x)
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
            relu_index = torch.greater_equal(x, 0)
            tTa = torch.where(relu_index, -a, a)
            tTb = torch.where(relu_index, -b, b)

            temp_l, temp_u = truncated_interval(tTa, tTb)

            output_l = torch.max(l, temp_l)
            output_u = torch.min(u, temp_u)

            output_x = torch.where(relu_index, x, torch.tensor(0, dtype=torch.float64))
            output_a = torch.where(relu_index, a, torch.tensor(0, dtype=torch.float64))
            output_b = torch.where(relu_index, b, torch.tensor(0, dtype=torch.float64))
        else:
            x = a
            output_x = torch.nn.functional.relu(x)
            output_a = output_x
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)

        return output_a, output_b, output_l, output_u


class Sigmoid(Layer):
    """
    Sigmoid activation function can be used only in the intermediate layers
    that are not subject to the test or the final layer of the input subject to the test.
    """

    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = torch.nn.functional.sigmoid(x)
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            output_a = a
            output_b = b
        else:
            output_a = a
            output_b = None
        return output_a, output_b, l, u


class Conv(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].to(torch.float64)
        self.bias = inputs[2].to(torch.float64) if len(inputs) > 2 else None
        self.strides = (
            tuple(self.attribute["strides"].ints) if "strides" in self.attribute else 1
        )
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )
        self.pads = (
            "same"
            if "pads" not in self.attribute and "auto_pad" in self.attribute
            else self.pads
        )
        # attribute["kernel_shape"] not necessary since it can be obtained from weight

        self.dilations = (
            tuple(self.attribute["dilations"].ints)
            if "dilations" in self.attribute
            else 1
        )
        self.group = self.attribute["group"].i if "group" in self.attribute else 1

    def forward(self, x):
        output = torch.nn.functional.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
            groups=self.group,
        )
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.nn.functional.conv2d(
            input=a,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
            groups=self.group,
        )
        if b is not None:
            output_b = torch.nn.functional.conv2d(
                input=b,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
                groups=self.group,
            )
        else:
            output_b = None
        return output_a, output_b, l, u


class ConvTranspose(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].to(torch.float64)
        self.bias = inputs[2].to(torch.float64) if len(inputs) > 2 else None
        self.strides = (
            self.attribute["strides"].ints[0] if "strides" in self.attribute else 1
        )
        self.pads = self.attribute["pads"].ints[0] if "pads" in self.attribute else 0
        self.output_padding = (
            self.attribute["output_padding"].ints[0]
            if "output_padding" in self.attribute
            else 0
        )
        self.dilations = (
            self.attribute["dilations"].ints[0] if "dilations" in self.attribute else 1
        )

    def forward(self, x):
        output = torch.nn.functional.conv_transpose2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            output_padding=self.output_padding,
            dilation=self.dilations,
        )
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.nn.functional.conv_transpose2d(
            input=a,
            weight=self.weight,
            bias=self.bias,
            stride=self.strides,
            padding=self.pads,
            output_padding=self.output_padding,
            dilation=self.dilations,
        )
        if b is not None:
            output_b = torch.nn.functional.conv_transpose2d(
                input=b,
                weight=self.weight,
                bias=None,
                stride=self.strides,
                padding=self.pads,
                output_padding=self.output_padding,
                dilation=self.dilations,
            )
        else:
            output_b = None
        return output_a, output_b, l, u


class MaxPool(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.kernel_shape = tuple(self.attribute["kernel_shape"].ints)
        self.strides = tuple(self.attribute["strides"].ints)
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )
        self.dilations = (
            tuple(self.attribute["dilations"].ints)
            if "dilations" in self.attribute
            else (1, 1)
        )
        self.ceil_mode = (
            bool(self.attribute["ceil_mode"].i)
            if "ceil_mode" in self.attribute
            else False
        )

    def forward(self, x):
        if x.dim() == 3:
            output = nn.functional.max_pool1d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.stride,
                padding=self.pads,
                dilation=self.dilations,
                ceil_mode=self.ceil_mode,
            )
            return output
        elif x.dim() == 4:
            output = nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
                ceil_mode=self.ceil_mode,
            )
            return output
        else:
            raise ValueError(
                "Input dimension must be 3 (for 1D) or 4 (for 2D) but got {}".format(
                    x.dim()
                )
            )

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
        else:
            x = a
        B, C, H, W = x.shape

        if self.ceil_mode:
            H_out = (
                torch.ceil(
                    torch.tensor(
                        (
                            H
                            + 2 * self.pads
                            - self.dilations[0] * (self.kernel_shape[0] - 1)
                            - 1
                        )
                        / self.strides[0]
                        + 1
                    )
                )
                .int()
                .item()
            )
            W_out = (
                torch.ceil(
                    torch.tensor(
                        (
                            W
                            + 2 * self.pads
                            - self.dilations[1] * (self.kernel_shape[1] - 1)
                            - 1
                        )
                        / self.strides[1]
                        + 1
                    )
                )
                .int()
                .item()
            )
        else:
            H_out = (
                H + 2 * self.pads - self.dilations[0] * (self.kernel_shape[0] - 1) - 1
            ) // self.strides[0] + 1
            W_out = (
                W + 2 * self.pads - self.dilations[1] * (self.kernel_shape[1] - 1) - 1
            ) // self.strides[1] + 1

        x_im2coled = nn.functional.unfold(
            x,
            kernel_size=self.kernel_shape,
            stride=self.strides,
            padding=self.pads,
            dilation=self.dilations,
        )
        x_im2coled_reshaped = x_im2coled.view(
            B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
        )

        max_index = x_im2coled_reshaped.argmax(dim=2).unsqueeze(2)
        output_x = x_im2coled_reshaped.gather(dim=2, index=max_index)
        output_x = output_x.view(B, C, H_out, W_out)

        if b is not None:
            a_im2coled = nn.functional.unfold(
                a,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
            )
            a_im2coled_reshaped = a_im2coled.view(
                B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
            )
            output_a = a_im2coled_reshaped.gather(dim=2, index=max_index)

            b_im2coled = nn.functional.unfold(
                b,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                dilation=self.dilations,
            )
            b_im2coled_reshaped = b_im2coled.view(
                B, C, self.kernel_shape[0] * self.kernel_shape[1], H_out * W_out
            )
            output_b = b_im2coled_reshaped.gather(dim=2, index=max_index)

            tTa = a_im2coled_reshaped - output_a
            tTb = b_im2coled_reshaped - output_b

            temp_l, temp_u = truncated_interval(tTa, tTb)

            l = torch.maximum(l, temp_l)
            u = torch.minimum(u, temp_u)

            output_a = output_a.view(B, C, H_out, W_out)
            output_b = output_b.view(B, C, H_out, W_out)
        else:
            output_a = output_x
            output_b = None

        return output_a, output_b, l, u


class AveragePool(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.kernel_shape = tuple(self.attribute["kernel_shape"].ints)
        self.strides = tuple(self.attribute["strides"].ints)
        self.pads = (
            int(self.attribute["pads"].ints[0]) if "pads" in self.attribute else 0
        )
        self.ceil_mode = (
            bool(self.attribute["ceil_mode"].i)
            if "ceil_mode" in self.attribute
            else False
        )
        self.count_include_pad = (
            bool(self.attribute["count_include_pad"].i)
            if "count_include_pad" in self.attribute
            else 1
        )

    def forward(self, x):
        if x.dim() == 3:
            output = nn.functional.avg_pool1d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        elif x.dim() == 4:
            output = nn.functional.avg_pool2d(
                x,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
        else:
            x = a

        if x.dim() == 3:  # not tested
            output_a = nn.functional.avg_pool1d(
                a,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            if b is not None:
                output_b = nn.functional.avg_pool1d(
                    b,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
            else:
                output_b = None
        elif x.dim() == 4:
            output_a = nn.functional.avg_pool2d(
                a,
                kernel_size=self.kernel_shape,
                stride=self.strides,
                padding=self.pads,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
            )
            if b is not None:
                output_b = nn.functional.avg_pool2d(
                    b,
                    kernel_size=self.kernel_shape,
                    stride=self.strides,
                    padding=self.pads,
                    ceil_mode=self.ceil_mode,
                    count_include_pad=self.count_include_pad,
                )
            else:
                output_b = None
        return output_a, output_b, l, u


class GlobalAveragePool(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
        else:
            x = a
        output_a = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        if b is not None:
            output_b = nn.functional.adaptive_avg_pool2d(b, (1, 1))
        else:
            output_b = None
        return output_a, output_b, l, u


class Gemm(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.weight = inputs[1].detach().to(torch.float64)
        self.bias = inputs[2].detach().to(torch.float64) if len(inputs) > 2 else None
        self.alpha = self.attribute["alpha"].f if "alpha" in self.attribute else 1.0
        self.beta = self.attribute["beta"].f if "beta" in self.attribute else 1.0
        self.transA = self.attribute["transA"].i if "transA" in self.attribute else 0
        self.transB = self.attribute["transB"].i if "transB" in self.attribute else 0

    def forward(self, x):
        if self.transA:
            x = x.t()
        if self.transB:
            self.weight = self.weight.t()
        output = self.alpha * torch.mm(x, self.weight) + self.beta * self.bias
        return output

    def forward_si(self, z, a, b, l, u):
        if self.transA:
            a = a.t()
            if b is not None:
                b = b.t()
            else:
                b = None
        if self.transB:
            self.weight = self.weight.t()

        output_a = self.alpha * torch.mm(a, self.weight) + self.beta * self.bias
        if b is not None:
            output_b = self.alpha * torch.mm(b, self.weight)
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Transpose(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.perm = self.attribute["perm"].ints

    def forward(self, x):
        output = x.permute(tuple(self.perm))
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = a.permute(tuple(self.perm))
        if b is not None:
            output_b = b.permute(tuple(self.perm))
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Shape(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.end = self.attribute["end"].i if "end" in self.attribute else None
        self.start = self.attribute["start"].i if "start" in self.attribute else 0

    def forward(self, x):
        shape = x.shape
        rank = len(shape)

        if self.start < 0:
            start = max(0, rank + self.start)
        else:
            start = min(rank, self.start)

        if self.end is None or self.end >= rank:
            end = rank
        elif self.end < 0:
            end = max(0, rank + self.end)
        else:
            end = min(rank, self.end)

        output = torch.tensor(shape[start:end], dtype=torch.int64)
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
        else:
            x = a
        output_a = self.forward(x)
        output_b = None

        return output_a, output_b, l, u


# Now fixing
class Slice(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.starts = list(inputs[1]) if inputs[1] != [] else None
        self.ends = list(inputs[2]) if inputs[2] != [] else None
        self.axes = list(inputs[3]) if len(inputs) > 3 and inputs[3] != [] else None
        self.steps = list(inputs[4]) if len(inputs) > 4 and inputs[4] != [] else None

    def forward(self, x):
        slices = [slice(None)] * x.dim()

        if self.axes is None:
            axes = list(range(x.dim()))
        else:
            axes = self.axes

        for i, axis in enumerate(axes):
            start = (
                self.starts[i]
                if self.starts is not None and i < len(self.starts)
                else None
            )
            end = self.ends[i] if self.ends is not None and i < len(self.ends) else None
            step = (
                self.steps[i]
                if self.steps is not None and i < len(self.steps)
                else None
            )

            slices[axis] = slice(start, end, step)
        output = x[tuple(slices)]
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            x = a + b * z
        else:
            x = a
        slices = [slice(None)] * x.dim()

        if self.axes is None:
            axes = list(range(x.dim()))
        else:
            axes = self.axes

        for i, axis in enumerate(axes):
            start = (
                self.starts[i]
                if self.starts is not None and i < len(self.starts)
                else None
            )
            end = self.ends[i] if self.ends is not None and i < len(self.ends) else None
            step = (
                self.steps[i]
                if self.steps is not None and i < len(self.steps)
                else None
            )

            slices[axis] = slice(start, end, step)

        output_a = a[tuple(slices)]
        if b is not None:
            output_b = b[tuple(slices)]
        else:
            output_b = None

        return output_a, output_b, l, u


class Exp(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)

    def forward(self, x):
        output = torch.exp(x)
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.exp(a)
        output_b = None
        return output_a, output_b, l, u


class RandomNormalLike(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.mean = self.attribute["mean"].f if "mean" in self.attribute else 0.0
        self.scale = self.attribute["scale"].f if "scale" in self.attribute else 1.0
        self.shape = inputs[0].shape

    def forward(self, x, rng, mean=None, scale=None):
        output = torch.tensor(
            rng.normal(self.mean, self.scale, size=self.shape),
            dtype=torch.float64,
        )
        return output

    def forward_si(self, z, output_x):
        output_a, output_b = output_x, None
        l = torch.tensor(-float("inf"), dtype=torch.float64)
        u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, l, u


class Flatten(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.axis = self.attribute["axis"].i if "axis" in self.attribute else 1

    def forward(self, x):
        output = torch.flatten(x, start_dim=self.axis)
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.flatten(a, start_dim=self.axis)
        if b is not None:
            output_b = torch.flatten(b, start_dim=self.axis)
        else:
            output_b = None
        return output_a, output_b, l, u


class ConstantOfShape(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.value = self.attribute["value"].f if "value" in self.attribute else 0.0

    def forward(self, x):
        output = torch.full(x, self.value, dtype=torch.float64)
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.full(a, self.value, dtype=torch.float64)
        output_b = None
        return output_a, output_b, l, u


class EyeLike(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.dtype = (
            self.attribute["dtype"].i if "dtype" in self.attribute else torch.float64
        )
        self.k = self.attribute["k"].i if "k" in self.attribute else 0
        if self.k != 0:
            raise ValueError("k must be 0 in EyeLike layer")

    def forward(self, x):
        output = torch.eye(x.shape[0], dtype=self.dtype)
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.eye(a.shape[0], dtype=self.dtype)
        output_b = None
        return output_a, output_b, l, u


class Reciprocal(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)

    def forward(self, x):
        output = torch.reciprocal(x)
        return output

    def forward_si(self, z, a, b, l, u):
        output_a = torch.reciprocal(a)
        output_b = None
        return output_a, output_b, l, u


class Reshape(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.shape = node_output[node.input[1]]

    def forward(self):
        output = torch.reshape(self.input, self.shape)
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = node_output_si[node.input[0]][0]
        b = node_output_si[node.input[0]][1]
        l = node_output_si[node.input[0]][2]
        u = node_output_si[node.input[0]][3]

        output_a = torch.reshape(a, self.shape)
        if b is not None:
            output_b = torch.reshape(b, self.shape)
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Resize(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.roi = (
            node_output[node.input[1]]
            if len(node.input) > 1 and node.input[1] != ""
            else None
        )
        self.scales = (
            node_output[node.input[2]]
            if len(node.input) > 2 and node.input[2] != ""
            else None
        )
        self.sizes = (
            node_output[node.input[3]]
            if len(node.input) > 3 and node.input[3] != ""
            else None
        )

        if isinstance(self.sizes, torch.Tensor):
            self.sizes = tuple(map(int, self.sizes))[2:]

        self.mode = (
            self.attribute["mode"].s.decode() if "mode" in self.attribute else "nearest"
        )
        self.coordinate_transformation_mode = (
            self.attribute["coordinate_transformation_mode"].s.decode()
            if "coordinate_transformation_mode" in self.attribute
            else "half_pixel"
        )
        self.antialias = (
            self.attribute["antialias"].i if "antialias" in self.attribute else 0
        )

        if self.input.dim() == 4:
            if self.scales is not None:
                self.scales = (float(self.scales[2]), float(self.scales[3]))
            if self.mode == "linear":
                self.mode = "bilinear"
            elif self.mode == "cubic":
                self.mode = "bicubic"
        elif self.input.dim() == 3:
            if self.scales is not None:
                self.scales = (float(self.scales[1]),)

    def forward(self):
        output = torch.nn.functional.interpolate(
            input=self.input,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.coordinate_transformation_mode == "align_corners",
            recompute_scale_factor=None,
            antialias=self.antialias,
        )
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = node_output_si[node.input[0]][0]
        b = node_output_si[node.input[0]][1]
        l = node_output_si[node.input[0]][2]
        u = node_output_si[node.input[0]][3]
        output_a = torch.nn.functional.interpolate(
            a,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.coordinate_transformation_mode == "align_corners",
            recompute_scale_factor=None,
            antialias=self.antialias,
        )

        if b is not None:
            output_b = torch.nn.functional.interpolate(
                b,
                size=self.sizes,
                scale_factor=self.scales,
                mode=self.mode,
                align_corners=self.coordinate_transformation_mode == "align_corners",
                recompute_scale_factor=None,
                antialias=self.antialias,
            )
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)

        return output_a, output_b, output_l, output_u


class Concat(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [
            torch.as_tensor(node_output[input_name]) for input_name in node.input
        ]
        self.axis = self.attribute["axis"].i

    def forward(self):
        output = torch.cat(self.inputs, dim=self.axis)
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = [
            torch.as_tensor(node_output_si[input_name][0]) for input_name in node.input
        ]
        b = [
            (
                torch.as_tensor(node_output_si[input_name][1])
                if node_output_si[input_name][1] is not None
                else torch.zeros_like(
                    torch.as_tensor(node_output[input_name]), dtype=torch.float64
                )
            )
            for input_name in node.input
        ]
        l = torch.tensor([node_output_si[input_name][2] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][3] for input_name in node.input])

        output_a = torch.cat(a, dim=self.axis)
        output_b = torch.cat(b, dim=self.axis)
        output_l, output_u = l.max(), u.min()
        return output_a, output_b, output_l, output_u


class Add(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]

    def forward(self):
        output = torch.add(self.inputs[0], self.inputs[1])
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = [(node_output_si[input_name][0]) for input_name in node.input]
        output_a = torch.add(a[0], a[1])

        if any(node_output_si[input_name][1] is not None for input_name in node.input):
            b = [
                (
                    node_output_si[input_name][1]
                    if node_output_si[input_name][1] is not None
                    else torch.zeros_like(node_output[input_name], dtype=torch.float64)
                )
                for input_name in node.input
            ]
            output_a = torch.add(a[0], a[1])
            output_b = torch.add(b[0], b[1])
        else:
            output_b = None

        l = torch.tensor([node_output_si[input_name][2] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        output_l, output_u = l.max(), u.min()
        return output_a, output_b, output_l, output_u


class Sub(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]

    def forward(self):
        output = torch.sub(self.inputs[0], self.inputs[1])
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = [
            (
                node_output_si[input_name][0]
                if node_output_si[input_name][0] is not None
                else node_output[input_name]
            )
            for input_name in node.input
        ]

        output_a = torch.sub(a[0], a[1])

        if any(node_output_si[input_name][1] is not None for input_name in node.input):
            b = [
                (
                    node_output_si[input_name][1]
                    if node_output_si[input_name][1] is not None
                    else torch.zeros_like(node_output[input_name], dtype=torch.float64)
                )
                for input_name in node.input
            ]
            output_a = torch.sub(a[0], a[1])
            output_b = torch.sub(b[0], b[1])
        else:
            output_b = None

        l = torch.tensor([node_output_si[input_name][2] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        output_l, output_u = l.max(), u.min()
        return output_a, output_b, output_l, output_u


class Split(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.inputs = [node_output[input_name] for input_name in node.input]
        self.axis = self.attribute["axis"].i
        self.num_outputs = (
            self.attribute["num_outputs"].i if "num_outputs" in self.attribute else None
        )

    def forward(self) -> tuple[torch.Tensor]:
        if len(self.inputs) > 1:
            output = torch.split(
                tensor=self.inputs[0],
                split_size_or_sections=self.inputs[1],
                dim=self.axis,
            )
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = node_output_si[node.input[0]][0]
        b = node_output_si[node.input[0]][1]

        l = torch.tensor([node_output_si[input_name][2] for input_name in node.input])
        u = torch.tensor([node_output_si[input_name][3] for input_name in node.input])
        l = torch.tensor([l.max() for _ in range(len(node.output))])
        u = torch.tensor([u.min() for _ in range(len(node.output))])

        output_a = torch.split(
            tensor=a, split_size_or_sections=self.inputs[1], dim=self.axis
        )

        if b is not None:
            output_b = torch.split(
                tensor=b, split_size_or_sections=self.inputs[1], dim=self.axis
            )
        else:
            output_b = None

        return output_a, output_b, l, u


class BatchNormalization(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.scale = node_output[node.input[1]]
        self.B = node_output[node.input[2]]
        self.input_mean = node_output[node.input[3]]
        self.input_var = node_output[node.input[4]]
        self.epsilon = (
            self.attribute["epsilon"].f if "epsilon" in self.attribute else 1e-5
        )
        self.momentum = (
            self.attribute["momentum"].f if "momentum" in self.attribute else 0.9
        )
        self.training_mode = (
            self.attribute["training_mode"].i
            if "training_mode" in self.attribute
            else 1
        )

    def forward(self):
        if self.training_mode:
            raise NotImplementedError(
                "Training mode is not supported. Please save the model in evaluation mode."
            )
        else:
            output = (self.input - self.input_mean) / torch.sqrt(
                self.input_var + self.epsilon
            ) * self.scale + self.B
        return output

    def forward_si(self, node, node_output, node_output_si):
        if self.training_mode:
            raise NotImplementedError(
                "Training mode is not supported. Please save the model in evaluation mode."
            )
        else:
            a = node_output_si[node.input[0]][0]
            b = node_output_si[node.input[0]][1]
            l = node_output_si[node.input[0]][2]
            u = node_output_si[node.input[0]][3]
            output_a = (a - self.input_mean) / torch.sqrt(
                self.input_var + self.epsilon
            ) * self.scale + self.B
            if b is not None:
                output_b = b / torch.sqrt(self.input_var + self.epsilon) * self.scale
                output_l = l
                output_u = u
            else:
                output_b = None
                output_l = torch.tensor(-float("inf"), dtype=torch.float64)
                output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Mul(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.mul(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_a = node_output_si[node.input[0]][0]
        A_b = node_output_si[node.input[0]][1]
        A_l = node_output_si[node.input[0]][2]
        A_u = node_output_si[node.input[0]][3]
        B_a = node_output_si[node.input[1]][0]
        B_b = node_output_si[node.input[1]][1]
        B_l = node_output_si[node.input[1]][2]
        B_u = node_output_si[node.input[1]][3]
        output_a = torch.mul(A_a, B_a)
        if A_b is not None or B_b is not None:
            if A_b is None:
                A_b = self.A
            else:
                B_b = self.B
            output_b = torch.mul(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class MatMul(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.matmul(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_a = node_output_si[node.input[0]][0]
        A_b = node_output_si[node.input[0]][1]
        A_l = node_output_si[node.input[0]][2]
        A_u = node_output_si[node.input[0]][3]
        B_a = node_output_si[node.input[1]][0]
        B_b = node_output_si[node.input[1]][1]
        B_l = node_output_si[node.input[1]][2]
        B_u = node_output_si[node.input[1]][3]
        # vector a of constant variable is equal to constant variable
        output_a = torch.matmul(A_a, B_a)
        # If one is a random variable, the other is a constant
        if A_b is not None or B_b is not None:
            if A_b is None:
                # If A is a random variable, treat B as a constant
                A_b = self.A
            else:
                # If B is a random variable, treat A as a constant
                B_b = self.B
            # If A is a random variable: output_x = matmul((A_a+A_b*z),B)
            # If B is a random variable: output_x = matmul(A,(B_a+B_b*z))
            # So,
            output_b = torch.matmul(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else: # If both are constants
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u



class Div(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.div(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_a = node_output_si[node.input[0]][0]
        A_b = node_output_si[node.input[0]][1]
        A_l = node_output_si[node.input[0]][2]
        A_u = node_output_si[node.input[0]][3]
        B_a = node_output_si[node.input[1]][0]
        B_b = node_output_si[node.input[1]][1]
        B_l = node_output_si[node.input[1]][2]
        B_u = node_output_si[node.input[1]][3]
        output_a = torch.div(A_a, B_a)
        if A_b is not None or B_b is not None:
            if A_b is None:
                A_b = self.A
            else:
                B_b = self.B
            output_b = torch.div(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class ReduceSum(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.axes = node_output[node.input[1]] if len(node.input) > 1 else None
        self.keepdims = (
            bool(self.attribute["keepdims"].i)
            if "keepdims" in self.attribute
            else False
        )
        self.noop_with_empty_axes = (
            self.attribute["noop_with_empty_axes"].i
            if "noop_with_empty_axes" in self.attribute
            else 0
        )  # Not tested

    def forward(self):
        output = torch.sum(self.input, dim=self.axes, keepdim=self.keepdims)
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = node_output_si[node.input[0]][0]
        b = node_output_si[node.input[0]][1]
        l = node_output_si[node.input[0]][2]
        u = node_output_si[node.input[0]][3]
        output_a = torch.sum(a, dim=self.axes, keepdim=self.keepdims)
        if b is not None:
            output_b = torch.sum(b, dim=self.axes, keepdim=self.keepdims)
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Equal(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.eq(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_a = node_output_si[node.input[0]][0]
        A_b = node_output_si[node.input[0]][1]
        A_l = node_output_si[node.input[0]][2]
        A_u = node_output_si[node.input[0]][3]
        B_a = node_output_si[node.input[1]][0]
        B_b = node_output_si[node.input[1]][1]
        B_l = node_output_si[node.input[1]][2]
        B_u = node_output_si[node.input[1]][3]
        output_a = torch.eq(A_a, B_a)
        if A_b is not None or B_b is not None:
            if A_b is None:
                A_b = self.A
            else:
                B_b = self.B
            output_b = torch.eq(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Greater(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.A = node_output[node.input[0]]
        self.B = node_output[node.input[1]]

    def forward(self):
        output = torch.gt(self.A, self.B)
        return output

    def forward_si(self, node, node_output, node_output_si):
        A_a = node_output_si[node.input[0]][0]
        A_b = node_output_si[node.input[0]][1]
        A_l = node_output_si[node.input[0]][2]
        A_u = node_output_si[node.input[0]][3]
        B_a = node_output_si[node.input[1]][0]
        B_b = node_output_si[node.input[1]][1]
        B_l = node_output_si[node.input[1]][2]
        B_u = node_output_si[node.input[1]][3]
        output_a = torch.gt(A_a, B_a)
        if A_b is not None or B_b is not None:
            if A_b is None:
                A_b = self.A
            else:
                B_b = self.B
            output_b = torch.gt(A_b, B_b)
            output_l = torch.max(A_l, B_l)
            output_u = torch.min(A_u, B_u)
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Squeeze(Layer):
    def __init__(self, inputs, node, node_output):
        super().__init__(inputs, node)
        self.input = node_output[node.input[0]]
        self.axes = node_output[node.input[1]].tolist()

    def forward(self):
        output = torch.squeeze(self.input, dim=self.axes)
        return output

    def forward_si(self, node, node_output, node_output_si):
        a = node_output_si[node.input[0]][0]
        b = node_output_si[node.input[0]][1]
        l = node_output_si[node.input[0]][2]
        u = node_output_si[node.input[0]][3]
        output_a = torch.squeeze(a, dim=self.axes)
        if b is not None:
            output_b = torch.squeeze(b, dim=self.axes)
            output_l = l
            output_u = u
        else:
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        return output_a, output_b, output_l, output_u


class Constant(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        self.dims = self.attribute["value"].t.dims
        self.data_type = self.attribute["value"].t.data_type
        self.raw_data = self.attribute["value"].t.raw_data

    def forward(self):
        if self.data_type == 1:
            x = np.frombuffer(self.raw_data, dtype=np.float32).astype(np.float64)
            x = x.reshape(self.dims)
            output = torch.tensor(x)
        elif self.data_type == 7:
            x = np.frombuffer(self.raw_data, dtype=np.int64)
            x = x.reshape(self.dims)
            if x.ndim == 0:
                output = torch.tensor(int(x), dtype=torch.int64)
            else:
                output = tuple(x)
        else:
            raise NotImplementedError(
                "data_type {} is not supported".format(self.data_type)
            )
        return output

    def forward_si(self):
        if self.data_type == 1:
            output_x = np.frombuffer(self.raw_data, dtype=np.float32).astype(np.float64)
            output_x = output_x.reshape(self.dims)
            output_x = torch.tensor(output_x, dtype=torch.float64)
            output_a = output_x
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        elif self.data_type == 7:
            output_x = np.frombuffer(self.raw_data, dtype=np.int64)
            output_x = output_x.reshape(self.dims)
            if output_x.ndim == 0:
                output_x = torch.tensor(int(output_x), dtype=torch.int64)
            else:
                output_x = tuple(output_x)
            output_a = output_x
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)
        else:
            raise NotImplementedError(
                "data_type {} is not supported".format(self.data_type)
            )
        return output_a, output_b, output_l, output_u

class ReduceMean(Layer):
    def __init__(self, inputs, node):
        super().__init__(inputs, node)
        pass

    def forward(self, x):
        output = torch.mean(x, dim=1)
        return output

    def forward_si(self, z, a, b, l, u):
        if b is not None:
            output_a = torch.mean(a, dim=1)
            output_b = torch.mean(b, dim=1)
            output_l = l
            output_u = u
            #assert np.allclose(output_a+output_b*z,self.forward(a+b*z))
        else: # If input is a constant
            output_a = torch.mean(a, dim=1)
            output_b = None
            output_l = torch.tensor(-float("inf"), dtype=torch.float64)
            output_u = torch.tensor(float("inf"), dtype=torch.float64)

        return output_a, output_b, output_l, output_u
