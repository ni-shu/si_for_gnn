import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from layers import truncated_interval, Relu
from typing import Union, List


class GradThresholding:

    def __init__(
        self,
        # model: pytorch model,
        thr: Union[List[float], float],
        torch_model: torch.nn.Module = None,
        apply_norm: bool = False,
        use_sigmoid: bool = False,
        apply_relu: bool = False,
        input_product: bool = False,
    ):
        self.torch_model = torch_model
        self.use_sigmoid = use_sigmoid
        self.apply_norm = apply_norm
        self.apply_relu = apply_relu
        self.input_product = input_product

        if not isinstance(thr, list):
            thr_list = [thr]
        else:
            thr_list = thr

        if self.use_sigmoid:
            self.tau_list = [torch.logit(t) for t in thr_list]
        else:
            self.tau_list = thr_list
        

        # self.tau = thr

    def _filter(self, grad):
        return grad

    def calc_grad(self, A, X):
        #! ∂ logit/ ∂X
        X = X.detach().clone()
        # もし値があれば初期化しないといけない
        assert X.grad is None
        A = A.detach()
        X.requires_grad = True
        cam, logits = self.torch_model(A, X)
        logits[0][1].backward()
        grad = X.grad.detach().clone()
        X = X.detach()
        return self._filter(grad)

    def forward(self, A, X):

        #! 勾配 ∂ logit/ ∂X を計算
        grad = self.calc_grad(A, X)
        self.grad = grad

        #! 勾配に対してReLUと正規化を適用
        grad = grad * X if self.input_product else grad
        x = torch.relu(grad) if self.apply_relu else grad

        #! 正規化して閾値判定
        x_max = torch.max(x)
        x_min = torch.min(x)
        threshold_index_list = []
        for tau in self.tau_list:
            if self.apply_norm:
                threshold_index = (x - x_min) / (x_max - x_min) > tau
            else:
                threshold_index = x > tau
            threshold_index = torch.flatten(threshold_index).int()
            threshold_index_list.append(threshold_index)
        return (
            threshold_index_list[0]
            if len(threshold_index_list) == 1
            else torch.stack(threshold_index_list)
        )
    def _relu_foward_si(self, a, b, l, u, z):
        x = a + b * z
        relu_index = torch.greater_equal(x, 0)
        tTa = torch.where(relu_index, -a, a)
        tTb = torch.where(relu_index, -b, b)

        temp_l, temp_u = truncated_interval(tTa, tTb)

        output_l = torch.max(l, temp_l)
        output_u = torch.min(u, temp_u)

        output_a = torch.where(relu_index, a, torch.tensor(0, dtype=torch.float64))
        output_b = torch.where(relu_index, b, torch.tensor(0, dtype=torch.float64))
        return output_a, output_b, output_l, output_u, z

    def forward_si(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
    ):

        # X == a + b * z
        threshold_index = self.forward(A, X)
        
        if not self.input_product:
            return threshold_index, l, u
        else:
            # 勾配 ∂ logit/ ∂X を計算
            grad = self.calc_grad(A, X)

            # input積
            a = grad * a
            b = grad * b

            # ReLU
            if self.apply_relu:
                a, b, l, u, z = self._relu_foward_si(a, b, l, u, z)

            # input (X) * grad
            x = a + b * z
            a = a.flatten()
            b = b.flatten()
            x = x.flatten()

            # 正規化
            if self.apply_norm:
                # Find the interval where max(X)=max(a+bz)
                max_index = torch.argmax(x)
                delta = lambda x: x - x[max_index]
                l_positive, u_positive = truncated_interval(delta(a), delta(b))
                l = torch.max(l, l_positive)
                u = torch.min(u, u_positive)
                assert l <= z
                assert z <= u

                # Find the interval where min(X)=min(a+bz)
                min_index = torch.argmin(x)
                delta = lambda x: x[min_index] - x 
                l_positive, u_positive = truncated_interval(delta(a), delta(b))
                l = torch.max(l, l_positive)
                u = torch.min(u, u_positive)
                assert l <= z
                assert z <= u

                for tau in self.tau_list:
                    positive_index = (x-torch.min(x))/(torch.max(x)-torch.min(x)) > tau

                    a_ = a -  (tau * a[max_index] + (1-tau) * a[min_index])
                    b_ = b - (tau * b[max_index] + (1-tau) * b[min_index])

                    delta = lambda x: torch.where(positive_index, -x, x)
                    tTa = delta(a_)
                    tTb = delta(b_)

                    l_positive, u_positive = truncated_interval(tTa, tTb)
                    l = torch.max(l, l_positive)
                    u = torch.min(u, u_positive)

                    assert l <= z, (format(l.item(), '.20f'), format(z.item(), '.20f'))
                    assert z <= u, (z,u)

            else:
                for tau in self.tau_list:
                    positive_index = x > tau
                    delta = lambda x: torch.where(positive_index, -x, x)
                    tTa = delta(a - tau)
                    tTb = delta(b)
                    l_positive, u_positive = truncated_interval(tTa, tTb)
                    l = torch.max(l, l_positive)
                    u = torch.min(u, u_positive)
                    assert l <= z
                    assert z <= u

            l = float(l.item())
            u = float(u.item())
            assert l <= z
            assert z <= u
            return threshold_index, l, u


class NormilzedThresholding:

    def __init__(
        self,
        thr: Union[List[float], float],
        apply_norm: bool = False,
        use_sigmoid: bool = False,
    ):
        self.use_sigmoid = use_sigmoid
        self.apply_norm = apply_norm

        if not isinstance(thr, list):
            thr_list = [thr]
        else:
            thr_list = thr

        if self.use_sigmoid:
            self.tau_list = [torch.logit(t) for t in thr_list]
        else:
            self.tau_list = thr_list

        # self.tau = thr

    def forward(self, x):
        x_max = torch.max(x)
        x_min = torch.min(x)
        threshold_index_list = []
        for tau in self.tau_list:
            if self.apply_norm:
                threshold_index = (x - x_min) / (x_max - x_min) > tau
            else:
                threshold_index = x > tau
            threshold_index = torch.flatten(threshold_index).int()
            threshold_index_list.append(threshold_index)
        return (
            threshold_index_list[0]
            if len(threshold_index_list) == 1
            else torch.stack(threshold_index_list)
        )

    def forward_si(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        l: torch.Tensor,
        u: torch.Tensor,
        z: torch.Tensor,
    ):

        x = a + b * z
        threshold_index = self.forward(x)

        if self.apply_norm:

            # Find the interval where max(X)=max(a+bz)
            max_index = torch.argmax(x)
            delta = lambda x: x - x[max_index]
            l_positive, u_positive = truncated_interval(delta(a), delta(b))
            l = torch.max(l, l_positive)
            u = torch.min(u, u_positive)
            assert l <= z, (format(l.item(), '.20f'), format(z.item(), '.20f'))
            assert z <= u, (format(z.item(), '.20f'), format(u.item(), '.20f'))

            # Find the interval where min(X)=min(a+bz)
            min_index = torch.argmin(x)
            delta = lambda x: x[min_index] - x 
            l_positive, u_positive = truncated_interval(delta(a), delta(b))
            l = torch.max(l, l_positive)
            u = torch.min(u, u_positive)
            assert l <= z, (format(l.item(), '.20f'), format(z.item(), '.20f'))
            assert z <= u, (format(z.item(), '.20f'), format(u.item(), '.20f'))

            for tau in self.tau_list:
                positive_index = (x - torch.min(x)) / (
                    torch.max(x) - torch.min(x)
                ) > tau

                a_ = a - (tau * a[max_index] + (1 - tau) * a[min_index])
                b_ = b - (tau * b[max_index] + (1 - tau) * b[min_index])

                delta = lambda x: torch.where(positive_index, -x, x)
                tTa = delta(a_)
                tTb = delta(b_)

                l_positive, u_positive = truncated_interval(tTa, tTb)
                l = torch.max(l, l_positive)
                u = torch.min(u, u_positive)

                assert l <= z, (format(l.item(), '.20f'), format(z.item(), '.20f'))
                assert z <= u, (format(z.item(), '.20f'), format(u.item(), '.20f'))
        else:
            for tau in self.tau_list:
                positive_index = x > tau
                delta = lambda x: torch.where(positive_index, -x, x)
                tTa = delta(a - tau)
                tTb = delta(b)
                l_positive, u_positive = truncated_interval(tTa, tTb)
                l = torch.max(l, l_positive)
                u = torch.min(u, u_positive)
                assert l <= z
                assert z <= u

        l = float(l.item())
        u = float(u.item())
        assert l <= z
        assert z <= u
        return threshold_index, l, u

def thresholding(
    thr: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    l: torch.Tensor,
    u: torch.Tensor,
    z: torch.Tensor,
    use_abs: bool = False,
    use_sigmoid: bool = False,
):
    """
    Threshold the input tensor x with the given threshold (thr)
    and return the indices greater than thr.
    Args:
        thr (torch.Tensor): threshold tensor, dtype=torch.float64
        x (torch.Tensor): input tensor
        a (torch.Tensor): a tensor
        b (torch.Tensor): b tensor
        apply_abs (bool): apply abs to the input tensor or not
        use_sigmoid (bool): use sigmoid or not in the final output layer
    Returns:
        threshold_index (torch.Tensor): threshold index
        l (float): lower bound of the truncated interval
        u (float): upper bound of the truncated interval
    """

    if use_sigmoid:
        tau = torch.logit(thr)
    else:
        tau = thr

    x = a + b * z

    if use_abs:
        abs_x = torch.abs(x)
        threshold_index = abs_x > tau
        threshold_index = torch.flatten(threshold_index).int()

        negative_index = x < -tau
        tTb = torch.where(negative_index, b, -b)
        tTa = a + tau
        tTa = torch.where(negative_index, tTa, -tTa)
        l_negative, u_negative = truncated_interval(tTa, tTb)
        l = torch.max(l, l_negative)
        u = torch.min(u, u_negative)
    else:
        threshold_index = x > tau
        threshold_index = torch.flatten(threshold_index).int()

    positive_index = x > tau
    tTa = a - tau
    tTa = torch.where(positive_index, -tTa, tTa)
    tTb = torch.where(positive_index, -b, b)
    l_positive, u_positive = truncated_interval(tTa, tTb)
    l = torch.max(l, l_positive)
    u = torch.min(u, u_positive)
    l = float(l.item())
    u = float(u.item())

    assert l < u
    return threshold_index, l, u
