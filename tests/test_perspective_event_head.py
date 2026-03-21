import torch

from apollo.perspective_event_head import (
    BottleneckStack,
    MiniModelBank,
    PerspectiveEventHead,
)


def test_perspective_head_shapes_and_softmax():
    n, p, d = 4, 5, 3
    head = PerspectiveEventHead(n=n, p=p, d=d)
    C = torch.randn(n, p)
    abd = torch.randn(d)
    y = head(C, abd)
    assert y.shape == (2,)
    assert torch.allclose(y.sum(), torch.tensor(1.0), atol=1e-6)
    assert (y >= 0).all()


def test_batched_forward():
    n, p, d, B = 3, 4, 2, 5
    head = PerspectiveEventHead(n=n, p=p, d=d)
    C = torch.randn(B, n, p)
    abd = torch.randn(B, d)
    y = head.forward_batched(C, abd)
    assert y.shape == (B, 2)
    assert torch.allclose(y.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_forward_all_contexts():
    n, p, d, J = 2, 3, 4, 3
    head = PerspectiveEventHead(n=n, p=p, d=d)
    C = torch.randn(n, p)
    stack = torch.randn(J, d)
    Y = head.forward_all_contexts(C, stack, pool="none")
    assert Y.shape == (J, 2)
    pooled = head.forward_all_contexts(C, stack, pool="mean")
    assert pooled.shape == (2,)


def test_init_align_identity():
    n, p, d = 2, 4, 4
    head = PerspectiveEventHead(n=n, p=p, d=d, init_align_identity=True)
    abd = torch.randn(d)
    aligned = torch.nn.functional.linear(abd, head.align.weight, head.align.bias)
    assert torch.allclose(aligned, abd, atol=1e-5)


def test_gradient_flow():
    n, p, d = 3, 4, 5
    head = PerspectiveEventHead(n=n, p=p, d=d)
    C = torch.randn(n, p, requires_grad=True)
    abd = torch.randn(d, requires_grad=True)
    y = head(C, abd)
    y[0].backward()
    assert C.grad is not None and C.grad.abs().sum() > 0
    assert abd.grad is not None and abd.grad.abs().sum() > 0


def test_bottleneck_stack_shape():
    n = 5
    stack = BottleneckStack(n)
    v = torch.randn(n)
    out = stack(v)
    assert out.shape == (n,)


def test_mini_bank_row_independence():
    n, p = 2, 3
    bank = MiniModelBank(n, p)
    C = torch.randn(n, p)
    E = bank(C)
    assert E.shape == (n, p)
