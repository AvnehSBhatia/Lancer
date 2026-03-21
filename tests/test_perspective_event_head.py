import torch
import torch.nn.functional as F

from apollo.perspective_event_head import BottleneckStack, PerspectiveEventHead


def test_forward_logits_matches_softmax():
    n, p, q = 3, 4, 4
    head = PerspectiveEventHead(n=n, p=p, q=q)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abn = torch.randn(p)
    d_n = torch.randn(q)
    logits = head.forward_logits(C, p_hat, abn, d_n)
    probs = head(C, p_hat, abn, d_n)
    assert logits.shape == (2,)
    torch.testing.assert_close(F.softmax(logits, dim=-1), probs, rtol=1e-5, atol=1e-5)


def test_perspective_head_shapes_unbatched():
    n, p, q = 4, 5, 5
    head = PerspectiveEventHead(n=n, p=p, q=q)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abn = torch.randn(p)
    d_n = torch.randn(q)
    y = head(C, p_hat, abn, d_n)
    assert y.shape == (2,)
    assert torch.allclose(y.sum(), torch.tensor(1.0), atol=1e-6)
    assert (y >= 0).all()


def test_perspective_head_batched():
    n, p, q, B = 3, 4, 4, 5
    head = PerspectiveEventHead(n=n, p=p, q=q)
    C = torch.randn(B, n, p)
    p_hat = F.softmax(torch.randn(B, n, 2), dim=-1)
    abn = torch.randn(B, p)
    d_n = torch.randn(B, q)
    y = head(C, p_hat, abn, d_n)
    assert y.shape == (B, 2)
    assert torch.allclose(y.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_forward_with_abdn():
    n, p = 2, 3
    head = PerspectiveEventHead(n=n, p=p)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abdn = torch.randn(p)
    y = head.forward_with_abdn(C, p_hat, abdn)
    assert y.shape == (2,)


def test_forward_all_contexts():
    n, p, q, J = 2, 3, 3, 3
    head = PerspectiveEventHead(n=n, p=p, q=q)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abn = torch.randn(p)
    d_stack = torch.randn(J, q)
    Y = head.forward_all_contexts(C, p_hat, abn, d_stack, pool="none")
    assert Y.shape == (J, 2)
    pooled = head.forward_all_contexts(C, p_hat, abn, d_stack, pool="mean")
    assert pooled.shape == (2,)


def test_context_proj_when_q_ne_p():
    n, p, q = 2, 4, 7
    head = PerspectiveEventHead(n=n, p=p, q=q)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abn = torch.randn(p)
    d_n = torch.randn(q)
    y = head(C, p_hat, abn, d_n)
    assert y.shape == (2,)


def test_gradient_flow():
    n, p = 3, 4
    head = PerspectiveEventHead(n=n, p=p)
    C = torch.randn(n, p, requires_grad=True)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    abn = torch.randn(p, requires_grad=True)
    d_n = torch.randn(p, requires_grad=True)
    y = head(C, p_hat, abn, d_n)
    y[0].backward()
    assert C.grad is not None and C.grad.abs().sum() > 0
    assert abn.grad is not None and abn.grad.abs().sum() > 0


def test_bottleneck_stack_batched():
    n = 5
    stack = BottleneckStack(n)
    v = torch.randn(3, n)
    out = stack(v)
    assert out.shape == (3, n)
    v1 = torch.randn(n)
    out1 = stack(v1)
    assert out1.shape == (n,)


def test_build_E_matches_manual():
    n, p = 2, 3
    head = PerspectiveEventHead(n=n, p=p)
    C = torch.randn(n, p)
    p_hat = F.softmax(torch.randn(n, 2), dim=-1)
    E = head.build_E(C, p_hat)
    assert E.shape == (n, p)
    c_tilde = F.softmax(float(n) * C, dim=-1)
    c_prime = c_tilde * head.w_sharp + head.b_sharp
    want = torch.tanh(p_hat[:, 0:1] * c_prime) + torch.tanh(p_hat[:, 1:2] * c_prime)
    torch.testing.assert_close(E, want)
