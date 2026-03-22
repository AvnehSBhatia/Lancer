import torch

from apollo.perspective_stages import Config, build_model, compute_abn


def test_forward_stage4_abdj_matches_full_forward():
    cfg = Config(n=5, d=32, p=16, q=24)
    model = build_model(cfg)
    B = 4
    a = torch.randn(B, cfg.d)
    b = torch.randn(B, cfg.d)
    c = torch.randn(B, cfg.n, cfg.p)
    d_ctx = torch.randn(B, cfg.n, cfg.q)

    with torch.no_grad():
        out = model(a, b, c, d_ctx)
        abdj = model.forward_stage4_abdj(a, b, d_ctx)

    assert abdj.shape == (B, cfg.n, cfg.d)
    torch.testing.assert_close(abdj, out.abdj, rtol=0, atol=0)


def test_compute_abn_batch_and_single():
    a = torch.randn(3, 32)
    b = torch.randn(3, 32)
    ab = compute_abn(a, b)
    assert ab.shape == (3, 32)
    a1 = torch.randn(32)
    b1 = torch.randn(32)
    ab1 = compute_abn(a1, b1)
    assert ab1.shape == (32,)


def test_forward_stage4_abdj_shape():
    cfg = Config(n=3, d=8, p=8, q=8)
    model = build_model(cfg)
    a = torch.randn(2, 8)
    b = torch.randn(2, 8)
    d_ctx = torch.randn(2, 3, 8)
    abdj = model.forward_stage4_abdj(a, b, d_ctx)
    assert abdj.shape == (2, 3, 8)
