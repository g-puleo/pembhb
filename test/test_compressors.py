import torch

from pembhb.autoencoder import ChannelizedMLPCompressor, ConvCompressor

N_CH, N_F = 2, 1000


def test_conv_is_channelized_variant():
    conv = ConvCompressor(N_CH, N_F, bottleneck_dim=32, hidden_channels=(8, 16), kernel_size=5)
    mlp = ChannelizedMLPCompressor(N_CH, N_F, hidden_dim_per_channel=16, out_dim_per_channel=8)
    assert isinstance(conv, ChannelizedMLPCompressor)
    # same preprocessing buffers and input layout: 2 * n_channels real channels
    assert conv.n_real_channels == mlp.n_real_channels == 2 * N_CH
    assert conv.mean_whitened.shape == mlp.mean_whitened.shape

    z = torch.randn(3, N_CH, N_F, dtype=torch.complex64)
    for model, dim in ((conv, 32), (mlp, 2 * N_CH * 8)):
        model.eval()
        out = model.encode(model.preprocess(z))
        assert out.shape == (3, dim) and model.bottleneck_dim == dim


def test_conv_encoder_has_no_running_stats():
    # LayerNorm (GroupNorm(1, C)): train and eval compute the same function, so a
    # single observation is summarised identically in both modes.
    torch.manual_seed(0)
    for residual in (False, True):
        conv = ConvCompressor(N_CH, N_F, bottleneck_dim=32, hidden_channels=(8, 16),
                              kernel_size=5, residual=residual)
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                       for m in conv.modules())
        x = conv.preprocess(torch.randn(4, N_CH, N_F, dtype=torch.complex64))
        conv.train()
        out_train = conv.encode(x)
        conv.eval()
        torch.testing.assert_close(conv.encode(x), out_train)
        torch.testing.assert_close(conv.encode(x[:1]), out_train[:1])


def test_per_bin_std_standardises_each_bin():
    torch.manual_seed(0)
    mlp = ChannelizedMLPCompressor(N_CH, N_F, hidden_dim_per_channel=16, out_dim_per_channel=8,
                                   amplitude_normalise=False, per_bin_std=True)
    offset = 100 * torch.rand(N_CH, N_F)
    scale = torch.logspace(-1, 2, N_F)
    cplx = lambda re, im: torch.complex(re, im)
    batches = [{"wave_fd": cplx(offset + scale * torch.randn(500, N_CH, N_F),
                                -offset + scale * torch.randn(500, N_CH, N_F)),
                "noise_fd": cplx(torch.randn(500, N_CH, N_F), torch.randn(500, N_CH, N_F))}
               for _ in range(4)]
    mlp.fit_white_normalisation(batches)
    # the std is that of the network input wave_fd + noise_fd
    x = torch.cat([mlp.preprocess(b["wave_fd"] + b["noise_fd"]) for b in batches])
    assert x.mean(0).abs().max() < 0.1
    torch.testing.assert_close(x.std(0, correction=0), torch.ones(2 * N_CH, N_F),
                               rtol=1e-4, atol=1e-4)


def test_per_bin_std_flag_checks():
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        ChannelizedMLPCompressor(N_CH, N_F, amplitude_normalise=True, per_bin_std=True)
    with pytest.raises(ValueError, match="requires subtract_mean_whitened"):
        ChannelizedMLPCompressor(N_CH, N_F, amplitude_normalise=False,
                                 subtract_mean_whitened=False, per_bin_std=True)
