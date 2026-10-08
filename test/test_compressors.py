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
