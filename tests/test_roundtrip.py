import torch

import torchac


def test_encode_decode_roundtrip():
    c, h, w = 3, 10, 10

    l = 3  # noqa: E741
    lp = l + 1

    cdf = torch.linspace(0, 1, lp, dtype=torch.float32).reshape(1, -1)
    cdf = torch.cat([cdf for _ in range(c * h * w)]).reshape(c, h, w, -1)
    assert cdf.shape[-1] == lp

    sym = torch.arange(c * h * w, dtype=torch.int16) % l
    sym = sym.reshape(c, h, w)

    byte_stream = torchac.encode(sym, cdf, check_input_bounds=True)
    sym_dec = torchac.decode(byte_stream, cdf)
    assert sym_dec.equal(sym)
