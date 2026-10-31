# Torchac

This library implements fast [arithmetic coding](https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html) for PyTorch,
encoding tensors of symbols into a byte stream given a per-symbol cumulative distribution function (CDF).
It is meant for learned compression, where a model predicts a distribution for every symbol and the coder turns it into bits.

The coder itself is a C++ extension that runs on CPU with 16-bit precision.
Because every symbol carries its own CDF, the distributions can vary freely across channels and spatial positions.
It is derived from [fab-jul/torchac](https://github.com/fab-jul/torchac), repackaged with a typed Python API and a modern build setup.

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.15+
- A C++20 compiler (MSVC on Windows, GCC or Clang elsewhere), since the extension is built at install time

### Install

Using [uv](https://github.com/astral-sh/uv), which builds the extension against the installed PyTorch:

```bash
uv add git+https://github.com/akanelite/torchac.git
```

With pip, install PyTorch first and disable build isolation so the build can see it:

```bash
pip install torch
pip install --no-build-isolation git+https://github.com/akanelite/torchac.git
```

## Quick Start

```python
import torch
import torchac

# 300 symbols, each taking one of L = 3 values, so each CDF has L + 1 = 4 entries.
sym = (torch.arange(300, dtype=torch.int16) % 3).reshape(3, 10, 10)    # (C, H, W), int16
cdf = torch.linspace(0, 1, 4).expand(3, 10, 10, 4).contiguous()        # (C, H, W, L + 1), float in [0, 1]

byte_stream = torchac.encode(sym, cdf, check_input_bounds=True)        # bytes
decoded = torchac.decode(byte_stream, cdf)                             # (3, 10, 10), int16
assert decoded.equal(sym)
```

The CDF has one more dimension than the symbols, and its leading dimensions must match the symbol shape.
Each row starts at 0, ends at 1, and is non-decreasing.
Both tensors must be on CPU.

## License

Licensed under the [GNU General Public License v3.0](LICENSE).
The C++ backend and parts of the Python wrapper are derived from [fab-jul/torchac](https://github.com/fab-jul/torchac), Copyright 2020 ETH Zurich.
