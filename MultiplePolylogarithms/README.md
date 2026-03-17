# harmonic-polylog

Arbitrary-precision evaluation of **Harmonic Polylogarithms (HPL)**,
**Multiple Polylogarithms (MPL)**, and **Multiple Zeta Values (MZV)**.

## Project layout

```
harmonic-polylog/
├── README.md
├─pyproject.toml          ← uv-managed project & dependencies
├─.venv/                  ← virtual environment (created by uv)
├─notebooks/
│   └── exploration.ipynb   ← interactive overview with examples & plots
└── src/
    └── hpl.py              ← core library
```

## Quick start

```bash
uv venv --system-site-packages .venv
source .venv/bin/activate
uv add mpmath sympy matplotlib numpy jupyter
jupyter notebook notebooks/exploration.ipynb
```

## Core API

```python
from src.hpl import n_hpl, n_mzv, function_expand_hpl

from mpmath import mp
mp.dps = 50
print(n_hpl((1, 2), 0.5))       # H({1,2}; 0.5) at 50 d.p.
print(n_mzv((3, 1)))             # Z(3,1) = π⁴/360

import sympy as sp
x = sp.Symbol('x')
print(function_expand_hpl((2, -1), x))   # symbolic expansion
```

## References

- Remiddi & Vermaseren (2000), *Harmonic polylogarithms*, Int. J. Mod. Phys. A 15, 725
- Maitre (2006), *HPL, a Mathematica implementation*, Comput. Phys. Commun. 174, 222
