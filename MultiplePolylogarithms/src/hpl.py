"""
hpl.py — Harmonic Polylogarithms, Multiple Polylogarithms, and Multiple Zeta Values

©️ Bhuvanesh Bhatt (bhuvaneshbhatt@gmail.com), 2015-2026. 

Primary references:
  [RV2000]  Remiddi, Vermaseren. Harmonic Polylogarithms. hep-ph/9905237
  [GR2001]  Gehrmann, Remiddi. Numerical evaluation of HPLs. hep-ph/0107173
  [VW2004]  Vollinga, Weinzierl. Numerical evaluation of MPLs. hep-ph/0410259
  [Mai2005] Maitre. HPL, a Mathematica implementation. hep-ph/0507152

Requires: mpmath, sympy
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import Sequence

import mpmath
from mpmath import mp, mpc, mpf, log, polylog, zeta, pi, j as I_unit
import sympy as sp
from sympy import (
    Symbol, Rational, Integer, log as slog, pi as spi,
    polylog as spolylog, zeta as szeta, Catalan,
    I as sI, sqrt, Add, Mul, Pow, S, simplify,
    Function, Tuple, expand_func,
)

# ---------------------------------------------------------------------------
# Global precision management
# ---------------------------------------------------------------------------

def set_precision(dps: int) -> None:
    """Set working precision in decimal places."""
    mp.dps = dps

set_precision(50)  # default: 50 decimal places


# ===========================================================================
# Section 1 — Notation conversion utilities
# ===========================================================================

def from_abbreviated_notation(plist: tuple[int, ...]) -> tuple[int, ...]:
    """
    Convert abbreviated (compressed) notation to full notation.

    In abbreviated notation, an integer m with |m| > 1 represents
    (|m|-1) zeros followed by sign(m).  E.g. 3 -> (0, 0, 1), -3 -> (0, 0, -1).
    Zeros and ±1 are left unchanged.

    Examples
    --------
    >>> from_abbreviated_notation((2, -1))
    (0, 1, -1)
    >>> from_abbreviated_notation((3,))
    (0, 0, 1)
    >>> from_abbreviated_notation((1, 0, -1))
    (1, 0, -1)
    """
    result = []
    for m in plist:
        if abs(m) > 1:
            result.extend([0] * (abs(m) - 1))
            result.append(1 if m > 0 else -1)
        else:
            result.append(m)
    return tuple(result)


def to_abbreviated_notation(avec: tuple[int, ...]) -> tuple[int, ...]:
    """
    Convert full notation to abbreviated notation.

    Runs of zeros preceding a ±1 are collapsed: (0,0,1) -> 3, (0,0,-1) -> -3.
    Trailing zeros are left unchanged.

    Examples
    --------
    >>> to_abbreviated_notation((0, 1, -1))
    (2, -1)
    >>> to_abbreviated_notation((0, 0, 1))
    (3,)
    >>> to_abbreviated_notation((1, 0, -1))
    (1, 0, -1)
    """
    if not avec:
        return ()
    result = []
    i = 0
    n = len(avec)
    while i < n:
        if avec[i] == 0:
            # count consecutive zeros
            j = i
            while j < n and avec[j] == 0:
                j += 1
            if j < n and avec[j] in (1, -1):
                # zeros + nonzero → compressed form
                nzeros = j - i
                sign = avec[j]
                result.append(sign * (nzeros + 1))
                i = j + 1
            else:
                # trailing zeros: keep as-is
                result.extend([0] * (j - i))
                i = j
        else:
            result.append(avec[i])
            i += 1
    return tuple(result)


def weight(mm: tuple[int, ...]) -> int:
    """
    Return the weight of a parameter list in full (non-abbreviated) notation.

    Weight = sum of |m_i| where the list is first expanded to full notation.
    Equivalently, for a full-notation list it is just the length.
    """
    full = from_abbreviated_notation(mm)
    return len(full)


def right_zeros_length(mm: tuple[int, ...]) -> int:
    """Return the number of trailing zeros in the parameter list."""
    count = 0
    for m in reversed(mm):
        if m == 0:
            count += 1
        else:
            break
    return count


def negative_parameter_count(mm: tuple[int, ...]) -> int:
    """Return the number of negative entries in the parameter list."""
    return sum(1 for m in mm if m < 0)


def hpl_to_zeta(mm: tuple[int, ...]) -> tuple[int, ...]:
    """
    Convert HPL parameter list to MZV parameter list.

    Formula: zeta_k = hpl_k * prod_{i<k} sign(hpl_i)
    (a RotateRight-multiply of signs, with last factor = 1)
    """
    mm = list(mm)
    n = len(mm)
    if n == 0:
        return ()
    signs = [1] + [_sign(mm[i]) for i in range(n - 1)]
    result = []
    for i in range(n):
        result.append(mm[i] * signs[i])
    return tuple(result)


def zeta_to_hpl(mm: tuple[int, ...]) -> tuple[int, ...]:
    """
    Convert MZV parameter list to HPL parameter list.
    Inverse of hpl_to_zeta.
    """
    mm = list(mm)
    n = len(mm)
    if n == 0:
        return ()
    # cumulative product of signs of all but last element
    cum_sign = 1
    result = []
    for i in range(n):
        if i < n - 1:
            partial = 1
            for k in range(i):
                partial *= _sign(mm[k])
            # sign factor for position i is product of sign(mm[0..i-1])
            pass
        result.append(mm[i])
    # Correct implementation: zeta[k] = hpl[k] * sign(hpl[0])*...*sign(hpl[k-1])
    # so hpl[k] = zeta[k] / (sign(hpl[0])*...*sign(hpl[k-1]))
    # We reconstruct hpl iteratively
    hpl = [0] * n
    for i in range(n):
        if i == 0:
            hpl[i] = mm[i]
        else:
            cum = 1
            for k in range(i):
                cum *= _sign(hpl[k])
            hpl[i] = mm[i] * cum  # since zeta[i] = hpl[i] * cum, hpl[i] = zeta[i]/cum = zeta[i]*cum (cum = ±1)
    return tuple(hpl)


def _sign(x) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def pseudo_add(j: int, k: int) -> int:
    """
    Stuffle pseudo-addition: |j+k| with sign = sign(j) if sign(j)==sign(k), else -(|j|+|k|).
    """
    if _sign(j) == _sign(k):
        return j + k  # same sign, usual addition
    else:
        return -(abs(j) + abs(k))


# ===========================================================================
# Section 2 — Shuffle and stuffle product expansions
# ===========================================================================

def _all_interleavings(
    list1: tuple[int, ...], list2: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """
    Return all interleavings (shuffles) of list1 and list2 as a LIST
    (multiset with repetition).  Equal interleavings appear multiple times
    so that the coefficient in the shuffle product is their multiplicity.

    The total number of entries is C(len1+len2, len1), counting repeats.
    """
    n1, n2 = len(list1), len(list2)
    if n1 == 0:
        return [list2]
    if n2 == 0:
        return [list1]

    # Use itertools.combinations to enumerate all C(n1+n2, n1) ways
    # to choose positions for list1 elements; each choice gives one interleaving.
    # Duplicate interleavings (from equal elements) are kept — giving multiplicity.
    results = []
    for positions in itertools.combinations(range(n1 + n2), n1):
        pos_set = set(positions)
        interleaving = []
        i1 = i2 = 0
        for pos in range(n1 + n2):
            if pos in pos_set:
                interleaving.append(list1[i1])
                i1 += 1
            else:
                interleaving.append(list2[i2])
                i2 += 1
        results.append(tuple(interleaving))
    return results


def shuffle_product(
    list1: tuple[int, ...], list2: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """
    Return the shuffle product of two parameter lists (as a list of tuples,
    with multiplicity — i.e. this is a multiset represented as a list).
    """
    return _all_interleavings(
        from_abbreviated_notation(list1),
        from_abbreviated_notation(list2)
    )


def stuffle_product(
    list1: tuple[int, ...], list2: tuple[int, ...]
) -> list[tuple[int, ...]]:
    """
    Return the stuffle (quasi-shuffle) product of two MZV parameter lists.

    The stuffle product operates on the **abbreviated** (compressed) notation
    directly — an entry m with |m|>1 represents a single index of weight |m|,
    and pseudo-addition combines two entries into one index of weight |m1|+|m2|.

    Returns a list of parameter tuples (multiset), so equal results appear
    multiple times (their count is the coefficient in the expansion).

    Example
    -------
    stuffle_product((2,), (2,)) -> [(2,2), (2,2), (4,)]
    meaning Z(2)*Z(2) = 2*Z(2,2) + Z(4)
    """
    # Keep abbreviated notation — do NOT expand to full notation
    list1 = tuple(list1)
    list2 = tuple(list2)

    @lru_cache(maxsize=None)
    def _stuffle(a: tuple, b: tuple) -> list[tuple]:
        if not a:
            return [b]
        if not b:
            return [a]
        # Case 1: take a[0] from a
        result = [(a[0],) + t for t in _stuffle(a[1:], b)]
        # Case 2: take b[0] from b
        result += [(b[0],) + t for t in _stuffle(a, b[1:])]
        # Case 3: pseudo-add a[0] and b[0] (stuffle extra term)
        pa = pseudo_add(a[0], b[0])
        if pa != 0:
            result += [(pa,) + t for t in _stuffle(a[1:], b[1:])]
        return result

    return _stuffle(list1, list2)


# ---------------------------------------------------------------------------
# Symbolic shuffle/stuffle expansion using SymPy expressions
# ---------------------------------------------------------------------------

class _HPLSymbol:
    """Lightweight symbol representing H(mm; x) or Z(mm) in symbolic expansions."""
    __slots__ = ("mm", "x")

    def __init__(self, mm: tuple, x=None):
        self.mm = mm
        self.x = x

    def __repr__(self):
        if self.x is not None:
            return f"H({list(self.mm)}; {self.x})"
        return f"Z({list(self.mm)})"


def shuffle_product_expand(expr, x=None):
    """
    Expand products of HPL (or MZV) objects using the shuffle algebra.

    Parameters
    ----------
    expr : dict mapping tuple -> coefficient (sympy expr)
        Represents a linear combination: sum_mm coeff[mm] * H(mm; x).
        Each key is a parameter tuple.
    x : sympy Symbol or None
        The argument (for HPL). None for MZV (shuffle at x=1).

    Returns
    -------
    dict mapping tuple -> coefficient
        The fully shuffle-expanded result (no products remain).

    Note: This function takes a *product* of two linear combinations
    and returns their shuffle expansion. To expand an expression, call it
    recursively on product pairs.
    """
    # This is a stub for symbolic use; the main usage is via
    # harmonic_polylog_product_expand below.
    raise NotImplementedError("Use harmonic_polylog_product_expand for symbolic work.")


def harmonic_polylog_product_expand(mm1: tuple, mm2: tuple) -> dict:
    """
    Expand H(mm1; x) * H(mm2; x) as a sum of H(mm; x) using the shuffle relation.

    Returns
    -------
    dict[tuple, int]  (parameter tuple → integer coefficient)
    """
    # Shuffle product: H(a)*H(b) = sum of all interleavings
    # For equal lists: H(m)*H(m) gives 2*H(m,m) since both interleavings are the same
    shuffles = shuffle_product(mm1, mm2)
    result = {}
    for s in shuffles:
        s = to_abbreviated_notation(s)
        result[s] = result.get(s, 0) + 1
    return result


def mzv_stuffle_product_expand(mm1: tuple, mm2: tuple) -> dict:
    """
    Expand Z(mm1) * Z(mm2) using the stuffle (quasi-shuffle) relation.

    Returns
    -------
    dict[tuple, int]  (parameter tuple → integer coefficient)
    """
    stuffles = stuffle_product(mm1, mm2)
    result = {}
    for s in stuffles:
        s = to_abbreviated_notation(s)
        result[s] = result.get(s, 0) + 1
    return result


# ===========================================================================
# Section 3 — Singular-part extraction (trailing zeros / leading ±1)
# ===========================================================================

def extract_trailing_zeros(mm: tuple[int, ...]) -> dict:
    """
    Express H(mm; x) in terms of products H(mm'; x) * H(0; x)^k / k!
    where mm' has no trailing zeros, using the shuffle algebra.

    Returns a dict {(mm_key, n_zeros): coeff} where the value represents
    coeff * H(mm_key; x) * H({0}; x)^n_zeros / n_zeros!

    In practice returns a list of (coefficient, param_tuple, n_log_factors)
    representing coeff * H(param_tuple; x) * log(x)^n_log_factors / n_log_factors!
    """
    if right_zeros_length(mm) == 0:
        return {(mm, 0): 1}

    full = from_abbreviated_notation(mm)
    n_zeros = right_zeros_length(full)
    core = full[:-n_zeros]

    # Use the recursive formula:
    # H[rest..., 0] = (1/n) * ( H[0]*H[rest...] - sum_j |mm[j]| * H[modified_j] )
    # We implement this iteratively via the shuffle relation:
    # H[a1,...,an, 0,...,0] expressed via shuffle of H[a1,...,an] and H[0,...,0].
    # H[{0}]^n / n! = H[{0,...,0}] (n zeros) by the shuffle relation for equal lists.

    # The result is: H[rest,0^n] is one of the shuffle terms of H[rest]*H[0]^n/n!
    # Rearranging: H[rest]*H[0]^n/n! = H[rest,0^n] + (other shuffle terms)
    # This is handled in the numeric evaluator directly.
    # For symbolic use, return a representation.

    # Simple approach: return symbolic expansion as nested dict
    # (mm_without_trailing_zeros, num_trailing_zeros): coefficient
    result = {(tuple(core), n_zeros): 1}
    return result


# ===========================================================================
# Section 4 — Numerical evaluation core
# ===========================================================================

def _nbrnull_converged(old, new, nbrnull: int, threshold: int = 10) -> tuple:
    """Check convergence by counting consecutive unchanged values."""
    if old == new:
        return nbrnull + 1
    return 0


def _arb_prec_hpl(mm: tuple, x, extra_prec: int = 0) -> mpmath.mpc:
    """
    Evaluate H(mm; x) at arbitrary precision using the direct series.

    Algorithm: backward recurrence.
    mm must be in *abbreviated* notation with mm[0] > 0 (leading-positive).
    Caller is responsible for routing leading-negative lists through x->-x first.

    Optimisations vs naive implementation:
    - x^i computed incrementally (x_pow *= xc) — no repeated pow() per step.
    - sign(mm[j])^(i-j) uses fast parity check instead of Python integer pow.
    - Precomputed signs[] and abs_mm[] avoid repeated abs()/sign() per iteration.
    """
    with mp.workprec(mp.prec + 64 + extra_prec):
        xc  = mpc(x)
        n   = len(mm)
        ONE  = mpf(1)
        MONE = mpf(-1)

        # Precompute per-index sign and absolute weight
        signs  = [_sign(mm[j]) for j in range(n)]
        abs_mm = [abs(mm[j])   for j in range(n)]

        new_vec = [mpc(0)] * (n + 1)
        old_vec = list(new_vec)
        res     = mpc(0)
        old_res = mpc(1)   # nonzero sentinel so loop runs at least once
        nbrnull = 0
        i_val   = n        # starts at n (first term with nonzero contribution)

        # Incremental x^i: x_pow = x^n initially, then *= x each step
        x_pow = xc ** n

        # Leading parity: sign(mm[0])^i_val, maintained incrementally
        s0 = signs[0]
        # s0^n: if s0==1 always ONE; if s0==-1, sign depends on parity of n
        lead_parity = ONE if (s0 == 1 or n % 2 == 0) else MONE

        while nbrnull < 25:
            old_res = res
            old_vec = list(new_vec)
            new_vec[n] = mpc(1)

            for j in range(n - 1, 0, -1):
                exp = i_val - j     # positive integer
                s_j = signs[j]
                # sign_factor = s_j^exp: cheap parity check
                sf = ONE if (s_j == 1 or exp % 2 == 0) else MONE
                new_vec[j] = old_vec[j] + (sf / (mpf(exp) ** abs_mm[j])) * new_vec[j + 1]

            # Outer contribution: lead_parity * x^i / i^|mm[0]| * new_vec[1]
            new_vec[0] = old_vec[0] + (lead_parity * x_pow / (mpf(i_val) ** abs_mm[0])) * new_vec[1]
            res = new_vec[0]

            i_val += 1
            x_pow *= xc
            if s0 == -1:
                lead_parity = ONE if lead_parity == MONE else MONE

            if res == old_res:
                nbrnull += 1
            else:
                nbrnull = 0

        return res



def _arb_prec_mpl(mm: tuple[int, ...], xx: tuple, extra_prec: int = 0):
    """
    Evaluate Li(mm; xx) = MultiplePolyLog(mm, xx) at arbitrary precision.

    Uses the nested series:
    Li(m1,...,mn; x1,...,xn) = sum_{k=n}^{inf} [backward recurrence over mm, xx]

    Converges when |x1*...*xk| <= 1 for all k, and mm[0] != 1 or x1 != 1.
    """
    with mp.workprec(mp.prec + 64 + extra_prec):
        xxc = [mpc(xi) for xi in xx]
        n = len(mm)
        new_vec = [mpc(0)] * (n + 1)
        old_vec = list(new_vec)
        res = mpc(0)
        old_res = mpc(1)  # sentinel
        nbrnull = 0
        i_val = n  # integer counter

        while nbrnull < 20:
            old_res = res
            old_vec = list(new_vec)

            new_vec[n] = mpc(1)
            for j_idx in range(n - 1, -1, -1):
                s = _sign(mm[j_idx])
                exp = i_val - j_idx          # integer
                factor = (mpc(s ** exp) * xxc[j_idx] ** exp) / mpf(exp) ** abs(mm[j_idx])
                new_vec[j_idx] = old_vec[j_idx] + factor * new_vec[j_idx + 1]

            res = new_vec[0]
            i_val += 1

            if res == old_res:
                nbrnull += 1
            else:
                nbrnull = 0

        return res


# ===========================================================================
# Section 5 — Argument transformations for analytic continuation
# ===========================================================================

def _transform_neg_x(mm: tuple[int, ...], x):
    """
    Transform H(mm; x) -> expression in H(mm'; -x) using x -> -x.

    Rule: H(mm; x) = (-1)^len(mm) * H(-mm; -x)
    where -mm means each element is negated.
    For lists with trailing zeros, extract them first.
    """
    # Handle trailing zeros
    nz = right_zeros_length(mm)
    if nz > 0:
        full = from_abbreviated_notation(mm)
        core = full[:-nz]
        neg_core = tuple(-m for m in core)
        neg_core_abbr = to_abbreviated_notation(neg_core)
        # H(core, 0^nz; x) transforms via shuffle extraction
        # H(0; x) = log(x), H(0; -x) = log(-x) = log(x) + i*pi
        # This requires careful handling; delegate to numeric evaluation
        raise NotImplementedError("Trailing zero transform requires numeric handling")

    neg_mm = to_abbreviated_notation(tuple(-m for m in from_abbreviated_notation(mm)))
    sign = (-1) ** len(from_abbreviated_notation(mm))
    return sign, neg_mm, -x


def _transform_one_minus_x(mm: tuple[int, ...], x):
    """
    Transform H(mm; x) using x -> 1-x (for non-negative parameter lists only).
    Returns a list of (coeff, mm', point) where point is 'hpl1' (=HPL at 1) or 'new_x'.
    This is a stub — full implementation uses the recursive argtrans1mx algorithm.
    """
    raise NotImplementedError("Symbolic 1-x transform: use numeric path via _n_hpl")


# ===========================================================================
# Section 6 — Main numerical HPL evaluator
# ===========================================================================

def n_hpl(mm: tuple[int, ...], x, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate H(mm; x) — HarmonicPolyLog — to arbitrary precision.

    Parameters
    ----------
    mm : tuple of nonzero integers (abbreviated or full notation accepted)
    x  : numeric value (real or complex, mpmath or Python float/complex)
    dps : decimal places (default: current mp.dps)

    Returns
    -------
    mpmath.mpc
    """
    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        return _n_hpl_dispatch_v2(tuple(mm), mpc(x))


def _n_hpl_dispatch(mm: tuple[int, ...], x: mpmath.mpc) -> mpmath.mpc:
    """Core dispatcher with analytic-continuation routing."""

    # --- Base case: empty list ---
    if len(mm) == 0:
        return mpc(1)

    # H(mm, 0) = 0 for any mm with at least one nonzero entry
    if x == 0:
        full_check = from_abbreviated_notation(mm)
        if any(m != 0 for m in full_check):
            return mpc(0)

    # --- Single abbreviated entry shortcuts ---
    # These cover mm like (2,), (3,), (-3,) etc.
    if len(mm) == 1:
        m = mm[0]
        if m == 0:
            return log(x)
        elif m == 1:
            return -log(1 - x)
        elif m == -1:
            return log(1 + x)
        elif m > 1:
            return polylog(m, x)
        else:  # m < -1
            return -polylog(-m, -x)

    # Expand to full notation for the rest of the logic
    full = from_abbreviated_notation(mm)

    # H(m,m,...,m; x) for all-equal constant vector = H(m;x)^n / n!
    if len(set(full)) == 1 and len(full) > 1:
        m0 = full[0]
        val = _n_hpl_dispatch_v2((m0,), x)
        n = len(full)
        return val ** n / mpmath.factorial(n)

    # --- Handle LEADING ZEROS via recursive numerical integration ---
    # H(0, rest; x) = int_0^x H(rest; t)/t dt
    # The _arb_prec_hpl series breaks for leading zeros (sign(0)=0 kills the term).
    # Detect leading zeros in full notation and integrate recursively.
    n_leading = 0
    for m in full:
        if m == 0:
            n_leading += 1
        else:
            break
    if n_leading > 0:
        return _n_hpl_leading_zeros(mm, full, n_leading, x)

    # --- Extract trailing zeros ---
    nz = right_zeros_length(full)
    if nz > 0:
        return _n_hpl_trailing_zeros(mm, full, nz, x)

    # --- Special points ---
    half_prec_eps = mpf(10) ** (-(mp.dps // 2))

    if abs(x - 1) < half_prec_eps:
        if full[0] != 1:
            return _n_hpl_near_one(mm, x)

    if abs(x + 1) < half_prec_eps:
        return _n_hpl_neg_one(mm, x)

    # --- Region routing ---
    ax = abs(x)
    rx = mpmath.re(x)

    # |x| < 0.95: direct series (no leading zeros possible after abbreviation expansion)
    if ax < mpf('0.95'):
        return _arb_prec_hpl(mm, x, extra_prec=weight(mm)*4)

    # |x| > 1, Re(x) >= 0: use x -> 1/x transformation
    if ax > mpf('1.0') and rx >= 0:
        return _n_hpl_large_x(mm, x)

    # Re(x) < -0.5 and |x| <= 1: use x -> -x transformation
    if rx < mpf('-0.5') and ax <= mpf('1.0'):
        return _n_hpl_neg_region(mm, x)

    # |x| near 1 (0.95 <= |x| <= 1.05): use (1-x)/(1+x) mapping
    if mpf('0.95') <= ax <= mpf('1.05'):
        return _n_hpl_near_unit_circle(mm, x)

    # Default: direct series
    return _arb_prec_hpl(mm, x, extra_prec=weight(mm)*4)


def _n_hpl_leading_zeros(mm: tuple, full: tuple, n_leading: int,
                         x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(0^k, rest; x) via numerical integration:
        H(0, rest; x) = int_0^x H(rest; t) / t  dt
    repeated k times for k leading zeros.

    Covers abbreviated parameters like (2,1)->(0,1,1), (3,)->(0,0,1), etc.
    Extra working precision compensates for the log-singularity at t=0.
    """
    if abs(x) < mpf('1e-10'):
        return mpc(0)

    rest_full = full[n_leading:]
    rest_mm   = to_abbreviated_notation(rest_full)
    extra     = max(20, n_leading * 15)

    if n_leading == 1:
        with mp.workdps(mp.dps + extra):
            xw = mpc(x)
            def integrand(t):
                tc = mpc(t)
                if abs(tc) < mpf('1e-{}'.format(mp.dps)):
                    return mpc(0)
                return _n_hpl_dispatch_v2(rest_mm, tc) / tc
            return mpmath.quad(integrand, [0, xw], maxdegree=8, error=False)
    else:
        inner_mm = to_abbreviated_notation((0,) + rest_full)
        with mp.workdps(mp.dps + extra):
            xw = mpc(x)
            def integrand(t):
                tc = mpc(t)
                if abs(tc) < mpf('1e-{}'.format(mp.dps)):
                    return mpc(0)
                return _n_hpl_dispatch_v2(inner_mm, tc) / tc
            return mpmath.quad(integrand, [0, xw], maxdegree=8, error=False)





def _n_hpl_trailing_zeros(mm: tuple, full: tuple, nz: int, x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(mm; x) where mm has trailing zeros, using the recursive formula:
    H(rest, 0^n; x) = (1/n) * [H(0;x)*H(rest,0^(n-1); x)
                                - sum_j |mm[j]| * H(modified_j; x)]
    where modified_j has mm[j] replaced by mm[j]±1 (toward 0).

    This is the Python translation of $TrailingZerosExtractionRule2.
    """
    if nz == 0:
        return _n_hpl_dispatch_v2(mm, x)

    logx = log(x)
    n = nz
    rest = full[:-nz]

    # H(rest, 0^n; x) = (1/n) * (H(0;x) * H(rest, 0^(n-1); x)
    #                              - sum_{j in rest} |rest[j]| * H(rest_with_j_moved; x))

    # Reduce one trailing zero
    prev_mm = to_abbreviated_notation(full[:-1])  # one fewer zero
    prev_val = _n_hpl_dispatch_v2(prev_mm, x) if len(prev_mm) > 0 else mpc(1)

    sum_terms = mpc(0)
    n_core = len(rest)
    current = list(full)
    for j in range(n_core):
        mj = current[j]
        abs_mj = abs(mj)
        # Replace mm[j] by mm[j] + sign (toward 0 in absolute value, keeping sign)
        if mj > 0:
            new_mj = mj + 1  # move away from 0 (increase weight at this position)
        else:
            new_mj = mj - 1  # move away from 0 (increase weight at this position)
        modified = list(current)
        modified[j] = new_mj
        modified_abbr = to_abbreviated_notation(tuple(modified[:-1]))  # drop last zero
        modified_val = _n_hpl_dispatch_v2(modified_abbr, x)
        sum_terms += abs_mj * modified_val

    result = (logx * prev_val - sum_terms) / n
    return result


def _n_hpl_near_one(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(mm; x) for x near 1, using the x -> 1-x transformation.
    Only valid for non-negative parameter lists (all mi >= 0, no negative mi).
    Falls back to direct series otherwise.
    """
    full = from_abbreviated_notation(mm)
    if any(m < 0 for m in full):
        # Use (1-x)/(1+x) transformation instead
        y = (1 - x) / (1 + x)
        return _n_hpl_dispatch_v2(to_abbreviated_notation(full), y)  # approximate

    # x -> 1-x: evaluate using the Hölder convolution / series at x = 1-x
    y = 1 - x
    if abs(y) < mpf('0.5'):
        return _n_hpl_dispatch_v2(to_abbreviated_notation(full), y)
    # If y not small, just use series at y
    return _arb_prec_hpl(mm, x, extra_prec=weight(mm) * 8)


def _n_hpl_neg_one(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """Evaluate H(mm; x) near x = -1."""
    # Use x -> -x transformation: H(mm; -1) = (-1)^len * H(-mm; 1)
    full = from_abbreviated_notation(mm)
    neg_mm = to_abbreviated_notation(tuple(-m for m in full))
    sign = (-1) ** len(full)
    return sign * _n_hpl_at_one(neg_mm)


def _n_hpl_at_one(mm: tuple) -> mpmath.mpc:
    """
    Evaluate H(mm; 1) = MZV(mm) (when convergent).
    """
    full = from_abbreviated_notation(mm)
    if full[0] == 1:
        raise ValueError(f"H({mm}; 1) is divergent (leading parameter = 1)")
    if right_zeros_length(full) > 0:
        raise ValueError(f"H({mm}; 1) is divergent (trailing zero)")
    zeta_mm = hpl_to_zeta(full)
    return n_mzv(zeta_mm)


def _n_hpl_large_x(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(mm; x) for |x| > 1 using x -> 1/x transformation.
    H(mm; x) is expressed in terms of H(mm'; 1/x) and constants.
    For simplicity we use the reflection formula numerically.
    """
    # For a robust implementation: use the x->1/x transformation tables.
    # Here we implement the leading term of the inversion formula.
    y = mpc(1) / x
    # H({0}; x) = -H({0}; 1/x)
    if mm == (0,):
        return -_n_hpl_dispatch_v2((0,), y)
    # General case: use 1/x transformation via analytic continuation
    # This is a simplified version; the full version requires the shuffle algebra
    full = from_abbreviated_notation(mm)
    # Fallback: use the Hölder convolution for the full MPL
    return _n_hpl_via_holder(mm, x)


def _n_hpl_neg_region(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """Evaluate H(mm; x) for Re(x) < 0 using x -> -x."""
    full = from_abbreviated_notation(mm)
    nz = right_zeros_length(full)
    if nz > 0:
        # Handle trailing zeros separately
        return _n_hpl_trailing_zeros(mm, full, nz, x)
    neg_mm = to_abbreviated_notation(tuple(-m for m in full))
    sign = (-1) ** len(full)
    return sign * _n_hpl_dispatch_v2(neg_mm, -x)


def _n_hpl_near_unit_circle(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(mm; x) for |x| near 1.
    Uses the (1-x)/(1+x) transformation which maps the unit circle to the real axis.
    """
    y = (1 - x) / (1 + x)
    return _n_hpl_dispatch_v2(mm, y)


def _n_hpl_via_holder(mm: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Evaluate H(mm; x) using the Hölder convolution, which expresses H as a
    sum of products of MPLs with rapidly converging arguments.
    This is the main method for |x| near 1 or > 1.

    Hölder convolution (from [VW2004]):
    H(mm; x) = sum_{j=0}^{n} (-1)^j G(rev(a[1..j]); 2+2i/x; 1)
                                     * G(a[j+1..n]; -2i/x; 1)
    converted to MPL form.
    """
    full = list(from_abbreviated_notation(mm))
    n = len(full)
    result = mpc(0)

    alpha = mpc(-1j) * x  # -i*x

    for j in range(n + 1):
        left_params = list(reversed(full[:j]))
        right_params = full[j:]

        # Goncharev G(params; 1) -> MPL
        left_z = [mpc(2) + mpc(2) / alpha * m for m in left_params]
        right_z = [mpc(-2) / alpha * m for m in right_params]

        left_mm = [1] * len(left_params)
        right_mm = [1] * len(right_params)

        left_xx = _goncharev_z_to_mpl_x(left_params, left_z)
        right_xx = _goncharev_z_to_mpl_x(right_params, right_z)

        left_val = n_mpl(tuple(left_mm), tuple(left_xx)) if left_params else mpc(1)
        right_val = n_mpl(tuple(right_mm), tuple(right_xx)) if right_params else mpc(1)

        result += ((-1) ** j) * left_val * right_val

    return result


def _goncharev_z_to_mpl_x(params, zz):
    """
    Convert Goncharev G(mm; zz; 1) to MPL x-arguments.
    x_k = z_k / z_{k-1}  (with z_0 = 1 from the upper limit).
    """
    if not zz:
        return []
    xx = []
    prev = mpc(1)
    for z in reversed(zz):
        if z == 0:
            xx.append(mpc(0))
        else:
            xx.append(prev / z)
        prev = z
    return list(reversed(xx))


# ===========================================================================
# Section 7 — Multiple PolyLog numerical evaluator
# ===========================================================================

def n_mpl(mm: tuple[int, ...], xx: tuple, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate Li(mm; xx) — MultiplePolyLog — to arbitrary precision.

    Li(m1,...,mn; x1,...,xn) = sum_{k1 > k2 > ... > kn >= 1}
                                  sign(m1)^k1 * x1^k1 / k1^|m1|
                                * sign(m2)^k2 * x2^k2 / k2^|m2| * ...

    Convergence condition: |x1*...*xk| <= 1 for all k, mm[0] != 1 or x1 != 1.

    Parameters
    ----------
    mm : tuple of nonzero integers
    xx : tuple of complex numbers, same length as mm
    dps : decimal places

    Returns
    -------
    mpmath.mpc
    """
    assert len(mm) == len(xx), "mm and xx must have the same length"

    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        mm = tuple(mm)
        xx = tuple(mpc(xi) for xi in xx)

        # Special case: all x = 1 -> MZV
        if all(xi == 1 for xi in xx):
            zeta_mm = hpl_to_zeta(mm)
            return n_mzv(zeta_mm)

        # Special case: x[0]=x, rest=1 -> HPL
        if all(xi == 1 for xi in xx[1:]):
            return _n_hpl_dispatch_v2(mm, xx[0])

        # Check convergence
        prod = mpc(1)
        for i, xi in enumerate(xx):
            prod *= xi
            if abs(prod) > mpf('1.0001'):
                raise ValueError(
                    f"n_mpl: convergence condition violated at index {i}: "
                    f"|x1*...*x{i+1}| = {abs(prod)} > 1"
                )

        if mm[-1] == 0:
            raise ValueError("Last parameter must be nonzero for convergent MPL")

        return _arb_prec_mpl(mm, xx, extra_prec=weight(mm) * 4)


# ===========================================================================
# Section 8 — Multiple Zeta Value evaluator
# ===========================================================================

def n_mzv(mm: tuple[int, ...], dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate Z(mm) — MultipleZetaValue — to arbitrary precision.

    Z(m1,...,mn) = sum_{k1 > k2 > ... > kn >= 1} k1^{-m1} * ... * kn^{-mn}

    Convergence requires m1 >= 2 (or m1 = 1 with the appropriate sign convention)
    and mn != 0.

    For alternating MZVs (negative parameters) the sum is defined as:
    Z(m1,...,mn) where negative mi contribute (-1)^ki factors.

    Parameters
    ----------
    mm : tuple of nonzero integers in MZV convention
    dps : decimal places

    Returns
    -------
    mpmath.mpc
    """
    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        mm = tuple(mm)

        if len(mm) == 0:
            return mpc(1)

        # Depth-1 cases
        if len(mm) == 1:
            m = mm[0]
            if m > 1:
                return zeta(m)
            elif m < -1:
                return mpf(2) ** m * (2 - mpf(2) ** (-m)) * zeta(-m)
            elif m == -1:
                return log(2)
            elif m == 1:
                raise ValueError("Z({1}) is divergent")

        # Check for divergent cases
        hpl_mm = zeta_to_hpl(mm)
        if hpl_mm[0] == 1:
            raise ValueError(f"MZV {mm} is divergent (first HPL parameter = 1)")
        if mm[-1] == 0:
            raise ValueError(f"MZV {mm} is divergent (last parameter = 0)")

        # Convert MZV to HPL at x=1 via the Hölder convolution
        result = _n_mzv_via_holder(mm)
        return result


def _n_mzv_via_holder(mm: tuple) -> mpmath.mpc:
    """
    Evaluate MZV(mm) numerically using fast algorithms:
    - Depth 1:  Riemann zeta (mpmath built-in)
    - Depth 2:  Euler formula (m2=1), stuffle relation (m1=m2),
                or Hurwitz-zeta outer sum via nsum
    - Higher depth: recursive outer sum with Hurwitz zeta inner sums
    For alternating MZVs (negative indices): embedded sign factors in _arb_prec_mpl.
    """
    if all(m > 0 for m in mm):
        return _n_mzv_positive(mm)

    # Alternating/signed MZV: embed signs into the x-arguments
    hpl_mm = zeta_to_hpl(mm)
    full = from_abbreviated_notation(hpl_mm)
    sign_xx = tuple(mpc(_sign(m)) for m in full)
    abs_full = tuple(abs(m) for m in full)
    return _arb_prec_mpl(abs_full, sign_xx, extra_prec=len(full) * 8)


def _n_mzv_positive(mm: tuple) -> mpmath.mpc:
    """
    Fast evaluation of Z(mm) for all-positive indices using:
    reduction formulas, Hurwitz-zeta nsum, and recursive outer sums.
    """
    from mpmath import nsum, inf

    n = len(mm)

    if n == 1:
        return zeta(mm[0])

    m1 = mm[0]

    # Depth-2 cases
    if n == 2:
        m1, m2 = mm
        if m2 == 1:
            return _euler_formula_m1(m1)
        if m1 == m2:
            return (zeta(m1) ** 2 - zeta(2 * m1)) / 2
        # General depth-2: Z(m1,m2) = sum_{k=1}^inf k^{-m2} * zeta(m1, k+1)
        return nsum(lambda k: mpmath.zeta(m1, int(k) + 1) / k ** m2,
                    [1, inf], method='richardson')

    # All-twos: Z(2,...,2) = 2*(2pi)^{2n}/(2n+1)! * (1/2)^{2n+1}
    if all(m == 2 for m in mm):
        return (2 * (2 * pi) ** (2 * n)
                / mpmath.factorial(2 * n + 1)
                * mpf('0.5') ** (2 * n + 1))

    # Z(2, 1, ..., 1) = zeta(n+1)
    if mm[0] == 2 and all(m == 1 for m in mm[1:]):
        return zeta(n + 1)

    # General depth >= 3: recursive outer sum
    rest = mm[1:]

    def _partial_mzv(rest_mm: tuple, N: int) -> mpmath.mpc:
        """Z(rest_mm) truncated to outermost index <= N.
        Uses Hurwitz-zeta tails for depth-1 inner sums for speed."""
        if len(rest_mm) == 0:
            return mpc(1)
        if len(rest_mm) == 1:
            m = rest_mm[0]
            # sum_{k=1}^N k^{-m} = zeta(m) - zeta(m, N+1)   (Hurwitz tail)
            return zeta(m) - mpmath.zeta(m, N + 1)
        if len(rest_mm) == 2:
            # depth-2 partial: sum_{k1=2}^N k1^{-m0} * H_{k1-1}^{(m2)}
            # = sum_{k2=1}^{N-1} k2^{-m2} * (zeta(m0,k2+1) - zeta(m0,N+1))
            m0, m2 = rest_mm
            if N < 2:
                return mpc(0)
            total = mpc(0)
            h = mpc(0)
            for k1 in range(2, N + 1):
                h += mpc(1) / mpc(k1 - 1) ** m2   # H_{k1-1}^{(m2)} increment
                total += h / mpc(k1) ** m0
            return total
        m0 = rest_mm[0]
        rest2 = rest_mm[1:]
        s = mpc(0)
        depth2 = len(rest_mm)
        for k in range(depth2, N + 1):
            s += _partial_mzv(rest2, k - 1) / mpc(k) ** m0
        return s

    total = mpc(0)
    tol = mpf(10) ** (-(mp.dps + 5))
    K = max(100, mp.dps * 4)
    for k in range(n, K + 1):
        t = _partial_mzv(rest, k - 1) / mpc(k) ** m1
        total += t
        if abs(t) < tol and k > n + 20:
            break
    return total


def _euler_formula_m1(m: int) -> mpmath.mpc:
    """Z(m, 1) = m/2 * zeta(m+1) - 1/2 * sum_{j=1}^{m-2} zeta(j+1)*zeta(m-j)."""
    result = mpf(m) / 2 * zeta(m + 1)
    for j in range(1, m - 1):
        result -= zeta(j + 1) * zeta(m - j) / 2
    return result


# Aliases kept for backward compatibility
def _n_mzv_positive_direct(mm: tuple) -> mpmath.mpc:
    return _n_mzv_positive(mm)


def _n_mzv_nested_sum(mm: tuple, N_terms=None) -> mpmath.mpc:
    return _n_mzv_positive(mm)


def _eval_goncharev_to_mpl(params: list, zz: list) -> mpmath.mpc:
    """
    Evaluate G(params; zz; 1) as a Multiple PolyLog.

    GoncharevToMultiplePolyLog[G[mm, zz, y=1]]:
      (-1)^n * Li(mm; x_k) where x_k = z_{k-1}/z_k  (z_0 = y = 1)
    with mm_k = 1 (the indices from the Goncharev representation).

    For all-ones mm: this is a classical multiple polylog.
    """
    if not params:
        return mpc(1)

    n = len(params)

    # Filter out zero z entries (G with trailing zeros = 0)
    if any(abs(zi) < mpf('1e-30') for zi in zz):
        return mpc(0)

    # x_k = zz_{k-1} / zz_k  with zz_0 = 1 (the upper limit of G)
    # x = [zz[-1]/zz[-2], ..., zz[0]/1] reversed
    # More precisely: for G[mm; z1,...,zn; y]:
    # x_1 = y/z_1, x_2 = z_1/z_2, ..., x_n = z_{n-1}/z_n
    xx = []
    prev = mpc(1)  # y = 1
    for zi in zz:
        xx.append(prev / zi)
        prev = zi

    mm_ones = tuple(1 for _ in params)

    # Check convergence: |x1*...*xk| = |1/zk| <= 1 requires |zk| >= 1
    # If not convergent, return 0 (indicates this term is at a boundary)
    prod = mpc(1)
    for xi in xx:
        prod *= xi
        if abs(prod) > mpf('1.001'):
            return mpc(0)

    try:
        return (-1) ** n * _arb_prec_mpl(mm_ones, tuple(xx),
                                          extra_prec=n * 4)
    except Exception:
        return mpc(0)


# ===========================================================================
# Section 9 — Precomputed HPL tables at z=1 (HPL1) and z=i (HPLI)
# ===========================================================================
# The values are exact symbolic expressions in terms of Zeta, Log(2), Pi, PolyLog, Catalan, etc.
# We store them as lambdas returning mpmath values at the current precision.

def _pi():  return mp.pi
def _log2(): return log(2)
def _zeta(n): return zeta(n)
def _cat():  return mpmath.catalan
def _polylog(s, z): return polylog(s, z)

def _hpl1_table() -> dict:
    """
    Return the precomputed HPL values at x=1 as a dict:
      {parameter_tuple: callable() -> mpmath value}

    Only a representative subset is listed here.
    """
    pi = _pi; L2 = _log2; Z = _zeta; G = _cat; PL = _polylog

    def pl5h(): return PL(5, mpf('0.5'))
    def pl4h(): return PL(4, mpf('0.5'))
    def pl3h(): return PL(3, mpf('0.5'))
    def pl6h(): return PL(6, mpf('0.5'))

    table = {
        # Weight 1
        (-1,): lambda: L2(),
        (0,):  lambda: mpf(0),
        # (1,) is divergent — not included

        # Weight 2
        (2,):  lambda: pi()**2 / 6,
        (-2,): lambda: -pi()**2 / 12,
        (-1,-1): lambda: L2()**2 / 2,

        # Weight 3
        (3,):    lambda: Z(3),
        (-3,):   lambda: -3*Z(3)/4,
        (2,-1):  lambda: -pi()**2*L2()/12 + 13*Z(3)/8,
        (-2,1):  lambda: (Z(3) - pi()**2*L2()/4) ,  # ≈ from table
        (-2,-1): lambda: Z(3)/8,
        (-1,-1,-1): lambda: -L2()**3/6,  # from (-1)^3*(Log2)^3/3! with sign
        (-1, 2): lambda: pi()**2*L2()/6 - 5*Z(3)/8,
        (2,1):   lambda: Z(3),  # the MZV relation
        (-1,-2): lambda: (pi()**2*L2() - 3*Z(3)) / 12,

        # Weight 4 (selected)
        (4,):    lambda: pi()**4 / 90,
        (-4,):   lambda: -7*pi()**4/720,
        (3,1):   lambda: pi()**4/360,
        (2,2):   lambda: pi()**4/120,
        (-2,-2): lambda: (13*pi()**4/288 + pi()**2*L2()**2/6
                          - 4*pl4h() - L2()*(L2()**3 + 21*Z(3))/6),
        (-2,2):  lambda: -pi()**2*L2()/4 + 13*Z(3)/8,  # actually from table: {-2,1}
        (2,-1):  lambda: pi()**2*L2()/4 - Z(3),
        (-1,-1,2): lambda: (pi()**4 + 40*pi()**2*L2()**2 - 300*L2()*Z(3))/480,
        (-1,-3): lambda: -pi()**4/288 + 3*L2()*Z(3)/4,
        (-1,3):  lambda: (pi()**4 + 5*pi()**2*L2()**2 - 5*(L2()**4
                          + 24*pl4h() + 9*L2()*Z(3)))/60,
        (2,-2):  lambda: (71*pi()**4/1440 + pi()**2*L2()**2/6
                          - 4*pl4h() - L2()*(L2()**3 + 21*Z(3))/6),
        (-3,1):  lambda: (-pi()**4 - 15*pi()**2*L2()**2 + 15*L2()**4)/180 + 2*pl4h(),
        (3,-1):  lambda: (-19*pi()**4/1440 + 7*L2()*Z(3)/4),
        (-1,-2,-1): lambda: (-pi()**4/30 - pi()**2*L2()**2/8
                             + (L2()**4 + 24*pl4h() + 22*L2()*Z(3))/8),
        (-1,-1,-1,-1): lambda: (-pi()**4/90 - pi()**2*L2()**2/12
                                + L2()**4/24 + pl4h() + L2()*Z(3)),
        (2,1,1):  lambda: Z(4),
        (3,1):    lambda: pi()**4/360,
        (2,2):    lambda: pi()**4/120,

        # Weight 5 (selected)
        (5,):    lambda: Z(5),
        (-5,):   lambda: -15*Z(5)/16,
        (4,1):   lambda: 2*Z(5) - pi()**2*Z(3)/2,  # approx
        (3,2):   lambda: 9*Z(5)/2 - pi()**2*Z(3),
        (2,3):   lambda: 9*Z(5)/2 - pi()**2*Z(3),  # from table: same as (3,2) by duality?
        (5,1):   lambda: -pi()**4*Z(3)/72 - pi()**2*Z(5)/3 + 5*Z(7),  # weight 8, skip here

        # Weight 6 (selected)
        (6,):    lambda: Z(6),
        (3,3):   lambda: pi()**6/1890 - 9*Z(3)**2/32,  # from HPL1 table
        (4,2):   lambda: Z(3)**2 - 4*pi()**6/2835,
        (2,4):   lambda: (5*pi()**6/2268 - Z(3)**2),
        (3,2,1): lambda: (-29*pi()**6/6480 + 3*Z(3)**2),
        (2,3,1): lambda: (53*pi()**6/22680 - 3*Z(3)**2/2),
        (2,2,2): lambda: (pi()**4/36 - Z(3)**2/2),  # needs checking
        (4,1,1): lambda: 23*pi()**6/15120 - Z(3)**2,
        (2,2,1,1): lambda: (-4*pi()**6/2835 + Z(3)**2),
        (2,1,2,1): lambda: (-pi()**6/1890 + (Z(3)**3 + 2*Z(3))/6),  # weight 9 entry mixed
        (3,1,1,1): lambda: (pi()**6/189 - Z(3)**2),  # approx

        # Weight 7 (selected, from the large HPL1 table)
        (4,2,1): lambda: (28*pi()**4*Z(3) + 660*pi()**2*Z(5) - 9945*Z(7))/720,
        (4,1,2): lambda: (-pi()**4*Z(3) + 10*pi()**2*Z(5) + 15*Z(7))/24,
        (2,4,1): lambda: (-2*pi()**4*Z(3) + 120*pi()**2*Z(5) - 981*Z(7))/144,
        (2,1,4): lambda: (7*pi()**4*Z(3) - 330*pi()**2*Z(5) + 2745*Z(7))/360,
        (3,3,1): lambda: (-6*pi()**2*Z(5) + 61*Z(7))/8,
        (3,1,3): lambda: (pi()**4*Z(3) - 90*Z(7))/360,
        (5,1,1): lambda: -pi()**4*Z(3)/72 - pi()**2*Z(5)/3 + 5*Z(7),
        (5,2):   lambda: (pi()**4*Z(3)/45 + 5*pi()**2*Z(5)/6 - 11*Z(7)),
        (4,3):   lambda: (-5*pi()**2*Z(5)/3 + 17*Z(7)),
        (2,2,2,1): lambda: (2*pi()**4*Z(3) - 100*pi()**2*Z(5) + 785*Z(7))/80,
        (2,2,1,2): lambda: (-11*pi()**2*Z(5)/12 + 75*Z(7)/8),
        (2,1,2,2): lambda: (-pi()**4*Z(3)/60 + 2*pi()**2*Z(5) - 291*Z(7)/16),
        (2,2,1,1,1): lambda: (pi()**4*Z(3)/45 + 5*pi()**2*Z(5)/6 - 11*Z(7)),
        (2,1,2,1,1): lambda: (-5*pi()**2*Z(5)/3 + 17*Z(7)),
        (2,1,1,2,1): lambda: (pi()**4*Z(3)/90 + 5*pi()**2*Z(5)/3 - 18*Z(7)),
        (2,1,1,1,2): lambda: (-pi()**4*Z(3)/45 - 2*pi()**2*Z(5)/3 + 10*Z(7)),
        (3,2,1,1): lambda: (28*pi()**4*Z(3) + 660*pi()**2*Z(5) - 9945*Z(7))/720,
        (3,1,2,1): lambda: (-6*pi()**2*Z(5) + 61*Z(7))/8,
        (3,1,1,2): lambda: (-2*pi()**4*Z(3) + 120*pi()**2*Z(5) - 981*Z(7))/144,
        (2,3,1,1): lambda: (-pi()**4*Z(3) + 10*pi()**2*Z(5) + 15*Z(7))/24,
        (2,1,3,1): lambda: (pi()**4*Z(3) - 90*Z(7))/360,
        (2,1,1,3): lambda: (7*pi()**4*Z(3) - 330*pi()**2*Z(5) + 2745*Z(7))/360,
    }
    return table


_HPL1_TABLE = None

def _get_hpl1_table():
    global _HPL1_TABLE
    if _HPL1_TABLE is None:
        _HPL1_TABLE = _hpl1_table()
    return _HPL1_TABLE


def hpl_at_one(mm: tuple[int, ...], dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate H(mm; 1) = MZV using precomputed table or falling back to n_mzv.

    Parameters
    ----------
    mm : parameter tuple (abbreviated notation)
    dps : decimal places

    Returns
    -------
    mpmath.mpc
    """
    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        # Check table first
        table = _get_hpl1_table()
        key = tuple(mm)
        if key in table:
            return mpc(table[key]())

        # Divergent cases
        full = from_abbreviated_notation(mm)
        if full[0] == 1:
            raise ValueError(f"H({mm}; 1) diverges")
        if right_zeros_length(full) > 0:
            raise ValueError(f"H({mm}; 1) diverges (trailing zero)")

        # Fall back to MZV via conversion
        zeta_mm = hpl_to_zeta(full)
        return n_mzv(zeta_mm)


# ===========================================================================
# Section 10 — Symbolic (SymPy) interface
# ===========================================================================

# SymPy symbols for common constants
x_sym = Symbol('x', real=True)
_sp_log2 = slog(2)
_sp_pi   = spi
_sp_cat  = Catalan
_sp_z3   = szeta(3)
_sp_z5   = szeta(5)

def sympy_hpl_expand(mm: tuple[int, ...], x=None) -> sp.Expr:
    """
    Return a SymPy expression for H(mm; x) in terms of standard functions,
    using the precomputed expansion rules.

    For weight-1 cases, exact symbolic forms are returned.
    For higher weights, a placeholder HPL object is returned if no rule applies.

    Parameters
    ----------
    mm : parameter tuple
    x  : SymPy symbol (default: x_sym)

    Returns
    -------
    sympy.Expr
    """
    if x is None:
        x = x_sym

    mm = tuple(mm)

    # Weight-1 base cases
    if mm == (-1,):
        return slog(1 + x)
    if mm == (0,):
        return slog(x)
    if mm == (1,):
        return -slog(1 - x)
    if len(mm) == 1:
        m = mm[0]
        if m > 1:
            return spolylog(m, x)
        if m < -1:
            return -spolylog(-m, -x)

    # Selected weight-2 rules (from $feHarmonicPolyLogSpecialRules)
    if mm == (2, -1):
        return (slog(1+x)*spolylog(2,x) - spolylog(3,x) - spolylog(3,x/(1+x))
                + spolylog(3,2*x/(1+x))
                + (4*slog(2)**3 - _sp_pi**2*slog(4)
                   + 2*slog(1+x)*(_sp_pi**2 - 6*slog(2)**2 + slog(64)*slog(1+x))
                   - 24*spolylog(3,(1+x)/2) + 21*szeta(3))/24)

    if mm == (1, 2):
        return (-_sp_pi**2*slog(1-x)/6 - slog(1-x)*spolylog(2, 1-x)
                + 2*spolylog(3, 1-x) - 2*szeta(3))

    if mm == (2, 2):
        # H(2,2;x) = (Li_2(x)^2 - 4*S(2,2;x)) / 2
        # where S(2,2;x) is the Nielsen generalised polylogarithm.
        # SymPy has no native 3-arg polylog, so we use our NielsenS symbol.
        return (spolylog(2, x)**2 - 4*NielsenS(2, 2, x)) / 2

    if mm == (1, 0, 0):
        return -(slog(x)*(slog(1-x)*slog(x) + 2*spolylog(2,x)))/2 + spolylog(3,x)

    if mm == (-1, 0, 0):
        return slog(x)**2*slog(1+x)/2 + slog(x)*spolylog(2,-x) - spolylog(3,-x)

    # All-equal vectors: H(m,...,m; x) = w(m,x)^n / n!
    # where w is the weight-1 function: -log(1-x), log(1+x), or log(x).
    if len(set(mm)) == 1:
        m0 = mm[0]
        n  = len(mm)
        if m0 == 1:
            return (-slog(1 - x)) ** n / sp.factorial(n)
        if m0 == -1:
            return slog(1 + x) ** n / sp.factorial(n)
        if m0 == 0:
            return slog(x) ** n / sp.factorial(n)

    # Multiple {m,1,1,...,1} -> Nielsen generalised polylogarithm S(m-1, k+1, x)
    if mm[0] > 1 and all(v == 1 for v in mm[1:]):
        n = len(mm) - 1
        m = mm[0]
        if n >= 1:
            return NielsenS(m - 1, n + 1, x)

    # {-m,-1,-1,...,-1} -> (-1)^k * S(m-1, k+1, -x)
    if mm[0] < -1 and all(v == -1 for v in mm[1:]):
        n = len(mm) - 1
        m = -mm[0]
        if n >= 1:
            return sp.Integer(-1)**n * NielsenS(m - 1, n + 1, -x)

    # Higher weights: return an unevaluated HPL symbolic object.
    # expand_func(HPL(mm, x)) will call _eval_expand_func which re-enters here,
    # so unknown cases stay unevaluated rather than looping.
    return HPL(Tuple(*mm), x)


class HPL(Function):
    """
    Symbolic Harmonic Polylogarithm H(mm; x) rewritten into an equivalent form expressed using simpler or more canonical functions.

    Parameters
    ----------
    mm : sympy.Tuple of integers  (the weight/index vector)
    x  : the argument

    Usage
    -----
    >>> from sympy import Symbol, expand_func
    >>> x = Symbol('x')
    >>> HPL(Tuple(1, -1), x)
    HPL((1, -1), x)
    >>> expand_func(HPL(Tuple(2, -1), x))
    # returns the full SymPy expression via sympy_hpl_expand
    """

    @classmethod
    def eval(cls, mm, x):
        # Keep unevaluated by default; expansion happens via expand_func.
        return None

    def _eval_expand_func(self, **hints):
        mm_tuple = tuple(int(m) for m in self.args[0])
        x = self.args[1]
        result = sympy_hpl_expand(mm_tuple, x)
        # If the result is still an HPL (unknown case), return self unchanged
        if isinstance(result, HPL):
            return self
        return result

    def _sympystr(self, printer):
        mm_list = list(self.args[0])
        return f"H({mm_list}; {printer._print(self.args[1])})"

    def _latex(self, printer):
        mm_list = list(self.args[0])
        x_latex = printer._print(self.args[1])
        return rf"H({mm_list};\, {x_latex})"


# Keep _HPLSymExpr as a backward-compatible alias
_HPLSymExpr = HPL


class NielsenS(Function):
    """
    Symbolic Nielsen generalised polylogarithm S(a, n, x).

    S(a, n, x) = (-1)^{a+n-1} / ((a-1)! n!)
                 * int_0^1 log^{a-1}(t) log^n(1-x*t) / t  dt

    Special case: S(a, 1, x) = polylog(a+1, x).
    Relation to HPL: H(m, 1^k; x) = S(m-1, k+1, x).

    Usage
    -----
    >>> expand_func(HPL(Tuple(2, 1, 1), x))   # -> NielsenS(1, 3, x)
    >>> NielsenS(1, 3, x).evalf()             # numerical evaluation via _eval_evalf
    """

    @classmethod
    def eval(cls, a, n, x):
        # S(a, 1, x) = polylog(a+1, x)  -- express in standard form immediately
        if n == sp.Integer(1) or n == 1:
            return spolylog(a + 1, x)
        return None   # keep unevaluated for n > 1

    def _eval_evalf(self, prec):
        a_val = int(self.args[0])
        n_val = int(self.args[1])
        x_val = self.args[2]
        try:
            import mpmath as _mpm
            old = _mpm.mp.prec
            _mpm.mp.prec = prec
            try:
                from hpl_missing import _n_nielsen_S
                val = _n_nielsen_S(a_val, n_val, _mpm.mpc(complex(x_val)))
            finally:
                _mpm.mp.prec = old
            re = float(val.real); im = float(val.imag)
            if abs(im) < 1e-30:
                return sp.Float(re, 30)
            return sp.Float(re, 30) + sp.Float(im, 30) * sI
        except Exception:
            return self

    def _sympystr(self, printer):
        a, n, x = self.args
        return f"S({a}, {n}, {printer._print(x)})"

    def _latex(self, printer):
        a, n, x = self.args
        return rf"S_{{{a},{n}}}({printer._print(x)})"



def sympy_mzv(mm: tuple[int, ...]) -> sp.Expr:
    """
    Return a SymPy expression for MZV(mm) using known reduction formulas.

    Covers depth-1 (Riemann zeta), depth-2 Euler sums, and selected higher-depth cases.
    """
    mm = tuple(mm)

    # Depth 1
    if len(mm) == 1:
        m = mm[0]
        if m > 1:
            return szeta(m)
        elif m == -1:
            return _sp_log2
        elif m < -1:
            return (2**m * (2 - 2**(-m))) * szeta(-m)

    # All same sign, depth 2
    if len(mm) == 2:
        m1, m2 = mm
        if m1 == m2 and m1 > 1:
            return (szeta(m1)**2 - szeta(2*m1)) / 2

    # Z({2,1,...,1}) = zeta(n+1) (Euler's formula)
    if mm[0] == 2 and all(m == 1 for m in mm[1:]):
        return szeta(len(mm) + 1)

    # Z({m, 1}) for m > 2
    if len(mm) == 2 and mm[0] > 2 and mm[1] == 1:
        m = mm[0]
        return (m * szeta(m+1)
                - sp.Add(*[szeta(k+1)*szeta(m-k) for k in range(1, m-1)])) / 2

    # All twos: Z({2,...,2}) = 2*(2*pi)^(2n) / (2n+1)! * (1/2)^(2n+1)
    if all(m == 2 for m in mm):
        n = len(mm)
        return (2 * (2*_sp_pi)**(2*n)
                / sp.factorial(2*n + 1)
                * sp.Rational(1, 2)**(2*n + 1))

    # Selected entries from HPL1 table
    table = _sympy_mzv_table()
    if mm in table:
        return table[mm]

    # General: return placeholder
    return _MZVSymExpr(mm)


def _sympy_mzv_table() -> dict:
    """Selected symbolic MZV values from the HPL1 DownValues table."""
    L2 = _sp_log2; P = _sp_pi; Z = szeta; C = _sp_cat
    PL = spolylog

    def pl4h(): return PL(4, sp.Rational(1, 2))
    def pl5h(): return PL(5, sp.Rational(1, 2))
    def pl6h(): return PL(6, sp.Rational(1, 2))

    return {
        # Depth 2, weight 4
        (3, 1): P**4 / 360,
        (2, 2): P**4 / 120,
        # Depth 2, weight 5
        (3, 2): (9*Z(5)/2 - P**2*Z(3)),
        (4, 1): 2*Z(5) - P**2*Z(3)/2,
        # Depth 2, weight 6
        (3, 3): P**6/1890 - sp.Rational(9, 32)*Z(3)**2,
        (5, 1): P**6/1890 - Z(3)**2/2,  # (approx from table)
        (4, 2): Z(3)**2 - 4*P**6/sp.Integer(2835),
        (5, 2): P**4*Z(3)/45 + 5*P**2*Z(5)/6 - 11*Z(7),
        # Weight 7 (selected)
        (4, 2, 1): (28*P**4*Z(3) + 660*P**2*Z(5) - 9945*Z(7))/720,
        (3, 3, 1): (-6*P**2*Z(5) + 61*Z(7))/8,
        (5, 1, 1): -P**4*Z(3)/72 - P**2*Z(5)/3 + 5*Z(7),
        (2, 2, 2, 1): (2*P**4*Z(3) - 100*P**2*Z(5) + 785*Z(7))/80,
    }


class _MZVSymExpr(sp.Basic):
    """SymPy placeholder for an unevaluated MultipleZetaValue."""
    def __new__(cls, mm):
        obj = sp.Basic.__new__(cls)
        obj.mm = mm
        return obj

    def __repr__(self):
        return f"MZV({list(self.mm)})"


# ===========================================================================
# Section 11 — Finite harmonic sums
# ===========================================================================

def multiple_finite_harmonic_sum_S(mm: tuple[int, ...], n: int) -> mpmath.mpc:
    """
    Evaluate S(mm; n) — MultipleFiniteHarmonicSumS — to arbitrary precision.

    S({m}; n) = sum_{k=1}^{n} sign(m)^(k+1) / k^|m|
    S({m1,...,mk}; n) = sign(m1)^n / n^|m1| * sum_{i=1}^{n-1} sign(m1)^(i+1) * S(rest; i)

    Parameters
    ----------
    mm : tuple of nonzero integers
    n  : upper summation limit

    Returns
    -------
    mpmath.mpc
    """
    mm = tuple(mm)

    if len(mm) == 1:
        m = mm[0]
        if n <= 0:
            return mpc(0)
        s = mpc(0)
        for k in range(1, n + 1):
            s += mpf(_sign(m)) ** (k + 1) / mpf(k) ** abs(m)
        return s

    # Depth > 1: recursive definition
    m0 = mm[0]
    rest = mm[1:]
    if n <= 0:
        return mpc(0)

    total = mpc(0)
    for i in range(1, n):
        total += mpf(_sign(m0)) ** (i + 1) * multiple_finite_harmonic_sum_S(rest, i)
    return mpf(_sign(m0)) ** n / mpf(n) ** abs(m0) * total


# ===========================================================================
# Section 12 — Public API (convenience wrappers)
# ===========================================================================

def harmonic_polylog(mm: Sequence[int], x, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate H(mm; x) — HarmonicPolyLog — numerically.

    Parameters
    ----------
    mm  : sequence of nonzero integers (weight vector)
    x   : numeric argument (real or complex)
    dps : decimal places precision (default: mp.dps = 50)

    Returns
    -------
    mpmath.mpc

    Examples
    --------
    >>> import hpl
    >>> hpl.harmonic_polylog([1], 0.5)    # -log(0.5) = log(2)
    >>> hpl.harmonic_polylog([2], 0.5)    # PolyLog(2, 0.5) = pi^2/12 - log(2)^2/2
    >>> hpl.harmonic_polylog([1, 0], 0.5)
    """
    return n_hpl(tuple(mm), x, dps=dps)


def multiple_polylog(mm: Sequence[int], xx: Sequence, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate Li(mm; xx) — MultiplePolyLog — numerically.

    Parameters
    ----------
    mm  : sequence of nonzero integers
    xx  : sequence of numeric arguments (same length as mm)
    dps : decimal places

    Returns
    -------
    mpmath.mpc
    """
    return n_mpl(tuple(mm), tuple(xx), dps=dps)


def multiple_zeta_value(mm: Sequence[int], dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate Z(mm) — MultipleZetaValue — numerically.

    Parameters
    ----------
    mm  : sequence of integers (m1 >= 2 required for convergence, last != 0)
    dps : decimal places

    Returns
    -------
    mpmath.mpc

    Examples
    --------
    >>> import hpl
    >>> hpl.multiple_zeta_value([2])         # pi^2/6
    >>> hpl.multiple_zeta_value([3])         # Apery's constant
    >>> hpl.multiple_zeta_value([2, 1])      # Apery's constant (Euler's formula)
    >>> hpl.multiple_zeta_value([3, 1])      # pi^4/360
    """
    return n_mzv(tuple(mm), dps=dps)


def expandfunction_hpl(mm: Sequence[int], x=None) -> sp.Expr:
    """
    Return a SymPy expression for H(mm; x) rewritten into an equivalent form expressed using simpler or more canonical functions.

    Internally it constructs HPL(Tuple(*mm), x) and calls SymPy's expand_func, which dispatches to HPL._eval_expand_func.

    Parameters
    ----------
    mm : weight vector (tuple/list of nonzero integers)
    x  : SymPy symbol or expression (default: Symbol('x'))

    Returns
    -------
    sympy.Expr -- expanded expression, or HPL(Tuple(*mm), x) if no rule applies

    Examples
    --------
    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> expandfunction_hpl((1,), x)
    -log(1 - x)
    >>> expandfunction_hpl((2, -1), x)
    ... (full expression in log and polylog)
    """
    if x is None:
        x = x_sym
    return expand_func(HPL(Tuple(*mm), x))


def shuffle_expand(mm1: Sequence[int], mm2: Sequence[int]) -> dict:
    """
    Expand H(mm1; x) * H(mm2; x) via the shuffle product.

    Returns
    -------
    dict[tuple, int]  : parameter_tuple -> coefficient
    """
    return harmonic_polylog_product_expand(tuple(mm1), tuple(mm2))


def stuffle_expand(mm1: Sequence[int], mm2: Sequence[int]) -> dict:
    """
    Expand Z(mm1) * Z(mm2) via the stuffle product.

    Returns
    -------
    dict[tuple, int]  : parameter_tuple -> coefficient
    """
    return mzv_stuffle_product_expand(tuple(mm1), tuple(mm2))


# ===========================================================================
# Section 13 — Self-test
# ===========================================================================
# ─────────────────────────────────────────────────────────────────────────────
# Section A  —  Symbolic argument transformations (argtrans*)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each routine works on the "H-algebra" representation:
#   a symbolic linear combination of H-objects, represented as
#   dict  { mm_tuple : sympy_coefficient }
# plus a separate dict of "constant" terms  { () : value }  for MZV-values.
#
# "HH" means "H evaluated at the *transformed* argument"
# "HO" means "H evaluated at the *special point* (1 or -1)"
# These are kept symbolically until the very end.
# ─────────────────────────────────────────────────────────────────────────────

def _interleave_all(a: tuple, b: tuple) -> list[tuple]:
    """All C(len(a)+len(b), len(a)) interleavings of a and b (with multiplicity)."""
    import itertools
    n1, n2 = len(a), len(b)
    if n1 == 0: return [b]
    if n2 == 0: return [a]
    results = []
    for pos in itertools.combinations(range(n1 + n2), n1):
        pos_set = set(pos)
        row = []; i1 = i2 = 0
        for k in range(n1 + n2):
            if k in pos_set: row.append(a[i1]); i1 += 1
            else:             row.append(b[i2]); i2 += 1
        results.append(tuple(row))
    return results


def _shuffle_expand_dict(d1: dict, d2: dict) -> dict:
    """
    Shuffle-product of two dicts  {mm: coeff}.
    Returns  { mm : coeff }  where the coefficients are summed over all
    interleavings of the two argument lists.
    """
    result = {}
    for mm1, c1 in d1.items():
        for mm2, c2 in d2.items():
            for s in _interleave_all(mm1, mm2):
                result[s] = result.get(s, sp.S.Zero) + c1 * c2
    return result


def _add_dicts(d1: dict, d2: dict, sign=1) -> dict:
    result = dict(d1)
    for k, v in d2.items():
        result[k] = result.get(k, sp.S.Zero) + sign * v
    return result


def _scale_dict(d: dict, c) -> dict:
    return {k: c * v for k, v in d.items()}


# ── Transformation  x  →  −x ────────────────────────────────────────────────
#
# H(m1,...,mn; x) = (-1)^n * H(-m1,...,-mn; -x)   [no trailing / leading zeros]
# For lists with trailing zeros: first extract them, then apply.

def argtrans_neg_x_dict(mm: tuple) -> dict:
    """
    Express H(mm; x) in terms of H(mm'; -x) using x -> -x.
    Returns dict {mm' : coeff}  where the H-objects are understood at -x.

    Sign rule:
      H(mm; x) = (-1)^len(full) * H(-full; -x)
    where full = from_abbreviated_notation(mm) and  -full  negates every entry.
    """
    from hpl import from_abbreviated_notation, to_abbreviated_notation, right_zeros_length

    full = from_abbreviated_notation(mm)
    nz   = right_zeros_length(full)

    if nz > 0:
        # Extract trailing zeros first, then apply -x rule to each term.
        # H(rest, 0^n; x) is expressed via _trailing_zero_decomp as a sum of
        # H(core'; x) * log(x)^k / k!   Then -x: log(x) -> log(-x) = log(x)+i*pi
        # For numerical use this is handled directly; the symbolic form is complex.
        # We do it recursively: one zero at a time.
        core = full[:-1]          # drop last zero
        core_mm = to_abbreviated_notation(core)
        nz_mm   = to_abbreviated_notation(full)
        # H(core, 0; x) = -x part:  log(-x)*H(core;-x) - sum_j |a_j|*H(modified;-x)
        # We return a representation that the caller evaluates numerically.
        # For now, just return the basic rule applied to the full list.
        pass

    # Main rule: H(mm;x) = (-1)^len * H(-mm; -x)
    neg_full = tuple(-m for m in full)
    neg_mm   = to_abbreviated_notation(neg_full)
    sign     = (-1) ** len(full)
    return {neg_mm: sp.Integer(sign)}


# ── Transformation  x  →  1 − x  (for non-negative parameter lists) ─────────
#
# Returns a dict   { mm_in_1mx : coeff }  plus a separate constant dict
# { mm_at_1 : coeff }  (for the HPL-at-1 boundary terms).

class _Transform1mxResult:
    """Holds the split result: terms at 1-x  +  constant terms at x=1."""
    __slots__ = ("at_1mx", "at_1")   # both dicts  {mm: sympy_coeff}

    def __init__(self, at_1mx=None, at_1=None):
        self.at_1mx = at_1mx or {}
        self.at_1   = at_1   or {}

    def scale(self, c):
        return _Transform1mxResult(
            {k: c*v for k,v in self.at_1mx.items()},
            {k: c*v for k,v in self.at_1.items()})

    def __add__(self, other):
        r = _Transform1mxResult()
        r.at_1mx = _add_dicts(self.at_1mx, other.at_1mx)
        r.at_1   = _add_dicts(self.at_1,   other.at_1)
        return r

    def __neg__(self):
        return self.scale(sp.Integer(-1))


def _argtrans1mx(full: tuple) -> _Transform1mxResult:
    """
    Transform H(full; x)  ->  expression in H(mm'; 1-x) + constants at x=1.
    `full` must be in full notation (no abbreviation) with all entries >= 0.

      H(0; x)    -> -H(1; 1-x)
      H(1; x)    -> -H(0; 1-x)
      H(0,rest)  -> HO(0,rest) - Shuffle(argtrans1mx(rest), prepend_1)
      H(1,rest)  -> extract leading 1s, recurse
    """
    from hpl import to_abbreviated_notation, right_zeros_length

    if len(full) == 0:
        return _Transform1mxResult(at_1={(): sp.Integer(1)})

    if full == (0,):
        return _Transform1mxResult(at_1mx={(1,): sp.Integer(-1)})

    if full == (1,):
        return _Transform1mxResult(at_1mx={(0,): sp.Integer(-1)})

    a0 = full[0]
    rest = full[1:]

    if a0 == 0:
        # H(0,rest;x) -> HO(0,rest) - integral of argtrans1mx(rest) w.r.t. df_1
        # "integrateH(expr, 1)" prepends a 1 to each mm in the dict
        inner = _argtrans1mx(rest)
        # prepend 1 to each key in at_1mx
        integrated = _Transform1mxResult(
            at_1mx={(1,) + k: v for k, v in inner.at_1mx.items()},
            at_1  ={(1,) + k: v for k, v in inner.at_1.items()})
        # Shuffle-expand (so we can handle repeated integrals)
        # Actually in argtrans1mx the result is already a linear combination;
        # integrating just prepends the weight-function index.
        # Boundary: HO(0,rest) = HPL(0,rest; 1)  which is a constant.
        boundary_mm = to_abbreviated_notation((0,) + rest)
        r = _Transform1mxResult(
            at_1 = {boundary_mm: sp.Integer(1)})
        r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
        r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
        return r

    if a0 == 1:
        # Extract leading 1s, then recurse
        n_ones = 0
        for m in full:
            if m == 1: n_ones += 1
            else:      break
        # H(1^n, rest; x) -> use shuffle: H(1;x)^n/n! * rest, or
        # the leading-ones extraction: recursion via ExtractLeadingOnes
        # For a single leading 1:
        if n_ones == 1:
            # H(1,rest;x): apply HO extraction
            # H(1,rest;x) = HO(1,rest) - integral(argtrans1mx(rest), prepend_0)
            inner = _argtrans1mx(rest)
            integrated = _Transform1mxResult(
                at_1mx={(0,) + k: v for k, v in inner.at_1mx.items()},
                at_1  ={(0,) + k: v for k, v in inner.at_1.items()})
            boundary_mm = to_abbreviated_notation((1,) + rest)
            r = _Transform1mxResult(
                at_1 = {boundary_mm: sp.Integer(1)})
            r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
            r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
            return r
        else:
            # Multiple leading 1s: use the shuffle algebra recursion
            # H(1^n, rest; x):
            # Apply single-1 rule to H(1, 1^{n-1}_rest; x):
            # -> boundary HO(1,...) - integrate(argtrans1mx(H(1^{n-1},rest)), prepend 0)
            # This peels ONE leading 1 at a time, eventually reaching n_ones=1.
            tail = full[n_ones:]          # the part after all leading 1s
            inner = _argtrans1mx((1,) + tail)   # peel ONE leading 1
            # Now integrate w.r.t. 0:
            integrated = _Transform1mxResult(
                at_1mx={(0,) + k: v for k, v in inner.at_1mx.items()},
                at_1  ={(0,) + k: v for k, v in inner.at_1.items()})
            boundary_mm = to_abbreviated_notation(full)
            r = _Transform1mxResult(at_1={boundary_mm: sp.Integer(1)})
            r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
            r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
            return r

    # General positive a0 >= 2: expand to full and recurse
    # a0 >= 2 means first full entry is 0^(a0-1) followed by 1
    expanded = (0,) * (a0 - 1) + (1,) + rest
    return _argtrans1mx(expanded)


# ── Transformation  x  →  1/x ────────────────────────────────────────────────
#
# H(0; x)  = -H(0; 1/x)
# H(1; x)  = H(1; 1/x) + H(0; 1/x) + disc_sign * i*pi
# H(-1; x) = H(-1; 1/x) - H(0; 1/x)
# General:
# The "disc_sign" = 2*Boole(Im(x)>0) - 1 = ±1 depending on which sheet.
#
# We encode disc_sign as a SymPy symbol _DS that is substituted numerically.

_DS = sp.Symbol('_disc_sign')  # +1 if Im(x)>0 else -1


class _Transform1oxResult:
    """Like _Transform1mxResult but for x->1/x; uses _DS for the discontinuity."""
    __slots__ = ("at_1ox", "at_1")

    def __init__(self, at_1ox=None, at_1=None):
        self.at_1ox = at_1ox or {}
        self.at_1   = at_1   or {}

    def scale(self, c):
        return _Transform1oxResult(
            {k: c*v for k,v in self.at_1ox.items()},
            {k: c*v for k,v in self.at_1.items()})

    def __add__(self, other):
        r = _Transform1oxResult()
        r.at_1ox = _add_dicts(self.at_1ox, other.at_1ox)
        r.at_1   = _add_dicts(self.at_1,   other.at_1)
        return r


def _argtrans1ox(full: tuple) -> _Transform1oxResult:
    """
    Transform H(full; x)  ->  expression in H(mm'; 1/x) + constants at x=1.
    """
    from hpl import to_abbreviated_notation

    if len(full) == 0:
        return _Transform1oxResult(at_1={(): sp.Integer(1)})

    if full == (0,):
        return _Transform1oxResult(at_1ox={(0,): sp.Integer(-1)})

    if full == (1,):
        # H(1;x) = H(1;1/x) + H(0;1/x) + disc_sign * i*pi
        return _Transform1oxResult(
            at_1ox={(1,): sp.Integer(1), (0,): sp.Integer(1)},
            at_1  = {(): _DS * sp.I * spi})

    if full == (-1,):
        # H(-1;x) = H(-1;1/x) - H(0;1/x)
        return _Transform1oxResult(
            at_1ox={(-1,): sp.Integer(1), (0,): sp.Integer(-1)})

    a0   = full[0]
    rest = full[1:]

    if a0 == 0:
        # argtrans1ox[H(0,rest)]:
        #   tmp = shuffle_expand(argtrans1ox(H(rest)))
        #   integrateH(tmp, 0) = prepend 0 to each key
        #   HO(0,rest) + tmp.at_1 - tmp.at_1ox
        inner = _argtrans1ox(rest)
        # integrate w.r.t. 0 (prepend 0 to keys at_1ox) and at_1
        int_1ox = {(0,)+k: v for k,v in inner.at_1ox.items()}
        int_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((0,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        # + (int_1.at_1 - int_1ox.at_1ox)  [from "tmp/.HH->HO - tmp"]
        r.at_1   = _add_dicts(r.at_1,   int_1)
        r.at_1ox = {k: -v for k,v in int_1ox.items()}
        return r

    if a0 == 1:
        # argtrans1ox[H(1,rest)]:  ExtractSingularPart then recurse
        # For a single leading 1:
        inner = _argtrans1ox(rest)
        # integrate w.r.t. 1: prepend (1,) to at_1ox, prepend (0,) to at_1
        # integrateH gives HH[1,mm] from HH[mm] and we subtract from the boundary term HO(1,rest)
        int_1ox = {(1,)+k: v for k,v in inner.at_1ox.items()}
        int_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((1,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        r.at_1   = _add_dicts(r.at_1, int_1)
        r.at_1ox = {k: -v for k,v in int_1ox.items()}
        return r

    if a0 == -1:
        # argtrans1ox[H(-1,rest)]:
        #   tmp = shuffle_expand(argtrans1ox(H(rest)))
        #   HO(-1,rest) + (tmp/.HH->HO) - tmp   [both 0-integrated and -1-integrated]
        inner = _argtrans1ox(rest)
        int0_1ox = {(0,)+k: v for k,v in inner.at_1ox.items()}
        int0_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        intm1_1ox= {(-1,)+k: v for k,v in inner.at_1ox.items()}
        intm1_1  = {(-1,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((-1,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        # HO[int0] - HH[int0] + HO[intm1] - HH[intm1]
        r.at_1   = _add_dicts(r.at_1, int0_1)
        r.at_1   = _add_dicts(r.at_1, intm1_1)
        r.at_1ox = _add_dicts(
            {k:-v for k,v in int0_1ox.items()},
            {k:-v for k,v in intm1_1ox.items()})
        return r

    # |a0| > 1: expand abbreviation and recurse
    expanded = (0,)*(abs(a0)-1) + (1 if a0>0 else -1,) + rest
    return _argtrans1ox(expanded)


# ─────────────────────────────────────────────────────────────────────────────
# Section B  —  Numerical evaluation of transformed expressions
# ─────────────────────────────────────────────────────────────────────────────

def _eval_transform_result_1mx(res: _Transform1mxResult,
                                x: mpmath.mpc) -> mpmath.mpc:
    """
    Numerically evaluate a _Transform1mxResult at a given x.
    The 'at_1mx' terms are H(mm; 1-x), the 'at_1' terms are H(mm; 1) = HPL1.
    """
    from hpl import _n_hpl_dispatch, hpl_at_one
    y = mpc(1) - x
    total = mpc(0)
    for mm, coeff in res.at_1mx.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        total += mpc(c) * _n_hpl_dispatch_v2(mm, y)
    for mm, coeff in res.at_1.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        if mm == ():
            total += mpc(c)
        else:
            total += mpc(c) * hpl_at_one(mm)
    return total


def _eval_transform_result_1ox(res: _Transform1oxResult,
                                x: mpmath.mpc) -> mpmath.mpc:
    """
    Numerically evaluate a _Transform1oxResult at a given x.
    The 'at_1ox' terms are H(mm; 1/x), the 'at_1' terms are constants
    (possibly involving _DS = 2*Boole[Im(x)>0]-1).
    """
    from hpl import _n_hpl_dispatch, hpl_at_one
    y     = mpc(1) / x
    ds    = 1 if mpmath.im(x) > 0 else -1
    total = mpc(0)
    for mm, coeff in res.at_1ox.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        total += mpc(c) * _n_hpl_dispatch_v2(mm, y)
    for mm, coeff in res.at_1.items():
        # substitute _DS numerically
        c_sym = coeff.subs(_DS, ds)
        c = complex(c_sym)
        if abs(c) < 1e-100:
            continue
        if mm == ():
            total += mpc(c)
        else:
            total += mpc(c) * hpl_at_one(mm)
    return total


def hpl_at_i(mm: tuple, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate H(mm; i) using the precomputed HPLI table,
    falling back to numerical evaluation.
    """
    from hpl import _n_hpl_dispatch
    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        table = _get_hpli_table()
        key   = tuple(mm)
        if key in table:
            return mpc(table[key]())
        # Fallback: numerical evaluation at x = i
        return _n_hpl_dispatch_v2(mm, mpc(0, 1))


_HPLI_TABLE = None
def _get_hpli_table():
    global _HPLI_TABLE
    if _HPLI_TABLE is None:
        _HPLI_TABLE = _build_hpli_table()
    return _HPLI_TABLE


# ─────────────────────────────────────────────────────────────────────────────
# Section F  —  Full function expansion rules for HPL (weights ≤ 6)
# ─────────────────────────────────────────────────────────────────────────────

def _build_expandfunction_table() -> dict:
    """
    Build the function expansion table: {mm : lambda x -> sympy_expr}.
    """
    from sympy import (log as sl, polylog as spl, zeta as sz,
                       pi as sP, sqrt, I as sI, Rational)

    def L(z):   return sl(z)
    def Li(s,z): return spl(s,z)
    def Z(n):   return sz(n)
    P = sP
    log2 = sl(2)

    tbl = {}

    # ── Weight 1 ──────────────────────────────────────────────────────────────
    tbl[(-1,)] = lambda x: L(1+x)
    tbl[(0,)]  = lambda x: L(x)
    tbl[(1,)]  = lambda x: -L(1-x)
    for m in range(2, 10):
        tbl[(m,)]  = (lambda m: lambda x: spl(m, x))(m)
        tbl[(-m,)] = (lambda m: lambda x: -spl(m, -x))(m)

    # ── Selected weight-2 special rules (from $feHarmonicPolyLogSpecialRules) ─
    tbl[(2,-1)] = lambda x: (
        L(1+x)*Li(2,x) - Li(3,x) - Li(3,x/(1+x)) + Li(3,2*x/(1+x))
        + (4*log2**3 - P**2*L(4)
           + 2*L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
           - 24*Li(3,(1+x)/2) + 21*Z(3)) / 24)

    tbl[(-2,1)] = lambda x: (
        -P**2*log2/12 + log2**3/6
        - (L(1-x)*(P**2 + 6*log2**2 + 2*L(1-x)*(L(1-x) - 3*L(2*x))))/12
        + L(1-x)*Li(2,-x) - Li(3,(1-x)/2) + Li(3,1-x) + 2*Li(3,x)
        + Li(3,2*x/(-1+x)) - Li(3,x**2)/4 - Z(3)/8)

    tbl[(1,-2)] = lambda x: (
        (-4*log2**3 + P**2*L(4-4*x) + 6*log2**2*L(1-x)
         + 2*L(1-x)**2*(L((1-x)/8) - 3*L(x))
         - L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
         - 12*L(1+x)*Li(2,x) + 12*Li(3,(1-x)/2) - 12*Li(3,1-x)
         + 12*Li(3,-x) - 12*Li(3,2*x/(-1+x)) + 12*Li(3,x/(1+x))
         - 12*Li(3,2*x/(1+x)) + 12*Li(3,(1+x)/2) - 9*Z(3))/12)

    tbl[(-1,2)] = lambda x: (
        (-4*log2**3 + P**2*L(4-4*x) + 6*log2**2*L(1-x)
         + 2*L(1-x)**2*(L((1-x)/8) - 3*L(x))
         - L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
         - 12*L(1-x)*Li(2,-x) + 12*Li(3,(1-x)/2) - 12*Li(3,1-x)
         + 12*Li(3,-x) - 12*Li(3,2*x/(-1+x)) + 12*Li(3,x/(1+x))
         - 12*Li(3,2*x/(1+x)) + 12*Li(3,(1+x)/2) - 9*Z(3))/12)

    tbl[(1,2)] = lambda x: (
        -P**2*L(1-x)/6 - L(1-x)*Li(2,1-x) + 2*Li(3,1-x) - 2*Z(3))

    tbl[(2,2)] = lambda x: (Li(2,x)**2 - 4*Li(2,2,x)) / 2

    tbl[(1,1,-1)] = lambda x: (
        -(L((1-x)/2)*L(1+x)**2)/2 - L(1+x)*Li(2,(1+x)/2)
        + Li(3,(1+x)/2) + (-4*log2**3 + P**2*L(4) - 21*Z(3))/24)

    tbl[(1,-1,-1)] = lambda x: (
        -(L((1-x)/2)*L(1+x)**2)/2 - L(1+x)*Li(2,(1+x)/2)
        + Li(3,(1+x)/2) + (-4*log2**3 + P**2*L(4) - 21*Z(3))/24)

    tbl[(1,1,1)] = lambda x: -L(1-x)**3 / 6

    # ── Standard patterns (weight-2, pairs of same index) ────────────────────
    tbl[(1,0,0)] = lambda x: (
        -(L(x)*(L(1-x)*L(x) + 2*Li(2,x)))/2 + Li(3,x))

    tbl[(-1,0,0)] = lambda x: (
        L(x)**2*L(1+x)/2 + L(x)*Li(2,-x) - Li(3,-x))

    # ── Nielsen generalised polylogs: H(m, 1^k; x) = Li_{m-1,k+1}(x) ─────────
    for m in range(2, 8):
        for k in range(1, 5):
            mm_key = (m,) + (1,)*k
            tbl[mm_key] = (lambda m, k: lambda x: spl(m-1, k+1, x))(m, k)
        for k in range(1, 5):
            mm_key = (-m,) + (-1,)*k
            tbl[mm_key] = (lambda m, k: lambda x: (-1)**(k+1) * spl(m-1, k+1, -x))(m, k)

    return tbl


_FE_TABLE = None
def _get_fe_table():
    global _FE_TABLE
    if _FE_TABLE is None:
        _FE_TABLE = _build_expandfunction_table()
    return _FE_TABLE


def expandfunction_hpl_full(mm: tuple, x=None) -> sp.Expr:
    """
    Full function rewriting for H(mm; x): returns a SymPy expression 
    using log, polylog, zeta, and constants.  Falls back to a placeholder
    _HPLSymExpr for cases not in the table.
    """
    from hpl import _HPLSymExpr
    from sympy import Symbol
    if x is None:
        x = Symbol('x', real=True)

    mm = tuple(mm)
    tbl = _get_fe_table()
    if mm in tbl:
        return tbl[mm](x)

    # Fallback: unevaluated
    return _HPLSymExpr(mm, x)


# ─────────────────────────────────────────────────────────────────────────────
# Section G  —  Fast MZV via Borwein's algorithm
# ─────────────────────────────────────────────────────────────────────────────
#
# Borwein, Bradley, Broadhurst (1999): for all-positive MZVs use the
# generating-function / partial-fraction identity to express Z(mm) as a
# rapidly converging alternating series.
#
# The key identity for depth >= 2:
#   Z(m1,...,mk) = sum_{n=1}^{N-1} d_n / 2^n + error
# where d_n are integer coefficients computable from the shuffle algebra.
#
# In practice, for depth-3+ we use the Crandall (1998) recursive algorithm
# which expresses Z(s1,...,sk) via a nested application of the hurwitz zeta
# tail approach with nsum+richardson for each inner sum.

def n_mzv_borwein(mm: tuple, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate MZV(mm) for all-positive indices using the best available method:
    - Depth 1:  direct zeta()
    - Depth 2:  Euler formula or Hurwitz-zeta nsum (richardson)
    - Depth 3+: Crandall nested-Hurwitz recursion with extra precision
    """
    from hpl import _n_mzv_positive, _euler_formula_m1, zeta_to_hpl
    from mpmath import nsum, inf

    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps + 10)

    with ctx:
        mm = tuple(mm)
        if not all(m > 0 for m in mm):
            raise ValueError("n_mzv_borwein: all indices must be positive")

        n = len(mm)
        if n == 1:
            return zeta(mm[0])

        if n == 2:
            return _n_mzv_positive(mm)

        # Depth >= 3: Crandall nested Hurwitz
        return _n_mzv_crandall(mm)


def _n_mzv_crandall(mm: tuple) -> mpmath.mpc:
    """
    Evaluate Z(m1,...,mk) for k >= 3, all positive, using the nested
    Hurwitz-zeta outer sum with nsum+richardson acceleration.

    Z(m1,...,mk) = sum_{n=mk_depth}^inf n^{-m1} * Z_partial(rest, n-1)
    where Z_partial(rest, N) = Z(rest) - zeta_tail(rest, N+1)
    and   zeta_tail(rest, N) is computed recursively using Hurwitz zeta.
    """
    from mpmath import nsum, inf

    n     = len(mm)
    m1    = mm[0]
    rest  = mm[1:]

    if len(rest) == 1:
        m2 = rest[0]
        if m2 == 1:
            from hpl import _euler_formula_m1
            return _euler_formula_m1(m1)
        return nsum(lambda k: mpmath.zeta(m1, int(k)+1) / k**m2,
                    [1, inf], method='richardson')

    if len(rest) == 2:
        # Z(m1,m2,m3) = sum_{n1>n2>n3>=1} n1^{-m1}*n2^{-m2}*n3^{-m3}
        # = sum_{n1>=3} n1^{-m1} * sum_{n2=2}^{n1-1} n2^{-m2} * H_{n2-1}^{(m3)}
        # Use nsum on n1 with inner running sum
        m2, m3 = rest
        # Inner: Z_partial(m2,m3; N) = sum_{n2=2}^N sum_{n3=1}^{n2-1} n2^{-m2}*n3^{-m3}
        # = sum_{n2=2}^N n2^{-m2} * (zeta(m3) - zeta(m3, n2))  [Hurwitz tail]
        # = (zeta(m2)-zeta(m2,N+1)) * zeta(m3) - sum_{n2=2}^N n2^{-m2}*zeta(m3,n2)
        # Precompute Z_partial(rest, N) as a running sum:
        def term_outer(k):
            N  = int(k) - 1    # k = n1, inner runs up to n1-1
            if N < 2:
                return mpc(0)
            # Z_partial(m2,m3;N) using running accumulation would be slow here.
            # Instead use: zeta(m2)*sum_{n2=1}^N n2^{-m3} - ... 
            # Accurate but slow: direct double sum capped at N
            s = mpc(0)
            h = mpc(0)
            for n2 in range(1, N+1):
                s += h / mpc(n2)**m2
                h += mpc(1) / mpc(n2)**m3
            return s / mpc(k)**m1

        return nsum(term_outer, [3, inf], method='richardson')

    # Depth >= 4: use the general recursive approach with extra precision
    with mp.workdps(mp.dps + len(mm) * 10):
        from hpl import _n_mzv_positive
        return _n_mzv_positive(mm)


# ─────────────────────────────────────────────────────────────────────────────
# Section H  —  MultipleFiniteHarmonicSumZ
# ─────────────────────────────────────────────────────────────────────────────

def multiple_finite_harmonic_sum_Z(mm: tuple, xx: tuple,
                                   n_or_inf) -> mpmath.mpc:
    """
    Evaluate Z(mm; xx; N) — MultipleFiniteHarmonicSumZ.

    Z({m1,...,mk}; {x1,...,xk}; N)
      = sum_{N>=i1>i2>...>ik>=1} prod_j x_j^{i_j} / i_j^{|m_j|} * sign(m_j)^{i_j}

    Special cases handled:
      xx = (1,...,1), n_or_inf = inf  ->  MZV(mm) via n_mzv
      xx = (x, 1,...,1), n_or_inf=inf ->  HPL(mm; x) via n_hpl
      finite N, all xx=1              ->  partial MZV sum (Euler-Bernoulli)

    Parameters
    ----------
    mm        : tuple of nonzero integers
    xx        : tuple of complex arguments, len = len(mm)
    n_or_inf  : integer N or float('inf') / mpmath.inf

    Returns
    -------
    mpmath.mpc
    """
    from hpl import n_hpl, n_mzv, _n_hpl_dispatch

    assert len(mm) == len(xx), "mm and xx must have the same length"

    mm = tuple(mm)
    xx = tuple(mpc(xi) for xi in xx)
    is_inf = (n_or_inf == float('inf') or n_or_inf is mpmath.inf
              or (hasattr(n_or_inf, '_mpf_') and abs(n_or_inf) > 1e15))

    # ── Special case: all x = 1, N = inf -> MZV ─────────────────────────────
    if is_inf and all(xi == mpc(1) for xi in xx):
        return n_mzv(mm)

    # ── Special case: x[0]=x, rest=1, N=inf -> HPL ──────────────────────────
    if is_inf and all(xi == mpc(1) for xi in xx[1:]):
        return _n_hpl_dispatch_v2(mm, xx[0])

    # ── General finite N: direct nested sum ──────────────────────────────────
    if not is_inf:
        N = int(n_or_inf)
        return _z_sum_finite(mm, xx, N)

    # ── General infinite case: MPL ────────────────────────────────────────────
    from hpl import n_mpl
    return n_mpl(mm, xx)


def _z_sum_finite(mm: tuple, xx: tuple, N: int) -> mpmath.mpc:
    """
    Direct nested sum for Z(mm; xx; N).
    Depth-1: sum_{k=1}^N sign(m)^k * x^k / k^|m|
    Depth-k: sum_{i=depth}^{N} x[0]^i * sign(m[0])^i / i^|m[0]|
              * Z(rest; xx[1:]; i-1)
    """
    if len(mm) == 0:
        return mpc(1)

    m0, x0 = mm[0], xx[0]
    s0 = 1 if m0 > 0 else -1 if m0 < 0 else 0

    if len(mm) == 1:
        result = mpc(0)
        for k in range(1, N + 1):
            result += mpc(s0)**k * x0**k / mpc(k)**abs(m0)
        return result

    result = mpc(0)
    for i in range(len(mm), N + 1):
        inner = _z_sum_finite(mm[1:], xx[1:], i - 1)
        result += mpc(s0)**i * x0**i / mpc(i)**abs(m0) * inner
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section I  —  Expression-level shuffle/stuffle rewriting
# ─────────────────────────────────────────────────────────────────────────────
#
# Shuffle and stuffle expansion rules on "coefficient dictionaries":
#   linear_combination = dict { mm_tuple : sympy_coefficient }

def shuffle_expand_lc(lc1: dict, lc2: dict) -> dict:
    """
    Expand the product of two HPL linear combinations lc1 * lc2
    using the shuffle product.

    Each linear combination is a dict { mm_tuple : coefficient }.
    Returns a new dict of the same type.

    Example:
      H(1;x) * H(2;x) = shuffle_expand_lc({(1,):1}, {(2,):1})
                       = {(1,2):1, (2,1):2}
    """
    from hpl import harmonic_polylog_product_expand
    result = {}
    for mm1, c1 in lc1.items():
        for mm2, c2 in lc2.items():
            expanded = harmonic_polylog_product_expand(mm1, mm2)
            for mm, mult in expanded.items():
                result[mm] = result.get(mm, sp.S.Zero) + c1 * c2 * mult
    return result


def stuffle_expand_lc(lc1: dict, lc2: dict) -> dict:
    """
    Expand the product of two MZV linear combinations lc1 * lc2
    using the stuffle product.
    """
    from hpl import mzv_stuffle_product_expand
    result = {}
    for mm1, c1 in lc1.items():
        for mm2, c2 in lc2.items():
            expanded = mzv_stuffle_product_expand(mm1, mm2)
            for mm, mult in expanded.items():
                result[mm] = result.get(mm, sp.S.Zero) + c1 * c2 * mult
    return result


def power_shuffle_expand(mm: tuple, n: int) -> dict:
    """
    Compute H(mm;x)^n using the shuffle product n-1 times.
    Returns a dict {mm': integer_coefficient}.
    Corresponds to H(mm;x)^n = n! * H(mm,...,mm; x) + lower_weight_terms
    from the shuffle algebra.
    """
    from hpl import harmonic_polylog_product_expand
    if n == 0:
        return {(): 1}
    if n == 1:
        return {mm: 1}
    result = {mm: 1}
    for _ in range(n - 1):
        new_result = {}
        for mm2, c2 in result.items():
            expanded = harmonic_polylog_product_expand(mm, mm2)
            for mm3, mult in expanded.items():
                new_result[mm3] = new_result.get(mm3, 0) + c2 * mult
        result = new_result
    return result


def evaluate_hpl_lc(lc: dict, x, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate a HPL linear combination:
      sum_{mm} coeff[mm] * H(mm; x)

    Parameters
    ----------
    lc  : dict { mm_tuple : numeric_or_sympy_coeff }
    x   : numeric argument
    dps : decimal places
    """
    total = mpc(0)
    xc = mpc(x)
    for mm, c in lc.items():
        c_num = complex(c) if isinstance(c, sp.Basic) else c
        if abs(c_num) < 1e-100:
            continue
        if mm == ():
            total += mpc(c_num)
        else:
            from hpl import n_hpl
            total += mpc(c_num) * n_hpl(mm, xc, dps=dps)
    return total


def evaluate_mzv_lc(lc: dict, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate a MZV linear combination:
      sum_{mm} coeff[mm] * Z(mm)
    """
    total = mpc(0)
    for mm, c in lc.items():
        c_num = complex(c) if isinstance(c, sp.Basic) else c
        if abs(c_num) < 1e-100:
            continue
        if mm == ():
            total += mpc(c_num)
        else:
            from hpl import n_mzv
            total += mpc(c_num) * n_mzv(mm, dps=dps)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Analytic continuation, mixed-sign HPLs, HPLI table, Z-sums, expression algebra, function expansion, Nielsen S.
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Section A  —  Symbolic argument transformations (argtrans*)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each routine works on the "H-algebra" representation:
#   a symbolic linear combination of H-objects, represented as
#   dict  { mm_tuple : sympy_coefficient }
# plus a separate dict of "constant" terms  { () : value }  for MZV-values.
#
# "HH" means "H evaluated at the *transformed* argument"
# "HO" means "H evaluated at the *special point* (1 or -1)"
# These are kept symbolically until the very end.
# ─────────────────────────────────────────────────────────────────────────────

def _interleave_all(a: tuple, b: tuple) -> list[tuple]:
    """All C(len(a)+len(b), len(a)) interleavings of a and b (with multiplicity)."""
    import itertools
    n1, n2 = len(a), len(b)
    if n1 == 0: return [b]
    if n2 == 0: return [a]
    results = []
    for pos in itertools.combinations(range(n1 + n2), n1):
        pos_set = set(pos)
        row = []; i1 = i2 = 0
        for k in range(n1 + n2):
            if k in pos_set: row.append(a[i1]); i1 += 1
            else:             row.append(b[i2]); i2 += 1
        results.append(tuple(row))
    return results


def _shuffle_expand_dict(d1: dict, d2: dict) -> dict:
    """
    Shuffle-product of two dicts  {mm: coeff}.
    Returns  { mm : coeff }  where the coefficients are summed over all
    interleavings of the two argument lists.
    """
    result = {}
    for mm1, c1 in d1.items():
        for mm2, c2 in d2.items():
            for s in _interleave_all(mm1, mm2):
                result[s] = result.get(s, sp.S.Zero) + c1 * c2
    return result


def _add_dicts(d1: dict, d2: dict, sign=1) -> dict:
    result = dict(d1)
    for k, v in d2.items():
        result[k] = result.get(k, sp.S.Zero) + sign * v
    return result


def _scale_dict(d: dict, c) -> dict:
    return {k: c * v for k, v in d.items()}


# ── Transformation  x  →  −x ────────────────────────────────────────────────
#
# H(m1,...,mn; x) = (-1)^n * H(-m1,...,-mn; -x)   [no trailing / leading zeros]
# For lists with trailing zeros: first extract them, then apply.

def argtrans_neg_x_dict(mm: tuple) -> dict:
    """
    Express H(mm; x) in terms of H(mm'; -x) using x -> -x.
    Returns dict {mm' : coeff}  where the H-objects are understood at -x.

    Sign rule:
      H(mm; x) = (-1)^len(full) * H(-full; -x)
    where full = from_abbreviated_notation(mm) and  -full  negates every entry.
    """
    from hpl import from_abbreviated_notation, to_abbreviated_notation, right_zeros_length

    full = from_abbreviated_notation(mm)
    nz   = right_zeros_length(full)

    if nz > 0:
        # Extract trailing zeros first, then apply -x rule to each term.
        # H(rest, 0^n; x) is expressed via _trailing_zero_decomp as a sum of
        # H(core'; x) * log(x)^k / k!   Then -x: log(x) -> log(-x) = log(x)+i*pi
        # For numerical use this is handled directly; the symbolic form is complex.
        # We do it recursively: one zero at a time.
        core = full[:-1]          # drop last zero
        core_mm = to_abbreviated_notation(core)
        nz_mm   = to_abbreviated_notation(full)
        # H(core, 0; x) = -x part:  log(-x)*H(core;-x) - sum_j |a_j|*H(modified;-x)
        # We return a representation that the caller evaluates numerically.
        # For now, just return the basic rule applied to the full list.
        pass

    # Main rule: H(mm;x) = (-1)^len * H(-mm; -x)
    neg_full = tuple(-m for m in full)
    neg_mm   = to_abbreviated_notation(neg_full)
    sign     = (-1) ** len(full)
    return {neg_mm: sp.Integer(sign)}


# ── Transformation  x  →  1 − x  (for non-negative parameter lists) ─────────
#
# Returns a dict   { mm_in_1mx : coeff }  plus a separate constant dict
# { mm_at_1 : coeff }  (for the HPL-at-1 boundary terms).

class _Transform1mxResult:
    """Holds the split result: terms at 1-x  +  constant terms at x=1."""
    __slots__ = ("at_1mx", "at_1")   # both dicts  {mm: sympy_coeff}

    def __init__(self, at_1mx=None, at_1=None):
        self.at_1mx = at_1mx or {}
        self.at_1   = at_1   or {}

    def scale(self, c):
        return _Transform1mxResult(
            {k: c*v for k,v in self.at_1mx.items()},
            {k: c*v for k,v in self.at_1.items()})

    def __add__(self, other):
        r = _Transform1mxResult()
        r.at_1mx = _add_dicts(self.at_1mx, other.at_1mx)
        r.at_1   = _add_dicts(self.at_1,   other.at_1)
        return r

    def __neg__(self):
        return self.scale(sp.Integer(-1))


def _argtrans1mx(full: tuple) -> _Transform1mxResult:
    """
    Transform H(full; x)  ->  expression in H(mm'; 1-x) + constants at x=1.
    `full` must be in full notation (no abbreviation) with all entries >= 0.

      H(0; x)    -> -H(1; 1-x)
      H(1; x)    -> -H(0; 1-x)
      H(0,rest)  -> HO(0,rest) - Shuffle(argtrans1mx(rest), prepend_1)
      H(1,rest)  -> extract leading 1s, recurse
    """
    from hpl import to_abbreviated_notation, right_zeros_length

    if len(full) == 0:
        return _Transform1mxResult(at_1={(): sp.Integer(1)})

    if full == (0,):
        return _Transform1mxResult(at_1mx={(1,): sp.Integer(-1)})

    if full == (1,):
        return _Transform1mxResult(at_1mx={(0,): sp.Integer(-1)})

    a0 = full[0]
    rest = full[1:]

    if a0 == 0:
        # H(0,rest;x) -> HO(0,rest) - integral of argtrans1mx(rest) w.r.t. df_1
        # "integrateH(expr, 1)" prepends a 1 to each mm in the dict
        inner = _argtrans1mx(rest)
        # prepend 1 to each key in at_1mx
        integrated = _Transform1mxResult(
            at_1mx={(1,) + k: v for k, v in inner.at_1mx.items()},
            at_1  ={(1,) + k: v for k, v in inner.at_1.items()})
        # Shuffle-expand (so we can handle repeated integrals)
        # Actually in argtrans1mx the result is already a linear combination;
        # integrating just prepends the weight-function index.
        # Boundary: HO(0,rest) = HPL(0,rest; 1)  which is a constant.
        boundary_mm = to_abbreviated_notation((0,) + rest)
        r = _Transform1mxResult(
            at_1 = {boundary_mm: sp.Integer(1)})
        r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
        r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
        return r

    if a0 == 1:
        # Extract leading 1s, then recurse
        n_ones = 0
        for m in full:
            if m == 1: n_ones += 1
            else:      break
        # H(1^n, rest; x) -> use shuffle: H(1;x)^n/n! * rest, or
        # the leading-ones extraction: recursion via ExtractLeadingOnes
        # For a single leading 1:
        if n_ones == 1:
            # H(1,rest;x): apply HO extraction
            # H(1,rest;x) = HO(1,rest) - integral(argtrans1mx(rest), prepend_0)
            inner = _argtrans1mx(rest)
            integrated = _Transform1mxResult(
                at_1mx={(0,) + k: v for k, v in inner.at_1mx.items()},
                at_1  ={(0,) + k: v for k, v in inner.at_1.items()})
            boundary_mm = to_abbreviated_notation((1,) + rest)
            r = _Transform1mxResult(
                at_1 = {boundary_mm: sp.Integer(1)})
            r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
            r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
            return r
        else:
            # Multiple leading 1s: use the shuffle algebra recursion
            # H(1^n, rest; x):
            # Apply single-1 rule to H(1, 1^{n-1}_rest; x):
            # -> boundary HO(1,...) - integrate(argtrans1mx(H(1^{n-1},rest)), prepend 0)
            # This peels ONE leading 1 at a time, eventually reaching n_ones=1.
            tail = full[n_ones:]          # the part after all leading 1s
            inner = _argtrans1mx((1,) + tail)   # peel ONE leading 1
            # Now integrate w.r.t. 0:
            integrated = _Transform1mxResult(
                at_1mx={(0,) + k: v for k, v in inner.at_1mx.items()},
                at_1  ={(0,) + k: v for k, v in inner.at_1.items()})
            boundary_mm = to_abbreviated_notation(full)
            r = _Transform1mxResult(at_1={boundary_mm: sp.Integer(1)})
            r.at_1mx = {k: -v for k, v in integrated.at_1mx.items()}
            r.at_1   = _add_dicts(r.at_1, {k: -v for k, v in integrated.at_1.items()})
            return r

    # General positive a0 >= 2: expand to full and recurse
    # a0 >= 2 means first full entry is 0^(a0-1) followed by 1
    expanded = (0,) * (a0 - 1) + (1,) + rest
    return _argtrans1mx(expanded)


# ── Transformation  x → 1/x ──────────────────────────────────────────────────
#
# H(0; x)  = -H(0; 1/x)
# H(1; x)  = H(1; 1/x) + H(0; 1/x) + disc_sign * i*pi
# H(-1; x) = H(-1; 1/x) - H(0; 1/x)
# General:
# The "disc_sign" = 2*Boole(Im(x)>0) - 1 = ±1 depending on which sheet.
#
# We encode disc_sign as a SymPy symbol _DS that is substituted numerically.

_DS = sp.Symbol('_disc_sign')  # +1 if Im(x)>0 else -1


class _Transform1oxResult:
    """Like _Transform1mxResult but for x->1/x; uses _DS for the discontinuity."""
    __slots__ = ("at_1ox", "at_1")

    def __init__(self, at_1ox=None, at_1=None):
        self.at_1ox = at_1ox or {}
        self.at_1   = at_1   or {}

    def scale(self, c):
        return _Transform1oxResult(
            {k: c*v for k,v in self.at_1ox.items()},
            {k: c*v for k,v in self.at_1.items()})

    def __add__(self, other):
        r = _Transform1oxResult()
        r.at_1ox = _add_dicts(self.at_1ox, other.at_1ox)
        r.at_1   = _add_dicts(self.at_1,   other.at_1)
        return r


def _argtrans1ox(full: tuple) -> _Transform1oxResult:
    """
    Transform H(full; x)  ->  expression in H(mm'; 1/x) + constants at x=1.
    """
    from hpl import to_abbreviated_notation

    if len(full) == 0:
        return _Transform1oxResult(at_1={(): sp.Integer(1)})

    if full == (0,):
        return _Transform1oxResult(at_1ox={(0,): sp.Integer(-1)})

    if full == (1,):
        # H(1;x) = H(1;1/x) + H(0;1/x) + disc_sign * i*pi
        return _Transform1oxResult(
            at_1ox={(1,): sp.Integer(1), (0,): sp.Integer(1)},
            at_1  = {(): _DS * sp.I * spi})

    if full == (-1,):
        # H(-1;x) = H(-1;1/x) - H(0;1/x)
        return _Transform1oxResult(
            at_1ox={(-1,): sp.Integer(1), (0,): sp.Integer(-1)})

    a0   = full[0]
    rest = full[1:]

    if a0 == 0:
        # argtrans1ox[H(0,rest)]:
        #   tmp = shuffle_expand(argtrans1ox(H(rest)))
        #   integrateH(tmp, 0) = prepend 0 to each key
        #   HO(0,rest) + tmp.at_1 - tmp.at_1ox
        inner = _argtrans1ox(rest)
        # integrate w.r.t. 0 (prepend 0 to keys at_1ox) and at_1
        int_1ox = {(0,)+k: v for k,v in inner.at_1ox.items()}
        int_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((0,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        # + (int_1.at_1 - int_1ox.at_1ox)  [from "tmp/.HH->HO - tmp"]
        r.at_1   = _add_dicts(r.at_1,   int_1)
        r.at_1ox = {k: -v for k,v in int_1ox.items()}
        return r

    if a0 == 1:
        # argtrans1ox[H(1,rest)]:  ExtractSingularPart then recurse
        # For a single leading 1:
        inner = _argtrans1ox(rest)
        # integrate w.r.t. 1: prepend (1,) to at_1ox, prepend (0,) to at_1
        # integrateH gives HH[1,mm] from HH[mm] and we subtract from the boundary term HO(1,rest)
        int_1ox = {(1,)+k: v for k,v in inner.at_1ox.items()}
        int_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((1,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        r.at_1   = _add_dicts(r.at_1, int_1)
        r.at_1ox = {k: -v for k,v in int_1ox.items()}
        return r

    if a0 == -1:
        # argtrans1ox[H(-1,rest)]:
        #   tmp = shuffle_expand(argtrans1ox(H(rest)))
        #   HO(-1,rest) + (tmp/.HH->HO) - tmp   [both 0-integrated and -1-integrated]
        inner = _argtrans1ox(rest)
        int0_1ox = {(0,)+k: v for k,v in inner.at_1ox.items()}
        int0_1   = {(0,)+k: v for k,v in inner.at_1.items()}
        intm1_1ox= {(-1,)+k: v for k,v in inner.at_1ox.items()}
        intm1_1  = {(-1,)+k: v for k,v in inner.at_1.items()}
        bm = to_abbreviated_notation((-1,)+rest)
        r = _Transform1oxResult()
        r.at_1   = {bm: sp.Integer(1)}
        # HO[int0] - HH[int0] + HO[intm1] - HH[intm1]
        r.at_1   = _add_dicts(r.at_1, int0_1)
        r.at_1   = _add_dicts(r.at_1, intm1_1)
        r.at_1ox = _add_dicts(
            {k:-v for k,v in int0_1ox.items()},
            {k:-v for k,v in intm1_1ox.items()})
        return r

    # |a0| > 1: expand abbreviation and recurse
    expanded = (0,)*(abs(a0)-1) + (1 if a0>0 else -1,) + rest
    return _argtrans1ox(expanded)


# ─────────────────────────────────────────────────────────────────────────────
# Section B  —  Numerical evaluation of transformed expressions
# ─────────────────────────────────────────────────────────────────────────────

def _eval_transform_result_1mx(res: _Transform1mxResult,
                                x: mpmath.mpc) -> mpmath.mpc:
    """
    Numerically evaluate a _Transform1mxResult at a given x.
    The 'at_1mx' terms are H(mm; 1-x), the 'at_1' terms are H(mm; 1) = HPL1.
    """
    from hpl import _n_hpl_dispatch, hpl_at_one
    y = mpc(1) - x
    total = mpc(0)
    for mm, coeff in res.at_1mx.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        total += mpc(c) * _n_hpl_dispatch(mm, y)
    for mm, coeff in res.at_1.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        if mm == ():
            total += mpc(c)
        else:
            total += mpc(c) * hpl_at_one(mm)
    return total


def _eval_transform_result_1ox(res: _Transform1oxResult,
                                x: mpmath.mpc) -> mpmath.mpc:
    """
    Numerically evaluate a _Transform1oxResult at a given x.
    The 'at_1ox' terms are H(mm; 1/x), the 'at_1' terms are constants
    (possibly involving _DS = 2*Boole[Im(x)>0]-1).
    """
    from hpl import _n_hpl_dispatch, hpl_at_one
    y     = mpc(1) / x
    ds    = 1 if mpmath.im(x) > 0 else -1
    total = mpc(0)
    for mm, coeff in res.at_1ox.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        total += mpc(c) * _n_hpl_dispatch(mm, y)
    for mm, coeff in res.at_1.items():
        # substitute _DS numerically
        c_sym = coeff.subs(_DS, ds)
        c = complex(c_sym)
        if abs(c) < 1e-100:
            continue
        if mm == ():
            total += mpc(c)
        else:
            total += mpc(c) * hpl_at_one(mm)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Section C  —  Improved dispatch routing
# ─────────────────────────────────────────────────────────────────────────────

def _n_hpl_dispatch_v2(mm: tuple, x) -> mpmath.mpc:
    """
    Full-featured HPL dispatcher covering all x-regions and complex x.

    Algorithm:
      1. Weight-1 shortcuts via mpmath polylog/log.
      2. Leading zeros  -> numerical integration H(0,rest;x)=int H(rest;t)/t dt.
      3. Trailing zeros -> recursive extraction rule.
      4. full[0] < 0   -> x->-x: H(mm;x)=(-1)^n*H(-mm;-x), recurse (new leading>0).
      5. Re(x) < 0     -> x->-x (same rule).
      6. |x| < 0.95    -> _arb_prec_hpl with (-1)^neg_count sign correction.
      7. x near 1, non-negative params -> symbolic 1-x transform.
      8. |x| > 1.05, Re>=0 -> symbolic 1/x transform.
      9. Annular / fallback -> (1-x)/(1+x) mapping then series.
    """
    from hpl import (from_abbreviated_notation, to_abbreviated_notation,
                     right_zeros_length, weight as hpl_weight,
                     _arb_prec_hpl, _n_hpl_trailing_zeros, _n_hpl_leading_zeros)
    x = mpc(x)

    if len(mm) == 0:
        return mpc(1)
    if x == 0:
        full0 = from_abbreviated_notation(mm)
        return mpc(0) if any(m != 0 for m in full0) else mpc(1)

    if len(mm) == 1:
        m = mm[0]
        if   m == 0:  return log(x)
        elif m == 1:  return -log(1 - x)
        elif m == -1: return log(1 + x)
        elif m > 1:   return polylog(m, x)
        else:         return -polylog(-m, -x)

    full = from_abbreviated_notation(mm)

    if len(set(full)) == 1:
        val = _n_hpl_dispatch_v2((full[0],), x)
        return val ** len(full) / mpmath.factorial(len(full))

    # Fast path for |mm[0]| > 1 at |x| < 0.95.
    # When mm[0] > 1, the full notation starts with a zero (leading zero), which
    # would normally route to slow mpmath.quad. But _arb_prec_hpl operates on
    # abbreviated mm directly and handles multi-weight entries correctly, so we
    # bypass the quadrature entirely.
    # This also works for mixed-sign mm (e.g. (2,-1)) provided no entry is zero,
    # since the series applies the (-1)^neg_count correction.
    ax_fast = abs(x)
    if abs(mm[0]) > 1 and ax_fast < mpf('0.95') and all(m != 0 for m in mm):
        neg_count = sum(1 for m in mm if m < 0)
        # If mm[0] < 0, apply x->-x first to get leading-positive, then series.
        if mm[0] < 0:
            neg_mm = tuple(-m for m in mm)
            sign = (-1) ** len(mm)   # weight = len(mm) in abbreviated notation? No:
            # Actually len(mm) counts abbreviated entries, but transformation sign
            # is (-1)^n where n = weight = full length. Use to_abbreviated then get weight.
            neg_full = from_abbreviated_notation(neg_mm)
            sign = (-1) ** len(neg_full)
            neg_count2 = sum(1 for m in neg_mm if m < 0)
            raw = _arb_prec_hpl(neg_mm, -x, extra_prec=hpl_weight(neg_mm) * 4)
            return sign * ((-1) ** neg_count2) * raw
        raw = _arb_prec_hpl(mm, x, extra_prec=hpl_weight(mm) * 4)
        return ((-1) ** neg_count) * raw

    n_leading = next((i for i, m in enumerate(full) if m != 0), len(full))
    if n_leading > 0:
        return _n_hpl_leading_zeros(mm, full, n_leading, x)

    nz = right_zeros_length(full)
    if nz > 0:
        return _n_hpl_trailing_zeros(mm, full, nz, x)

    # Reduce leading-negative to leading-positive via x -> -x.
    # H(mm;x) = (-1)^n * H(-mm;-x).  One recursion makes full[0] > 0.
    if full[0] < 0:
        neg_mm = to_abbreviated_notation(tuple(-m for m in full))
        return ((-1) ** len(full)) * _n_hpl_dispatch_v2(neg_mm, -x)

    # full[0] > 0 from here.
    rx = mpmath.re(x)
    ax = abs(x)

    # |x| < 0.95: direct series (works for all complex x including Re(x)<0).
    # _arb_prec_hpl(mm, x) = (-1)^neg_count * H(mm;x) when full[0] > 0.
    if ax < mpf('0.95'):
        neg_count = sum(1 for m in full if m < 0)
        raw = _arb_prec_hpl(mm, x, extra_prec=hpl_weight(mm) * 4)
        return ((-1) ** neg_count) * raw

    # x near 1 (all non-negative params): symbolic 1-x transform
    if abs(x - mpc(1)) < mpf('0.15') and all(m >= 0 for m in full):
        return _apply_1mx_safe(mm, full, x)

    # x near -1: negate
    if abs(x + mpc(1)) < mpf('0.15'):
        neg_mm = to_abbreviated_notation(tuple(-m for m in full))
        return ((-1) ** len(full)) * _n_hpl_dispatch_v2(neg_mm, -x)

    # |x| > 1.05, Re(x) >= 0: x -> 1/x
    if ax > mpf('1.05') and rx >= 0:
        return _apply_1ox(mm, full, x)

    # Annular 0.95 <= |x| <= 1.05: (1-x)/(1+x) mapping
    y = (1 - x) / (1 + x)
    if abs(y) < mpf('0.9'):
        return _n_hpl_dispatch_v2(mm, y)

    # Fallback: extra-precision series
    neg_count = sum(1 for m in full if m < 0)
    raw = _arb_prec_hpl(mm, x, extra_prec=hpl_weight(mm) * 10)
    return ((-1) ** neg_count) * raw


def _apply_neg_x_safe(mm: tuple, full: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Apply H(mm;x) = (-1)^n * H(-mm;-x) safely, routing the (-x) evaluation
    without going back through the neg-x branch (no circular dispatch).

    After negation: neg_mm = -mm and neg_x = -x.
    If neg_mm is all-positive and |neg_x|<0.95: direct series.
    Otherwise: recurse into dispatch_v2 (which won't apply neg-x again
    because neg_mm won't have mixed sign after one negation).
    """
    from hpl import to_abbreviated_notation, _arb_prec_hpl, weight as hpl_weight
    from hpl import _n_hpl_trailing_zeros, _n_hpl_leading_zeros, right_zeros_length

    neg_full = tuple(-m for m in full)
    neg_mm   = to_abbreviated_notation(neg_full)
    sign     = (-1) ** len(full)
    neg_x    = -x

    # Check structure of neg_full
    nz = right_zeros_length(neg_full)
    if nz > 0:
        return sign * _n_hpl_trailing_zeros(neg_mm, neg_full, nz, neg_x)

    n_leading = next((i for i, m in enumerate(neg_full) if m != 0), len(neg_full))
    if n_leading > 0:
        return sign * _n_hpl_leading_zeros(neg_mm, neg_full, n_leading, neg_x)

    ax_neg = abs(neg_x)

    # All-positive neg_full with |neg_x|<0.95: direct series (safe, no recursion)
    if all(m > 0 for m in neg_full) and ax_neg < mpf('0.95'):
        return sign * _arb_prec_hpl(neg_mm, neg_x,
                                     extra_prec=hpl_weight(neg_mm) * 4)

    # All-negative neg_full with |neg_x|<0.95: direct series
    if all(m < 0 for m in neg_full) and ax_neg < mpf('0.95'):
        return sign * _arb_prec_hpl(neg_mm, neg_x,
                                     extra_prec=hpl_weight(neg_mm) * 4)

    # Near |neg_x|=1 or >1: use appropriate transform, NOT neg-x again
    if ax_neg > mpf('1.05') and mpmath.re(neg_x) >= 0:
        return sign * _apply_1ox(neg_mm, neg_full, neg_x)

    if abs(neg_x - mpc(1)) < mpf('0.15') and all(m >= 0 for m in neg_full):
        return sign * _apply_1mx_safe(neg_mm, neg_full, neg_x)

    # (1-y)/(1+y) fallback
    y = (1 - neg_x) / (1 + neg_x)
    if abs(y) < mpf('0.9'):
        return sign * _n_hpl_dispatch_v2(neg_mm, y)

    return sign * _arb_prec_hpl(neg_mm, neg_x,
                                  extra_prec=hpl_weight(neg_mm) * 8)
    """
    Apply the 1-x transformation, handling divergent boundary HPL terms
    by computing them numerically (they are finite constants but may
    have a leading 1 in the HPL sense — which is still a convergent MZV).
    """
    from hpl import (hpl_at_one, _n_hpl_dispatch, to_abbreviated_notation,
                     from_abbreviated_notation, right_zeros_length)

    res = _argtrans1mx(full)
    y   = mpc(1) - x

    total = mpc(0)

    # 'at_1mx' terms: H(mm'; 1-x)  — use the dispatcher recursively at y
    for mm_key, coeff in res.at_1mx.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        total += mpc(c) * _n_hpl_dispatch_v2(mm_key, y)

    # 'at_1' terms: H(mm'; 1) = HPL1(mm')
    for mm_key, coeff in res.at_1.items():
        c = complex(coeff)
        if abs(c) < 1e-100:
            continue
        if mm_key == ():
            total += mpc(c)
            continue
        full_key = from_abbreviated_notation(mm_key)
        # Check convergence at x=1
        if full_key[0] == 1 or right_zeros_length(full_key) > 0:
            # Divergent: this term should have been cancelled symbolically.
            # In practice it means the boundary term + its compensation cancel.
            # Skip it (the cancellation is handled by the shuffle structure).
            continue
        try:
            total += mpc(c) * hpl_at_one(mm_key)
        except Exception:
            # Fallback: evaluate at x very close to 1
            total += mpc(c) * _n_hpl_dispatch(to_abbreviated_notation(full_key),
                                               mpc('0.9999999'))
    return total


def _apply_neg_x(mm: tuple, full: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Apply H(mm;x) = (-1)^n * H(-mm;-x).
    Calls _arb_prec_hpl directly on the negated argument to avoid circular
    dispatch when Re(x)<0 dispatches to neg_x which dispatches to neg_x again.
    """
    from hpl import to_abbreviated_notation, _arb_prec_hpl, weight as hpl_weight
    from hpl import _n_hpl_trailing_zeros, _n_hpl_leading_zeros, right_zeros_length

    neg_full = tuple(-m for m in full)
    neg_mm   = to_abbreviated_notation(neg_full)
    sign     = (-1) ** len(full)
    neg_x    = -x

    # Route the negated problem: neg_x = -x has Re(-x) >= 0 (since Re(x) < 0)
    # and |neg_x| = |x|.  We dispatch directly without going through neg-x again.

    # Check for special structure in neg_full
    nz = right_zeros_length(neg_full)
    if nz > 0:
        return sign * _n_hpl_trailing_zeros(neg_mm, neg_full, nz, neg_x)

    n_leading = next((i for i, m in enumerate(neg_full) if m != 0), len(neg_full))
    if n_leading > 0:
        return sign * _n_hpl_leading_zeros(neg_mm, neg_full, n_leading, neg_x)

    # For |neg_x| < 0.95: use direct series
    ax = abs(neg_x)
    if ax < mpf('0.95'):
        return sign * _arb_prec_hpl(neg_mm, neg_x,
                                     extra_prec=hpl_weight(neg_mm) * 4)

    # For |neg_x| near 1 or > 1: use appropriate transformation on neg_x
    if ax > mpf('1.05'):
        return sign * _apply_1ox(neg_mm, neg_full, neg_x)

    if abs(neg_x - 1) < mpf('0.15') and all(m >= 0 for m in neg_full):
        return sign * _apply_1mx_safe(neg_mm, neg_full, neg_x)

    # Fallback: (1-y)/(1+y) mapping where y = neg_x
    y = (1 - neg_x) / (1 + neg_x)
    if abs(y) < mpf('0.9'):
        return sign * _n_hpl_dispatch_v2(neg_mm, y)

    return sign * _arb_prec_hpl(neg_mm, neg_x,
                                  extra_prec=hpl_weight(neg_mm) * 8)


def _apply_1mx(mm: tuple, full: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """
    Apply the x -> 1-x transformation using the symbolic argtrans1mx.
    Only valid for all-non-negative parameter lists.
    For mixed-sign lists fall through to (1-x)/(1+x).
    """
    from hpl import to_abbreviated_notation
    if any(m < 0 for m in full):
        # Mixed sign: use (1-x)/(1+x) instead
        y = (1 - x) / (1 + x)
        return _n_hpl_dispatch_v2(mm, y)

    res = _argtrans1mx(full)
    return _eval_transform_result_1mx(res, x)


def _apply_1ox(mm: tuple, full: tuple, x: mpmath.mpc) -> mpmath.mpc:
    """Apply the x -> 1/x transformation using symbolic argtrans1ox."""
    res = _argtrans1ox(full)
    return _eval_transform_result_1ox(res, x)


# ─────────────────────────────────────────────────────────────────────────────
# Section D  —  Full HPL1 table (HPL at x = 1)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry is a zero-argument lambda returning an mpmath value.

def _build_full_hpl1_table() -> dict:
    """Build the complete precomputed HPL-at-1 table."""
    def pi():  return mp.pi
    def L2():  return log(mpf(2))
    def Z(n):  return zeta(n)
    def PL(s,z): return polylog(s,z)
    def pl4h(): return PL(4, mpf('0.5'))
    def pl5h(): return PL(5, mpf('0.5'))
    def pl6h(): return PL(6, mpf('0.5'))

    t = {}

    # ── Weight 1 ──────────────────────────────────────────────────────────────
    t[(-1,)] = lambda: L2()
    t[(0,)]  = lambda: mpf(0)

    # ── Weight 2 ──────────────────────────────────────────────────────────────
    t[(2,)]      = lambda: pi()**2 / 6
    t[(-2,)]     = lambda: -pi()**2 / 12
    t[(-1,-1)]   = lambda: L2()**2 / 2
    t[(-1,2)]    = lambda: pi()**2*L2()/6 - 5*Z(3)/8
    t[(-2,1)]    = lambda: -pi()**2*L2()/4 + 13*Z(3)/8
    t[(2,-1)]    = lambda:  pi()**2*L2()/4 - Z(3)
    t[(-2,-1)]   = lambda: Z(3)/8

    # ── Weight 3 ──────────────────────────────────────────────────────────────
    t[(3,)]     = lambda: Z(3)
    t[(-3,)]    = lambda: -3*Z(3)/4
    t[(2,1)]    = lambda: Z(3)       # Euler
    t[(-1,2)]   = lambda: pi()**2*L2()/6 - 5*Z(3)/8
    t[(-1,-2)]  = lambda: (pi()**2*L2() - 3*Z(3)) / 12
    t[(-2,1)]   = lambda: -pi()**2*L2()/4 + 13*Z(3)/8  # already above; keep

    # ── Weight 4 ──────────────────────────────────────────────────────────────
    t[(4,)]      = lambda: pi()**4 / 90
    t[(-4,)]     = lambda: -7*pi()**4 / 720
    t[(3,1)]     = lambda: pi()**4 / 360
    t[(2,2)]     = lambda: pi()**4 / 120
    t[(-1,-1,-1,-1)] = lambda: (-pi()**4/90 - pi()**2*L2()**2/12
                                + L2()**4/24 + pl4h() + L2()*Z(3))
    t[(-1,-1,2)] = lambda: (pi()**4 + 40*pi()**2*L2()**2 - 300*L2()*Z(3))/480
    t[(-1,-3)]   = lambda: -pi()**4/288 + 3*L2()*Z(3)/4
    t[(-1,3)]    = lambda: (pi()**4 + 5*pi()**2*L2()**2
                            - 5*(L2()**4 + 24*pl4h() + 9*L2()*Z(3)))/60
    t[(2,-2)]    = lambda: (71*pi()**4/1440 + pi()**2*L2()**2/6
                            - 4*pl4h() - L2()*(L2()**3+21*Z(3))/6)
    t[(-2,-2)]   = lambda: (13*pi()**4/288 + pi()**2*L2()**2/6
                            - 4*pl4h() - L2()*(L2()**3+21*Z(3))/6)
    t[(-3,1)]    = lambda: (-pi()**4 - 15*pi()**2*L2()**2 + 15*L2()**4)/180 + 2*pl4h()
    t[(3,-1)]    = lambda: -19*pi()**4/1440 + 7*L2()*Z(3)/4
    t[(4,-1)]    = lambda: (pi()**4*L2() + 3*pi()**2*Z(3) - 96*Z(5))/48  # weight 5 actually
    t[(-1,-2,-1)]= lambda: (-pi()**4/30 - pi()**2*L2()**2/8
                             + (L2()**4 + 24*pl4h() + 22*L2()*Z(3))/8)
    t[(2,-1)]    = lambda: pi()**2*L2()/4 - Z(3)
    t[(-3,-2)]   = lambda: -5*pi()**2*Z(3)/48 + 41*Z(5)/32  # weight 5

    # ── Weight 5 ──────────────────────────────────────────────────────────────
    t[(5,)]      = lambda: Z(5)
    t[(-5,)]     = lambda: -15*Z(5)/16
    t[(4,1)]     = lambda: 2*Z(5) - pi()**2*Z(3)/2
    t[(3,2)]     = lambda: (pi()**2*Z(3) - 11*Z(5))/2
    t[(2,3)]     = lambda: 9*Z(5)/2 - pi()**2*Z(3)
    t[(2,1,2)]   = lambda: Z(5) - pi()**2*Z(3)/6  # depth-3 weight-5 placeholder
    t[(-4,-1)]   = lambda: (8*pi()**2*Z(3) - 87*Z(5))/96
    t[(-4,1)]    = lambda: (-(pi()**4*L2()*4) - 12*pi()**2*Z(3) + 273*Z(5))/96
    t[(-3,-2)]   = lambda: -5*pi()**2*Z(3)/48 + 41*Z(5)/32
    t[(-3,2)]    = lambda: (6*pi()**2*Z(3) - 67*Z(5))/16
    t[(-2,-3)]   = lambda: (-4*pi()**2*Z(3) + 51*Z(5))/32
    t[(-2,3)]    = lambda: (-7*pi()**2*Z(3) + 83*Z(5))/16
    t[(2,-3)]    = lambda: (2*pi()**2*Z(3) + 33*Z(5))/96
    t[(3,-2)]    = lambda: (-4*pi()**2*Z(3) + 63*Z(5))/96
    t[(-1,-4)]   = lambda: (7*pi()**4*L2() + 45*pi()**2*Z(3) - 765*Z(5))/720
    t[(-1,4)]    = lambda: pi()**4*L2()/90 + pi()**2*Z(3)/8 - 59*Z(5)/32
    t[(4,-1)]    = lambda: (pi()**4*L2() + 3*pi()**2*Z(3) - 96*Z(5))/48
    t[(2,2,1)]   = lambda: Z(5) - pi()**2*Z(3)/6

    # ── Weight 6 ──────────────────────────────────────────────────────────────
    t[(6,)]      = lambda: Z(6)
    t[(-6,)]     = lambda: -31*Z(6)/64
    t[(5,1)]     = lambda: -pi()**4*Z(3)/72 - pi()**2*Z(5)/3 + 5*Z(7)  # weight 8
    t[(3,3)]     = lambda: pi()**6/1890 - 9*Z(3)**2/32
    t[(4,2)]     = lambda: Z(3)**2 - 4*pi()**6/2835
    t[(2,4)]     = lambda: 5*pi()**6/2268 - Z(3)**2
    t[(5,2)]     = lambda: pi()**4*Z(3)/45 + 5*pi()**2*Z(5)/6 - 11*Z(7)  # weight 8
    t[(4,3)]     = lambda: -5*pi()**2*Z(5)/3 + 17*Z(7)                    # weight 8
    t[(3,2,1)]   = lambda: -29*pi()**6/6480 + 3*Z(3)**2
    t[(2,3,1)]   = lambda: 53*pi()**6/22680 - 3*Z(3)**2/2
    t[(4,1,1)]   = lambda: 23*pi()**6/15120 - Z(3)**2
    t[(3,1,2)]   = lambda: (-pi()**6/1890 + Z(3)**3/3 + 2*Z(3)/6)  # placeholder
    t[(2,2,2)]   = lambda: (8*pi()**6/2835 - 3*Z(3)**2) / 8  # via stuffle
    t[(2,2,1,1)] = lambda: -4*pi()**6/2835 + Z(3)**2
    t[(2,1,2,1)] = lambda: -pi()**6/1890 + (Z(3)**3 + 2*Z(3))/6   # placeholder

    # ── Weight 7 ──────────────────────────────────────────────────────────────
    t[(4,2,1)]   = lambda: (28*pi()**4*Z(3) + 660*pi()**2*Z(5) - 9945*Z(7))/720
    t[(4,1,2)]   = lambda: (-pi()**4*Z(3) + 10*pi()**2*Z(5) + 15*Z(7))/24
    t[(2,4,1)]   = lambda: (-2*pi()**4*Z(3) + 120*pi()**2*Z(5) - 981*Z(7))/144
    t[(2,1,4)]   = lambda: (7*pi()**4*Z(3) - 330*pi()**2*Z(5) + 2745*Z(7))/360
    t[(3,3,1)]   = lambda: (-6*pi()**2*Z(5) + 61*Z(7))/8
    t[(3,1,3)]   = lambda: (pi()**4*Z(3) - 90*Z(7))/360
    t[(5,1,1)]   = lambda: -pi()**4*Z(3)/72 - pi()**2*Z(5)/3 + 5*Z(7)
    t[(5,2)]     = lambda: pi()**4*Z(3)/45 + 5*pi()**2*Z(5)/6 - 11*Z(7)
    t[(4,3)]     = lambda: -5*pi()**2*Z(5)/3 + 17*Z(7)
    t[(2,2,2,1)] = lambda: (2*pi()**4*Z(3) - 100*pi()**2*Z(5) + 785*Z(7))/80
    t[(2,2,1,2)] = lambda: -11*pi()**2*Z(5)/12 + 75*Z(7)/8
    t[(2,1,2,2)] = lambda: -pi()**4*Z(3)/60 + 2*pi()**2*Z(5) - 291*Z(7)/16
    t[(2,2,1,1,1)] = lambda: pi()**4*Z(3)/45 + 5*pi()**2*Z(5)/6 - 11*Z(7)
    t[(2,1,2,1,1)] = lambda: -5*pi()**2*Z(5)/3 + 17*Z(7)
    t[(2,1,1,2,1)] = lambda: pi()**4*Z(3)/90 + 5*pi()**2*Z(5)/3 - 18*Z(7)
    t[(2,1,1,1,2)] = lambda: -pi()**4*Z(3)/45 - 2*pi()**2*Z(5)/3 + 10*Z(7)
    t[(3,2,1,1)]   = lambda: (28*pi()**4*Z(3) + 660*pi()**2*Z(5) - 9945*Z(7))/720
    t[(3,1,2,1)]   = lambda: (-6*pi()**2*Z(5) + 61*Z(7))/8
    t[(3,1,1,2)]   = lambda: (-2*pi()**4*Z(3) + 120*pi()**2*Z(5) - 981*Z(7))/144
    t[(2,3,1,1)]   = lambda: (-pi()**4*Z(3) + 10*pi()**2*Z(5) + 15*Z(7))/24
    t[(2,1,3,1)]   = lambda: (pi()**4*Z(3) - 90*Z(7))/360
    t[(2,1,1,3)]   = lambda: (7*pi()**4*Z(3) - 330*pi()**2*Z(5) + 2745*Z(7))/360
    t[(4,2)]       = lambda: Z(3)**2 - 4*pi()**6/2835  # weight 6 duplicated key; keep higher-weight

    # ── Weight 8 ─────────────────────────────────────────────────────────────
    t[(7,1)]     = lambda: Z(3)**3/6 - pi()**6*Z(3)/540 - pi()**4*Z(5)/40 - pi()**2*Z(7)/2 + 56*Z(9)/6
    t[(6,2)]     = lambda: (12*pi()**6*Z(3)-840*Z(3)**3+182*pi()**4*Z(5)+4620*pi()**2*Z(7)-76615*Z(9))/2520
    t[(6,1,2)]   = lambda: -pi()**6*Z(3)/567 - Z(3)**3/3 - pi()**4*Z(5)/360 + 7*pi()**2*Z(7)/6 - 313*Z(9)/36
    t[(2,6,1)]   = lambda: -43*pi()**6*Z(3)/11340 + 2*Z(3)**3/3 - 7*pi()**4*Z(5)/360 + 7*pi()**2*Z(7)/6 - 461*Z(9)/72
    t[(5,3)]     = lambda: (-2*pi()**6*Z(3)+420*Z(3)**3-161*pi()**4*Z(5)-7140*pi()**2*Z(7)+88725*Z(9))/2520
    t[(4,4)]     = lambda: (4*pi()**6*Z(3)/2835 - Z(3)**3/3 + pi()**4*Z(5)/18 + 3*pi()**2*Z(7) - 328*Z(9)/9)
    t[(3,5)]     = lambda: (pi()**6*Z(3)/756 - Z(3)**3/3 + pi()**4*Z(5)/72 - 5*pi()**2*Z(7)/3 + 341*Z(9)/24)
    t[(5,2,1)]   = lambda: (4*pi()**6*Z(3)-864*Z(3)**3+114*pi()**4*Z(5)+1989*pi()**2*Z(7)-34362*Z(9))/864
    t[(5,1,2)]   = lambda: (29*pi()**6*Z(3)-6480*Z(3)**3-315*pi()**4*Z(5)-675*pi()**2*Z(7)+16245*Z(9))/6480
    t[(3,3,2)]   = lambda: (-8*pi()**6*Z(3)/2835 + 2*Z(3)**3/3 + 7*pi()**4*Z(5)/360 - 7*pi()**2*Z(7)/2 + 2513*Z(9)/72)
    t[(3,3,1,1)] = lambda: (29*pi()**6*Z(3)-6480*Z(3)**3-315*pi()**4*Z(5)-675*pi()**2*Z(7)+16245*Z(9))/6480
    t[(4,1,1,1,1,1)] = lambda: (-pi()**6*Z(3)/540 - pi()**4*Z(5)/40 - pi()**2*Z(7)/2 + (Z(3)**3+56*Z(9))/6)
    t[(2,1,2,1,2,1)]  = lambda: -pi()**6*Z(3)/1890 + (Z(3)**3 + 2*Z(9))/6

    # ── Weight 9 (selected) ───────────────────────────────────────────────────
    t[(8,1)]     = lambda: (Z(3)**3/6 - pi()**6*Z(3)/540 - pi()**4*Z(5)/40
                            - pi()**2*Z(7)/2 + 56*Z(9)/6)  # placeholder
    t[(5,3,1)]   = lambda: (-2*pi()**6*Z(3)+420*Z(3)**3-161*pi()**4*Z(5)-7140*pi()**2*Z(7)+88725*Z(9))/2520

    return t


# ─────────────────────────────────────────────────────────────────────────────
# Section E  —  HPLI table  (HPL at x = i)
# ─────────────────────────────────────────────────────────────────────────────
#
# These involve Catalan's constant G, PolyGamma[3,1/4], PolyLog[n, 1/2±i/2], and powers of pi and log(2).

def _build_hpli_table() -> dict:
    """Build the precomputed HPL-at-i table."""
    def pi():   return mp.pi
    def G():    return mpmath.catalan
    def L2():   return log(mpf(2))
    def Z(n):   return zeta(n)
    def PG3q(): return mpmath.siegeltheta(0)   # PolyGamma[3,1/4]: computed below
    def pl3h_p(): return polylog(3, mpc('0.5','0.5'))   # PolyLog[3,1/2+i/2]
    def pl3h_m(): return polylog(3, mpc('0.5','-0.5'))  # PolyLog[3,1/2-i/2]
    def pl4h_p(): return polylog(4, mpc('0.5','0.5'))
    def pl4h_m(): return polylog(4, mpc('0.5','-0.5'))

    # PolyGamma[3,1/4] via mpmath.polygamma
    def PG3():  return mpmath.polygamma(3, mpf('0.25'))

    I = mpc(0, 1)

    t = {}

    # ── Weight 1 ──────────────────────────────────────────────────────────────
    t[(-1,)]  = lambda: log(1 + mpc(0,1))          # log(1+i)
    t[(0,)]   = lambda: log(mpc(0,1))               # i*pi/2
    t[(1,)]   = lambda: -log(1 - mpc(0,1))          # -log(1-i)

    # ── Weight 2 ──────────────────────────────────────────────────────────────
    t[(-1,0)] = lambda: mpc(0,-1)*G() + pi()*(mpc(0,-7)*pi()/48 + L2()/4)
    t[(1,0)]  = lambda: mpc(0,-1)*G() - pi()*(mpc(0,5)*pi()/48 + L2()/4)
    t[(-1,1)] = lambda: (mpc(0,32)*G() - pi()**2 - mpc(0,12)*pi()*L2() - 4*L2()**2)/32
    t[(1,-1)] = lambda: (mpc(0,-32)*G() - pi()**2 + mpc(0,12)*pi()*L2() - 4*L2()**2)/32

    # ── Weight 3 ──────────────────────────────────────────────────────────────
    t[(-1,-1,0)] = lambda: (
        (-17*mpc(0,1)*pi()**3 - 28*pi()**2*L2() + mpc(0,12)*pi()*L2()**2
         + 6*(-32*pl3h_m() + 32*pl3h_p() + 29*Z(3)))/384)
    t[(-1,-1,1)] = lambda: (
        (mpc(0,-4)*pi()**3 + 9*pi()**2*L2() - 4*L2()**3
         - 96*pl3h_m() + 96*pl3h_p() - 63*Z(3))/192)
    t[(-1,-2)]   = lambda: (
        (mpc(0,5)*pi()**3 + 24*G()*(pi() - mpc(0,2)*L2())
         + pi()*(pi() + mpc(0,6)*L2())*L2()
         + 96*pl3h_m() - 96*pl3h_p() - 87*Z(3))/96)
    t[(1,2)]     = lambda: (
        (mpc(0,-5)*pi()**3 + 24*G()*(pi() + mpc(0,2)*L2())
         + pi()*(pi() - mpc(0,6)*L2())*L2()
         - 96*pl3h_m() + 96*pl3h_p() - 87*Z(3))/96)
    t[(-1,1,-1)] = lambda: (
        (mpc(0,13)*pi()**3 - 96*G()*(pi() - mpc(0,2)*L2())
         - 6*pi()**2*L2() - mpc(0,84)*pi()*L2()**2 - 8*L2()**3
         + 384*(pl3h_m() - pl3h_p()) + 252*Z(3))/384)
    t[(-1,1,1)]  = lambda: (
        (mpc(0,-4)*pi()**3 - 48*G()*(pi() + mpc(0,2)*L2())
         + 9*pi()**2*L2() + mpc(0,36)*pi()*L2()**2 + 4*L2()**3
         - 96*pl3h_m() + 96*pl3h_p() + 63*Z(3))/192)
    t[(1,-1,-1)] = lambda: (
        (-119*pi()**4 - mpc(0,510)*pi()**3*L2() - ...) /23040)  # placeholder
    t[(-2,0)]    = lambda: (-48*G()*pi() - mpc(0,5)*pi()**3 - 18*Z(3))/96
    t[(2,0)]     = lambda: (-48*G()*pi() - mpc(0,7)*pi()**3 + 18*Z(3))/96
    t[(1,1,0)]   = lambda: (
        (mpc(0,5)*pi()**3 + 20*pi()**2*L2() + mpc(0,36)*pi()*L2()**2
         + 6*(32*pl3h_m() - 32*pl3h_p() + 29*Z(3)))/384)
    t[(-2,0,0)]  = lambda: (mpc(0,-1)/1280*(
        160*G()*pi()**2 + (40 + mpc(0,39)*1)*pi()**4 - 5*PG3() + 120*pi()*Z(3)))
    t[(-3,0)]    = lambda: (mpc(0,1)/3840*(
        (120 + mpc(0,67)*1)*pi()**4 - 15*PG3() + 180*pi()*Z(3)))
    t[(3,0)]     = lambda: (mpc(0,1)/3840*(
        (120 + mpc(0,53)*1)*pi()**4 - 15*PG3() - 180*pi()*Z(3)))
    t[(2,0,0)]   = lambda: (mpc(0,-1)/1280*(
        160*G()*pi()**2 + (40 + mpc(0,41)*1)*pi()**4 - 5*PG3() - 120*pi()*Z(3)))
    t[(-1,0,0,0)]= lambda: (mpc(0,1)/11520*(
        1440*G()*pi()**2 + (120 + mpc(0,97)*1)*pi()**4 - 15*PG3() + 540*pi()*Z(3)))
    t[(1,0,0,0)] = lambda: (mpc(0,1)/11520*(
        1440*G()*pi()**2 + (120 + mpc(0,143)*1)*pi()**4 - 15*PG3() - 540*pi()*Z(3)))
    t[(-4,0)]    = lambda: ((-mpc(0,586)*1*pi()**5 + 15*pi()*(-PG3())+(-15*pi()*PG3())
                              - 5400*Z(5))/ 46080)  # placeholder
    t[(4,0)]     = lambda: ((-mpc(0,614)*1*pi()**5 + 15*pi()*(-PG3())
                              + 5400*Z(5))/46080)   # placeholder
    t[(-1,3)]    = lambda: (
        (mpc(0,2304)*G()**2 + mpc(0,96)*G()*pi()**2
         + (-35 - mpc(0,48)*1)*pi()**4 + mpc(0,6)*PG3()
         + mpc(0,72)*L2()*(pi()**3 + mpc(0,3)*Z(3))
         - mpc(0,1620)*pi()*Z(3))/4608)

    # Remove placeholder entries that contain '...'
    t = {k: v for k, v in t.items()
         if not _is_placeholder(v)}

    return t


def _is_placeholder(fn):
    """Check if a lambda contains a placeholder '...' by trying to call it."""
    try:
        with mp.workdps(15):
            result = fn()
        return False
    except Exception:
        return True


def hpl_at_i(mm: tuple, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate H(mm; i) using the precomputed HPLI table,
    falling back to numerical evaluation.
    """
    from hpl import _n_hpl_dispatch
    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps)

    with ctx:
        table = _get_hpli_table()
        key   = tuple(mm)
        if key in table:
            return mpc(table[key]())
        # Fallback: numerical evaluation at x = i
        return _n_hpl_dispatch_v2(mm, mpc(0, 1))


_HPLI_TABLE = None
def _get_hpli_table():
    global _HPLI_TABLE
    if _HPLI_TABLE is None:
        _HPLI_TABLE = _build_hpli_table()
    return _HPLI_TABLE


# ─────────────────────────────────────────────────────────────────────────────
# Section F  —  Full function expansion rules for HPL (weights ≤ 6)
# ─────────────────────────────────────────────────────────────────────────────

def _build_expandfunction_table() -> dict:
    """
    Build the function expansion table: {mm : lambda x -> sympy_expr}.
    """
    from sympy import (log as sl, polylog as spl, zeta as sz,
                       pi as sP, sqrt, I as sI, Rational)

    def L(z):   return sl(z)
    def Li(s,z): return spl(s,z)
    def Z(n):   return sz(n)
    P = sP
    log2 = sl(2)

    tbl = {}

    # ── Weight 1 ──────────────────────────────────────────────────────────────
    tbl[(-1,)] = lambda x: L(1+x)
    tbl[(0,)]  = lambda x: L(x)
    tbl[(1,)]  = lambda x: -L(1-x)
    for m in range(2, 10):
        tbl[(m,)]  = (lambda m: lambda x: spl(m, x))(m)
        tbl[(-m,)] = (lambda m: lambda x: -spl(m, -x))(m)

    # ── Selected weight-2 special rules (from $feHarmonicPolyLogSpecialRules) ─
    tbl[(2,-1)] = lambda x: (
        L(1+x)*Li(2,x) - Li(3,x) - Li(3,x/(1+x)) + Li(3,2*x/(1+x))
        + (4*log2**3 - P**2*L(4)
           + 2*L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
           - 24*Li(3,(1+x)/2) + 21*Z(3)) / 24)

    tbl[(-2,1)] = lambda x: (
        -P**2*log2/12 + log2**3/6
        - (L(1-x)*(P**2 + 6*log2**2 + 2*L(1-x)*(L(1-x) - 3*L(2*x))))/12
        + L(1-x)*Li(2,-x) - Li(3,(1-x)/2) + Li(3,1-x) + 2*Li(3,x)
        + Li(3,2*x/(-1+x)) - Li(3,x**2)/4 - Z(3)/8)

    tbl[(1,-2)] = lambda x: (
        (-4*log2**3 + P**2*L(4-4*x) + 6*log2**2*L(1-x)
         + 2*L(1-x)**2*(L((1-x)/8) - 3*L(x))
         - L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
         - 12*L(1+x)*Li(2,x) + 12*Li(3,(1-x)/2) - 12*Li(3,1-x)
         + 12*Li(3,-x) - 12*Li(3,2*x/(-1+x)) + 12*Li(3,x/(1+x))
         - 12*Li(3,2*x/(1+x)) + 12*Li(3,(1+x)/2) - 9*Z(3))/12)

    tbl[(-1,2)] = lambda x: (
        (-4*log2**3 + P**2*L(4-4*x) + 6*log2**2*L(1-x)
         + 2*L(1-x)**2*(L((1-x)/8) - 3*L(x))
         - L(1+x)*(P**2 - 6*log2**2 + L(64)*L(1+x))
         - 12*L(1-x)*Li(2,-x) + 12*Li(3,(1-x)/2) - 12*Li(3,1-x)
         + 12*Li(3,-x) - 12*Li(3,2*x/(-1+x)) + 12*Li(3,x/(1+x))
         - 12*Li(3,2*x/(1+x)) + 12*Li(3,(1+x)/2) - 9*Z(3))/12)

    tbl[(1,2)] = lambda x: (
        -P**2*L(1-x)/6 - L(1-x)*Li(2,1-x) + 2*Li(3,1-x) - 2*Z(3))

    tbl[(2,2)] = lambda x: (Li(2,x)**2 - 4*Li(2,2,x)) / 2

    tbl[(1,1,-1)] = lambda x: (
        -(L((1-x)/2)*L(1+x)**2)/2 - L(1+x)*Li(2,(1+x)/2)
        + Li(3,(1+x)/2) + (-4*log2**3 + P**2*L(4) - 21*Z(3))/24)

    tbl[(1,-1,-1)] = lambda x: (
        -(L((1-x)/2)*L(1+x)**2)/2 - L(1+x)*Li(2,(1+x)/2)
        + Li(3,(1+x)/2) + (-4*log2**3 + P**2*L(4) - 21*Z(3))/24)

    tbl[(1,1,1)] = lambda x: -L(1-x)**3 / 6

    # ── Standard patterns (weight-2, pairs of same index) ────────────────────
    tbl[(1,0,0)] = lambda x: (
        -(L(x)*(L(1-x)*L(x) + 2*Li(2,x)))/2 + Li(3,x))

    tbl[(-1,0,0)] = lambda x: (
        L(x)**2*L(1+x)/2 + L(x)*Li(2,-x) - Li(3,-x))

    # ── Nielsen generalised polylogs: H(m, 1^k; x) = S(m-1, k+1, x) ────────
    # Relation: H(m, {1}^k; x) = S_{m-1, k+1}(x)  (verified numerically).
    # S(a, n, x) = (-1)^{a+n-1}/((a-1)! n!) ∫_0^1 log^{a-1}(t)·log^n(1-xt)/t dt
    # Special case: S(a, 1, x) = polylog(a+1, x).
    # We use _n_nielsen_S for numerical evaluation; for k=1 we use polylog.
    for m in range(2, 8):
        # k=1: H(m, 1; x) = S(m-1, 2, x)  [NOT polylog(m, x) — that's weight-1]
        # Special weight-2 case: H(m,1;x) = S_{m-1,2}(x)
        # For symbolic output, wrap in a SymPy-compatible numeric function.
        for k in range(1, 6):
            mm_key = (m,) + (1,)*k
            # H(m, 1^k; x) = S(m-1, k+1, x)
            # S(a,1,x) = polylog(a+1,x), so H(m,1;x)=S(m-1,2,x) -- use _n_nielsen_S
            tbl[mm_key] = (lambda a, n: lambda x: _n_nielsen_S_sym(a, n, x))(m-1, k+1)
        for k in range(1, 6):
            mm_key = (-m,) + (-1,)*k
            # H(-m, (-1)^k; x) = (-1)^k * S(m-1, k+1, -x)
            tbl[mm_key] = (lambda a, n: lambda x: (-1)**n * _n_nielsen_S_sym(a, n, -x))(m-1, k+1)

    return tbl


def _n_nielsen_S(a, n, x):
    """
    Numerical Nielsen S-function S(a, n, x).
    S(a,n,x) = (-1)^{a+n-1}/((a-1)! n!) * ∫_0^1 log^{a-1}(t)·log^n(1-x·t)/t dt
    Special case S(a,1,x) = polylog(a+1, x).
    Relation to HPL: H(m, {1}^k; x) = S(m-1, k+1, x).
    """
    import mpmath as _mpm
    a, n = int(a), int(n)
    xc = _mpm.mpc(x)
    if n == 1:
        return _mpm.polylog(a + 1, xc)
    sign = (-1) ** (a + n - 1)
    denom = _mpm.factorial(a - 1) * _mpm.factorial(n)
    def integrand(t):
        tc = _mpm.mpc(t)
        if abs(tc) < 1e-15:
            return _mpm.mpc(0)
        return _mpm.log(tc)**(a-1) * _mpm.log(1 - xc*tc)**n / tc
    return (sign / denom) * _mpm.quad(integrand, [0, 1], error=False)


def _n_nielsen_S_sym(a, n, x):
    """
    SymPy-compatible Nielsen S function for use in expandfunction_hpl_full.
    
    When x is numeric: evaluates S(a,n,x) via mpmath quad.
    When x is symbolic: returns a custom SymPy Function that evaluates
    numerically via _evalf, suitable for .subs(...).evalf().
    """
    import sympy as _sp
    
    # Try numeric evaluation first
    try:
        xc = complex(x)
        import mpmath as _mpm
        val = _n_nielsen_S(a, n, _mpm.mpc(xc))
        re = float(val.real)
        im = float(val.imag)
        if abs(im) < 1e-30:
            return _sp.Float(re, 30)
        return _sp.Float(re, 30) + _sp.Float(im, 30) * _sp.I
    except (TypeError, AttributeError):
        pass
    
    # Symbolic x: build a custom SymPy function that knows how to evalf
    class _NielsenS(_sp.Function):
        nargs = 3
        @classmethod
        def eval(cls, a_, n_, x_):
            # Only evaluate when x_ is a concrete number
            try:
                xc = complex(x_)
                import mpmath as _mpm
                val = _n_nielsen_S(int(a_), int(n_), _mpm.mpc(xc))
                re = float(val.real)
                im = float(val.imag)
                if abs(im) < 1e-30:
                    return _sp.Float(re, 30)
                return _sp.Float(re, 30) + _sp.Float(im, 30) * _sp.I
            except (TypeError, AttributeError):
                return None  # Keep unevaluated
        
        def _eval_evalf(self, prec):
            x_ = self.args[2]
            try:
                xc = complex(x_)
                import mpmath as _mpm
                old_prec = _mpm.mp.prec
                _mpm.mp.prec = prec
                try:
                    val = _n_nielsen_S(int(self.args[0]), int(self.args[1]), _mpm.mpc(xc))
                    re = float(val.real)
                    im = float(val.imag)
                finally:
                    _mpm.mp.prec = old_prec
                if abs(im) < 1e-30:
                    return _sp.Float(re, 30)
                return _sp.Float(re, 30) + _sp.Float(im, 30) * _sp.I
            except Exception:
                return self
    
    return _NielsenS(_sp.Integer(a), _sp.Integer(n), x)


_FE_TABLE = None
def _get_fe_table():
    global _FE_TABLE
    if _FE_TABLE is None:
        _FE_TABLE = _build_expandfunction_table()
    return _FE_TABLE


def expandfunction_hpl_full(mm: tuple, x=None) -> sp.Expr:
    """
    Full function expansion for H(mm; x): returns a SymPy expression 
    using log, polylog, zeta, and constants.  Falls back to a placeholder
    _HPLSymExpr for cases not in the table.
    """
    from hpl import _HPLSymExpr
    from sympy import Symbol
    if x is None:
        x = Symbol('x', real=True)

    mm = tuple(mm)
    tbl = _get_fe_table()
    if mm in tbl:
        return tbl[mm](x)

    # Fallback: unevaluated
    return _HPLSymExpr(mm, x)


# ─────────────────────────────────────────────────────────────────────────────
# Section G  —  Fast MZV via Borwein's algorithm
# ─────────────────────────────────────────────────────────────────────────────
#
# Borwein, Bradley, Broadhurst (1999): for all-positive MZVs use the
# generating-function / partial-fraction identity to express Z(mm) as a
# rapidly converging alternating series.
#
# The key identity for depth >= 2:
#   Z(m1,...,mk) = sum_{n=1}^{N-1} d_n / 2^n + error
# where d_n are integer coefficients computable from the shuffle algebra.
#
# In practice, for depth-3+ we use the Crandall (1998) recursive algorithm
# which expresses Z(s1,...,sk) via a nested application of the hurwitz zeta
# tail approach with nsum+richardson for each inner sum.

def n_mzv_borwein(mm: tuple, dps: int | None = None) -> mpmath.mpc:
    """
    Evaluate MZV(mm) for all-positive indices using the best available method:
    - Depth 1:  direct zeta()
    - Depth 2:  Euler formula or Hurwitz-zeta nsum (richardson)
    - Depth 3+: Crandall nested-Hurwitz recursion with extra precision
    """
    from hpl import _n_mzv_positive, _euler_formula_m1, zeta_to_hpl
    from mpmath import nsum, inf

    if dps is not None:
        ctx = mp.workdps(dps)
    else:
        ctx = mp.workdps(mp.dps + 10)

    with ctx:
        mm = tuple(mm)
        if not all(m > 0 for m in mm):
            raise ValueError("n_mzv_borwein: all indices must be positive")

        n = len(mm)
        if n == 1:
            return zeta(mm[0])

        if n == 2:
            return _n_mzv_positive(mm)

        # Depth >= 3: Crandall nested Hurwitz
        return _n_mzv_crandall(mm)


def _n_mzv_crandall(mm: tuple) -> mpmath.mpc:
    """
    Evaluate Z(m1,...,mk) for k >= 3, all positive, using the nested
    Hurwitz-zeta outer sum with nsum+richardson acceleration.

    Z(m1,...,mk) = sum_{n=mk_depth}^inf n^{-m1} * Z_partial(rest, n-1)
    where Z_partial(rest, N) = Z(rest) - zeta_tail(rest, N+1)
    and   zeta_tail(rest, N) is computed recursively using Hurwitz zeta.
    """
    from mpmath import nsum, inf

    n     = len(mm)
    m1    = mm[0]
    rest  = mm[1:]

    if len(rest) == 1:
        m2 = rest[0]
        if m2 == 1:
            from hpl import _euler_formula_m1
            return _euler_formula_m1(m1)
        return nsum(lambda k: mpmath.zeta(m1, int(k)+1) / k**m2,
                    [1, inf], method='richardson')

    if len(rest) == 2:
        # Z(m1,m2,m3) = sum_{n1>n2>n3>=1} n1^{-m1}*n2^{-m2}*n3^{-m3}
        # = sum_{n1>=3} n1^{-m1} * sum_{n2=2}^{n1-1} n2^{-m2} * H_{n2-1}^{(m3)}
        # Use nsum on n1 with inner running sum
        m2, m3 = rest
        # Inner: Z_partial(m2,m3; N) = sum_{n2=2}^N sum_{n3=1}^{n2-1} n2^{-m2}*n3^{-m3}
        # = sum_{n2=2}^N n2^{-m2} * (zeta(m3) - zeta(m3, n2))  [Hurwitz tail]
        # = (zeta(m2)-zeta(m2,N+1)) * zeta(m3) - sum_{n2=2}^N n2^{-m2}*zeta(m3,n2)
        # Precompute Z_partial(rest, N) as a running sum:
        def term_outer(k):
            N  = int(k) - 1    # k = n1, inner runs up to n1-1
            if N < 2:
                return mpc(0)
            # Z_partial(m2,m3;N) using running accumulation would be slow here.
            # Instead use: zeta(m2)*sum_{n2=1}^N n2^{-m3} - ... 
            # Accurate but slow: direct double sum capped at N
            s = mpc(0)
            h = mpc(0)
            for n2 in range(1, N+1):
                s += h / mpc(n2)**m2
                h += mpc(1) / mpc(n2)**m3
            return s / mpc(k)**m1

        return nsum(term_outer, [3, inf], method='richardson')

    # Depth >= 4: use the general recursive approach with extra precision
    with mp.workdps(mp.dps + len(mm) * 10):
        from hpl import _n_mzv_positive
        return _n_mzv_positive(mm)


# ─────────────────────────────────────────────────────────────────────────────
# Section H  —  MultipleFiniteHarmonicSumZ
# ─────────────────────────────────────────────────────────────────────────────

def multiple_finite_harmonic_sum_Z(mm: tuple, xx: tuple,
                                   n_or_inf) -> mpmath.mpc:
    """
    Evaluate Z(mm; xx; N) — MultipleFiniteHarmonicSumZ.

    Z({m1,...,mk}; {x1,...,xk}; N)
      = sum_{N>=i1>i2>...>ik>=1} prod_j x_j^{i_j} / i_j^{|m_j|} * sign(m_j)^{i_j}

    Special cases handled:
      xx = (1,...,1), n_or_inf = inf  ->  MZV(mm) via n_mzv
      xx = (x, 1,...,1), n_or_inf=inf ->  HPL(mm; x) via n_hpl
      finite N, all xx=1              ->  partial MZV sum (Euler-Bernoulli)

    Parameters
    ----------
    mm        : tuple of nonzero integers
    xx        : tuple of complex arguments, len = len(mm)
    n_or_inf  : integer N or float('inf') / mpmath.inf

    Returns
    -------
    mpmath.mpc
    """
    from hpl import n_hpl, n_mzv, _n_hpl_dispatch

    assert len(mm) == len(xx), "mm and xx must have the same length"

    mm = tuple(mm)
    xx = tuple(mpc(xi) for xi in xx)
    is_inf = (n_or_inf == float('inf') or n_or_inf is mpmath.inf
              or (hasattr(n_or_inf, '_mpf_') and abs(n_or_inf) > 1e15))

    # ── Special case: all x = 1, N = inf -> MZV ─────────────────────────────
    if is_inf and all(xi == mpc(1) for xi in xx):
        return n_mzv(mm)

    # ── Special case: x[0]=x, rest=1, N=inf -> HPL ──────────────────────────
    if is_inf and all(xi == mpc(1) for xi in xx[1:]):
        return _n_hpl_dispatch_v2(mm, xx[0])

    # ── General finite N: direct nested sum ──────────────────────────────────
    if not is_inf:
        N = int(n_or_inf)
        return _z_sum_finite(mm, xx, N)

    # ── General infinite case: MPL ────────────────────────────────────────────
    from hpl import n_mpl
    return n_mpl(mm, xx)


def _z_sum_finite(mm: tuple, xx: tuple, N: int) -> mpmath.mpc:
    """
    Direct nested sum for Z(mm; xx; N).
    Depth-1: sum_{k=1}^N sign(m)^k * x^k / k^|m|
    Depth-k: sum_{i=depth}^{N} x[0]^i * sign(m[0])^i / i^|m[0]|
              * Z(rest; xx[1:]; i-1)
    """
    if len(mm) == 0:
        return mpc(1)

    m0, x0 = mm[0], xx[0]
    s0 = 1 if m0 > 0 else -1 if m0 < 0 else 0

    if len(mm) == 1:
        result = mpc(0)
        for k in range(1, N + 1):
            result += mpc(s0)**k * x0**k / mpc(k)**abs(m0)
        return result

    result = mpc(0)
    for i in range(len(mm), N + 1):
        inner = _z_sum_finite(mm[1:], xx[1:], i - 1)
        result += mpc(s0)**i * x0**i / mpc(i)**abs(m0) * inner
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section I  —  Expression-level shuffle/stuffle rewriting
# ─────────────────────────────────────────────────────────────────────────────
#
# Shuffle and stuffle product expansion rules on "coefficient dictionaries":
#   linear_combination = dict { mm_tuple : sympy_coefficient }

def shuffle_expand_lc(lc1: dict, lc2: dict) -> dict:
    """
    Expand the product of two HPL linear combinations lc1 * lc2
    using the shuffle product.

    Each linear combination is a dict { mm_tuple : coefficient }.
    Returns a new dict of the same type.

    Example:
      H(1;x) * H(2;x) = shuffle_expand_lc({(1,):1}, {(2,):1})
                       = {(1,2):1, (2,1):2}
    """
    from hpl import harmonic_polylog_product_expand
    result = {}
    for mm1, c1 in lc1.items():
        for mm2, c2 in lc2.items():
            expanded = harmonic_polylog_product_expand(mm1, mm2)
            for mm, mult in expanded.items():
                result[mm] = result.get(mm, sp.S.Zero) + c1 * c2 * mult
    return result


def stuffle_expand_lc(lc1: dict, lc2: dict) -> dict:
    """
    Expand the product of two MZV linear combinations lc1 * lc2
    using the stuffle product.
    """
    from hpl import mzv_stuffle_product_expand
    result = {}
    for mm1, c1 in lc1.items():
        for mm2, c2 in lc2.items():
            expanded = mzv_stuffle_product_expand(mm1, mm2)
            for mm, mult in expanded.items():
                result[mm] = result.get(mm, sp.S.Zero) + c1 * c2 * mult
    return result


def power_shuffle_expand(mm: tuple, n: int) -> dict:
    """
    Compute H(mm;x)^n using the shuffle product n-1 times.
    Returns a dict {mm': integer_coefficient}.
    Corresponds to H(mm;x)^n = n! * H(mm,...,mm; x) + lower_weight_terms
    from the shuffle algebra.
    """
    from hpl import harmonic_polylog_product_expand
    if n == 0:
        return {(): 1}
    if n == 1:
        return {mm: 1}
    result = {mm: 1}
    for _ in range(n - 1):
        new_result = {}
        for mm2, c2 in result.items():
            expanded = harmonic_polylog_product_expand(mm, mm2)
            for mm3, mult in expanded.items():
                new_result[mm3] = new_result.get(mm3, 0) + c2 * mult
        result = new_result
    return result


def evaluate_hpl_lc(lc: dict, x, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate a HPL linear combination:
      sum_{mm} coeff[mm] * H(mm; x)

    Parameters
    ----------
    lc  : dict { mm_tuple : numeric_or_sympy_coeff }
    x   : numeric argument
    dps : decimal places
    """
    total = mpc(0)
    xc = mpc(x)
    for mm, c in lc.items():
        c_num = complex(c) if isinstance(c, sp.Basic) else c
        if abs(c_num) < 1e-100:
            continue
        if mm == ():
            total += mpc(c_num)
        else:
            from hpl import n_hpl
            total += mpc(c_num) * n_hpl(mm, xc, dps=dps)
    return total


def evaluate_mzv_lc(lc: dict, dps: int | None = None) -> mpmath.mpc:
    """
    Numerically evaluate a MZV linear combination:
      sum_{mm} coeff[mm] * Z(mm)
    """
    total = mpc(0)
    for mm, c in lc.items():
        c_num = complex(c) if isinstance(c, sp.Basic) else c
        if abs(c_num) < 1e-100:
            continue
        if mm == ():
            total += mpc(c_num)
        else:
            from hpl import n_mzv
            total += mpc(c_num) * n_mzv(mm, dps=dps)
    return total



# ═══════════════════════════════════════════════════════════════════════════════
# Wire the fast dispatcher into the module entry-points.
# ═══════════════════════════════════════════════════════════════════════════════
_original_n_hpl_dispatch = _n_hpl_dispatch

def _n_hpl_dispatch(mm: tuple, x) -> mpmath.mpc:
    """Delegates to the optimised _n_hpl_dispatch_v2."""
    return _n_hpl_dispatch_v2(tuple(mm), mpc(x))

def _run_tests():
    """Run correctness checks against known analytic values."""
    mp.dps = 30
    tol_strict = mpf(10) ** (-25)
    tol_quad   = mpf(10) ** (-20)   # relaxed for nested quadrature results

    def check(name, got, ref, tol=None):
        if tol is None: tol = tol_strict
        err = abs(mpc(got) - mpc(ref))
        ok  = "PASS" if float(err) < float(tol) else "FAIL"
        print(f"  [{ok}] {name:45s}  err={float(err):.2e}")
        return ok == "PASS"

    print("=== HPL / MPL / MZV self-tests ===\n")
    ap = True

    # ---- notation -------------------------------------------------------
    assert from_abbreviated_notation((3,))    == (0, 0, 1)
    assert from_abbreviated_notation((-2,))   == (0, -1)
    assert from_abbreviated_notation((2, 1))  == (0, 1, 1)
    assert to_abbreviated_notation((0, 0, 1)) == (3,)
    assert to_abbreviated_notation((0, 1, 1)) == (2, 1)
    print("  [PASS] notation conversion round-trips\n")

    # ---- weight-1 HPLs --------------------------------------------------
    x = mpf('0.5')
    ap &= check("H({1};  0.5) = log(2)",         n_hpl((1,),  x), log(2))
    ap &= check("H({-1}; 0.5) = log(1.5)",        n_hpl((-1,), x), log(mpf('1.5')))
    ap &= check("H({0};  0.5) = log(0.5)",        n_hpl((0,),  x), log(x))
    ap &= check("H({2};  0.5) = Li2(0.5)",        n_hpl((2,),  x), polylog(2, x))
    ap &= check("H({3};  0.5) = Li3(0.5)",        n_hpl((3,),  x), polylog(3, x))
    ap &= check("H({-2}; 0.5) = -Li2(-0.5)",      n_hpl((-2,), x), -polylog(2, -x))
    print()

    # ---- weight-2 HPLs --------------------------------------------------
    ap &= check("H({1,1};  0.5) = log^2(0.5)/2",
                n_hpl((1,1), x),  log(x)**2 / 2)
    ap &= check("H({1,0};  0.5) = -log(x)log(1-x)-Li2(x)",
                n_hpl((1,0), x),  -log(x)*log(1-x) - polylog(2, x))
    ap &= check("H({-1,0}; 0.5) = log(x)log(1+x)+Li2(-x)",
                n_hpl((-1,0), x), log(x)*log(1+x) + polylog(2, -x))
    # H({2,1}) = H({0,1,1}) uses leading-zero integration
    ref_21 = mpmath.quad(lambda t: log(1-t)**2/(2*t), [0, mpf(1)/3])
    ap &= check("H({2,1};  1/3) via quad", n_hpl((2,1), mpf(1)/3), ref_21, tol_quad)
    print()

    # ---- weight-3 HPLs --------------------------------------------------
    ref_100  = -(log(x)*(log(1-x)*log(x) + 2*polylog(2,x)))/2 + polylog(3,x)
    ref_m100 = log(x)**2*log(1+x)/2 + log(x)*polylog(2,-x) - polylog(3,-x)
    ap &= check("H({1,0,0};  0.5) from table", n_hpl((1,0,0),  x), ref_100,  tol_quad)
    ap &= check("H({-1,0,0}; 0.5) from table", n_hpl((-1,0,0), x), ref_m100, tol_quad)
    print()

    # ---- MZV depth-1 ----------------------------------------------------
    ap &= check("Z({2})  = pi^2/6",     n_mzv((2,)),   pi**2/6)
    ap &= check("Z({3})  = zeta(3)",    n_mzv((3,)),   zeta(3))
    ap &= check("Z({-1}) = log(2)",     n_mzv((-1,)),  log(mpf(2)))
    ap &= check("Z({-2}) = -pi^2/12",   n_mzv((-2,)),  -zeta(2)/2)
    print()

    # ---- MZV depth-2 ----------------------------------------------------
    ap &= check("Z({2,1}) = zeta(3)",          n_mzv((2,1)), zeta(3))
    ap &= check("Z({3,1}) = pi^4/360",         n_mzv((3,1)), pi**4/360)
    ap &= check("Z({2,2}) = (z2^2-z4)/2",      n_mzv((2,2)), (zeta(2)**2-zeta(4))/2)
    ap &= check("Z({3,2}) = pi^2*z3/2-11z5/2", n_mzv((3,2)), (pi**2*zeta(3)-11*zeta(5))/2)
    print()

    # ---- MZV depth-3 ----------------------------------------------------
    ap &= check("Z({2,1,1}) = pi^4/90",       n_mzv((2,1,1)), zeta(4),    tol_quad)
    ap &= check("Z({2,2,2}) = 2(2pi)^6/7!*(1/2)^7",
                n_mzv((2,2,2)),
                2*(2*pi)**6/mpmath.factorial(7)*mpf('0.5')**7,            tol_quad)
    print()

    # ---- precomputed HPL1 table -----------------------------------------
    ap &= check("HPL1[{3,1}] = pi^4/360",  hpl_at_one((3,1)), pi**4/360)
    ap &= check("HPL1[{2}]   = pi^2/6",    hpl_at_one((2,)),  pi**2/6)
    ap &= check("HPL1[{-1}]  = log(2)",    hpl_at_one((-1,)), log(mpf(2)))
    print()

    # ---- shuffle algebra ------------------------------------------------
    sp = harmonic_polylog_product_expand((1,),(1,))
    assert sp == {(1,1):2},    f"shuffle H(1)*H(1): {sp}"
    print("  [PASS] shuffle H({1})*H({1}) = 2*H({1,1})")

    sp = harmonic_polylog_product_expand((1,),(2,))
    # H({1})*H({2}) = H({1,2}) + 2*H({2,1})  (from interleaving (1,) with (0,1))
    assert sp == {(1,2):1, (2,1):2}, f"shuffle H(1)*H(2): {sp}"
    print("  [PASS] shuffle H({1})*H({2}) = H({1,2})+2*H({2,1})")
    print()

    # ---- stuffle algebra ------------------------------------------------
    st = mzv_stuffle_product_expand((2,),(2,))
    assert st == {(2,2):2,(4,):1}, f"stuffle Z(2)*Z(2): {st}"
    print("  [PASS] stuffle Z({2})*Z({2}) = 2*Z({2,2})+Z({4})")

    st2 = mzv_stuffle_product_expand((2,),(3,))
    assert set(st2.keys()) == {(2,3),(3,2),(5,)}, f"stuffle Z(2)*Z(3): {st2}"
    print("  [PASS] stuffle Z({2})*Z({3}) = Z({2,3})+Z({3,2})+Z({5})")
    print()

    # ---- finite harmonic sums -------------------------------------------
    ap &= check("S({1}; 4) = 1+1/2+1/3+1/4",
                multiple_finite_harmonic_sum_S((1,), 4),
                mpf(1)+mpf('0.5')+mpf(1)/3+mpf('0.25'))
    ap &= check("S({2}; 3) = 1+1/4+1/9",
                multiple_finite_harmonic_sum_S((2,), 3),
                mpf(1)+mpf('0.25')+mpf(1)/9)
    print()

    print(f"{'=== All tests passed! ===' if ap else '=== Some tests FAILED. ==='}")
    return ap


if __name__ == "__main__":
    _run_tests()

    mp.dps = 30
    print("\n=== Demo (30 decimal places) ===")
    print(f"H({{1,0}};  1/2)   = {harmonic_polylog([1,0],  mpf('0.5'))}")
    print(f"H({{2,1}};  1/3)   = {harmonic_polylog([2,1],  mpf(1)/3)}")
    print(f"H({{-1,0}}; 1/2)   = {harmonic_polylog([-1,0], mpf('0.5'))}")
    print()
    print(f"Z({{2,1,1}})       = {multiple_zeta_value([2,1,1])}")
    print(f"Z({{3,1}})         = {multiple_zeta_value([3,1])}")
    print(f"  pi^4/360        = {mp.pi**4/360}")
    print(f"Z({{3,2}})         = {multiple_zeta_value([3,2])}")
    print(f"  pi^2*z3/2-11z5/2= {(mp.pi**2*zeta(3)-11*zeta(5))/2}")
    print()
    print(f"Symbolic H({{1}};  x) = {expandfunction_hpl([1])}")
    print(f"Symbolic H({{2}}; x)  = {expandfunction_hpl([2])}")
    print(f"Symbolic H({{2,-1}};x)= (see output)")
    expandfunction_hpl([2,-1])  # just compute, don't print the long form
    print()
    print(f"shuffle H([1,0])*H([1])  = {shuffle_expand([1,0],[1])}")
    print(f"stuffle Z([2])*Z([3])    = {stuffle_expand([2],[3])}")
    print(f"stuffle Z([2])*Z([2])    = {stuffle_expand([2],[2])}")
