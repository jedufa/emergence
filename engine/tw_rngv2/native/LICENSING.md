# Native RNG and mathematical source attribution

The Philox4x32-10 constants and permutation derive from D. E. Shaw Research
Random123 at commit `9545ff6413f258be2f04c1d319d99aaef7521150`.
The implementation and known-answer vectors retain the BSD terms in
[LICENSES/Random123.txt](../../../LICENSES/Random123.txt).
This applies to the Python implementation as well as the native source.

TWDMATH log/sin/cos coefficients and operation order derive from fdlibm,
including `e_log.c`, `k_sin.c`, and `k_cos.c`. They are covered by the Sun
Microsystems permission notice in [LICENSES/fdlibm.txt](../../../LICENSES/fdlibm.txt).
They are not described as public domain. The frozen arithmetic is a local
implementation and does not dynamically bind glibc or libdevice.

These upstream terms remain applicable independently of the first-party license.
