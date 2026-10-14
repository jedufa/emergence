# Third-party notices

The first-party license does not replace the following upstream terms.
This distribution ships research source, including optional native source;
it does not ship compiled native extensions, a CUDA toolkit, models or datasets.

| Included material | Upstream and applicable terms | Notice |
|---|---|---|
| Philox4x32-10 constants, permutation and known-answer vectors in `engine/tw_rngv2` | D. E. Shaw Research Random123, commit `9545ff6413f258be2f04c1d319d99aaef7521150`; BSD three-clause terms | [Full notice](LICENSES/Random123.txt) |
| TWDMATH log/sin/cos algorithms and coefficients in `engine/tw_rngv2` | fdlibm `e_log.c`, `k_sin.c`, `k_cos.c` and argument reduction; Sun Microsystems permission notice | [Full notice](LICENSES/fdlibm.txt) |
| `engine/option_d/native/glibc_2_43_fma_math.cuh` | Bounded CUDA adaptation of GNU glibc 2.43; LGPL-2.1-or-later; original IBM mathematical-library attribution retained | [License](LICENSES/LGPL-2.1.txt), [upstream notices](LICENSES/glibc-NOTICES.txt), [adaptation details](engine/option_d/native/LICENSING.md) |

The glibc adaptation retains its original file hashes and modification description.
All five hashes were matched to the GNU 2.43 source archive. Source redistribution
must retain its notices and LGPL terms. If you redistribute compiled products
using this optional material, assess the applicable LGPL linking, corresponding
source and relinking obligations for that product; this source release does not
supply a blanket binary redistribution clearance.

Random123 source and binary redistribution conditions, including attribution and
non-endorsement, remain applicable. fdlibm's notice-preservation condition remains
applicable; it is not described here as public domain.

Python dependencies are installed separately under their own licenses: NumPy,
Numba/llvmlite, PyTorch, PyYAML, jsonschema, pandas, psutil, packaging, and the
build/test tools. Optional GPU dependencies and device toolchains are not bundled.
See `pyproject.toml` for dependency declarations and `requirements-cpu.txt` for the
verified CPU environment. No upstream project endorses Emergence.

Primary sources:
- https://github.com/DEShawResearch/random123/tree/9545ff6413f258be2f04c1d319d99aaef7521150
- https://www.netlib.org/fdlibm/
- https://ftp.gnu.org/gnu/glibc/glibc-2.43.tar.xz
- https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html
