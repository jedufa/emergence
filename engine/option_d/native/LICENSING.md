# Option D native licensing

`glibc_2_43_fma_math.cuh` is a bounded CUDA adaptation of GNU C Library
2.43 FMA implementations and tables. It is licensed under
LGPL-2.1-or-later, as stated in its SPDX header. The adaptation was made on
2026-08-18 and is intentionally limited to the argument domains required by
the standalone Option D V1 Gaussian contract.

The complete LGPL 2.1 license is available from the
[GNU license archive](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
and, on Debian/Ubuntu systems, at `/usr/share/common-licenses/LGPL-2.1`.

The source-file SHA-256 bindings used to create the adaptation are recorded
in the header itself and in the D3 qualification artifact. The surrounding
Option D implementation is independently authored; this notice does not
relicense unrelated repository code.

The complete license and original file notices are included in
`LICENSES/LGPL-2.1.txt` and `LICENSES/glibc-NOTICES.txt`. The original
IBM Accurate Mathematical Library attribution is retained there. All five
upstream file hashes in the adapter header were independently matched against
the GNU glibc 2.43 source archive during public-source preparation.
