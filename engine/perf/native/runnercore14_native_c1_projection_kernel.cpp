#include <cstdint>

extern "C" int runnercore14_c1_project_rows(
    const int64_t n,
    const double* float_cols,
    const int64_t float_col_count,
    const int32_t* int_cols,
    const int64_t int_col_count,
    double* out_float_cols,
    int32_t* out_int_cols,
    double* out_checksum
) {
    if (n < 0 || float_col_count < 0 || int_col_count < 0) {
        return 1;
    }
    if (!float_cols || !int_cols || !out_float_cols || !out_int_cols || !out_checksum) {
        return 2;
    }

    double checksum = 0.0;
    for (int64_t row = 0; row < n; ++row) {
        const int64_t fbase = row * float_col_count;
        for (int64_t col = 0; col < float_col_count; ++col) {
            const double value = float_cols[fbase + col];
            out_float_cols[fbase + col] = value;
            checksum += value * static_cast<double>(col + 1);
        }

        const int64_t ibase = row * int_col_count;
        for (int64_t col = 0; col < int_col_count; ++col) {
            const int32_t value = int_cols[ibase + col];
            out_int_cols[ibase + col] = value;
            checksum += static_cast<double>(value) * static_cast<double>(101 + col);
        }
    }

    *out_checksum = checksum;
    return 0;
}
