"""Generate caller.cpp for a dynamic multicore rowsum or colsum kernel (fp32).

Usage: python caller.py --mode row|col
"""

def generate_caller(mode):
    return f"""\
#include "{mode}sum.cpp"

extern "C" void call_{mode}sum(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *y,
    uint32_t batch,
    uint32_t n_cols)
{{
    _kernel<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<float *>(x),
        reinterpret_cast<float *>(y),
        static_cast<int32_t>(batch),
        static_cast<int32_t>(n_cols));
}}
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["row", "col"], required=True)
    args = parser.parse_args()
    print(generate_caller(args.mode))
