# Provisional Frameshift Rates

`tools/get_indel_rates.py` extracts 1-base and 2-base insertion/deletion rates from a Badread error model.

## Current approximation

The following values are provisional:

```cpp
const Float INSERT1 = 0.0171;
const Float INSERT2 = 0.0018;
const Float DELETE1 = 0.0328;
const Float DELETE2 = 0.0083;
```

They come from the Badread `nanopore2020` model at **90% target identity**:

```text
1-insertion: 1.709757% per reference base -> 0.0171
2-insertion: 0.175268% per reference base -> 0.0018
1-deletion:  3.279012% per reference base -> 0.0328
2-deletion:  0.828603% per reference base -> 0.0083
```

Run the extractor with an explicit Badread checkout and model:

```bash
BADREAD_ROOT=/path/to/Badread
python3 tools/get_indel_rates.py \
  --badread-root "$BADREAD_ROOT" \
  --model "$BADREAD_ROOT/badread/error_models/nanopore2020.gz" \
  --identity 90
```

If Badread is installed as a Python package, `--badread-root` and `--model` can be omitted.  Alternatively, set `BADREAD_ROOT` in the environment.

The script estimates empirical event proportions from the k-mer alternatives, then scales those proportions by the requested overall error rate.  Its output is an occurrence rate per reference base, not a probability for one HMM transition.

This is intentionally a rough approximation for now. It is hypothesized that true rates are higher and current E-values are conservatively lower without the true rates. Future work will focus on ability to estimate frameshift transitions in the HMM construction process.
