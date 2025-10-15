import io
import importlib.resources as pkg_resources
import pandas as pd
from io import StringIO
from pathlib import Path
from sar_sampling import SarSampler

def _len_to_domain(l: int, lb: int, ub: int, step: int) -> list[int]:
    """Convert array length to a symmetric domain range with bounds and step size."""
    l = min(max(l, lb), ub) // (2 * step) * (2 * step)
    return list(range(-l//2, l//2+1, step))

def _clamp_range(fmin: int, fmax: int, lb: int, ub: int) -> tuple[int, int]:
    """Clamp frequency range to bounds and validate min < max."""
    fmin = min(max(fmin, lb), ub)
    fmax = min(max(fmax, lb), ub)
    if fmin >= fmax:
        raise ValueError(f"invalid frequency range: [{fmin}, {fmax}]")
    return fmin, fmax

def _main(
    n: int | float | str = '225', 
    w: int | float | str = '80',
    l: int | float | str = '160',
    fmin: int | float | str = '4',
    fmax: int | float | str = '10000',
    output_dir: str = '.',
    output_file: str = 'sarsample.csv',
) -> None:
    """
    Generate a SAR sampling dataset with configurable parameters.

    This function creates a Latin Hypercube sample for SAR testing scenarios using
    configurable dimensions, frequency ranges, and output settings. It loads input
    data from package resources, validates parameters, builds a sampling configuration,
    and saves the generated sample to a CSV file.

    Args:
        n: Number of samples to generate (default: 225)
        w: Width dimension for x-domain (default: 80, clamped to [40, 400])
        l: Length dimension for y-domain (default: 160, clamped to [40, 600])
        fmin: Minimum frequency in MHz (default: 4, clamped to [4, 10000])
        fmax: Maximum frequency in MHz (default: 10000, clamped to [4, 10000])
        output_dir: Output directory path (default: current directory)
        output_file: Output filename (default: 'sarsample.csv')

    Returns:
        None

    Raises:
        ValueError: If frequency range is invalid (fmin >= fmax)
        Exception: Any other errors during sampling or file operations
    """
    try:
        n = int(n)
        w = int(w)
        l = int(l)
        fmin = int(fmin)
        fmax = int(fmax)

        # validate and build config
        fmin, fmax = _clamp_range(fmin, fmax, 4, 10000)
        x_domain = _len_to_domain(w, 40, 400, 10)
        y_domain = _len_to_domain(l, 40, 600, 10)
        config = {
            'x_domain': x_domain,
            'y_domain': y_domain,
            'input_frequency_min': fmin,
            'input_frequency_max': fmax,
        }

        # build and run sampler
        csv_content = pkg_resources.read_text('sar_sampling.data', 'input_table.csv')
        input_df = pd.read_csv(StringIO(csv_content))
        sampler = SarSampler(input_df, config=config)
        sample = sampler(n_samples=n, method='maximin')

        # save sample
        output_path = Path(output_dir) / output_file
        sample.to_csv(str(output_path))

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def log(msg: str):
    """Add message to the log buffer."""
    if not hasattr(log, "buffer"):
        log.buffer = io.StringIO()
    log.buffer.write(msg + '\n')

def flush_log(filename: str | None):
    """Flush the log buffer to a file (if filename provided) and clear the buffer."""
    if filename is not None:
        with open(filename, "a") as f:
            f.write(log.buffer.getvalue())
    log.buffer.seek(0)
    log.buffer.truncate(0)

def main(*args, **kwargs):
    """
    Command-line interface for SAR sampling dataset generation.
    
    This function provides a CLI wrapper around the core sampling functionality,
    handling argument parsing, debug logging, and calling the main sampling function.
    
    Args:
        *args: Command-line arguments (typically sys.argv[1:])
        **kwargs: Additional keyword arguments (for programmatic usage)
    
    Returns:
        None
        
    CLI Arguments:
        --n: Number of samples to generate (default: 225)
        --w: Width dimension for x-domain (default: 80)
        --l: Length dimension for y-domain (default: 160)
        --fmin: Minimum frequency in MHz (default: 4)
        --fmax: Maximum frequency in MHz (default: 10000)
        --output_dir: Output directory path (default: current directory)
        --output_file: Output filename (default: sarsample.csv)
        --log_file: Debug log file path (default: None, no logging to file)
        
    Side Effects:
        - Creates a CSV file with the generated sample
        - Optionally creates a debug log file if --log_file is specified
        - Prints error messages to stdout if exceptions occur
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a SAR sampling dataset using Latin Hypercube Sampling."
    )
    parser.add_argument("--n", type=int, default=225)
    parser.add_argument("--w", type=int, default=80)
    parser.add_argument("--l", type=int, default=160)
    parser.add_argument("--fmin", type=int, default=4)
    parser.add_argument("--fmax", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--output_file", type=str, default="sarsample.csv")
    parser.add_argument("--log_file", type=str, default=None)

    # Debug logging (not saved if log_file is None)
    import sys
    log("=== SAR sampling debug log ===")
    log(f"DEBUG: sys.argv: {sys.argv}")
    log(f"DEBUG: main args: {args}")
    log(f"DEBUG: main kwargs: {kwargs}")

    # 1) If kwargs provided explicitly, use them (allow override)
    if kwargs:
        parsed_kwargs = kwargs
    # 2) If positional args passed, parse them with argparse
    elif args:
        # args might be a tuple of strings -- convert to list of str
        arg_list = [str(a) for a in args]
        parsed = parser.parse_args(arg_list)
        parsed_kwargs = vars(parsed)
    # 3) No args/kwargs -> parse from sys.argv as usual
    else:
        parsed = parser.parse_args()
        parsed_kwargs = vars(parsed)

    log(f"CLI kwargs parsed: {parsed_kwargs}")

    main_kwargs = {k: v for k, v in parsed_kwargs.items() if k != "log_file"}
    log(f"main kwargs: {main_kwargs}")
    
    flush_log(parsed_kwargs.get("log_file", None))
    _main(**main_kwargs)

if __name__ == "__main__":
    main()
