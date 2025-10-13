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

def main(
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

if __name__ == "__main__":
    main()
