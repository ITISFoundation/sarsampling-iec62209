import pandas as pd
import tempfile
from pathlib import Path
from sar_sampling.sarsampler import SarSampler

def input_table() -> pd.DataFrame:
    """Return a reference to the input table."""
    input_dict = {
        'antenna': [
            'C6', 'C13', 'C30', 'C64', 'C128', 'C150', 'C220', 'M300', 'M450', 'D750',
            'D835', 'D900', 'D1450', 'D1500', 'D1640', 'D1750', 'D1800', 'D1900', 'D1950', 'D2000',
            'D2100', 'D2300', 'D2450', 'D2600', 'D3000', 'D3500', 'D3700', 'D4200', 'D4600', 'D5000',
            'D5200', 'D5500', 'D5600', 'D5800', 'D7000', 'D9000', 'P6500', 'V750-V1', 'V835-V1', 'V1950-V1',
            'V3700-V1', 'V300-V2', 'V450-V2', 'V750-V2', 'V835-V2', 'V1950-V2'
        ],
        'frequency': [
            6, 13, 30, 64, 128, 150, 220, 300, 450, 750,
            835, 900, 1450, 1500, 1640, 1750, 1800, 1900, 1950, 2000,
            2100, 2300, 2450, 2600, 3000, 3500, 3700, 4200, 4600, 5000,
            5200, 5500, 5600, 5800, 7000, 9000, 6500, 750, 835, 1950,
            3700, 3000, 450, 750, 835, 1950
        ],
        '2mm': [
            14.4, 14.4, 14.4, 14.2, 15.8, 16.8, 16.2, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None
        ],
        '5mm': [
            None, None, None, None, None, None, None, None, None, None,
            4.6, 4.7, 5.4, 5.2, 5.2, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 4.9, 4.7, 4.9, 4.5, 4.6, 4.6, 4.6, 4.6, 4.1,
            4.1, 4.1, 4.1, 4.1, 4.0, 4.1, 5.2, 5.1, 5.1, 5.1,
            9.6, 2.6, 2.6, 2.6, 2.6, 2.5
        ],
        '10mm': [
            None, None, None, None, None, None, None, None, None, 15.4,
            15.0, 14.6, 9.4, 9.5, 9.2, 9.0, 8.9, 8.7, 8.7, 8.6,
            8.5, 8.1, 8.1, 8.1, 7.7, 7.8, 7.7, 7.6, 7.7, 7.0,
            7.0, 7.1, 7.0, 7.2, None, None, None, None, None, None,
            None, None, None, None, None, None
        ],
        '15mm': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 15.4,
            15.0, 14.6, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        '21mm': [
            0, 0, 0, 0, 0, 0, 0, 20.4, 20.6, 21.5,
            20.8, 20.4, 18.3, 18.1, 17.9, 17.7, 17.7, 17.6, 17.7, 17.4,
            17.3, 17.4, 17.4, 17.4, 17.1, 16.4, 15.8, 15.7, 15.2, 13.9,
            13.9, 13.9, 13.9, 14.0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        '25mm': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        'm1': [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1
        ],
        'm2': [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1
        ],
        'm3': [
            None, None, None, None, None, None, None, None, None, 1,
            None, None, None, None, None, None, None, None, None, None,
            None, 1, None, None, None, None, None, None, None, 1,
            1, 1, 1, 1, 1, 1, 1, None, None, None,
            None, None, None, None, None, None
        ],
        'M4': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, 1, None, 1, 1, 1, 1, None, 1, None,
            None, None, None, None, None, None, None, None, None, None,
            1, None, None, None, None, None
        ],
        'M5': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, 1, 1, 1, 1, None, None, None, None,
            None, None, 1, None, None, 1, 1, 1, 1, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None
        ],
        'M6': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, 1, 1, 1, None, None, 1, 1, None,
            None, None, None, None, None, None, None, None, None, None,
            1, None, None, None, None, None
        ],
        'M7': [
            None, None, None, None, None, None, None, 1, 1, None,
            None, None, 1, 1, 1, 1, None, None, None, None,
            None, 1, None, None, None, 1, 1, 1, 1, None,
            None, None, None, None, None, None, None, None, None, None,
            1, None, None, None, None, None
        ],
        'M8': [
            None, None, None, None, None, None, None, 1, 1, None,
            1, 1, None, None, None, None, 1, 1, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, 1, None,
            None, 1, 1, None, 1, None
        ],
        'M9': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, 1, 1,
            1, None, 1, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            1, None, None, None, None, None
        ],
        'M10': [
            None, None, None, None, None, None, None, 1, 1, None,
            1, 1, 1, 1, 1, None, 1, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, 1, None,
            None, 1, 1, 1, 1, None
        ],
        'M11': [
            None, None, None, None, None, None, None, 1, 1, 1,
            1, 1, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, 1, None,
            None, 1, 1, 1, 1, None
        ],
        'M12': [
            None, None, None, None, None, None, None, None, None, None,
            1, None, None, None, None, 1, 1, 1, 1, 1,
            1, None, None, 1, 1, None, None, None, None, None,
            None, None, None, None, None, None, None, None, 1, 1,
            None, None, None, None, 1, 1
        ],
        'M13': [
            None, None, None, None, None, None, None, None, None, 1,
            1, None, None, None, None, None, None, None, 1, 1,
            1, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, None, 1,
            None, None, None, 1, None, None
        ],
        'M14': [
            None, None, None, None, None, None, None, 1, 1, None,
            None, 1, 1, 1, 1, None, None, None, 1, 1,
            1, 1, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, None, 1,
            None, 1, 1, 1, None, None
        ],
        'M15': [
            None, None, None, None, None, None, None, 1, 1, None,
            None, 1, None, None, None, None, None, None, 1, 1,
            1, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, 1, 1, 1,
            None, 1, 1, 1, 1, 1
        ],
        'M16': [
            None, None, None, None, None, None, None, None, None, 1,
            1, None, 1, 1, 1, 1, 1, 1, None, None,
            None, 1, None, 1, 1, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, 1,
            None, None, None, None, None, 1
        ],
        'M17': [
            None, None, None, None, None, None, None, None, None, 1,
            None, None, 1, 1, 1, 1, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None
        ],
        'M18': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, 1,
            1, 1, 1, 1, 1, 1, 1, None, None, None,
            None, None, None, None, None, None
        ],
        'M19': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, 1,
            1, 1, 1, 1, 1, 1, 1, None, None, None,
            None, None, None, None, None, None
        ],
        'M20': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        'M21': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        'M22': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        'M23': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0
        ],
        'M24': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0
        ]
    }
    return pd.DataFrame(input_dict)

def test_read_input_table():
    """Test the _read_input_table static method with various CSV formats and error handling."""
    # basic case
    csv_content = "antenna,frequency,10mm,M1\nD750,750,15.4,1\nD835,835,15.0,0"
    with tempfile.TemporaryDirectory() as tempdir:
        csv_path = (Path(tempdir) / "test.csv").as_posix()
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        df = SarSampler._read_input_table(csv_path)
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert list(df.columns) == ['antenna', 'frequency', '10mm', 'M1'], f"Unexpected columns: {list(df.columns)}"
        assert df.iloc[0]['antenna'] == 'D750', f"Expected 'D750', got {df.iloc[0]['antenna']}"
    
    # case with whitespace
    csv_content_with_spaces = "antenna, frequency , 10mm , M1 \n D750 , 750 , 15.4 , 1 \n D835 , 835 , 15.0 , 0 "
    with tempfile.TemporaryDirectory() as tempdir:
        csv_path = (Path(tempdir) / "test_with_spaces.csv").as_posix()
        with open(csv_path, 'w') as f:
            f.write(csv_content_with_spaces)
        
        df = SarSampler._read_input_table(csv_path)
        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert df.iloc[0]['antenna'] == 'D750', f"Expected 'D750', got {df.iloc[0]['antenna']}"
    
    # test error handling for non-existent file
    try:
        SarSampler._read_input_table("nonexistent_file.csv")
        assert False, "Should have raised OSError for non-existent file"
    except OSError:
        pass  # Expected behavior

def test_validate_input_table():
    """Test the _validate_input_table static method for data type conversion and validation."""
    # Test modulation columns (M/m columns)
    df = pd.DataFrame({
        'antenna': ['D750', 'D835', 'D900', 'D1450'],
        'frequency': [750, 835, 900, 1450],
        'M1': ['1', '0', '', 'None'],
        'm2': [1, 0, '', 'None'],
        'M3': [1, 2, 3, 4],
        '10mm': [15.4, 15.0, 14.6, 14.2],
        '20mm': [20.5, 20.1, 19.7, 19.3],
    })
    
    validated_df = SarSampler._validate_input_table(df)
    
    # Check that M/m columns are converted to 0/1 integers
    assert validated_df['M1'].dtype == 'int64', f"Expected int64, got {validated_df['M1'].dtype}"
    assert validated_df['m2'].dtype == 'int64', f"Expected int64, got {validated_df['m2'].dtype}"
    assert validated_df['M3'].dtype == 'int64', f"Expected int64, got {validated_df['M3'].dtype}"
    
    # Check specific values
    assert validated_df['M1'].iloc[0] == 1, f"Expected 1, got {validated_df['M1'].iloc[0]}"
    assert validated_df['M1'].iloc[1] == 0, f"Expected 0, got {validated_df['M1'].iloc[1]}"
    assert validated_df['M1'].iloc[2] == 0, f"Expected 0, got {validated_df['M1'].iloc[2]}"
    assert validated_df['M1'].iloc[3] == 0, f"Expected 0, got {validated_df['M1'].iloc[3]}"
    
    # Check that mm columns are converted to float
    assert validated_df['10mm'].dtype == 'float64', f"Expected float64, got {validated_df['10mm'].dtype}"
    assert validated_df['20mm'].dtype == 'float64', f"Expected float64, got {validated_df['20mm'].dtype}"
    
    # Check specific values
    assert validated_df['10mm'].iloc[0] == 15.4, f"Expected 15.4, got {validated_df['10mm'].iloc[0]}"
    assert validated_df['20mm'].iloc[1] == 20.1, f"Expected 20.1, got {validated_df['20mm'].iloc[1]}"
    
    # Test error handling for invalid float values
    df_invalid = pd.DataFrame({
        'antenna': ['D750'],
        'frequency': [750],
        '10mm': ['invalid_float']
    })
    
    try:
        SarSampler._validate_input_table(df_invalid)
        assert False, "Should have raised ValueError for invalid float"
    except ValueError:
        pass  # Expected behavior
    
def _test_full_domain(nsamples: int | list[int]) -> None:
    """Test the sampling function on the full domain."""
    if not isinstance(nsamples, list):
        nsamples = [nsamples]

    sampler = SarSampler(input_table())
    for n in nsamples:
        sample = sampler(n)
        assert sample.xshape() == (n, 9), f"incorrect sample shape: {sample.xshape()}"
        nrows, ncols = sample.data.shape
        assert (nrows, ncols) == (n, 9), f"incorrect data sample shape: {(nrows, ncols)}"
        if n >= 10000:
            # check that all frequencies and radii from domain appear in sample
            freqs = sampler.config.get('frequency_domain', '[]')
            sample_freqs = set(sample.data['frequency'].unique())
            domain_freqs = set(freqs)
            assert sample_freqs.issubset(domain_freqs), \
                f"invalid frequencies in large sample: {sample_freqs - domain_freqs}"
            rads = sampler.config.get('radius_domain', '[]')
            sample_rads = set(sample.data['radius'].unique())
            domain_rads = set(rads)
            assert sample_rads.issubset(domain_rads), \
                f"invalid radii in large sample: {sample_rads - domain_rads}"

def _test_reduced_domain(nsamples: int | list[int]) -> None:
    """Test the sampling function on a reduced domain."""
    if not isinstance(nsamples, list):
        nsamples = [nsamples]

    df = input_table()
    df = df[df['antenna'].str.startswith('D')].copy()
    sampler = SarSampler(df)
    for n in nsamples:
        sample = sampler(n)
        assert sample.xshape() == (n, 9), f"incorrect sample shape: {sample.xshape()}"
        nrows, ncols = sample.data.shape
        assert (nrows, ncols) == (n, 9), f"incorrect data sample shape: {(nrows, ncols)}"

def test_full_domain():
    _test_full_domain([1, 10000])

def test_reduced_domain():
    _test_reduced_domain([1, 10, 100, 1000])

def main() -> None:
    print("Running Sampler validation tests...")
    test_full_domain()
    test_reduced_domain()
    test_read_input_table()
    test_validate_input_table()
    print("All Sampler validation tests passed.")

if __name__ == "__main__":
    main()
