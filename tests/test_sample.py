import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sar_sampling.sample import Sample

def sample_data() -> pd.DataFrame:
    """Return a reference to sample test data."""
    return pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'x2': [10.0, 20.0, 30.0, 40.0, 50.0],
        'y1': [100.0, 200.0, 300.0, 400.0, 500.0],
        'y2': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        'z': [0.1, 0.2, 0.3, 0.4, 0.5]
    })

def test_sample_data():
    """
    Test the basic data access methods: zdata(), xdata(), xshape(), and zshape().
    """
    df = sample_data()
    sample = Sample(df, xvar=['x1', 'x2'], zvar=['y1', 'y2'])
    
    # Test xdata() method
    xdata_result = sample.xdata()
    expected_xdata = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])
    assert np.array_equal(xdata_result, expected_xdata), f"xdata() returned incorrect values: {xdata_result}"
    
    # Test zdata() method
    zdata_result = sample.zdata()
    expected_zdata = np.array([[100.0, 1000.0], [200.0, 2000.0], [300.0, 3000.0], [400.0, 4000.0], [500.0, 5000.0]])
    assert np.array_equal(zdata_result, expected_zdata), f"zdata() returned incorrect values: {zdata_result}"
    
    # Test xshape() method
    xshape_result = sample.xshape()
    expected_xshape = (5, 2)
    assert xshape_result == expected_xshape, f"xshape() returned incorrect shape: {xshape_result}, expected {expected_xshape}"
    
    # Test zshape() method
    zshape_result = sample.zshape()
    expected_zshape = (5, 2)
    assert zshape_result == expected_zshape, f"zshape() returned incorrect shape: {zshape_result}, expected {expected_zshape}"
    
    # Test size() method
    size_result = sample.size()
    expected_size = 5
    assert size_result == expected_size, f"size() returned incorrect size: {size_result}, expected {expected_size}"

def test_sample_json():
    """
    Test the to_json() and from_json() methods by converting a sample to JSON and back.
    """
    df = sample_data()
    original_sample = Sample(df, xvar=['x1', 'x2'], zvar=['y1', 'y2'], metadata={'test': 'value'})
    
    # Convert to JSON
    json_data = original_sample.to_json()
    
    # Convert back from JSON
    restored_sample = Sample.from_json(json_data)
    
    # Assert that the restored sample is equal to the original
    assert restored_sample.xvar == original_sample.xvar, "xvar should be preserved"
    assert restored_sample.zvar == original_sample.zvar, "zvar should be preserved"
    assert restored_sample.mdata == original_sample.mdata, "metadata should be preserved"
    assert np.array_equal(restored_sample.xdata(), original_sample.xdata()), "xdata should be preserved"
    assert np.array_equal(restored_sample.zdata(), original_sample.zdata()), "zdata should be preserved"

def test_sample_csv():
    """
    Test the to_csv() and from_csv() methods by converting a sample to CSV and back.
    """
    df = sample_data()

    # test default case
    original_sample = Sample(df)
    with tempfile.TemporaryDirectory() as tempdir:
        csv_path = (Path(tempdir) / "sample.csv").as_posix()
        # convert to csv
        original_sample.to_csv(csv_path)
        # convert back from csv
        restored_sample = Sample.from_csv(csv_path)
    # Assert that the restored sample is equal to the original
    assert restored_sample.xvar == original_sample.xvar, "xvar should be preserved"
    assert restored_sample.zvar == original_sample.zvar, "zvar should be preserved"
    assert np.array_equal(restored_sample.xdata(), original_sample.xdata()), "xdata should be preserved"
    assert np.array_equal(restored_sample.zdata(), original_sample.zdata()), "zdata should be preserved"

    # test with xvar, zvar, and metadata
    xvar = ['x1', 'x2']
    zvar = ['y1', 'y2']
    mdata = {'test': 'value'}
    original_sample = Sample(df, xvar=xvar, zvar=zvar, metadata=mdata)
    with tempfile.TemporaryDirectory() as tempdir:
        csv_path = (Path(tempdir) / "sample.csv").as_posix()
        # convert to csv
        original_sample.to_csv(csv_path)
        # convert back from csv
        restored_sample = Sample.from_csv(csv_path)
        restored_sample.xvar = xvar
        restored_sample.zvar = zvar
        restored_sample.mdata = mdata
    # Assert that the restored sample is equal to the original
    assert np.array_equal(restored_sample.xdata(), original_sample.xdata()), "xdata should be preserved"
    assert np.array_equal(restored_sample.zdata(), original_sample.zdata()), "zdata should be preserved"

def test_sample_getitem():
    """
    Test the __getitem__ method for accessing sample data using square bracket notation.
    """
    df = sample_data()
    sample = Sample(df, xvar=['x1', 'x2'], zvar=['y1', 'y2'])
    
    # Test accessing single column
    x1_data = sample['x1']
    expected_x1 = pd.DataFrame({'x1': [1.0, 2.0, 3.0, 4.0, 5.0]})
    assert x1_data.equals(expected_x1), f"Single column access failed: {x1_data}"
    
    # Test accessing multiple columns
    x_columns = sample[['x1', 'x2']]
    expected_x_columns = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'x2': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    assert x_columns.equals(expected_x_columns), f"Multiple columns access failed: {x_columns}"
    
    # Test accessing z-variables
    y1_data = sample['y1']
    expected_y1 = pd.DataFrame({'y1': [100.0, 200.0, 300.0, 400.0, 500.0]})
    assert y1_data.equals(expected_y1), f"Z-variable access failed: {y1_data}"
    
    # Test accessing mixed x and z variables
    mixed_data = sample[['x1', 'y1']]
    expected_mixed = pd.DataFrame({
        'x1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y1': [100.0, 200.0, 300.0, 400.0, 500.0]
    })
    assert mixed_data.equals(expected_mixed), f"Mixed variables access failed: {mixed_data}"

def test_sample_copy():
    """
    Test the Sample.copy() method functionality.
    
    This test verifies that:
    1. Basic copying works correctly
    2. Copying with modified xdata works
    3. Copying with modified zdata works
    4. Copying with modified xvar/zvar works
    5. Copying with modified metadata works
    6. Shape validation works correctly
    7. Deep copying preserves independence
    """
    # Create original sample
    df = sample_data()
    original = Sample(df, xvar=['x1', 'x2'], zvar=['y1', 'y2'], metadata={'test': 'value'})
    
    # Test 1: Basic copy without modifications
    copy1 = original.copy()
    assert copy1 is not original, "Copy should be a different object"
    assert copy1.data is not original.data, "Data should be deep copied"
    assert copy1.xvar == original.xvar, "xvar should be preserved"
    assert copy1.zvar == original.zvar, "zvar should be preserved"
    assert copy1.mdata == original.mdata, "metadata should be preserved"
    assert np.array_equal(copy1.xdata(), original.xdata()), "xdata should be identical"
    assert np.array_equal(copy1.zdata(), original.zdata()), "zdata should be identical"
    
    # Test 2: Copy with modified xdata
    new_xdata = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0], [90.0, 100.0]])
    copy2 = original.copy(xdata=new_xdata)
    assert copy2.xvar == original.xvar, "xvar should be preserved when modifying xdata"
    assert np.array_equal(copy2.xdata(), new_xdata), "xdata should be updated"
    assert np.array_equal(copy2.zdata(), original.zdata()), "zdata should remain unchanged"
    
    # Test 3: Copy with modified zdata
    new_zdata = np.array([[1000.0, 2000.0], [3000.0, 4000.0], [5000.0, 6000.0], [7000.0, 8000.0], [9000.0, 10000.0]])
    copy3 = original.copy(zdata=new_zdata)
    assert copy3.zvar == original.zvar, "zvar should be preserved when modifying zdata"
    assert np.array_equal(copy3.zdata(), new_zdata), "zdata should be updated"
    assert np.array_equal(copy3.xdata(), original.xdata()), "xdata should remain unchanged"
    
    # Test 4: Copy with modified xvar/zvar
    copy4 = original.copy(xvar=['x1'], zvar=['y1'])
    assert copy4.xvar == ['x1'], "xvar should be updated"
    assert copy4.zvar == ['y1'], "zvar should be updated"
    assert copy4.xshape() == (5, 1), "xshape should reflect new xvar"
    assert copy4.zshape() == (5, 1), "zshape should reflect new zvar"
    
    # Test 5: Copy with modified metadata
    new_metadata = {'new_test': 'new_value', 'nested': {'key': 'value'}}
    copy5 = original.copy(metadata=new_metadata)
    assert copy5.mdata == new_metadata, "metadata should be updated"
    assert copy5.mdata is not new_metadata, "metadata should be deep copied"
    
    # Test 6: Shape validation
    try:
        # This should raise ValueError due to shape mismatch
        original.copy(xdata=np.array([[1.0, 2.0], [3.0, 4.0]]))  # Wrong shape
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError:
        pass  # Expected behavior
    
    try:
        # This should raise ValueError due to shape mismatch
        original.copy(zdata=np.array([[1.0, 2.0], [3.0, 4.0]]))  # Wrong shape
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError:
        pass  # Expected behavior
    
    # Test 7: Deep copy independence
    copy7 = original.copy()
    # Modify original data
    original.data.iloc[0, 0] = 999.0
    # Copy should remain unchanged
    assert copy7.data.iloc[0, 0] != 999.0, "Copy should be independent of original"
    
    # Test 8: Copy with string xvar/zvar
    copy8 = original.copy(xvar='x1', zvar='y1')
    assert copy8.xvar == ['x1'], "Single string xvar should be converted to list"
    assert copy8.zvar == ['y1'], "Single string zvar should be converted to list"

def main() -> None:
    """
    Main function to run Sample class tests.
    """
    print("Running Sample validation tests...")
    test_sample_copy()
    test_sample_data()
    test_sample_json()
    test_sample_csv()
    test_sample_getitem()
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()