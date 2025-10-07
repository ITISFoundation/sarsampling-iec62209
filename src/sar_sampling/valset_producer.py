import importlib.resources as pkg_resources
import pandas as pd
from io import StringIO
from sar_sampling import SarSampler

def main(arg=None) -> None:
    """
    Main function to generate a sample dataset using SAR sampling.
    
    This function loads input data from the package resources, creates a SarSampler
    instance, generates a sample of 225 data points, and saves the result to a CSV file.
    
    The function handles exceptions gracefully and provides error information if
    the sampling process fails.
    
    Returns:
        None
        
    Side Effects:
        - Creates a file named 'sample_225.csv' in the current directory
        - Prints status messages to stdout
        - Prints error messages and traceback to stdout if an exception occurs
    """
    try:
        csv_content = pkg_resources.read_text('sar_sampling.data', 'input_table.csv')
        input_df = pd.read_csv(StringIO(csv_content))
        sampler = SarSampler(input_df)
        sample = sampler(n_samples=225)
        sample.to_csv(f"sample_225.csv")
        print(f"Sample saved.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
