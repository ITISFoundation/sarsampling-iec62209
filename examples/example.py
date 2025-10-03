from sar_sampling import SarSampler
import pandas as pd

def example():
    """
    Basic usage example of the SarSampler class.
    
    This example demonstrates how to:
    1. Create an input table with antenna configurations
    2. Initialize a SarSampler with the input table
    3. Generate samples with different parameters
    4. Access and analyze the generated data
    """
    
    # Create a sample input table with antenna configurations
    # Note: This data should normally be loaded from a CSV file 
    # of appropriate format file using:
    # 
    # sampler = SarSampler('input_table.csv')
    # 
    # The CSV file should have the following format:
    # antenna,frequency,2mm,5mm,10mm,21mm,M1,M2,M3,M4
    # D750,750,,,15.4,21.5,1,1,1,0
    # D835,835,4.6,,15.0,20.8,1,1,0,0
    # D900,900,4.7,,14.6,20.4,1,1,0,0
    # D1450,1450,5.4,9.4,,18.3,1,1,0,0
    # D1500,1500,5.2,9.5,,18.1,1,1,0,0
    # D1640,1640,5.2,9.2,,17.9,1,1,0,0
    # D2300,2300,4.9,8.1,,17.4,1,1,1,1
    #
    # Required columns:
    # - 'antenna': Antenna identifiers
    # - 'frequency': Frequency values
    # - Distance columns ending with 'mm' (e.g., '10mm', '15mm', '20mm')
    # - Modulation columns starting with 'M' (e.g., 'M1', 'M2', 'M3')
    input_data = {
        'antenna': ['D750', 'D835', 'D900', 'D1450', 'D1500', 'D1640', 'D2300'],
        'frequency': [750, 835, 900, 1450, 1500, 1640, 2300],
        '2mm': [None, 4.6, 4.7, 5.4, 5.2, 5.2, 4.9],
        '5mm': [None, None, None, 9.4, 9.5, 9.2, 8.1],
        '10mm': [15.4, 15.0, 14.6, None, None, None, None],
        '15mm': [None, None, None, None, None, None, None],
        '21mm': [21.5, 20.8, 20.4, 18.3, 18.1, 17.9, 17.4],
        'M1': [1, 1, 1, 1, 1, 1, 1],
        'M2': [1, 1, 1, 1, 1, 1, 1],
        'M3': [1, 0, 0, 0, 0, 0, 1],
        'M4': [0, 0, 0, 0, 0, 0, 1],
    }
    
    input_df = pd.DataFrame(input_data)
    print("Input table:")
    print()
    print(input_df)
    print()
    
    # Initialize SarSampler with the input table
    sampler = SarSampler(input_df)
    freqs = sampler.config['frequency_domain']
    rads = sampler.config['radius_domain']
    
    print("Available frequencies:")
    for i in range(0, len(freqs), 10):
        print("  ", freqs[i:i+10])
    
    print("Available radii:")
    for i in range(0, len(rads), 10):
        print("  ", rads[i:i+10])
    print()
    
    # Generate small sample set
    print("=" * 80)
    sample1 = sampler(10, seed=42)
    print(f"Small sample with {sample1.size()} points:")
    print()
    print(sample1)
    print()
    
    # Larger sample with specific method
    sample2 = sampler(1000, method='maximin', seed=123)
    print("=" * 80)
    print(f"Large sample with {sample2.size()} points generated.")
    print()
    
    # Display sample statistics
    print("Sample statistics:")
    print(f"  X-coordinate range: {sample2.data['x'].min():.1f} to {sample2.data['x'].max():.1f}")
    print(f"  Y-coordinate range: {sample2.data['y'].min():.1f} to {sample2.data['y'].max():.1f}")
    print(f"  Frequency range: {sample2.data['frequency'].min():.0f} to {sample2.data['frequency'].max():.0f}")
    print(f"  Power range: {sample2.data['power'].min():.0f} to {sample2.data['power'].max():.0f}")
    print()
    
    # Show antenna distribution
    antenna_counts = sample2.data['antenna'].value_counts()
    print("Antenna distribution:")
    for antenna, count in antenna_counts.items():
        print(f"  {antenna}: {count} samples")
    print()
    
    # Show modulation distribution
    modulation_counts = sample2.data['modulation'].value_counts()
    print("Modulation distribution:")
    for modulation, count in modulation_counts.items():
        print(f"  {modulation}: {count} samples")
    print()
    
    # Save sample to CSV
    filename = f"sample_{sample2.size()}.csv"
    sample2.to_csv(filename)
    print(f"Sample saved to {filename}")

def main():
    """Run the basic usage example."""
    try:
        example()
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
