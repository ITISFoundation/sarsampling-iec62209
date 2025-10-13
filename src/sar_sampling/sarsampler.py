import numpy as np
import pandas as pd
from typing import Any
from io import StringIO
from sar_sampling.lhs import LhSampler
from sar_sampling.sample import Sample
from sar_sampling.voronoi import voronoi_areas


__all__ = ['SarSampler']

# ============================================================================
# SarSampler class:
class SarSampler:
    """
    A Latin Hypercube Sampler for SAR (Specific Absorption Rate) testing scenarios.
    
    This class generates structured samples for SAR testing by combining Latin Hypercube
    sampling with domain-specific constraints and mappings. It processes input tables
    containing antenna configurations, frequency/radius mappings, and modulation data
    to create realistic test scenarios.
    
    The sampler works by:
    1. Processing input tables to extract antenna, frequency, distance, and modulation data
    2. Building frequency-radius mappings and Voronoi diagrams for spatial distribution
    3. Using Latin Hypercube sampling to generate test points across multiple dimensions
    4. Mapping sampled values to actual antenna configurations and modulation schemes
    
    Attributes:
        config (dict): Configuration dictionary containing domain definitions, mappings,
                      and processing parameters
        dom (dict): Domain definitions for each sampling dimension (x, y, angle, power, etc.)
        hist (dict): Histogram data for frequency-radius distributions
        lhsampler (LhSampler): The underlying Latin Hypercube sampler instance
    
    Parameters:
        input_table (pd.DataFrame or str): Input data as DataFrame or path to CSV file.
                                         Must contain 'antenna' and 'frequency' columns,
                                         distance columns ending with 'mm', and modulation
                                         columns starting with 'M' or 'm'.
        config (dict, optional): Configuration overrides. Default domains include:
            - x_domain: X-coordinate range (-27 to 27, step 3)
            - y_domain: Y-coordinate range (-69 to 69, step 3)  
            - angle_domain: Angle values [0, 20, 40, 65, 90, 110, 130, 155]
            - power_domain: Power range (0 to 20, step 1)
            - frequency_max: Maximum frequency (default: 10000)
            - radius_max: Maximum radius (default: 25)
            - freqradvor_to_weight: Weighting function from frequency f, radius r and voronoi area v (default: v)
    
    Example:
        >>> sampler = SarSampler(input_table_df)
        >>> samples = sampler(n_samples=100, method='random', seed=42)
        >>> print(samples.dataframe)
    """
    def __init__(
        self, 
        input_table: pd.DataFrame | str, 
        config: dict[str, Any] = {},
    ) -> None:
        """
        Initialize the SarSampler with input data and configuration.
        
        This method sets up the sampler by:
        1. Establishing default configuration parameters
        2. Processing the input table to extract antenna, frequency, distance, and modulation data
        3. Building frequency-radius mappings and Voronoi diagrams
        4. Creating domain definitions and histogram data for sampling
        5. Initializing the underlying Latin Hypercube sampler
        
        Parameters:
            input_table (pd.DataFrame or str): Input data containing SAR testing configurations.
                                             Can be a pandas DataFrame or path to a CSV file.
                                             Must contain:
                                             - 'antenna' column: Antenna identifiers
                                             - 'frequency' column: Frequency values
                                             - Distance columns ending with 'mm' (e.g., '10mm', '20mm')
                                             - Modulation columns starting with 'M' or 'm' (e.g., 'M1', 'M2', 'm1', 'm2')
            config (dict, optional): Configuration overrides. Default configuration includes:
                - x_domain: X-coordinate sampling range [-27, -24, ..., 24, 27]
                - y_domain: Y-coordinate sampling range [-69, -66, ..., 66, 69]
                - angle_domain: Angle sampling values [0, 20, 40, 65, 90, 110, 130, 155]
                - power_domain: Power sampling range [0, 1, 2, ..., 20]
                - frequency_max: Maximum frequency limit (default: 10000)
                - radius_max: Maximum radius limit (default: 25)
                - freqradvor_to_weight: Weighting function from frequency f, radius r and voronoi area v (default: 1/r)
        
        Raises:
            ValueError: If input_table is not a DataFrame or string, or if required columns
                      are missing from the input data.
            Exception: If there are errors processing the input table (printed to console).
        
        Example:
            >>> # Using DataFrame
            >>> df = pd.DataFrame({
            ...     'antenna': ['A1', 'A2'],
            ...     'frequency': [1000, 2000],
            ...     '10mm': [1.5, 2.0],
            ...     'M1': [1, 1]
            ... })
            >>> sampler = SarSampler(df)
            
            >>> # Using CSV file
            >>> sampler = SarSampler('config.csv', config={'frequency_max': 5000})
        """
        self.config: dict[str, Any] = {
            'x_domain': list(range(-27, 28, 3)),
            'y_domain': list(range(-69, 70, 3)),
            'angle_domain': [0, 20, 40, 65, 90, 110, 130, 155],
            'power_domain': list(range(0, 21, 1)),
            'freqradvor_to_weight': lambda f, r, v: 1/r,
            'input_frequency_min': 1,
            'input_frequency_max': 10000,
            'frequency_margin': 0.1,
            'frequency_min': 0,
            'frequency_max': 10000,
            'radius_margin': 0.1,
            'radius_min': 0,
            'radius_max': 25,
        }

        # merge config with default config
        self.config = {**self.config, **config}

        try:
            if isinstance(input_table, str):
                input_table = SarSampler._read_input_table(input_table)
            input_table = SarSampler._filter_input_table(
                input_table, 
                self.config['input_frequency_min'], 
                self.config['input_frequency_max'])
            input_table = SarSampler._validate_input_table(input_table)
            tables = SarSampler._process_input_table(input_table)
            self.config = {**self.config, **tables}
        except Exception as e:
            print(f"error reading input table: {e}")

        # build voronoi table
        self.config['voronoi_table'] = SarSampler._make_voronoi_table(
            self.config['freqrad_map'],
            freq_margin=self.config['frequency_margin'],
            rad_margin=self.config['radius_margin'],
            min_freq=self.config['frequency_min'],
            min_rad=self.config['radius_min'],
            max_freq=self.config['frequency_max'],
            max_rad=self.config['radius_max']
        )
        # build freqrad histogram
        self.config['freqrad_hist'] = {
            k: self.config['freqradvor_to_weight'](k[0], k[1], v) for k, v in self.config['voronoi_table'].items()
        }

        # init domain and histogram
        self.dom: dict[str, None | list[float]] = {
            'x': self.config['x_domain'],
            'y': self.config['y_domain'],
            'angle': self.config['angle_domain'],
            'power': self.config['power_domain'],
            'radius': self.config['radius_domain'],
            'frequency': self.config['frequency_domain'],
            'modulation': None,
        }
        self.hist: dict[tuple[str, str], np.ndarray] = {
            ('frequency', 'radius') : SarSampler._dict2array(
            self.config['frequency_domain'], 
            self.config['radius_domain'], 
            self.config['freqrad_hist'])
        }

        # init lhs sampler
        self.lhsampler = LhSampler(self.dom, hist=self.hist) # type: ignore

    def __call__(
        self, 
        n_samples: int, 
        method: None | str = None, 
        seed: None | int = None,
    ) -> Sample:
        """
        Generate SAR test samples using Latin Hypercube sampling.
        
        Parameters:
            n_samples (int): Number of samples to generate
            method (str, optional): Sampling method for LhSampler
            seed (int, optional): Random seed for reproducibility
            
        Returns:
            Sample: Sample object containing the generated test data with columns:
                   ['antenna', 'frequency', 'modulation', 'distance', 'radius', 'power', 'angle', 'x', 'y']
        """
        lhsdf = self.lhsampler(n_samples, method=method, seed=seed)
        lhsdf = self._add_antdistmod(lhsdf)
        lhsdf = lhsdf.sort_values([
            'frequency', 'antenna', 'modulation', 'distance', 'power', 'angle',
        ]).reset_index(drop=True)
        dom = ['antenna', 'frequency', 'modulation', 'distance', 'radius', 'power', 'angle', 'x', 'y']
        return Sample(lhsdf[dom], xvar=dom)
    
    def _add_antdistmod(self, lhsdf: pd.DataFrame) -> pd.DataFrame:
        """Add antenna, distance, and modulation columns to the sampled dataframe."""
        # get antenna and distance from frequency and radius
        def get_ant_dist(row: pd.Series) -> pd.Series:
            # these are consistent because of the rounding in _process_input_table
            freq, rad = row['frequency'], row['radius']
            ant_dist_list = self.config['freqrad_map'].get((freq, rad), [])
            if not ant_dist_list:
                return pd.Series({'antenna': None, 'distance': None})
            # randomly select one pair from the list
            ant, dist = ant_dist_list[np.random.randint(len(ant_dist_list))]
            return pd.Series({'antenna': ant, 'distance': dist})
        ant_dist_df = lhsdf.apply(get_ant_dist, axis=1)
        lhsdf['antenna'] = ant_dist_df['antenna']
        lhsdf['distance'] = ant_dist_df['distance']
        # use antenna and convert modulation dimension to categorical dimension
        # convert normalized float to index in list
        def binidx(lst: list, u: float) -> int:
            k = len(lst)
            return min(int(u * k), k - 1)
        # for each row, map modulation value to the corresponding modulation name
        def get_mod_name(row: pd.Series) -> str:
            ant = row['antenna']
            mod_table = self.config['modulation_table']
            mod_list = mod_table[ant] if ant in mod_table else mod_table['']
            idx = binidx(mod_list, row['modulation'])
            return mod_list[idx]
        lhsdf['modulation'] = lhsdf.apply(get_mod_name, axis=1)
        return lhsdf

    @staticmethod
    def _read_input_table(filepath: str, **read_csv_kwargs) -> pd.DataFrame:
        """
        Loads a CSV file, removes all whitespace (spaces, tabs) from the content
        except line breaks, and returns a pandas DataFrame.

        Parameters:
            filepath (str): Path to the CSV file.
            **read_csv_kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            pd.DataFrame: Cleaned DataFrame with whitespace removed.

        Raises:
            OSError: If file cannot be read.
            ValueError: If DataFrame parsing fails or validation checks fail.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            raise OSError(f"Error reading file '{filepath}': {e}")

        # remove all whitespace except newlines
        cleaned_content = "".join(ch for ch in content if not ch.isspace() or ch == "\n")

        # try converting to DataFrame
        try:
            df = pd.read_csv(StringIO(cleaned_content), **read_csv_kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to parse cleaned content into DataFrame. "
                f"Check formatting of '{filepath}'. Original error: {e}"
            )
        
        return df

    @staticmethod
    def _filter_input_table(df: pd.DataFrame, fmin: float, fmax: float) -> pd.DataFrame:
        """Filter input table by frequency range."""
        if 'frequency' not in df.columns:
            raise ValueError("input table must contain 'frequency' column")
        if fmin < 0 or fmax < fmin:
            raise ValueError(f"invalid frequency range: [{fmin}, {fmax}]")
        return df[df['frequency'].between(fmin, fmax)]

    @staticmethod
    def _validate_input_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input table.

        Columns whose names start with 'M' or 'm' are treated as strings.
        -> Their values are normalized: "" / NaN / "0" => 0 (int), others => 1 (int).
        Columns whose names end with 'mm' must be valid floats.
        -> If parsing fails, a ValueError is raised.
        """
        # check required columns exist
        if 'antenna' not in df.columns:
            raise ValueError("input table must contain 'antenna' column")
        if 'frequency' not in df.columns:
            raise ValueError("input table must contain 'frequency' column")
        # check that there is at least one row of data
        if len(df) == 0:
            raise ValueError("input table must contain at least one row of data")

        # modulation columns
        m_columns = [col for col in df.columns if col.lower().startswith("m")]
        if len(m_columns) == 0:
            raise ValueError("no modulation columns found starting with 'M' or 'm'")
        for col in m_columns:
            # map NaN, 'None', '', numeric 0 to 0 and others to 1
            df[col] = df[col].map(
                lambda x: 0 if pd.isna(x) or (isinstance(x, (int, float)) and x == 0) or str(x) in ['None', '', '0'] else 1
                ).astype(int)
        # ensure at least one 1 in m_columns for each row
        m_sums = df[m_columns].sum(axis=1)
        if (m_sums == 0).any():
            raise ValueError("Each row must have at least one valid modulation")

        # distance columns
        mm_columns = [col for col in df.columns if col.endswith("mm")]
        if len(mm_columns) == 0:
            raise ValueError("no distance columns found ending in 'mm'")
        try:
            for col in mm_columns:
                df[col] = df[col].astype(float)
        except Exception as e:
            raise ValueError(f"Invalid value in column '{col}' : cannot be parsed as float")
        # ensure at least 4 distinct valid radii exist
        valid_rad = set()
        for col in mm_columns:
            valid_rad.update(df[col][df[col].notna() & (df[col] > 0)].unique())
        if len(valid_rad) < 4:
            raise ValueError("Input table must contain at least 4 distinct valid radii")

        return df


    @staticmethod
    def _process_input_table(input_table: pd.DataFrame | str) -> dict[str, Any]:
        """Process input table to extract antenna, frequency, distance, and modulation data."""
        tables: dict[str, Any] = {}
        inputdf: None | pd.DataFrame = None
        if isinstance(input_table, pd.DataFrame):
            inputdf = input_table.copy()
        elif isinstance(input_table, str):
            inputdf = pd.read_csv(input_table, header=0)
        else:
            raise ValueError("input_table must be a pandas DataFrame or path to csv file")

        # to avoid inconsistencies with radius values due to floating point errors
        inputdf = inputdf.map(SarSampler._round_float)

        # check antenna, frequency
        columns = list(inputdf.columns)
        if 'antenna' not in columns:
            raise ValueError("input table must contain an 'antenna' column")
        if 'frequency' not in columns:
            raise ValueError("input table must contain a 'frequency' column")

        # distance names end with 'mm'
        dist_map: dict[str, int] = {}
        dist_values: list[int] = []
        for col in columns:
            if col.endswith('mm'):
                try:
                    distval = int(col.replace('mm', ''))
                    dist_values.append(distval)
                    dist_map[col] = distval
                except ValueError:
                    raise ValueError(f"invalid distance column format: {col}")
        if not dist_map:
            raise ValueError("no distance columns found ending in 'mm'")
        dist_values = sorted(dist_values)
        tables['distance_map'] = dist_map
        tables['distance_domain'] = dist_values

        # modulation names start with 'M' or 'm'
        mod_names = [col for col in columns if col.startswith('M') or col.startswith('m')]
        if not mod_names:
            raise ValueError("no modulation columns found starting with 'M' or 'm'")
        tables['modulation_domain'] = mod_names

        # process input table row by row
        freq_set: set[float] = set()
        ant_set: set[str] = set()
        rad_set: set[float] = set()
        freqrad_map: dict[tuple[float, float], list[tuple[str, int]]] = {}
        mod_table: dict[str, list[str]] = {}
        for _, row in inputdf.iterrows():
            ant = row['antenna']
            freq = row['frequency']
            ant_set.add(ant)
            freq_set.add(freq)
            
            # get 3dbr with non-null and positive values
            for distname, distval in dist_map.items():
                rad = row[distname]
                if not pd.isna(rad) and rad > 0:
                    rad_set.add(rad)
                    if (freq, rad) in freqrad_map:
                        freqrad_map[(freq, rad)].append((ant, distval))
                    else:
                        freqrad_map[(freq, rad)] = [(ant, distval)]

            # build modulation table for this antenna with non-null and positive values
            mods: list[str] = []
            for mod in mod_names:
                if not pd.isna(row[mod]) and row[mod] > 0:
                    mods.append(mod)
            mod_table[ant] = mods
        tables['frequency_domain'] = sorted(list(set(freq_set)))
        tables['radius_domain'] = sorted(list(set(rad_set)))
        tables['freqrad_map'] = freqrad_map
        tables['modulation_table'] = mod_table
        return tables

    @staticmethod
    def _round_float(x: Any, decimals: int = 3) -> Any:
        """Round float values to specified decimal places, handling NaN and inf values."""
        if not isinstance(x, float) or pd.isna(x) or np.isinf(x):
            return x
        return round(x, decimals)

    @staticmethod
    def _make_voronoi_table(
        freqrad_map: dict[tuple[float, float], list[tuple[str, int]]], 
        freq_margin: float = 0.1,
        rad_margin: float = 0.1,
        min_freq: float = 0, 
        min_rad: float = 0,
        max_freq: float = 10000, 
        max_rad: float = 25,
    ) -> dict[tuple[float, float], float]:
        """Create Voronoi table for frequency-radius mapping with area calculations, using log-transformed radius coordinates."""

        min_freq, min_rad = max(0, min_freq), max(0, min_rad)
        max_freq, max_rad = max(0, max_freq), max(0, max_rad)
        freqrad_arr = np.array(list(freqrad_map.keys())).reshape(-1, 2)
        # calculate appropriate bounding box based on actual data
        freq_min, freq_max = freqrad_arr[:, 0].min(), freqrad_arr[:, 0].max()
        rad_min, rad_max = freqrad_arr[:, 1].min(), freqrad_arr[:, 1].max()
        # add padding to the bounding box
        freq_padding = (freq_max - freq_min) * freq_margin
        rad_padding = (rad_max - rad_min) * rad_margin
        bounding_box = (
            max(min_freq, freq_min - freq_padding),
            max(min_rad, rad_min - rad_padding),
            min(max_freq, freq_max + freq_padding),
            min(max_rad, rad_max + rad_padding)
        )
        
        # with log-transformed radius values
        log_freqrad_arr = freqrad_arr.copy()
        log_freqrad_arr[:, 1] = np.log(freqrad_arr[:, 1] + 1)
        log_bounding_box = (
            bounding_box[0], np.log(bounding_box[1] + 1), 
            bounding_box[2], np.log(bounding_box[3] + 1))
        freqrad_areas = voronoi_areas(log_freqrad_arr, bbox=log_bounding_box)
        vor_table = {tuple(key): area for key, area in zip(freqrad_arr, freqrad_areas)}
        return vor_table

    @staticmethod
    def _dict2array(
        l1: list[float], 
        l2: list[float], 
        data_dict: dict[tuple[float, float], float],
    ) -> np.ndarray:
        """Convert dictionary data to 2D numpy array using index mappings."""
        # Create index maps for fast lookup
        l1_index: dict[float, int] = {key: i for i, key in enumerate(l1)}
        l2_index: dict[float, int] = {key: j for j, key in enumerate(l2)}
        
        # Initialize array of zeros
        result = np.zeros((len(l1), len(l2)))

        # Fill in values where key exists in both l1 and l2
        for (k1, k2), value in data_dict.items():
            if k1 in l1_index and k2 in l2_index:
                i = l1_index[k1]
                j = l2_index[k2]
                result[i, j] = value

        return result
