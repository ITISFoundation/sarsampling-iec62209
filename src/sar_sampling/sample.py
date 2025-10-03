import copy as cp
import json as js
import numpy as np
import pandas as pd
from typing import Any


__all__ = ['Sample']

# ============================================================================
# Sample class

class Sample:
    """
    A container class for structured data samples with variable classification.
    
    This class wraps a pandas DataFrame and provides functionality to classify
    variables as either 'x-variables' (independent/input variables) or 'z-variables'
    (dependent/output variables). It includes methods for data manipulation,
    serialization, and statistical analysis.
    
    The class is designed for experimental design and sampling applications,
    particularly useful when working with Latin Hypercube samples and other
    structured sampling methods.
    
    Attributes:
        data (pd.DataFrame): The underlying pandas DataFrame containing all data
        xvar (list): List of column names classified as x-variables (independent)
        zvar (list): List of column names classified as z-variables (dependent)
        mdata (dict): Optional metadata dictionary for storing additional information
    
    Parameters:
        df (pd.DataFrame): Input data as a pandas DataFrame
        xvar (list or str, optional): Column names to classify as x-variables.
                                     Can be a string (single variable) or list of strings.
                                     Default: []
        zvar (list or str, optional): Column names to classify as z-variables.
                                     Can be a string (single variable) or list of strings.
                                     Default: []
        sortvar (bool, optional): Whether to sort variable lists alphabetically.
                                 Default: True
        metadata (dict, optional): Additional metadata to store with the sample.
                                  Default: None
    
    Key Features:
        - Variable classification (x vs z variables)
        - Data manipulation and subsetting
        - Serialization to/from JSON and CSV
        - Statistical analysis helpers
        - Deep copying with selective modifications
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'x1': [1, 2, 3],
        ...     'x2': [4, 5, 6],
        ...     'y': [7, 8, 9]
        ... })
        >>> sample = Sample(df, xvar=['x1', 'x2'], zvar='y')
        >>> print(sample.xdata())  # Get x-variables as numpy array
        >>> print(sample.zdata())  # Get z-variables as numpy array
        >>> sample.save('sample.json')  # Serialize to file
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        xvar: list[str] | str = [], 
        zvar: list[str] | str = [], 
        sortvar: bool = True, 
        metadata: None | dict[str, Any] = None,
    ) -> None:
        """
        Initialize a Sample object with data and variable classifications.
        
        Parameters:
            df (pd.DataFrame): Input data as a pandas DataFrame
            xvar (list or str, optional): Column names for x-variables (independent)
            zvar (list or str, optional): Column names for z-variables (dependent)
            sortvar (bool, optional): Whether to sort variable lists alphabetically
            metadata (dict, optional): Additional metadata to store with the sample
            
        Raises:
            ValueError: If df is not a pandas DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError
        if isinstance(xvar, str):
            xvar = [xvar]
        if isinstance(zvar, str):
            zvar = [zvar]
        self.xvar: list[str] = list(xvar)
        self.zvar: list[str] = list(zvar)
        if sortvar:
            self.xvar.sort()
            self.zvar.sort()
        self.data: pd.DataFrame = df
        self.mdata: None | dict[str, Any] = metadata

    def __str__(self) -> str:
        """Returns a string representation of the sample."""
        return self.data.to_string()

    def copy(
        self, 
        xdata: None | np.ndarray = None, 
        zdata: None | np.ndarray = None, 
        xvar: None | list[str] | str = None, 
        zvar: None | list[str] | str = None, 
        metadata: None | dict[str, Any] = None,
    ) -> 'Sample':
        """
        Returns a deep copy of self.

        The returned copy's content is restricted to all conditions defined by
        non None arguments.
        """
        xv: None | list[str] = None
        zv: None | list[str] = None
        if xvar is None:
            xv = self.xvar
        else:
            if isinstance(xvar, str):
                xvar = [xvar]
            xv = [x for x in xvar if x in list(self.data)]
        if zvar is None:
            zv = self.zvar
        else:
            if isinstance(zvar, str):
                zvar = [zvar]
            zv = [z for z in zvar if z in list(self.data)]
        data = self.data.copy()
        if xdata is not None:
            if (np.asarray(xdata).shape[0] != data.shape[0]) or (np.asarray(xdata).shape[1] != len(xv)):
                raise ValueError('invalid shape for xdata')
            df = pd.DataFrame(xdata, columns=xv)
            data.loc[:, xv] = df[xv]
        if zdata is not None:
            if (np.asarray(zdata).shape[0] != data.shape[0]) or (np.asarray(zdata).shape[1] != len(zv)):
                raise ValueError('invalid shape for zdata')
            df = pd.DataFrame(zdata, columns=zv)
            data.loc[:, zv] = df[zv]
        md: None | dict[str, Any] = None
        if metadata is None:
            if self.mdata is not None:
                md = cp.deepcopy(self.mdata)
        else:
            md = cp.deepcopy(metadata)
        return Sample(data, xvar=xv, zvar=zv, metadata=md)

    def xshape(self) -> tuple[int, int]:
        """Returns the shape of the sub-dataframe of x-variables."""
        return (self.data.shape[0], len(self.xvar))

    def zshape(self) -> tuple[int, int]:
        """Returns the shape of the sub-dataframe of z-variables."""
        return (self.data.shape[0], len(self.zvar))

    def size(self) -> int:
        """Returns the number of points in this sample."""
        return self.data.shape[0]

    def __getitem__(self, key: str | list[str]) -> pd.DataFrame:
        """Returns the sub DataFrame associated to key."""
        return pd.DataFrame(self.data[key])

    def xdata(self) -> np.ndarray:
        """Returns a numpy array of all x-variables values."""
        return self.data[self.xvar].to_numpy()

    def zdata(self) -> np.ndarray:
        """Returns a numpy array of all z-variables values."""
        return self.data[self.zvar].to_numpy()

    def to_json(self) -> dict[str, Any]:
        """Returns a json object (a dict) representation of self."""
        metadata : None | dict[str, Any] = None
        if self.mdata is not None:
            metadata = self.mdata.copy()
        return {
            'metadata':metadata,
            'xvar':list(self.xvar), 
            'zvar':list(self.zvar), 
            'data':dict(self.data.to_dict('list'))
        }
    
    @classmethod
    def from_json(cls, json: str | bytes | bytearray | dict[str, Any]) -> 'Sample':
        """Builds a Sample for the given json object."""
        
        if isinstance(json, dict):
            json_dict : dict[str, Any] = dict(json)
        else:
            json_dict : dict[str, Any] = js.loads(json)
        mdata = json_dict['metadata']
        data = pd.DataFrame(pd.DataFrame.from_dict(json_dict['data']))
        xvar = [x for x in json_dict['xvar'] if x in list(data)]
        zvar = [z for z in json_dict['zvar'] if z in list(data)]
        return Sample(data, xvar=xvar, zvar=zvar, metadata=mdata)

    def to_csv(self, filename: str, sep: str = ',', index: bool = False) -> None:
        """Saves self's dataframe to csv (without the xvar, zvar defintions)."""
        self.data.to_csv(filename, sep=sep, index=index, float_format='%.6g')

    @classmethod
    def from_csv(
        cls, 
        filename: str, 
        sep: str = ',', 
        xvar: list[str] | str = [], 
        zvar: list[str] | str = [],
    ) -> 'Sample':
        """Builds a Sample from the provided csv file."""
        if isinstance(xvar, str):
            xvar = [xvar]
        if isinstance(zvar, str):
            zvar = [zvar]
        df = pd.read_csv(filename, sep=sep)
        var = list(df)
        xn = [x for x in xvar if x in var]
        zn = [z for z in zvar if z in var]
        if len(xn) < len(xvar) or len(zn) < len(zvar):
            raise ValueError('non existent columns')
        return Sample(df, xvar=xn, zvar=zn)
