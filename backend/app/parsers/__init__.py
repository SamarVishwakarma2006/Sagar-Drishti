"""Ocean data ingestion parsers for NetCDF and Tabular formats."""
from .netcdf_parser import NetCDFParser
from .tabular_parser import TabularParser

__all__ = ["NetCDFParser", "TabularParser"]
