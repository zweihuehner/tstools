
import mikeio
from mikeio import EUMType, EUMUnit, ItemInfo
import numpy as np
import pandas as pd


def to_dfs0(variables: list[np.ndarray] | np.ndarray, time: np.ndarray, variable_names: list[str] | str, eumtypes: list[EUMType] | EUMType, eumunits:  list[EUMUnit] | EUMUnit, save_file: str):
    """Create a .dfs0 file.

    Args:
        variables list[np.array] | np.array: The variable value.
        variable_names (list[str] | str): The names of the variables.
        time (np.array): An numpy array with the time values.
        eumtypes (list[EUMType] | EUMType): The EUMType of the variables.
        eumunits (list[EUMUnit] | EUMUnit): The EUMUnit of the variable.
        save_file (str): Path of the save file. If None the file will be named like the .netcdf file. Defaults to None.
    """

    list_variables_dfs = []

    variable_names, eumtypes, eumunits = _parse_variabels_eums(names=variable_names, eumtypes=eumtypes,eumunits=eumunits)
    if type(variables) == np.ndarray:
        variables = (variables,)

    for variable, variable_name, eumtype, eumunit in zip(variables, variable_names, eumtypes, eumunits):

        item = ItemInfo(name = variable_name, itemtype = eumtype, unit = eumunit)
        dataarray = mikeio.DataArray(variable, time=time, item=item)
        list_variables_dfs.append(dataarray)

    ds_dfs = mikeio.Dataset(list_variables_dfs)

    print(f"Saving to '{save_file}'")
    ds_dfs.to_dfs(f"{save_file}")


def _parse_variabels_eums(names: list[str] | str, eumtypes: list[EUMType] | EUMType, eumunits:  list[EUMUnit] | EUMUnit) -> tuple[list[str], list[EUMType], list[EUMUnit]]:
    """Check if inputs are given as a str or a list. If str make it a list with a single element.

    Args:
        Names (list[str] | str): Variable names.
        eumtypes (list[EUMType] | EUMType): EUMTypes.
        eumunits (list[EUMUnit] | EUMUnit): EUMUnits.

    Returns:
        tuple[list[str], list[EUMType], list[EUMUnit]]: list of inputs.
    """
    if type(names) == str:
        names = (names,)
    if type(eumtypes) == EUMType:
        eumtypes = (eumtypes,)
    if type(eumunits) == EUMUnit:
        eumunits = (eumunits,)

    return names, eumtypes, eumunits