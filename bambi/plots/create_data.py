import itertools

import numpy as np
import pandas as pd

from bambi.models import Model
from bambi.plots.utils import (
    ConditionalInfo,
    enforce_dtypes,
    get_covariates,
    get_model_covariates,
    make_group_panel_values,
    make_main_values,
    set_default_values,
    VariableInfo,
)


def create_differences_data(
    condition_info: ConditionalInfo, variable_info: VariableInfo, user_passed: bool, kind: str
) -> pd.DataFrame:
    def _grid_level():
        """
        Creates a "grid" of data by using the covariates passed into the
        `conditional` argument. Values for the grid are either: (1) computed
        using a equally spaced grid, mean, and or mode (depending on the
        covariate dtype), and (2) a user specified value or range of values.
        """
        covariates = get_covariates(condition_info.covariates)

        if user_passed:
            data_dict = {**condition_info.conditional}
        else:
            main_values = make_main_values(condition_info.model.data[covariates.main])
            data_dict = {covariates.main: main_values}
            data_dict = make_group_panel_values(
                condition_info.model.data,
                data_dict,
                covariates.main,
                covariates.group,
                covariates.panel,
                kind=kind,
            )

        data_dict[variable_info.name] = variable_info.values
        comparison_data = set_default_values(condition_info.model, data_dict, kind=kind)
        # use cartesian product (cross join) to create pairwise grid
        keys, values = zip(*comparison_data.items())
        pairwise_grid = pd.DataFrame([dict(zip(keys, v)) for v in itertools.product(*values)])

        # cannot enfore dtypes if 'slopes' because it may remove floating point of dy/dx
        if kind == "comparisons":
            pairwise_grid = enforce_dtypes(condition_info.model.data, pairwise_grid)

        return pairwise_grid

    def _unit_level():
        """
        Creates the data for unit-level contrasts by using the observed (empirical)
        data. All covariates in the model are included in the data, except for the
        contrast predictor. The contrast predictor is replaced with either: (1) the
        default contrast value, or (2) the user specified contrast value.
        """
        covariates = get_model_covariates(variable_info.model)
        df = variable_info.model.data[covariates].drop(labels=variable_info.name, axis=1)

        variable_vals = variable_info.values

        if kind == "comparisons":
            variable_vals = np.array(variable_info.values)[..., None]
            variable_vals = np.repeat(variable_vals, variable_info.model.data.shape[0], axis=1)

        contrast_df_dict = {}
        for idx, value in enumerate(variable_vals):
            contrast_df_dict[f"contrast_{idx}"] = df.copy()
            contrast_df_dict[f"contrast_{idx}"][variable_info.name] = value

        return pd.concat(contrast_df_dict.values())

    return _unit_level() if not condition_info.covariates else _grid_level()


def create_cap_data(model: Model, covariates: dict) -> pd.DataFrame:
    """Create data for a Conditional Adjusted Predictions

    Parameters
    ----------
    model : bambi.Model
        An instance of a Bambi model
    covariates : dict
        A dictionary of length between one and three.
        Keys must be taken from ("horizontal", "color", "panel").
        The values indicate the names of variables.

    Returns
    -------
    pandas.DataFrame
        The data for the Conditional Adjusted Predictions dataframe and or
        plotting.
    """
    data = model.data
    covariates = get_covariates(covariates)
    main, group, panel = covariates.main, covariates.group, covariates.panel

    # Obtain data for main variable
    main_values = make_main_values(data[main])
    data_dict = {main: main_values}

    # Obtain data for group and panel variables if not None
    data_dict = make_group_panel_values(data, data_dict, main, group, panel, kind="predictions")
    data_dict = set_default_values(model, data_dict, kind="predictions")

    return enforce_dtypes(data, pd.DataFrame(data_dict))
