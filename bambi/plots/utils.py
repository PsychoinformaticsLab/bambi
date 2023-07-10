from dataclasses import dataclass, field
from statistics import mode
from typing import Union

import numpy as np
from formulae.terms.call import Call
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from bambi import Model
from bambi.utils import listify


@dataclass
class ContrastInfo:
    model: Model
    contrast: Union[str, dict, list]
    name: str = field(init=False)
    values: Union[int, float] = field(init=False)

    def __post_init__(self):
        """ """
        if isinstance(self.contrast, dict):
            self.values = list(self.contrast.values())[0]
            self.name = list(self.contrast.keys())[0]
        elif isinstance(self.contrast, (list, str)):
            if isinstance(self.contrast, list):
                self.name = " ".join(self.contrast)
            else:
                self.name = self.contrast
            self.values = set_default_contrast_values(self.model, self.name)
        elif not isinstance(self.contrast, (list, dict, str)):
            raise TypeError("`contrast` must be a list, dict, or string")


@dataclass
class ConditionalInfo:
    model: Model
    conditional: Union[str, dict, list]
    covariates: dict = field(init=False)
    user_passed: bool = field(init=False)

    def __post_init__(self):
        """
        Sets the covariates attributes based on if the user passed a dictionary
        or not.
        """
        covariate_kinds = ("main", "group", "panel")

        if not isinstance(self.conditional, dict):
            self.covariates = listify(self.conditional)
            self.covariates = dict(zip(covariate_kinds, self.covariates))
            self.user_passed = False
        elif isinstance(self.conditional, dict):
            self.covariates = {k: listify(v) for k, v in self.conditional.items()}
            self.covariates = dict(zip(covariate_kinds, self.conditional))
            self.user_passed = True


@dataclass
class Covariates:
    main: str
    group: Union[str, None]
    panel: Union[str, None]


def average_over(data: pd.DataFrame, covariate: Union[str, list, None]) -> pd.DataFrame:
    """
    Average estimates by specified covariate in the model. data.columns[-3:] are
    the columns: 'estimate', 'lower', and 'upper'.
    """
    if covariate is None:
        return pd.DataFrame(data[data.columns[-3:]].mean()).T
    else:
        return data.groupby(covariate, as_index=False)[data.columns[-3:]].mean()


def get_model_terms(model: Model) -> dict:
    """
    Loops through the distributional components of a bambi model and
    returns a dictionary of terms.
    """
    terms = {}
    for component in model.distributional_components.values():
        if component.design.common:
            terms.update(component.design.common.terms)

        if component.design.group:
            terms.update(component.design.group.terms)

    return terms


def get_model_covariates(model: Model):
    """
    Return covariates specified in the model.
    """

    terms = get_model_terms(model)
    names = []
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names.append([arg.name for arg in component.call.args])
                else:
                    names.append([component.name])

    return np.unique(names)


def get_covariates(covariates: dict) -> Covariates:
    """
    Obtain the main, group, and panel covariates from the user's
    conditional dict.
    """
    covariate_kinds = ("main", "group", "panel")
    if any(key in covariate_kinds for key in covariates.keys()):
        # default if user did not pass their own conditional dict
        main = covariates.get("main")
        group = covariates.get("group", None)
        panel = covariates.get("panel", None)
    else:
        # assign main, group, panel based on the number of variables
        # passed by the user in their conditional dict
        length = len(covariates.keys())
        if length == 1:
            main = covariates.keys()
            group = None
            panel = None
        elif length == 2:
            main, group = covariates.keys()
            panel = None
        elif length == 3:
            main, group, panel = covariates.keys()

    return Covariates(main, group, panel)


def enforce_dtypes(data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce dtypes of the original data to the new data.
    """
    observed_dtypes = data.dtypes
    for col in df.columns:
        if col in observed_dtypes.index:
            df[col] = df[col].astype(observed_dtypes[col])
    return df


def make_group_panel_values(
    data: pd.DataFrame,
    data_dict: dict,
    main: str,
    group: Union[str, None],
    panel: Union[str, None],
    kind: str,
    groups_n: int = 5,
) -> dict:
    """
    Compute group and panel values based on original data.
    """

    # If available, obtain groups for grouping variable
    if group:
        group_values = make_group_values(data[group], groups_n)
        group_n = len(group_values)

    # If available, obtain groups for panel variable. Same logic than grouping applies
    if panel:
        panel_values = make_group_values(data[panel], groups_n)
        panel_n = len(panel_values)

    main_values = data_dict[main]
    main_n = len(main_values)

    if kind == "predictions":
        if group and not panel:
            main_values = np.tile(main_values, group_n)
            group_values = np.repeat(group_values, main_n)
            data_dict.update({main: main_values, group: group_values})
        elif not group and panel:
            main_values = np.tile(main_values, panel_n)
            panel_values = np.repeat(panel_values, main_n)
            data_dict.update({main: main_values, panel: panel_values})
        elif group and panel:
            if group == panel:
                main_values = np.tile(main_values, group_n)
                group_values = np.repeat(group_values, main_n)
                data_dict.update({main: main_values, group: group_values})
            else:
                main_values = np.tile(np.tile(main_values, group_n), panel_n)
                group_values = np.tile(np.repeat(group_values, main_n), panel_n)
                panel_values = np.repeat(panel_values, main_n * group_n)
                data_dict.update({main: main_values, group: group_values, panel: panel_values})
    elif kind == "comparison":
        # for comparisons, we need unique values for numeric and categorical
        # group/panel covariates since we iterate over pairwise combinations of values
        if group and not panel:
            data_dict.update({group: np.unique(group_values)})
        elif group and panel:
            data_dict.update({group: np.unique(group_values), panel: np.unique(panel_values)})

    return data_dict


def set_default_values(model: Model, data_dict: dict, kind: str):
    """
    Set default values for each variable in the model if the user did not
    pass them in the data_dict.
    """
    assert kind in [
        "comparison",
        "predictions",
    ], "kind must be either 'comparison' or 'predictions'"

    terms = get_model_terms(model)

    # Get default values for each variable in the model
    # pylint: disable=R1702
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]
                for name in names:
                    if name not in data_dict:
                        # For numeric predictors, select the mean.
                        if component.kind == "numeric":
                            data_dict[name] = np.mean(model.data[name])
                        # For categoric predictors, select the most frequent level.
                        elif component.kind == "categoric":
                            data_dict[name] = mode(model.data[name])

    if kind == "comparison":
        # if value in dict is not a list then convert to a list
        for key, value in data_dict.items():
            if not isinstance(value, (list, np.ndarray)):
                data_dict[key] = [value]
        return data_dict
    elif kind == "predictions":
        return data_dict
    else:
        return None


def set_default_contrast_values(model: Model, contrast_predictor: str) -> Union[list, np.ndarray]:
    """
    Set the default contrast value for the contrast predictor based on the
    contrast predictor dtype.
    """

    def _numeric_difference(x):
        """
        Centered difference for numeric predictors results in a default contrast
        of a 1 unit increase
        """
        return np.array([x - 0.5, x + 0.5])

    terms = get_model_terms(model)
    contrast_dtype = model.data[contrast_predictor].dtype

    # Get default values for each variable in the model
    # pylint: disable=R1702
    for term in terms.values():
        if hasattr(term, "components"):
            for component in term.components:
                # If the component is a function call, use the argument names
                if isinstance(component, Call):
                    names = [arg.name for arg in component.call.args]
                else:
                    names = [component.name]
                for name in names:
                    if name == contrast_predictor:
                        # For numeric predictors, select the mean.
                        if component.kind == "numeric":
                            contrast = _numeric_difference(np.mean(model.data[name])).astype(
                                contrast_dtype
                            )
                        # For categoric predictors, select the most frequent level.
                        elif component.kind == "categoric":
                            contrast = get_unique_levels(model.data[name])

    return contrast


def make_main_values(x: np.ndarray, grid_n: int = 50) -> np.ndarray:
    """
    Compuet main values based on original data using a grid of evenly spaced
    values for numeric predictors and unique levels for categoric predictors.
    """
    if is_numeric_dtype(x):
        return np.linspace(np.min(x), np.max(x), grid_n)
    elif is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    raise ValueError("Main covariate must be numeric or categoric.")


def make_group_values(x: np.ndarray, groups_n: int = 5) -> np.ndarray:
    """
    Compute group values based on original data using unique levels for
    categoric predictors and quantiles for numeric predictors.
    """
    if is_string_dtype(x) or is_categorical_dtype(x):
        return np.unique(x)
    elif is_numeric_dtype(x):
        return np.quantile(x, np.linspace(0, 1, groups_n))
    raise ValueError("Group covariate must be numeric or categoric.")


def get_unique_levels(x: np.ndarray) -> Union[list, np.ndarray]:
    """
    Get unique levels of a categoric variable.
    """
    if hasattr(x, "dtype") and hasattr(x.dtype, "categories"):
        levels = list(x.dtype.categories)
    else:
        levels = np.unique(x)
    return levels


def get_group_offset(n, lower: float = 0.05, upper: float = 0.4) -> np.ndarray:
    # Complementary log log function, scaled.
    # See following code to have an idea of how this function looks like
    # lower, upper = 0.05, 0.4
    # x = np.linspace(2, 9)
    # y = get_group_offset(x, lower, upper)
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(x, y)
    # ax.axvline(2, color="k", ls="--")
    # ax.axhline(lower, color="k", ls="--")
    # ax.axhline(upper, color="k", ls="--")
    intercept, slope = 3.25, 1
    return lower + np.exp(-np.exp(intercept - slope * n)) * (upper - lower)


def identity(x):
    return x
