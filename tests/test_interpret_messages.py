import bambi as bmb
import pandas as pd
import pytest

from bambi.interpret import plot_comparisons, plot_predictions, plot_slopes


@pytest.fixture(scope="module")
def mtcars():
    "Model with common level effects only"
    data = bmb.load_data("mtcars")
    data["am"] = pd.Categorical(data["am"], categories=[0, 1], ordered=True)
    model = bmb.Model("mpg ~ hp * drat * am", data)
    idata = model.fit(tune=500, draws=500, random_seed=1234)
    return model, idata


# Use caplog fixture to capture log messages generated by the interpret logger
def test_predictions_list(mtcars, caplog):
    model, idata = mtcars
    caplog.set_level("INFO", logger="__bambi_interpret__")

    # List of values with no unspecified covariates
    conditional = ["hp", "drat", "am"]
    plot_predictions(model, idata, conditional)

    main_msg = "Default computed for main variable: hp"
    group_panel_msg = "Default computed for group/panel variable: drat, am"
    interpret_log_msgs = [r.message for r in caplog.records]

    assert main_msg in interpret_log_msgs
    assert group_panel_msg in interpret_log_msgs
    assert len(caplog.records) == 2


def test_predictions_list_unspecified(mtcars, caplog):
    model, idata = mtcars
    caplog.set_level("INFO", logger="__bambi_interpret__")

    # List of values with unspecified covariates
    conditional = ["hp", "drat"]
    plot_predictions(model, idata, conditional)

    main_msg = "Default computed for main variable: hp"
    group_msg = "Default computed for group/panel variable: drat"
    unspecified_msg = "Default computed for unspecified variable: am"
    interpret_log_msgs = [r.message for r in caplog.records]

    assert main_msg in interpret_log_msgs
    assert group_msg in interpret_log_msgs
    assert unspecified_msg in interpret_log_msgs
    assert len(caplog.records) == 3


def test_predictions_dict_unspecified(mtcars, caplog):
    model, idata = mtcars
    caplog.set_level("INFO", logger="__bambi_interpret__")

    # User passed values with unspecified covariates
    conditional = {"hp": [110, 175], "am": [0, 1]}
    plot_predictions(model, idata, conditional)

    unspecified_msg = "Default computed for unspecified variable: drat"
    interpret_log_msgs = [r.message for r in caplog.records]

    assert unspecified_msg in interpret_log_msgs
    assert len(caplog.records) == 1


# Since the 'predictions' test functions above test all three scenarios of
# 'conditional', i.e., grid (list of covariates names), user-passed (dict of
# covariate names and values), and unspecified (did not pass a covariate that
# was specified in the model formula), we only need to test the 'comparisons'
# and 'slopes' functions for default computation of 'contrast' and 'wrt'.


def test_comparisons_contrast_default(mtcars, caplog):
    model, idata = mtcars
    caplog.set_level("INFO", logger="__bambi_interpret__")

    # List of values with no unspecified covariates
    plot_comparisons(model, idata, "hp", conditional=None, average_by="am")

    contrast_msg = "Default computed for contrast variable: hp"
    interpret_log_msgs = [r.message for r in caplog.records]

    assert contrast_msg in interpret_log_msgs
    assert len(caplog.records) == 1


def test_slopes_wrt_default(mtcars, caplog):
    model, idata = mtcars
    caplog.set_level("INFO", logger="__bambi_interpret__")

    # List of values with no unspecified covariates
    plot_slopes(model, idata, "hp", conditional=None, average_by="am")

    wrt_msg = "Default computed for wrt variable: hp"
    interpret_log_msgs = [r.message for r in caplog.records]

    assert wrt_msg in interpret_log_msgs
    assert len(caplog.records) == 1
