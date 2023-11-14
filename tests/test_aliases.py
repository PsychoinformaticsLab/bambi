import pytest

import bambi as bmb
import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def my_data():
    return bmb.load_data("my_data")


@pytest.fixture(scope="module")
def anes():
    return bmb.load_data("ANES")


@pytest.fixture(scope="module")
def data_100():
    size = 100
    rng = np.random.default_rng(121195)
    data = pd.DataFrame(
        {
            "n1": rng.normal(size=size),
            "n2": rng.normal(size=size),
            "n3": rng.normal(size=size),
            "b0": rng.binomial(n=1, p=0.5, size=size),
            "b1": rng.choice(["a", "b"], size=size),
            "count1": rng.poisson(lam=2, size=size),
            "count2": rng.poisson(lam=2, size=size),
            "cat1": rng.choice(list("MNOP"), size=size),
            "cat2": rng.choice(list("FGHIJK"), size=size),
        }
    )
    return data


def test_non_distributional_model(my_data):
    # Plain model
    formula = bmb.Formula("y ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "y_obs"]
    assert set(idata.posterior.data_vars) == {"Intercept", "x", "y_mean", "y_sigma"}
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]

    # Model with alises
    model.set_alias({"Intercept": "a", "x": "b", "sigma": "s", "y": "response"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "response_obs"]
    assert set(idata.posterior.data_vars) == {"a", "b", "response_mean", "s"}
    assert list(idata.posterior["response_mean"].coords) == ["chain", "draw", "response_obs"]


def test_distributional_model(my_data):
    formula = bmb.Formula("y ~ x", "sigma ~ x")
    model = bmb.Model(formula, my_data)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "y_obs"]
    assert set(idata.posterior.data_vars) == {
        "Intercept",
        "x",
        "sigma_Intercept",
        "sigma_x",
        "y_sigma",
        "y_mean",
    }
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]
    assert list(idata.posterior["y_sigma"].coords) == ["chain", "draw", "y_obs"]

    aliases = {
        "y": {"Intercept": "y_a", "x": "y_b", "y": "response"},
        "sigma": {"Intercept": "sigma_a", "x": "sigma_b", "sigma": "s"},
    }
    model.set_alias(aliases)
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)

    assert list(idata.posterior.coords) == ["chain", "draw", "response_obs"]
    assert set(idata.posterior.data_vars) == {
        "response_mean",
        "y_a",
        "y_b",
        "sigma_a",
        "sigma_b",
        "s",
    }
    assert list(idata.posterior["response_mean"].coords) == ["chain", "draw", "response_obs"]
    assert list(idata.posterior["s"].coords) == ["chain", "draw", "response_obs"]


def test_non_distributional_model_with_categories(anes):
    model = bmb.Model("vote[clinton] ~ age + age:party_id", anes, family="bernoulli")
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "age:party_id_dim", "vote_obs"]
    assert set(idata.posterior.data_vars) == {"Intercept", "age", "age:party_id", "vote_mean"}
    assert list(idata.posterior["vote_mean"].coords) == ["chain", "draw", "vote_obs"]
    assert list(idata.posterior["age:party_id"].coords) == ["chain", "draw", "age:party_id_dim"]
    assert set(idata.posterior["age:party_id_dim"].values) == {"independent", "republican"}

    model.set_alias({"age": "β", "Intercept": "α", "age:party_id": "γ", "vote": "y"})
    idata = model.fit(tune=100, draws=100)
    model.predict(idata)
    assert list(idata.posterior.coords) == ["chain", "draw", "γ_dim", "y_obs"]
    assert set(idata.posterior.data_vars) == {"α", "β", "γ", "y_mean"}
    assert list(idata.posterior["y_mean"].coords) == ["chain", "draw", "y_obs"]
    assert list(idata.posterior["γ"].coords) == ["chain", "draw", "γ_dim"]
    assert set(idata.posterior["γ_dim"].values) == {"independent", "republican"}


def test_alias_equal_to_name(my_data):
    model = bmb.Model("y ~ 1 + x", my_data)
    model.set_alias({"sigma": "sigma"})
    idata = model.fit(tune=100, draws=100)
    set(idata.posterior.data_vars) == {"Intercept", "y_mean", "x", "sigma"}


def test_set_alias_warnings(my_data):
    # Create a model to use aliases on
    formula = bmb.Formula("y ~ x")
    model = bmb.Model(formula, my_data)

    # Define cases that throw the various warnings
    test_cases = [
        # Only one unused alias, explicitly tell user the name
        (
            {"unused_alias": "ua"},
            "The following names do not match any terms, "
            "their aliases were not assigned: unused_alias",
        ),
        # Many unused aliases, generic response
        (
            {f"unused_alias{i}": f"ua{i}" for i in range(6)},
            "There are 6 names that do not match any terms, so their aliases were not assigned.",
        ),
    ]

    # Evaluate each case
    for alias_dict, expected_warning in test_cases:
        with pytest.warns(UserWarning) as record:
            model.set_alias(alias_dict)
            print(model.constant_components)
        assert len(record) == 1
        assert str(record[0].message) == expected_warning


# FIXME: Move somewhere
def test_set_alias(data_100):
    model = bmb.Model("n1 ~ n2 + (n2|cat1)", data_100)
    aliases = {
        "Intercept": "α",
        "n2": "β",
        "1|cat1": "α_group",
        "n2|cat1": "β_group",
        "sigma": "σ",
    }
    model.set_alias(aliases)
    model.build()
    new_names = set(["α", "β", "α_group", "α_group_σ", "β_group", "β_group_σ", "σ"])
    assert new_names.issubset(set(model.backend.model.named_vars))
