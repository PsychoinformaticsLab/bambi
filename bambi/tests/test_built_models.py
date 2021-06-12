import arviz as az
import numpy as np
import pandas as pd
import pytest
import theano.tensor as tt

from bambi.models import Model
from bambi.priors import Prior


@pytest.fixture(scope="module")
def crossed_data():
    """
    Group specific effects:
    10 subjects, 12 items, 5 sites
    Subjects crossed with items, nested in sites
    Items crossed with sites

    common effects:
    A continuous predictor, a numeric dummy, and a three-level category
    (levels a,b,c)

    Structure:
    Subjects nested in dummy (e.g., gender), crossed with threecats
    Items crossed with dummy, nested in threecats
    Sites partially crossed with dummy (4/5 see a single dummy, 1/5 sees both
    dummies)
    Sites crossed with threecats
    """
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "crossed_random.csv"))
    return data


@pytest.fixture(scope="module")
def dm():
    """
    Data obtained from from https://github.com/jswesner/nps_emergence/tree/v2_nps_emerge
    and used in Gamma GLM
    """
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "dm.csv"))
    return data


def test_empty_model(crossed_data):
    model0 = Model("Y ~ 0", crossed_data)
    model0.fit(tune=0, draws=1)


def test_intercept_only_model(crossed_data):
    model0 = Model("Y ~ 1", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_slope_only_model(crossed_data):
    model0 = Model("Y ~ 0 + continuous", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_cell_means_parameterization(crossed_data):
    model0 = Model("Y ~ 0 + threecats", crossed_data)
    model0.fit(tune=0, draws=1, init=None)


def test_3x4_common_anova(crossed_data):
    # add a four-level category that's perfectly crossed with threecats
    crossed_data["fourcats"] = sum([[x] * 10 for x in ["a", "b", "c", "d"]], list()) * 3

    # with intercept
    model0 = Model("Y ~ threecats*fourcats", crossed_data)
    fitted0 = model0.fit(tune=0, draws=1, init=None)
    assert len(fitted0.posterior.data_vars) == 5

    # without intercept (i.e., 2-factor cell means model)
    model1 = Model("Y ~ 0 + threecats*fourcats", crossed_data)
    fitted1 = model1.fit(tune=0, draws=1)
    assert len(fitted1.posterior.data_vars) == 4


def test_cell_means_with_covariate(crossed_data):
    model0 = Model("Y ~ 0 + threecats + continuous", crossed_data)
    model0.fit(tune=0, draws=1, init=None)

    # check that threecats priors have finite variance
    assert not (np.isinf(model0.terms["threecats"].prior.args["sigma"])).all()


def test_many_common_many_group_specific(crossed_data):
    # This test is kind of a mess, but it is very important, it checks lots of things.
    # delete a few values to also test dropna=True functionality
    crossed_data_missing = crossed_data.copy()
    crossed_data_missing.loc[0, "Y"] = np.nan
    crossed_data_missing.loc[1, "continuous"] = np.nan
    crossed_data_missing.loc[2, "threecats"] = np.nan

    # Here I'm comparing implicit/explicit intercepts for group specific effects workY the same way.
    model0 = Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (1|item) + (0+continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model0.fit(
        init=None,
        tune=10,
        draws=10,
        chains=2,
    )

    model1 = Model(
        "Y ~ continuous + dummy + threecats + (threecats|subj) + (continuous|item) + (dummy|item) + (threecats|site)",
        crossed_data_missing,
        dropna=True,
    )
    model1.fit(
        tune=10,
        draws=10,
        chains=2,
    )
    # check that the group specific effects design matrices have the same shape
    X0 = pd.concat([pd.DataFrame(t.data) for t in model0.group_specific_terms.values()], axis=1)
    X1 = pd.concat([pd.DataFrame(t.data) for t in model1.group_specific_terms.values()], axis=1)
    assert X0.shape == X1.shape

    # check that the group specific effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )

    assert X0 == X1

    # check that models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])

    # check that fit and add models have same priors for group specific effects
    priors0 = {x.name: x.prior.args["sigma"].args for x in model0.group_specific_terms.values()}
    priors1 = {x.name: x.prior.args["sigma"].args for x in model1.group_specific_terms.values()}

    # check dictionary keys
    assert set(priors0) == set(priors1)

    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_cell_means_with_many_group_specific_effects(crossed_data):
    # Group specific intercepts are added in different way, but the final result should be the same.
    formula = "Y ~" + "+".join(
        [
            "0",
            "threecats",
            "(threecats|subj)",
            "(1|subj)",
            "(0 + continuous|item)",
            "(dummy|item)",
            "(0 + threecats|site)",
            "(1|site)",
        ]
    )
    model0 = Model(formula, crossed_data)
    model0.fit(tune=0, draws=1)

    formula = "Y ~" + "+".join(
        [
            "0",
            "threecats",
            "(threecats|subj)",
            "(continuous|item)",
            "(dummy|item)",
            "(threecats|site)",
        ]
    )
    model1 = Model(formula, crossed_data)
    model1.fit(tune=0, draws=1)

    # check that the group specific effects design matrices have the same shape
    X0 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.group_specific_terms.values()
        ],
        axis=1,
    )
    X1 = pd.concat(
        [
            pd.DataFrame(t.data)
            if not isinstance(t.data, dict)
            else pd.concat([pd.DataFrame(t.data[x]) for x in t.data.keys()], axis=1)
            for t in model0.group_specific_terms.values()
        ],
        axis=1,
    )
    assert X0.shape == X1.shape

    # check that the group specific effect design matrix contain the same columns,
    # even if term names / columns names / order of columns is different
    X0_set = set(tuple(X0.iloc[:, i]) for i in range(len(X0.columns)))
    X1_set = set(tuple(X1.iloc[:, i]) for i in range(len(X1.columns)))
    assert X0_set == X1_set

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    assert X0 == X1

    # check that fit and add models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    assert set(priors0) == set(priors1)

    # check that fit and add models have same priors for group specific effects
    priors0 = {
        x.name: x.prior.args["sigma"].args for x in model0.terms.values() if x.group_specific
    }
    priors1 = {
        x.name: x.prior.args["sigma"].args for x in model1.terms.values() if x.group_specific
    }
    assert set(priors0) == set(priors1)


@pytest.mark.skip(reason="We are correctly handling string links only, not functions.")
def test_logistic_regression(crossed_data):
    # Tests passing link="logit" is equivalent to using tt.nnet.sigmoid
    model0 = Model(
        "threecats['b'] ~ continuous + dummy", crossed_data, family="bernoulli", link="logit"
    )
    fitted0 = model0.fit(
        tune=0,
        draws=1000,
    )

    # build model using fit, pymc3 and theano link function
    model1 = Model(
        "threecats['b'] ~ continuous + dummy",
        crossed_data,
        family="bernoulli",
        link=tt.nnet.sigmoid,
    )
    fitted1 = model1.fit(
        tune=0,
        draws=1000,
    )

    # check that using a theano link function works
    assert np.allclose(az.summary(fitted0)["mean"], az.summary(fitted1)["mean"], atol=0.2)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )

    assert X0 == X1

    # check that models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_logistic_regression_empty_index():
    data = pd.DataFrame({"y": np.random.choice(["a", "b"], 50), "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_logistic_regression_good_numeric():
    data = pd.DataFrame({"y": np.random.choice([1, 0], 50), "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_logistic_regression_bad_numeric():
    data = pd.DataFrame({"y": np.random.choice([1, 2], 50), "x": np.random.normal(size=50)})
    with pytest.raises(ValueError):
        model = Model("y ~ x", data, family="bernoulli")
        model.fit()


def test_logistic_regression_categoric():
    y = pd.Series(np.random.choice(["a", "b"], 50), dtype="category")
    data = pd.DataFrame({"y": y, "x": np.random.normal(size=50)})
    model = Model("y ~ x", data, family="bernoulli")
    model.fit()


def test_poisson_regression(crossed_data):
    # build model using fit and pymc3
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model0 = Model("count ~ dummy + continuous + threecats", crossed_data, family="poisson")
    model0.fit(tune=0, draws=1)

    # build model using add
    model1 = Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    model1.fit(tune=0, draws=1)

    # check that term names agree
    assert set(model0.term_names) == set(model1.term_names)

    # check that common effect design matrices are the same,
    # even if term names / level names / order of columns is different
    X0 = set(
        [
            tuple(t.data[:, lev])
            for t in model0.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )
    X1 = set(
        [
            tuple(t.data[:, lev])
            for t in model1.common_terms.values()
            for lev in range(len(t.levels))
        ]
    )

    assert X0 == X1

    # check that models have same priors for common effects
    priors0 = {x.name: x.prior.args for x in model0.terms.values() if not x.group_specific}
    priors1 = {x.name: x.prior.args for x in model1.terms.values() if not x.group_specific}
    # check dictionary keys
    assert set(priors0) == set(priors1)
    # check dictionary values
    def dicts_close(a, b):
        if set(a) != set(b):
            return False
        else:
            return [np.allclose(a[x], b[x], atol=0, rtol=0.01) for x in a.keys()]

    assert all([dicts_close(priors0[x], priors1[x]) for x in priors0.keys()])


def test_laplace():
    data = pd.DataFrame(np.repeat((0, 1), (30, 60)), columns=["w"])
    priors = {"Intercept": Prior("Uniform", lower=0, upper=1)}
    model = Model("w ~ 1", data=data, family="bernoulli", priors=priors, link="identity")
    results = model.fit(method="laplace")
    mode_n = np.round(results["Intercept"][0], 2)
    std_n = np.round(results["Intercept"][1][0], 2)
    mode_a = data.mean()
    std_a = data.std() / len(data) ** 0.5
    np.testing.assert_array_almost_equal((mode_n, std_n), (mode_a.item(), std_a.item()), decimal=2)


def test_prior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model = Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    model.fit(tune=0, draws=2)
    pps = model.prior_predictive(draws=500)

    keys = ["Intercept", "threecats", "continuous", "dummy"]
    shapes = [(1, 500), (1, 500, 2), (1, 500), (1, 500)]

    for key, shape in zip(keys, shapes):
        assert pps.prior[key].shape == shape

    assert pps.prior_predictive["count"].shape == (1, 500, 120)
    assert pps.observed_data["count"].shape == (120,)

    pps = model.prior_predictive(draws=500, var_names=["count"])
    assert pps.groups() == ["prior_predictive", "observed_data"]

    pps = model.prior_predictive(draws=500, var_names=["Intercept"])
    assert pps.groups() == ["prior"]


def test_posterior_predictive(crossed_data):
    crossed_data["count"] = (crossed_data["Y"] - crossed_data["Y"].min()).round()
    model = Model("count ~ threecats + continuous + dummy", crossed_data, family="poisson")
    fitted = model.fit(tune=0, draws=2)
    pps = model.posterior_predictive(fitted, draws=500, inplace=False)

    assert pps.posterior_predictive["count"].shape == (1, 500, 120)

    pps = model.posterior_predictive(fitted, draws=500, inplace=True)

    assert pps is None
    assert fitted.posterior_predictive["count"].shape == (1, 500, 120)


def test_gamma_regression(dm):
    # simulated data
    np.random.seed(1234)
    N = 100
    x = np.random.uniform(-1, 1, N)
    a = 0.5
    b = 1.2
    y_true = np.exp(a + b * x)
    shape_true = 10  # alpha

    y = np.random.gamma(shape_true, y_true / shape_true, N)
    data = pd.DataFrame({"x": x, "y": y})
    model = Model("y ~ x", data, family="gamma", link="log")
    model.fit()

    # Real data, categorical predictor.
    data = dm[["order", "ind_mg_dry"]]
    model = Model("ind_mg_dry ~ order", data, family="gamma", link="log")
    model.fit()
