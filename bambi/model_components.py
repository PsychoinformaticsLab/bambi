import numpy as np
import xarray as xr

from bambi.families import univariate, multivariate
from bambi.priors import Prior
from bambi.terms import CommonTerm, GroupSpecificTerm, OffsetTerm, ResponseTerm


class ConstantComponent:
    def __init__(self, name, prior, response_name, spec):
        self.name = with_suffix(response_name, name)
        self.prior = prior
        self.spec = spec

    def update_priors(self, value):
        self.prior = value


class DistributionalComponent:
    """_summary_
    response_kind = ["data", "parameter"]
    """

    def __init__(self, design, priors, response_name, response_kind, spec):
        self.terms = {}
        self.design = design
        self.response_name = response_name
        self.response_kind = response_kind
        self.response_term = None
        self.spec = spec

        if self.response_kind == "data":
            self.prefix = ""
        else:
            self.prefix = response_name

        if self.design.common:
            self.add_common_terms(priors)

        if self.design.group:
            self.add_group_specific_terms(priors)

        if self.design.response:
            self.add_response_term()

    def add_common_terms(self, priors):
        for name, term in self.design.common.terms.items():
            prior = priors.pop(name, priors.get("common", None))
            if isinstance(prior, Prior):
                any_hyperprior = any(isinstance(x, Prior) for x in prior.args.values())
                if any_hyperprior:
                    raise ValueError(
                        f"Trying to set hyperprior on '{name}'. "
                        "Can't set a hyperprior on common effects."
                    )

            if term.kind == "offset":
                self.terms[name] = OffsetTerm(term, term.data, self.prefix)
            else:
                self.terms[name] = CommonTerm(term, prior, self.prefix)

    def add_group_specific_terms(self, priors):
        for name, term in self.design.group.terms.items():
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior, self.prefix)

    def add_response_term(self):
        """Add a response (or outcome/dependent) variable to the model."""
        response = self.design.response

        if hasattr(response.term.term.components[0], "reference"):
            reference = response.term.term.components[0].reference
        else:
            reference = None

        # NOTE: This is a historical feature.
        # I'm not sure how many family specific checks we should add in this type of places now
        if reference is not None and not isinstance(self.spec.family, univariate.Bernoulli):
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        if isinstance(self.spec.family, univariate.Bernoulli):
            if response.kind == "categoric" and response.levels is None and reference is None:
                raise ValueError("Categoric response must be binary for 'bernoulli' family.")
            if response.kind == "numeric" and not all(np.isin(response.design_matrix, [0, 1])):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

        self.response_term = ResponseTerm(response, self.spec.family)

    def update_priors(self, priors):
        """Update priors

        Parameters
        ----------
        priors : dict
            Names are terms, values a priors.
        """
        for name, value in priors.items():
            self.terms[name].prior = value

    def predict(self, idata, data=None, include_group_specific=True):
        linear_predictor = 0
        x_offsets = []
        posterior = idata.posterior
        in_sample = data is None
        family = self.spec.family

        # Prepare dims objects
        response_dim = self.spec.response_name + "_obs"
        response_levels_dim = self.spec.response_name + "_dim"
        linear_predictor_dims = ("chain", "draw", response_dim)
        to_stack_dims = ("chain", "draw")
        design_matrix_dims = (response_dim, "__variables__")

        if isinstance(self.spec.family, multivariate.MultivariateFamily):
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)

        if self.design.common:
            if in_sample:
                X = self.design.common.design_matrix
            else:
                X = self.design.common.evaluate_new_data(data).design_matrix

            # Add offset columns to their own design matrix and remove then from common matrix
            for term in self.offset_terms:
                term_slice = self.design.common.slices[term]
                x_offsets.append(X[:, term_slice])
                X = np.delete(X, term_slice, axis=1)

            # Create DataArray
            X_terms = [with_prefix(name, self.prefix) for name in self.common_terms]
            if self.intercept_term:
                X_terms.insert(0, with_prefix("Intercept", self.prefix))
            b = posterior[X_terms].to_stacked_array("__variables__", to_stack_dims)

            # Add contribution due to the common terms
            X = xr.DataArray(X, dims=design_matrix_dims)
            linear_predictor += xr.dot(X, b)

        if self.design.group and include_group_specific:
            if in_sample:
                Z = self.design.group.design_matrix
            else:
                Z = self.design.group.evaluate_new_data(data).design_matrix

            # Create DataArray
            Z_terms = [with_prefix(name, self.prefix) for name in self.group_specific_terms]
            u = posterior[Z_terms].to_stacked_array("__variables__", to_stack_dims)

            # Add contribution due to the group specific terms
            Z = xr.DataArray(Z, dims=design_matrix_dims)
            linear_predictor += xr.dot(Z, u)

        # If model contains offsets, add them directly to the linear predictor
        if x_offsets:
            linear_predictor += np.column_stack(x_offsets).sum(axis=1)[:, np.newaxis, np.newaxis]

        # Sort dimensions
        linear_predictor = linear_predictor.transpose(*linear_predictor_dims)

        # Add coordinates for the observation number
        obs_n = len(linear_predictor[response_dim])
        linear_predictor = linear_predictor.assign_coords({response_dim: list(range(obs_n))})

        if hasattr(family, "transform_linear_predictor"):
            linear_predictor = family.transform_linear_predictor(self.spec, linear_predictor)

        if hasattr(family, "UFUNC_KWARGS"):
            ufunc_kwargs = family.UFUNC_KWARGS
        else:
            ufunc_kwargs = {}

        if self.response_kind == "data":
            linkinv = family.link[family.likelihood.parent].linkinv
        else:
            linkinv = family.link[self.response_name].linkinv

        response = xr.apply_ufunc(linkinv, linear_predictor, kwargs=ufunc_kwargs)

        if hasattr(family, "transform_coords"):
            response = family.transform_coords(self.spec, response)

        return response

    @property
    def group_specific_groups(self):
        groups = {}
        for term_name in self.group_specific_terms:
            factor = term_name.split("|")[1]
            if factor not in groups:
                groups[factor] = [term_name]
            else:
                groups[factor].append(term_name)
        return groups

    @property
    def intercept_term(self):
        """Return the intercept term in a model component."""
        term = [
            v for v in self.terms.values() if isinstance(v, CommonTerm) and v.kind == "intercept"
        ]
        if term:
            return term[0]

    @property
    def common_terms(self):
        """Return dict of all common effects in the model component."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if isinstance(v, CommonTerm) and v.kind != "intercept"
        }

    @property
    def group_specific_terms(self):
        """Return dict of all group specific effects in model component."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, GroupSpecificTerm)}

    @property
    def offset_terms(self):
        """Return dict of all offset effects in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, OffsetTerm)}


def with_suffix(value, suffix):
    if suffix:
        return f"{value}_{suffix}"
    return value


def with_prefix(value, prefix):
    if prefix:
        return f"{prefix}_{value}"
    return value
