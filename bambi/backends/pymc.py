import re
from collections import OrderedDict

import numpy as np
import pymc3 as pm
from arviz import from_pymc3
from pymc3.model import TransformedRV
import theano
from bambi.priors import Prior

from .base import BackEnd


class PyMC3BackEnd(BackEnd):

    """
    PyMC3 model-fitting back-end.
    """

    # Available link functions
    links = {
        "identity": lambda x: x,
        "logit": theano.tensor.nnet.sigmoid,
        "inverse": theano.tensor.inv,
        "inverse_squared": lambda x: theano.tensor.inv(theano.tensor.sqrt(x)),
        "log": theano.tensor.exp,
    }

    dists = {"HalfFlat": pm.Bound(pm.Flat, lower=0)}

    def __init__(self):

        self.reset()

        # Attributes defined elsewhere
        self.mu = None  # build()
        self.spec = None  # build()
        self.trace = None  # build()
        self.advi_params = None  # build()

    def reset(self):
        """
        Reset PyMC3 model and all tracked distributions and parameters.
        """
        self.model = pm.Model()
        self.mu = None
        self.par_groups = {}

    def _build_dist(self, spec, label, dist, **kwargs):
        """ Build and return a PyMC3 Distribution. """
        if isinstance(dist, str):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            elif dist in self.dists:
                dist = self.dists[dist]
            else:
                raise ValueError(
                    "The Distribution class '%s' was not "
                    "found in PyMC3 or the PyMC3BackEnd." % dist
                )

        # Inspect all args in case we have hyperparameters
        def _expand_args(key, value, label):
            if isinstance(value, Prior):
                label = "%s_%s" % (label, key)
                return self._build_dist(spec, label, value.name, **value.args)
            return value

        kwargs = {k: _expand_args(k, v, label) for (k, v) in kwargs.items()}

        # Non-centered parameterization for hyperpriors
        if (
            spec.noncentered
            and "sd" in kwargs
            and "observed" not in kwargs
            and isinstance(kwargs["sd"], pm.model.TransformedRV)
        ):
            old_sd = kwargs["sd"]
            _offset = pm.Normal(label + "_offset", mu=0, sd=1, shape=kwargs["shape"])
            return pm.Deterministic(label, _offset * old_sd)

        return dist(label, **kwargs)

    def build(self, spec, reset=True):  # pylint: disable=arguments-differ
        """
        Compile the PyMC3 model from an abstract model specification.
        
        Parameters
        ----------
        spec : Bambi model
            A bambi Model instance containing the abstract specification of the model to compile.
        reset : Bool
            If True (default), resets the PyMC3BackEnd instance before compiling.
        """
        if reset:
            self.reset()

        with self.model:
            self.mu = 0.0

            for t in spec.terms.values():
                data = t.data
                label = t.name
                dist_name = t.prior.name
                dist_args = t.prior.args

                n_cols = t.data.shape[1]

                coef = self._build_dist(spec, label, dist_name, shape=n_cols, **dist_args)

                if t.random:
                    self.mu += coef[t.group_index][:, None] * t.predictor
                else:
                    self.mu += pm.math.dot(data, coef)[:, None]

            y = spec.y.data
            y_prior = spec.family.prior
            link_f = spec.family.link

            if isinstance(link_f, str):
                link_f = self.links[link_f]
            else:
                link_f = link_f
            y_prior.args[spec.family.parent] = link_f(self.mu)
            y_prior.args["observed"] = y
            self._build_dist(spec, spec.y.name, y_prior.name, **y_prior.args)
            self.spec = spec

    def _get_transformed_vars(self):
        rvs = self.model.unobserved_RVs
        # identify the variables that pymc3 back-transformed to original scale
        trans = [var.name for var in rvs if isinstance(var, TransformedRV)]
        # find the corresponding transformed variables
        trans = set(x.name for x in rvs if any([t in x.name for t in trans])) - set(trans)
        # add any "centered" random effects to the list
        for varname in [var.name for var in rvs]:
            if re.search(r"_offset$", varname) is not None:
                trans.add(varname)

        return trans

    # pylint: disable=arguments-differ, inconsistent-return-statements
    def run(self, start=None, method="mcmc", init="auto", n_init=50000, **kwargs):
        """
        Run the PyMC3 MCMC sampler.

        Parameters
        ----------
        start: dict, or array of dict
            Starting parameter values to pass to sampler; see ``'pm.sample()'`` for details.
        method: str
            The method to use for fitting the model. By default, 'mcmc', in which case the
            PyMC3 sampler will be used. Alternatively, 'advi', in which case the model will be fitted
            using  automatic differentiation variational inference as implemented in PyMC3.
        init: str
            Initialization method (see PyMC3 sampler documentation). Currently, this is
            ``'jitter+adapt_diag'``, but this can change in the future.
        n_init: int 
            Number of initialization iterations if init = 'advi' or 'nuts'. Default is kind of in
            PyMC3 for the kinds of models we expect to see run with bambi, so we lower it
            considerably.

        Returns
        -------
        An ArviZ InferenceData instance.
        """

        if method == "mcmc":
            samples = kwargs.pop("samples", 1000)
            cores = kwargs.pop("chains", 1)
            with self.model:
                self.trace = pm.sample(
                    samples, start=start, init=init, n_init=n_init, cores=cores, **kwargs
                )
            return from_pymc3(self.trace)

        elif method == "advi":
            with self.model:
                self.advi_params = pm.variational.ADVI(start, **kwargs)
            return (
                self.advi_params
            )  # this should return an InferenceData object (once arviz adds support for VI)
