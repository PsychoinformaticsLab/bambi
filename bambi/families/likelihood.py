from collections import namedtuple

from bambi.priors import Prior
from bambi.utils import multilinify, spacify


DistSettings = namedtuple("DistSettings", ["params", "parent"])

DISTRIBUTIONS = {
    "Bernoulli": DistSettings(params=("p",), parent="p"),
    "Beta": DistSettings(params=("mu", "kappa"), parent="mu"),
    "Binomial": DistSettings(params=("p",), parent="p"),
    "Categorical": DistSettings(params=("p",), parent="p"),
    "Gamma": DistSettings(params=("mu", "alpha"), parent="mu"),
    "Multinomial": DistSettings(params=("p",), parent="p"),
    "Normal": DistSettings(params=("mu", "sigma"), parent="mu"),
    "NegativeBinomial": DistSettings(params=("mu", "alpha"), parent="mu"),
    "Laplace": DistSettings(params=("mu", "b"), parent="mu"),
    "Poisson": DistSettings(params=("mu",), parent="mu"),
    "StudentT": DistSettings(params=("mu", "sigma"), parent="mu"),
    "VonMises": DistSettings(params=("mu", "kappa"), parent="mu"),
    "Wald": DistSettings(params=("mu", "lam"), parent="mu"),
}


class Likelihood:
    """Representation of a Likelihood function for a Bambi model.

    Notes:
    * ``parent`` must be in ``params``
    * ``parent`` is inferred from the ``name`` if it is a known name

    Parameters
    ----------
    name : str
        Name of the likelihood function. Must be a valid PyMC distribution name.
    params : Sequence[str]
        The name of the parameters the likelihood function accepts.
    parent : str
        Optional specification of the name of the mean parameter in the likelihood.
        This is the parameter whose transformation is modeled by the linear predictor.
    dist : pymc.distributions.distribution.DistributionMeta
        Optional custom PyMC distribution that will be used to compute the likelihood.
    """

    DISTRIBUTIONS = DISTRIBUTIONS

    def __init__(self, name, params=None, parent=None, dist=None):
        self.name = name
        self.params = params
        self.parent = parent
        self.dist = dist

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        if self.name in self.DISTRIBUTIONS:
            if value is None:
                value = self.DISTRIBUTIONS[self.name].params
            elif set(value) != set(self.DISTRIBUTIONS[self.name].params):
                raise ValueError(f"'{value}' does not match the parameters of '{self.name}'")
        # Otherwise, no check is done. At your own risk!
        self._params = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        # Checks are made when using a known distribution
        if self.name in self.DISTRIBUTIONS:
            if value is None:
                value = self.DISTRIBUTIONS[self.name].parent
            elif value not in self.DISTRIBUTIONS[self.name].params:
                raise ValueError(
                    f"'{value}' is not a valid parameter for the likelihood '{self.name}'"
                )
        elif value not in self.params:
            raise ValueError(f"'{value}' must be one of {self.params}")
        self._parent = value

    def __str__(self):
        args = [("name", self.name), ("params", self.params), ("parent", self.parent)]
        args = [f"{arg[0]}: {arg[1]}" for arg in args]
        return f"{self.__class__.__name__}({spacify(multilinify(args))}\n)"

    def __repr__(self):
        return self.__str__()


# def _check_priors(self, priors):
#     args = [param for param in self.params if param != self.parent]
#     if not args:
#         args = None

#     # The function requires priors but none were passed
#     if priors == {} and args is not None:
#         raise ValueError(f"'{self.name}' requires priors for the parameters {args}.")

#     # The function does not require priors, but at least one was passed
#     if priors != {} and args is None:
#         raise ValueError(f"'{self.name}' does not require any additional priors.")

#     # The function requires priors, priors were passed, but they differ from the required
#     if priors and args:
#         difference = set(args) - set(priors)
#         if len(difference):
#             raise ValueError(f"'{self.name}' misses priors for the parameters {difference}")

#         # And check priors passed are in fact of class Prior
#         check_all_are_priors(priors)

#     return priors

# def check_all_are_priors(priors):
#     """Checks if values in the supplied dictionary are all valid prior objects

#     An object is a valid prior if
#     * It is an instance of bambi.priors.Prior
#     * It is a number

#     Parameters
#     ----------
#     priors: dict
#         A dictionary whose values are tested to be valid priors
#     """
#     if any(not isinstance(prior, (Prior, int, float)) for prior in priors.values()):
#         raise ValueError("Prior distributions must be a 'Prior' instance or a numeric value")
