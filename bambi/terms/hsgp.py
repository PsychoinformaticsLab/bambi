import numpy as np

import formulae.terms

from bambi.terms.base import BaseTerm, VALID_PRIORS

GP_VALID_PRIORS = tuple(value for value in VALID_PRIORS if value is not None)

# pylint: disable = invalid-name
class HSGPTerm(BaseTerm):
    def __init__(self, term, prior, prefix=None):
        """Create a term for a HSGP model component

        Parameters
        ----------
        term : formulae.terms.terms.Term
            A term that was created with ``hsgp(...)``. The caller is an instance of ``HSGP()``.
        prior : dict
            The keys are the names of the parameters of the covariance function and the values are
            instances of ``bambi.Prior`` or other values that are accepted by the covariance
            function.
        prefix : str
            It is used to indicate the term belongs to the component of a non-parent parameter.
            Defaults to ``None``.
        """
        self.term = term
        self.prior = prior
        self.prefix = prefix
        self.hsgp_attributes = get_hsgp_attributes(term)
        self.hsgp = None

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, value):
        assert isinstance(value, formulae.terms.terms.Term)
        self._term = value

    @property
    def data(self):
        return self.term.data

    @property
    def data_centered(self):
        return self.term.data - self.hsgp_attributes["mean"]

    @property
    def m(self):
        return np.atleast_1d(self.hsgp_attributes["m"])

    @property
    def L(self):
        if self.c:
            S = np.max(np.abs(self.data_centered), axis=0)
            output = self.c * S
        else:
            output = self.hsgp_attributes["L"]
        return np.atleast_1d(output)

    @property
    def c(self):
        if self.hsgp_attributes["c"] is None:
            return None
        return np.atleast_1d(self.hsgp_attributes["c"])

    @property
    def cov(self):
        return self.hsgp_attributes["cov"]

    @property
    def centered(self):
        return self.hsgp_attributes["centered"]

    @property
    def drop_first(self):
        return self.hsgp_attributes["drop_first"]

    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, value):
        message = (
            "The priors for an HSGP term must be passed within a dictionary. "
            "Keys must the names of the parameters of the covariance function "
            "and values are instances of `bambi.Prior` or numeric constants."
        )
        if value is None:
            self._prior = value
        else:
            if not isinstance(value, dict):
                raise ValueError(message)
            for prior in value.values():
                assert isinstance(prior, GP_VALID_PRIORS), f"Prior must be one of {GP_VALID_PRIORS}"
            self._prior = value

    @property
    def coords(self):
        # NOTE: This has to depend on the 'by' argument.
        return {}

    @property
    def name(self):
        if self.prefix:
            return f"{self.prefix}_{self.term.name}"
        return self.term.name

    @property
    def shape(self):
        return self.data.shape

    @property
    def categorical(self):
        return False

    @property
    def levels(self):
        return None


def get_hsgp_attributes(term):
    """Extract HSGP attributes from a model matrix term

    Parameters
    ----------
    term : formulae.terms.terms.Term
        The formulae term that creates the HSGP term.

    Returns
    -------
    dict
        The attributes that will be passed to pm.gp.HSGP
    """
    names = ("m", "L", "c", "by", "cov", "drop_first", "centered", "mean")
    attrs_original = term.components[0].call.stateful_transform.__dict__
    attrs = {}
    for name in names:
        attrs[name] = attrs_original[name]
    return attrs
