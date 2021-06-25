import numpy as np

from .priors import Prior


class PriorScaler2:
    """Scale prior distributions parameters."""

    # Standard deviation multiplier.
    STD = 2.5

    def __init__(self, model):
        self.model = model
        self.has_intercept = any(term.type == "intercept" for term in self.model.terms.values())
        self.priors = {}

        # Compute mean and std of the response
        if self.model.family.name == "gaussian":
            self.response_mean = np.mean(model.response.data)
            self.response_std = np.std(self.model.response.data)
        else:
            self.response_mean = 0
            self.response_std = 1

    def scale_response(self):
        if self.model.response.prior.auto_scale:
            if self.model.family.name == "gaussian":
                lam = 1 / self.response_std
                self.model.response.prior.update(sigma=Prior("Exponential", lam=lam))
            # Add cases for other families

    def scale_intercept(self, term):
        if term.prior.name != "Normal":
            return
        mu, sigma = self.get_intercept_stats()
        term.prior.update(mu=mu, sigma=sigma)

    def scale_common(self, term):
        if term.prior.name != "Normal":
            return

        # As many zeros as columns in the data. It can be greater than 1 for categorical variables
        mu = np.zeros(term.data.shape[1])
        sigma = np.zeros(term.data.shape[1])

        # Iterate over columns in the data
        for i, x in enumerate(term.data.T):
            sigma[i] = self.get_slope_sigma(x)

        # Save and set prior
        self.priors.update({term.name: {"mu": mu, "sigma": sigma}})
        term.prior.update(mu=mu, sigma=sigma)

    def scale_group_specific(self, term):
        # these default priors are only defined for HalfNormal priors
        if term.prior.args["sigma"].name != "HalfNormal":
            return

        # Recreate the corresponding common effect data
        data_as_common = term.predictor

        # Handle intercepts
        if term.type == "intercept":
            _, sigma = self.get_intercept_stats()
        # Handle slopes
        else:
            sigma = np.zeros(data_as_common.shape[1])
            for i, x in enumerate(data_as_common.T):
                sigma[i] = self.get_slope_sigma(x)

        term.prior.args["sigma"].update(sigma=np.squeeze(np.atleast_1d(sigma)))

    def scale(self):
        # Scale response
        self.scale_response()

        # Scale intercept
        if self.has_intercept:
            term = [t for t in self.model.common_terms.values() if t.type == "intercept"][0]
            if term.prior.auto_scale:
                self.scale_intercept(term)

        # Scale common terms
        for term in self.model.common_terms.values():
            # maybe intercept shouldn't go in common terms?
            if term.type == "intercept":
                continue
            if term.prior.auto_scale:
                self.scale_common(term)

        # Scale group-specific terms
        for term in self.model.group_specific_terms.values():
            if term.prior.auto_scale:
                self.scale_group_specific(term)

    def get_intercept_stats(self):
        mu = self.response_mean
        sigma = self.STD * self.response_std
        return mu, sigma

    def get_slope_sigma(self, x):
        return self.STD * (self.response_std / np.std(x))
