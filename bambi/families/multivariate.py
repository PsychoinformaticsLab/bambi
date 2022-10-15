# pylint: disable=unused-argument
import numpy as np
import xarray as xr

from .family import Family


class MultivariateFamily(Family):
    def predict(self, model, posterior, linear_predictor):
        return NotImplemented

    def posterior_predictive(self, model, posterior, linear_predictor):
        return NotImplemented


class Categorical(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        # This is only 'softmax' for now.
        # Because of reference encoding, we need to padd with 0s for the first level of the
        # response variable
        # (1, 0): 1 new levels on the left, 0 new level on the right
        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        # The mean has the reference level in the dimension, a new name is needed
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        posterior[response_var] = mean
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        # https://stackoverflow.com/questions/34187130
        def draw_categorical_samples(probability_matrix, items):
            """
            probability_matrix is a matrix of shape (n_chain * n_draw, n_levels)
            """
            cumsum = probability_matrix.cumsum(axis=1)
            idx = np.random.rand(probability_matrix.shape[0])[:, np.newaxis]
            idx = (cumsum < idx).sum(axis=1)
            return items[idx]

        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        mean = mean.to_numpy()
        shape = mean.shape
        # Stack chains and draws
        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        pps = np.empty((draws_n, obs_n), dtype=int)
        response_levels = np.arange(len(model.response.levels))

        for idx in range(obs_n):
            pps[:, idx] = draw_categorical_samples(mean[:, idx, :], response_levels)

        pps = pps.reshape((shape[0], shape[1], obs_n))
        pps = xr.DataArray(
            pps,
            coords={
                "chain": np.arange(shape[0]),
                "draw": np.arange(shape[1]),
                response_dim: np.arange(obs_n),
            },
        )
        return pps


class Multinomial(MultivariateFamily):
    SUPPORTED_LINKS = ["softmax"]

    def predict(self, model, posterior, linear_predictor):
        response_var = model.response.name + "_mean"
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        # The mean has the reference level in the dimension, a new name is needed
        mean = mean.rename({response_levels_dim: response_levels_dim_complete})
        mean = mean.assign_coords({response_levels_dim_complete: model.response.levels})

        # Drop var/dim if already present
        if response_var in posterior.data_vars:
            posterior = posterior.drop_vars(response_var)

        if response_dim in posterior.dims:
            posterior = posterior.drop_dims(response_dim)

        posterior[response_var] = mean
        return posterior

    def posterior_predictive(self, model, posterior, linear_predictor):
        response_dim = model.response.name + "_obs"
        response_levels_dim = model.response.name + "_dim"
        response_levels_dim_complete = model.response.name + "_mean_dim"

        linear_predictor = linear_predictor.pad({response_levels_dim: (1, 0)}, constant_values=0)
        mean = xr.apply_ufunc(self.link.linkinv, linear_predictor, kwargs={"axis": -1})

        mean = mean.to_numpy()
        shape = mean.shape

        mean = mean.reshape((mean.shape[0] * mean.shape[1], mean.shape[2], mean.shape[3]))
        draws_n = mean.shape[0]
        obs_n = mean.shape[1]

        pps = np.empty(mean.shape, dtype=int)
        n = model.response.data.sum(1)

        # random.multinomial only accepts
        # * n : integer
        # * p : vector
        for i in range(obs_n):
            for j in range(draws_n):
                pps[j, i, :] = np.random.multinomial(n[i], mean[j, i, :])

        # Final shape is of (chain, draw, obs_n, response_n)
        pps = pps.reshape(shape)
        pps = xr.DataArray(
            pps,
            coords={
                "chain": np.arange(shape[0]),
                "draw": np.arange(shape[1]),
                response_dim: np.arange(obs_n),
                response_levels_dim_complete: model.response.levels,
            },
        )
        return pps
