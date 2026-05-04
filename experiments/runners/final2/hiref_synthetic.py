"""
Half-Moon and S-Curve synthetic data generation copied from HiRef's
``synthetic_experiments_sample_complexity_bench_GPU.ipynb``.

The default RNG seed is the notebook default: ``jax.random.key(0)``.
Only the Half-Moon/S-Curve branch needed for the final2 comparison is included.
"""

import dataclasses
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal, Optional

import numpy as np


@dataclasses.dataclass
class _Dataset:
    source_iter: Iterator
    target_iter: Iterator


@dataclasses.dataclass
class SklearnDistribution:
    name: Literal["moon", "s_curve"]
    theta_rotation: float = 0.0
    mean: Optional["jnp.ndarray"] = None
    noise: float = 0.01
    scale: float = 1.0
    batch_size: int = 1024
    rng: Optional["jax.Array"] = None

    def __iter__(self) -> Iterator["jnp.ndarray"]:
        return self._create_sample_generators()

    def _create_sample_generators(self) -> Iterator["jnp.ndarray"]:
        import jax
        import jax.numpy as jnp
        import sklearn.datasets

        rng = jax.random.key(0) if self.rng is None else self.rng
        rotation = jnp.array(
            [
                [jnp.cos(self.theta_rotation), -jnp.sin(self.theta_rotation)],
                [jnp.sin(self.theta_rotation), jnp.cos(self.theta_rotation)],
            ]
        )
        while True:
            rng, _ = jax.random.split(rng)
            seed = jax.random.randint(rng, [], minval=0, maxval=1e5).item()
            if self.name == "moon":
                samples, _ = sklearn.datasets.make_moons(
                    n_samples=(self.batch_size, 0),
                    random_state=seed,
                    noise=self.noise,
                )
            elif self.name == "s_curve":
                x, _ = sklearn.datasets.make_s_curve(
                    n_samples=self.batch_size,
                    random_state=seed,
                    noise=self.noise,
                )
                samples = x[:, [2, 0]]
            else:
                raise NotImplementedError(
                    f"SklearnDistribution `{self.name}` not implemented."
                )

            samples = jnp.asarray(samples, dtype=jnp.float32)
            samples = jnp.squeeze(jnp.matmul(rotation[None, :], samples.T).T)
            mean = jnp.zeros(2) if self.mean is None else self.mean
            samples = mean + self.scale * samples
            yield samples


def create_samplers(
    source_kwargs: Mapping[str, Any] = MappingProxyType({}),
    target_kwargs: Mapping[str, Any] = MappingProxyType({}),
    train_batch_size: int = 512,
    valid_batch_size: int = 512,
    rng: Optional["jax.Array"] = None,
):
    import jax

    rng = jax.random.key(0) if rng is None else rng
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    train_dataset = _Dataset(
        source_iter=iter(
            SklearnDistribution(
                rng=rng1, batch_size=train_batch_size, **source_kwargs
            )
        ),
        target_iter=iter(
            SklearnDistribution(
                rng=rng2, batch_size=train_batch_size, **target_kwargs
            )
        ),
    )
    valid_dataset = _Dataset(
        source_iter=iter(
            SklearnDistribution(
                rng=rng3, batch_size=valid_batch_size, **source_kwargs
            )
        ),
        target_iter=iter(
            SklearnDistribution(
                rng=rng4, batch_size=valid_batch_size, **target_kwargs
            )
        ),
    )
    dim_data = 2
    return train_dataset, valid_dataset, dim_data


def ret_halfmoon_scurve(n_points: int = 512, seed: int = 0):
    import jax
    import jax.numpy as jnp

    rng = jax.random.key(seed)
    train_dataset, _valid_dataset, _dim_data = create_samplers(
        source_kwargs={
            "name": "moon",
            "theta_rotation": jnp.pi / 6,
            "mean": jnp.array([0.0, -0.5]),
            "noise": 0.05,
        },
        target_kwargs={
            "name": "s_curve",
            "scale": 0.6,
            "mean": jnp.array([0.5, -2.0]),
            "theta_rotation": -jnp.pi / 6,
            "noise": 0.05,
        },
        train_batch_size=n_points,
        valid_batch_size=n_points,
        rng=rng,
    )
    eval_data_source = next(train_dataset.source_iter)
    eval_data_target = next(train_dataset.target_iter)
    return np.array(eval_data_source, dtype=np.float32), np.array(
        eval_data_target, dtype=np.float32
    )
