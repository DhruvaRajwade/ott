# Conditional Monge Gap: JIT-compatible loss + training estimator

The CMonge paper has been accepted to Nature Machine Intelligence. Picks up from #605 and fixes the JIT issue + adds a training estimator.

## Changes

**`cmonge_gap_from_samples`** -- replaced the `jnp.unique` loop (breaks `jax.jit`) with `_segment_interface` (pad + vmap), following @michalk8's suggestion. Two new required-for-JIT params: `num_segments`, `max_measure_size` (consistent with `segment_sinkhorn`). A `logger.warning` fires when any condition is padded >10x its actual size, since heavy padding can cause small numerical differences vs non-padded Sinkhorn.

**`ConditionalMongeGapEstimator`** -- training wrapper mirroring `MongeGapEstimator` for conditional maps `T(x, c)`:

```
loss = fitting(T(x,c), y) + lambda * regularizer(x, T(x,c), labels)
       |__ sinkdiv            |__ cmonge_gap_from_samples
```

**Numerical exactness caveat:** When all conditions have the same `n_k`, the segment-based result matches `monge_gap_from_samples` to ~1e-7. With unequal `n_k`, smaller conditions are zero-padded to `max_measure_size`, which changes the Sinkhorn geometry slightly. This is inherent to the segment/vmap approach -- the trade-off is JIT compatibility.

## Tests (26 passing)

All tests mirror `monge_gap_test.py` patterns whereever applicable: non-negativity (random + neural map targets), JIT consistency, cost function variants, estimator convergence. Three additional equivalence tests verify `cmonge_gap = mean(monge_gap_k)` for equal-size conditions, document the padding effect for unequal sizes, and check monotonic gap ordering by difficulty.

```bash
pytest tests/neural/methods/conditional_monge_gap_test.py -v  # 26 tests, ~80s
```

## Usage example

```python
from ott import datasets
from ott.neural.methods.conditional_monge_gap import (
    ConditionalMongeGapEstimator, cmonge_gap_from_samples,
)
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)
from ott.tools.sinkhorn_divergence import sinkdiv
import optax

num_cond, dim_data = 5, 25
train_ds, valid_ds, _, n_cond, max_ms = (
    datasets.create_conditional_gaussian_mixture_samplers(
        num_conditions=num_cond, dim=dim_data,
        train_batch_size=150, valid_batch_size=150,
    )
)

fitting_loss = lambda mapped, target: (sinkdiv(x=mapped, y=target)[0], None)
regularizer = lambda src, mapped, labels: (
    cmonge_gap_from_samples(src, mapped, labels,
        num_segments=n_cond, max_measure_size=max_ms), None)

model = ConditionalPerturbationNetwork(
    dim_hidden=[64, 64], dim_data=dim_data, dim_cond=num_cond,
    dim_cond_map=(32,), is_potential=False,
    context_entity_bonds=((0, num_cond),), num_contexts=1,
)
solver = ConditionalMongeGapEstimator(
    dim_data=dim_data, model=model,
    optimizer=optax.adam(learning_rate=1e-4),
    fitting_loss=fitting_loss, regularizer=regularizer,
    regularizer_strength=5.0, num_train_iters=2000,
    logging=True, valid_freq=50,
)
state, logs = solver.train_map_estimator(*train_ds, *valid_ds)
```

![Training curves](cmonge_training.png)
