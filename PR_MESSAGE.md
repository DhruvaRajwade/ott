# Conditional Monge Gap: JIT-compatible loss + training estimator

This PR adds full support for training condition-dependent optimal transport maps regularized by the conditional Monge gap. It consists of two parts:

1. **Fix `cmonge_gap_from_samples` for JIT** — replace the `jnp.unique` loop with the segment interface
2. **Add `ConditionalMongeGapEstimator`** — a training class mirroring `MongeGapEstimator` for conditional maps `T(x, c)`

---

## Part 1: JIT-compatible `cmonge_gap_from_samples`

### The problem

The original `cmonge_gap_from_samples` used a Python `for`-loop over `jnp.unique(condition)`. `jnp.unique` returns a dynamically-sized array, so iterating over it and indexing with `condition == c` creates data-dependent shapes that break `jax.jit`. This makes the function unusable in any JIT-traced training loop.

### The fix

Following @michalk8's suggestion, uses `_segment_interface` (the same machinery behind `segment_sinkhorn`) to:

1. Pad per-condition source/target point clouds to a fixed `max_measure_size`
2. `vmap` a per-segment `eval_fn` across all conditions in parallel

The `eval_fn` computes the Monge gap for a single padded segment:

```python
displacement_cost = Σ c(xᵢ, yᵢ) · wᵢ     # weighted pairwise cost (padding has w=0)
ot_cost           = ent_reg_cost            # from Sinkhorn on padded PointCloud
monge_gap         = displacement_cost - ot_cost
```

Uses `ent_reg_cost` (not `reg_ot_cost`) to match the definition in `monge_gap_from_samples`.

### API changes to `cmonge_gap_from_samples`

- Geometry parameters (`cost_fn`, `epsilon`, `relative_epsilon`, `scale_cost`) are now explicit kwargs instead of being passed opaquely through `**kwargs` → `monge_gap_from_samples` → `linear.solve`. This is clearer and matches the `monge_gap_from_samples` signature.
- Two new required-for-JIT parameters: `num_segments` (number of conditions) and `max_measure_size` (upper bound on points per condition), consistent with the `segment_sinkhorn` API.
- `return_output=True` now returns `(avg_gap, per_condition_gaps)` as a `jnp.ndarray` instead of `(avg_gap, List[SinkhornOutput], List[float])`. Individual `SinkhornOutput` objects aren't available through `vmap`, and the per-condition gap array is more useful and JIT-compatible.
- `**kwargs` now maps to `sinkhorn.Sinkhorn(**kwargs)` directly.

### Verification: segment-based matches original loop

```python
import jax
import jax.numpy as jnp
from ott.neural.methods.conditional_monge_gap import cmonge_gap_from_samples
from ott.neural.methods.monge_gap import monge_gap_from_samples

rng = jax.random.PRNGKey(42)
n, d = 60, 4
source = jax.random.normal(rng, (n, d))
target = source + 0.1 * jax.random.normal(jax.random.PRNGKey(1), (n, d))
condition = jnp.repeat(jnp.arange(3), 20)

# --- New segment-based implementation ---
new_gap, per_cond = cmonge_gap_from_samples(
    source, target, condition,
    num_segments=3, max_measure_size=20,
    return_output=True,
)

# --- Original loop-based implementation (for reference) ---
manual_gaps = []
for c in range(3):
    mask = condition == c
    gap = monge_gap_from_samples(source[mask], target[mask])
    manual_gaps.append(float(gap))
manual_avg = sum(manual_gaps) / len(manual_gaps)

print("Segment-based:", float(new_gap), "per-cond:", [float(x) for x in per_cond])
# Segment-based: 0.8573737144470215 per-cond: [0.651, 1.089, 0.832]
print("Manual loop:  ", manual_avg, "per-cond:", manual_gaps)
# Manual loop:   0.8573738137880961 per-cond: [0.651, 1.089, 0.832]
print("Difference:", abs(float(new_gap) - manual_avg))
# Difference: 9.93e-08

# --- JIT compilation works ---
jitted = jax.jit(
    lambda s, t, c: cmonge_gap_from_samples(
        s, t, c, num_segments=3, max_measure_size=20,
    )
)
jit_gap = jitted(source, target, condition)
print("JIT gap:", float(jit_gap))
# JIT gap: 0.8573290109634399

assert new_gap >= 0
```

---

## Part 2: `ConditionalMongeGapEstimator`

### Motivation

`cmonge_gap_from_samples` is the loss function; `ConditionalMongeGapEstimator` is the training harness. It mirrors `MongeGapEstimator` but handles condition-aware maps and per-condition regularization:

```
MongeGapEstimator                    ConditionalMongeGapEstimator
─────────────────                    ────────────────────────────
model: BasePotential                 model: ConditionalPerturbationNetwork
batch: {source, target}              batch: {source, target, condition, labels}
T(x)                                 T(x, c)
loss: Δ(T(x), y) + λ·R(x, T(x))    loss: Δ(T(x,c), y) + λ·R(x, T(x,c), labels)
      ↑ sinkdiv    ↑ monge_gap             ↑ sinkdiv       ↑ cmonge_gap
```

### Key design decisions

**Why not subclass `MongeGapEstimator`?** The differences touch the core `loss_fn` (3-arg regularizer, conditioned `apply_fn`) and `_generate_batch` (4 iterators). Subclassing would require overriding most methods. A standalone class (~200 lines) is clearer.

**Why separate `condition` (float) and `condition_labels` (int)?** The continuous condition vector (e.g. one-hot, embedding) is fed to `ConditionalPerturbationNetwork.__call__(x, c)`. The integer labels are segment ids for `cmonge_gap_from_samples`. A drug might have a 50-dim embedding as `condition` but integer `3` as its segment label.

**Why keep `fitting_loss` unconditional?** The conditional Monge gap regularizer already enforces per-condition map quality. An unconditional `sinkdiv(T(x,c), y)` works because the regularizer prevents the map from averaging across conditions. A per-condition Sinkhorn divergence can be passed by the user if needed.

### Files changed

| File | Change |
|------|--------|
| `src/ott/neural/methods/conditional_monge_gap.py` | JIT-fixed `cmonge_gap_from_samples` + `ConditionalMongeGapEstimator` class |
| `src/ott/neural/methods/__init__.py` | Export `conditional_monge_gap` module |
| `src/ott/neural/networks/__init__.py` | Export `conditional_perturbation_network` module |
| `src/ott/neural/networks/conditional_perturbation_network.py` | Formatting fixes (linter) |
| `src/ott/datasets.py` | `ConditionalDataset` namedtuple + `ConditionalGaussianMixture` sampler + `create_conditional_gaussian_mixture_samplers()` factory |
| `tests/neural/methods/conditional_monge_gap_test.py` | 26 tests (new file) |

### Test suite (26 tests, all passing)

**`TestConditionalMongeGap`** — unit tests for `cmonge_gap_from_samples`:

#### Core property tests (mirrored from `monge_gap_test.py`)

| Test | Parametrization | What it verifies |
|------|----------------|-----------------|
| `test_non_negativity` | ×8: `(n_samples, n_features, num_conditions)` | Gap ≥ 0 for random perturbations of source, across `(10,30) × (4,10) × (2,3)` |
| `test_non_negativity_neural_map` | ×4: `(n_samples, n_features)` | Gap ≥ 0 when target is produced by a `PotentialMLP` (learned nonlinear map), across `(10,30) × (4,10)`. Mirrors `monge_gap_test::test_monge_gap_non_negativity` |
| `test_identity_smaller_than_random` | — | Identity map (target=source) has strictly smaller gap than a random independent target. Sanity check that the gap measures transport quality |

#### JIT and implementation correctness

| Test | What it verifies |
|------|-----------------|
| `test_jit_consistency` | `jax.jit(cmonge_gap_from_samples)` matches eager execution to `rtol=1e-3` |
| `test_matches_loop_baseline` | Segment-based result matches manual per-condition `monge_gap_from_samples` loop to `atol=1e-5`. This is the fundamental correctness test: the new `_segment_interface` implementation gives the same result as the old Python loop |
| `test_return_output_shape` | `return_output=True` gives `(scalar, [K] array)` with `avg == mean(per_cond_gaps)` |

#### Cost function tests

| Test | Parametrization | What it verifies |
|------|----------------|-----------------|
| `test_different_costs` | ×2: `SqEuclidean`, `PNormP(p=1)` | Finite and non-negative with standard costs |
| `test_different_costs_give_different_values` | ×3: `PNormP(1)`, `RegTICost(L1(), lam=2.0)`, `RegTICost(STVS(gamma=3.0), lam=1.0)` | Each non-Euclidean cost produces a cmonge_gap that **differs** from the Euclidean baseline (asserted via `pytest.raises(AssertionError)`). Also verifies finiteness. Mirrors `monge_gap_test::test_monge_gap_from_samples_different_cost` |

#### Equivalence tests: cmonge_gap vs averaged monge_gap

These tests verify the mathematical relationship between the conditional and unconditional Monge gap.

| Test | What it verifies |
|------|-----------------|
| `test_uniform_conditions_equals_averaged_monge_gap` | **Core equivalence**: with K=3 equal-size conditions (n_k=20 each, distinct offsets [0.1, 1.0, 3.0]), `cmonge_gap_from_samples` equals `(1/K) Σ monge_gap_from_samples(source_k, target_k)` to `atol=1e-5`. Both the average and each per-condition gap match their individual `monge_gap_from_samples` calls exactly. This proves the segment interface is numerically equivalent to the loop when condition sizes are equal |
| `test_unequal_conditions_shifts_average` | **Unequal-size behavior**: K=2 conditions (easy: tiny noise, hard: +5.0 offset) with equal (30/30) vs unequal (50/10) splits. Verifies: (a) all gaps are finite and non-negative, (b) easy < hard ordering holds in both cases, (c) `avg == mean(per_cond_gaps)` always, (d) the overall average **shifts** when condition sizes change. Note: per-condition gaps do NOT exactly match non-padded `monge_gap_from_samples` when padding is significant — the segment interface pads smaller conditions to `max_measure_size`, creating a larger cost matrix that changes the Sinkhorn solution slightly |
| `test_per_condition_gaps_reflect_difficulty` | K=3 conditions with increasing offsets (0.0, 1.5, 5.0). Verifies strict monotonic ordering: `gaps[0] < gaps[1] < gaps[2]`. The harder the transport problem (larger displacement), the larger the Monge gap |

#### Design note on padding and numerical equivalence

When all conditions have the same `n_k`, the segment interface produces results identical to calling `monge_gap_from_samples` independently (verified by `test_uniform_conditions_equals_averaged_monge_gap` and `test_matches_loop_baseline`). When conditions have unequal sizes, the smaller conditions are zero-padded to `max_measure_size`. The Sinkhorn solve on the padded geometry (e.g. 50×50 with 40 zero-weight entries) can give slightly different results than a non-padded solve (10×10), because the regularization and convergence behavior depend on the matrix size. This is inherent to the segment/vmap approach — the trade-off is JIT compatibility at the cost of small numerical differences for unequal partitions.

---

**`TestConditionalMongeGapEstimator`** — integration tests:

| Test | What it verifies |
|------|-----------------|
| `test_estimator_convergence` | Loss decreases over 15 iterations with `sinkdiv` fitting loss + `cmonge_gap` regularizer; output shape matches input; mapped values are finite |
| `test_estimator_no_regularizer` | Training with `regularizer_strength=0` runs without errors; outputs are finite. Verifies the estimator gracefully handles the degenerate case |

**No regressions:** All 12 existing `monge_gap_test.py` tests continue to pass.

### Coverage cross-reference: `monge_gap_test.py` → `conditional_monge_gap_test.py`

Every test pattern in the upstream `monge_gap_test.py` has a corresponding conditional counterpart, plus three additional equivalence/structural tests unique to the conditional setting:

| `monge_gap_test.py` test | `conditional_monge_gap_test.py` counterpart | Notes |
|---|---|---|
| `test_monge_gap_non_negativity` (×6) — `PotentialMLP` target, parametrized `(n_samples, n_features)` | `test_non_negativity` (×8) + `test_non_negativity_neural_map` (×4) | Non-negativity with both random perturbations and `PotentialMLP`-generated targets. Parametrized over `(n_samples, n_features, num_conditions)` |
| `test_monge_gap_jit` — eager vs JIT consistency | `test_jit_consistency` | Same pattern: `jax.jit(cmonge_gap_from_samples)` matches eager to `rtol=1e-3` |
| `test_monge_gap_from_samples_different_cost` (×4) — `SqEuclidean`, `PNormP(1)`, `RegTICost(L1)`, `RegTICost(STVS)` | `test_different_costs` (×2) + `test_different_costs_give_different_values` (×3) | Split into two tests: basic finite/non-negative check for standard costs, and explicit "different from Euclidean" assertion (using `pytest.raises(AssertionError)`) for `PNormP(1)`, `RegTICost(L1(), lam=2.0)`, `RegTICost(STVS(gamma=3.0), lam=1.0)` |
| `test_map_estimator_convergence` | `test_estimator_convergence` + `test_estimator_no_regularizer` | Loss decrease + output shape/finiteness, plus degenerate `λ=0` case |
| *(no counterpart)* | `test_matches_loop_baseline` | Segment-based equals manual per-condition `monge_gap_from_samples` loop |
| *(no counterpart)* | `test_identity_smaller_than_random` | Identity map gap < random target gap |
| *(no counterpart)* | `test_return_output_shape` | `return_output=True` shape and mean consistency |
| *(no counterpart)* | `test_uniform_conditions_equals_averaged_monge_gap` | **Equivalence**: `(1/K) Σ monge_gap(source_k, target_k) == cmonge_gap(source, target, condition)` when all `n_k` are equal |
| *(no counterpart)* | `test_unequal_conditions_shifts_average` | **Unequal conditions**: average shifts when `n_k` changes; structural properties hold |
| *(no counterpart)* | `test_per_condition_gaps_reflect_difficulty` | Monotonic ordering: harder transport → larger per-condition gap |

### Detailed test descriptions

#### `test_non_negativity` (×8)

Parametrized over `(n_samples, n_features, num_conditions)` = `(10,30) × (4,10) × (2,3)`. For each combination, generates `source ~ N(0,1)` and `target = source + 0.5 * N(0,1)` (a random perturbation), assigns conditions as equal-size blocks via `jnp.repeat`, and asserts `cmonge_gap_from_samples(...) >= 0`. This is the most basic property: the Monge gap is non-negative by construction (displacement cost minus OT cost).

#### `test_non_negativity_neural_map` (×4)

Parametrized over `(n_samples, n_features)` = `(10,30) × (4,10)`. Instead of a random perturbation, uses a `PotentialMLP(dim_hidden=[8,8], is_potential=False)` neural network to generate targets. The MLP is randomly initialized (not trained), so it produces a deterministic nonlinear map `T(x)`. This mirrors `monge_gap_test::test_monge_gap_non_negativity` which uses the same `PotentialMLP` pattern. Verifies gap is both finite and non-negative.

#### `test_jit_consistency`

Creates `n=60, d=4, k=3` data with `target = source + 0.1 * noise`. Computes `cmonge_gap_from_samples` both eagerly and via `jax.jit(lambda s, t, c: cmonge_gap_from_samples(s, t, c, ...))`. Asserts the two values match to `rtol=1e-3`. This validates that the `_segment_interface`-based implementation is fully JIT-traceable — the original `jnp.unique` loop was not.

#### `test_matches_loop_baseline`

The fundamental correctness test. With `k=3` equal-size conditions (`n_k=20`), computes the segment-based `cmonge_gap_from_samples` and compares against a manual Python loop: `for c in range(k): monge_gap_from_samples(source[mask], target[mask])`, averaged. Asserts match to `atol=1e-5`. This proves the new implementation gives the same numerical result as the old approach.

#### `test_identity_smaller_than_random`

Uses `source` as both source and target (identity map) vs a random independent target scaled by 3.0. Asserts `identity_gap < random_gap`. The identity map has zero displacement beyond the OT cost, so its Monge gap should be small. A random target has high displacement cost but the OT cost can't compensate, so its gap is large.

#### `test_different_costs` (×2)

Parametrized over `SqEuclidean()` and `PNormP(p=1)`. Verifies that `cmonge_gap_from_samples` returns a finite, non-negative value for each cost function. Basic smoke test that the cost function parameter works.

#### `test_different_costs_give_different_values` (×3)

Parametrized over `PNormP(1)`, `RegTICost(L1(), lam=2.0)`, `RegTICost(STVS(gamma=3.0), lam=1.0)`. For each, computes the cmonge_gap with that cost and with `Euclidean()` on the same data. Asserts (via `pytest.raises(AssertionError)`) that the values **differ** beyond `rtol=1e-1, atol=1e-1` — i.e., the cost function genuinely changes the result. Also checks finiteness. Mirrors `monge_gap_test::test_monge_gap_from_samples_different_cost`.

#### `test_return_output_shape`

Calls `cmonge_gap_from_samples(..., return_output=True)` with `k=3` conditions. Asserts the result is a tuple `(avg_gap, per_cond_gaps)`, that `per_cond_gaps.shape == (3,)`, and that `avg_gap == mean(per_cond_gaps)` to `rtol=1e-5`.

#### `test_uniform_conditions_equals_averaged_monge_gap`

**The core equivalence test.** Creates K=3 conditions with equal `n_k=20` and distinct offsets `[0.1, 1.0, 3.0]` (so per-condition gaps are meaningfully different). Computes:
- Segmented: `cmonge_gap_from_samples(source, target, condition, return_output=True)`
- Manual: `for c in range(3): monge_gap_from_samples(sources[c], targets[c])`

Asserts:
1. The average matches to `atol=1e-5`
2. Each per-condition gap `per_cond_gaps[c]` matches its individual `monge_gap_from_samples` call to `atol=1e-5`

This proves mathematically that `cmonge_gap = (1/K) Σ monge_gap_k` when all conditions have the same number of samples.

#### `test_unequal_conditions_shifts_average`

**The unequal-conditions test.** Creates K=2 conditions — easy (target ≈ source + tiny noise, small gap) and hard (target = source + 5.0, large gap). Runs with:
- Equal sizes: 30 samples per condition (`max_measure_size=30`)
- Unequal sizes: 50 easy / 10 hard (`max_measure_size=50`)

Asserts structural properties:
- (a) All gaps are finite and non-negative in both cases
- (b) `easy < hard` ordering holds in both cases
- (c) `avg == mean(per_cond_gaps)` always (the function always does equal-weight averaging over conditions)
- (d) The overall averages **differ** between equal and unequal splits

Key insight documented in the test docstring: per-condition gaps do NOT exactly match non-padded `monge_gap_from_samples` when padding is significant. The segment interface pads the 10-sample condition to 50 entries (with zero weights), creating a 50×50 Sinkhorn cost matrix instead of 10×10. The Sinkhorn solve converges differently on the padded geometry, so the per-condition gap values shift. This is inherent to the `_segment_interface` / `vmap` approach — the trade-off is JIT compatibility.

#### `test_per_condition_gaps_reflect_difficulty`

Creates K=3 conditions with deterministic offsets `[0.0, 1.5, 5.0]` (target = source + offset, no noise). Asserts strict monotonic ordering: `gaps[0] < gaps[1] < gaps[2]`. Larger displacement means more transport work, which increases the Monge gap. This validates that per-condition gaps are semantically meaningful.

#### `test_estimator_convergence`

End-to-end training test. Creates a `ConditionalMongeGapEstimator` with `sinkdiv` fitting loss and `cmonge_gap_from_samples` regularizer, trains for 15 iterations on `ConditionalGaussianMixture` data (3 conditions, 2D). Asserts:
- `train_loss[0] > train_loss[-1]` (loss decreases)
- Mapped output shape matches source shape
- All mapped values are finite

#### `test_estimator_no_regularizer`

Same as above but with `regularizer_strength=0.0` (no Monge gap penalty). Verifies the estimator doesn't crash in this degenerate case: logs are non-empty and mapped outputs are finite.

### How to run the tests

```bash
# 1. Clone the fork and checkout the branch
git clone https://github.com/DhruvaRajwade/ott.git
cd ott
git checkout pr-605-extend

# 2. Install in development mode with neural and test dependencies
pip install -e ".[neural,test]"

# 3. Run the conditional Monge gap tests (26 tests, ~80s)
pytest tests/neural/methods/conditional_monge_gap_test.py -v

# 4. Run both conditional and original Monge gap tests together (38 tests, ~2min)
pytest tests/neural/methods/conditional_monge_gap_test.py tests/neural/methods/monge_gap_test.py -v

# 5. Run only the equivalence tests
pytest tests/neural/methods/conditional_monge_gap_test.py -v -k "uniform_conditions or unequal_conditions or matches_loop"

# 6. Run only the cost function tests
pytest tests/neural/methods/conditional_monge_gap_test.py -v -k "different_costs"

# 7. Run only the estimator integration tests
pytest tests/neural/methods/conditional_monge_gap_test.py -v -k "estimator"

# 8. Run with the @fast marker (all tests in this file are marked fast)
pytest tests/neural/methods/conditional_monge_gap_test.py -v -m fast
```

### Test results (rebased on latest `ott-jax/ott` main)

All 38 tests pass on branch `pr-605-extend` (rebased on upstream `main`):

```
$ pytest tests/neural/methods/conditional_monge_gap_test.py tests/neural/methods/monge_gap_test.py -v
================== 38 passed in 124.70s (0:02:04) ===================
```

<details>
<summary>Full test output (click to expand)</summary>

```
tests/neural/methods/conditional_monge_gap_test.py

  TestConditionalMongeGap
    test_non_negativity[2-4-10]                          PASSED
    test_non_negativity[2-4-30]                          PASSED
    test_non_negativity[2-10-10]                         PASSED
    test_non_negativity[2-10-30]                         PASSED
    test_non_negativity[3-4-10]                          PASSED
    test_non_negativity[3-4-30]                          PASSED
    test_non_negativity[3-10-10]                         PASSED
    test_non_negativity[3-10-30]                         PASSED
    test_jit_consistency                                 PASSED
    test_matches_loop_baseline                           PASSED
    test_identity_smaller_than_random                    PASSED
    test_different_costs[sqeucl]                         PASSED
    test_different_costs[pnorm-1]                        PASSED
    test_return_output_shape                             PASSED
    test_non_negativity_neural_map[4-10]                 PASSED
    test_non_negativity_neural_map[4-30]                 PASSED
    test_non_negativity_neural_map[10-10]                PASSED
    test_non_negativity_neural_map[10-30]                PASSED
    test_different_costs_give_different_values[pnorm-1]  PASSED
    test_different_costs_give_different_values[l1-lam2]  PASSED
    test_different_costs_give_different_values[stvs-lam1] PASSED
    test_uniform_conditions_equals_averaged_monge_gap    PASSED
    test_unequal_conditions_shifts_average               PASSED
    test_per_condition_gaps_reflect_difficulty            PASSED

  TestConditionalMongeGapEstimator
    test_estimator_convergence                           PASSED
    test_estimator_no_regularizer                        PASSED

tests/neural/methods/monge_gap_test.py

  TestMongeGap
    test_monge_gap_non_negativity[10-5]                  PASSED
    test_monge_gap_non_negativity[10-25]                 PASSED
    test_monge_gap_non_negativity[50-5]                  PASSED
    test_monge_gap_non_negativity[50-25]                 PASSED
    test_monge_gap_non_negativity[100-5]                 PASSED
    test_monge_gap_non_negativity[100-25]                PASSED
    test_monge_gap_jit                                   PASSED
    test_monge_gap_from_samples_different_cost[sqeucl]   PASSED
    test_monge_gap_from_samples_different_cost[pnorm-1]  PASSED
    test_monge_gap_from_samples_different_cost[l1-lam2]  PASSED
    test_monge_gap_from_samples_different_cost[stvs-lam2] PASSED

  TestMongeGapEstimator
    test_map_estimator_convergence                       PASSED
```

</details>

### Usage example

```python
from ott import datasets
from ott.neural.methods.conditional_monge_gap import (
    ConditionalMongeGapEstimator,
    cmonge_gap_from_samples,
)
from ott.neural.networks.conditional_perturbation_network import (
    ConditionalPerturbationNetwork,
)
from ott.tools.sinkhorn_divergence import sinkdiv

num_cond = 3
dim_data = 10
dim_cond = num_cond  # one-hot

# Data
train_ds, valid_ds, _, n_cond, max_ms = (
    datasets.create_conditional_gaussian_mixture_samplers(
        num_conditions=num_cond, dim=dim_data,
        train_batch_size=90, valid_batch_size=90,
    )
)

# Losses
def fitting_loss(mapped, target):
    div, _ = sinkdiv(x=mapped, y=target)
    return div, None

def regularizer(source, mapped, labels):
    gap, _ = cmonge_gap_from_samples(
        source, mapped, labels,
        num_segments=n_cond, max_measure_size=max_ms,
        return_output=True,
    )
    return gap, None

# Model
model = ConditionalPerturbationNetwork(
    dim_hidden=[64, 32], dim_data=dim_data, dim_cond=dim_cond,
    dim_cond_map=(32,), is_potential=False,
    context_entity_bonds=((0, dim_cond),), num_contexts=1,
)

# Train
solver = ConditionalMongeGapEstimator(
    dim_data=dim_data, model=model,
    fitting_loss=fitting_loss, regularizer=regularizer,
    regularizer_strength=1.0, num_train_iters=1000,
    logging=True, valid_freq=100,
)
state, logs = solver.train_map_estimator(*train_ds, *valid_ds)
```
