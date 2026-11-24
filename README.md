# selection-overpromise

Code and manuscript for *Selection Bias Explains Overpromise in Model-Guided Design*.

Model-guided design trains a surrogate on measured data, ranks a candidate library by
predicted value, and sends the top `K` candidates to the next experimental round. The
predicted value of that batch is routinely treated as a forecast of what the batch will
deliver. This work shows that the forecast is biased upward even when the surrogate is
correctly calibrated on the population, derives the bias in closed form, and validates
it on 217 deep mutational scanning assays.

## Main result

Write true fitness as `y = G(z)` with `z ~ N(0,1)` and `G` monotone, and expand `G` in
probabilists' Hermite polynomials with coefficients `c_k`. For a least-squares
calibrated surrogate of latent accuracy `rho` selecting the top `q` fraction,

```
E[shortfall] = - sum_{k >= 2} (c_k / k!) * rho^k * E[He_k(R)]
```

where `R` is the selected score, a truncated standard normal. Three consequences:

1. Gaussian fitness gives `c_k = 0` for `k >= 2`, so the bias is exactly zero at every
   accuracy.
2. As `rho -> 1` the bias saturates at a nonzero constant. A surrogate that ranks every
   candidate correctly still overpromises.
3. Any order-preserving recalibration leaves the selected batch unchanged, so no post
   hoc rescaling can recover the lost gain.

## Empirical findings

On 217 ProteinGym substitution assays with a ridge surrogate on additive features:

- 93% of assay and training-fraction cells overpromise.
- The shortfall holds near 23% of achievable gain across accuracies from 0.5 to 0.9,
  and does not shrink as the surrogate improves.
- A surrogate with held-out Spearman between 0.8 and 0.9 shares only 14% of its top-20
  picks with the true top-20.
- The tail-compression index predicts the bias (Spearman -0.470, p = 9.9e-35; partial
  -0.177 controlling for accuracy) and predicts its reversal: the fraction of cells
  that underpromise rises from 0% to 42% as the upper tail lengthens.
- Neither variance-based rescaling nor position-diverse batch selection recovers gain.

## Tail-compression index

A one-line statistic of the measured fitness values, computable before any model is fit:

```python
tau = ((q99 - q50) / (q75 - q50)) / 3.449049
```

where the quantiles are of the measured fitness and 3.449049 is the value the ratio
takes for a Gaussian. `tau = 1` means a Gaussian upper tail, `tau < 1` a compressed one,
`tau > 1` an extended one.

## Layout

```
paper/        manuscript source (ICML format)
theory/       derivation checks and simulations
experiments/  selection experiments on ProteinGym
```

- `theory/shortfall_theorem.py` checks the closed form against direct simulation.
- `theory/normalized_shortfall.py` evaluates the normalized bias against accuracy.
- `theory/tail_shape_simulation.py` relates tail shape to the sign of the bias.
- `experiments/tail_index.py` computes the index and its association with the bias.
- `experiments/recalibration_rules.py` evaluates rescaling corrections.
- `experiments/diversity_rules.py` evaluates position-diverse batch selection.

## Data

ProteinGym v1.3 substitution assays:
<https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip>

Assay annotations:
<https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv>

## Reproducing

```bash
python experiments/recalibration_rules.py DMS_ProteinGym_substitutions
python experiments/diversity_rules.py      DMS_ProteinGym_substitutions
python experiments/tail_index.py           DMS_ProteinGym_substitutions
python theory/shortfall_theorem.py
```

Requires numpy, pandas, scipy, and optionally biopython for BLOSUM62.

## License

MIT, see `LICENSE`.
