"""Every metric reported with uncertainty: value + 95% CI + p-value.

Reviewer request: a metric should not be a single number. This example shows the
three ways EquiMed-DSS attaches uncertainty to a metric:

1. Proportion metrics (CHR, IVI, DFR) now return a Wilson 95% CI and a one-sided
   score test against an acceptability threshold *directly* in their result dict.
2. Any other metric can be wrapped with `inference.bootstrap_metric` to get a
   percentile bootstrap CI over its observation sample (optionally cluster-aware,
   so repeated evaluations of the same patient/visit do not inflate precision).
3. A fairness gap between two groups gets a permutation-test p-value.

Run: python examples/example_uncertainty.py
"""
import numpy as np

from equimed_dss.domain1 import DecisionFlipRate
from equimed_dss.domain4 import ClinicalHallucinationRate, InstructionalVulnerabilityIndex
from equimed_dss.inference import bootstrap_metric, permutation_test

rng = np.random.RandomState(42)
line = "=" * 74


def fmt(name, value, ci_lo, ci_hi, p, n, ptxt="p(>thr)"):
    pstr = "<0.001" if p < 0.001 else f"{p:.3g}"
    print(f"  {name:5s} = {value:6.3f}   95% CI [{ci_lo:6.3f}, {ci_hi:6.3f}]   "
          f"{ptxt} = {pstr}   (n={n})")


print(line)
print("1. PROPORTION METRICS — value + CI + threshold p-value (native output)")
print(line)

# CHR: 274 of 285 claims unsupported
scores = np.full(285, 0.9)
scores[:274] = 0.1                         # below tau -> unsupported
chr_res = ClinicalHallucinationRate().calculate_chr(scores, tau=0.5, threshold=0.05)
fmt("CHR", chr_res["chr"], chr_res["ci_lower"], chr_res["ci_upper"],
    chr_res["p_value_above_threshold"], chr_res["n_claims"])

# IVI: 36 of 132 decisions flip under a biased instruction
neutral = list(range(132))
biased = list(range(132))
for i in range(36):
    biased[i] = -1
ivi_res = InstructionalVulnerabilityIndex().calculate_ivi(neutral, biased, threshold=0.05)
fmt("IVI", ivi_res["ivi_flip_rate"], ivi_res["ci_lower"], ivi_res["ci_upper"],
    ivi_res["p_value_above_threshold"], ivi_res["n_pairs"])

# DFR: 3 of 10 decisions flip under a counterfactual swap
orig = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
cf   = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
dfr_res = DecisionFlipRate().calculate_dfr(orig, cf, threshold=0.05)
fmt("DFR", dfr_res["flip_rate"], dfr_res["ci_lower"], dfr_res["ci_upper"],
    dfr_res["p_value_above_threshold"], dfr_res["n_samples"])

print("\n  Full interpretation string (carries the same stats):")
print("   ", chr_res["interpretation"])

print()
print(line)
print("2. ANY METRIC via bootstrap_metric — CI over the observation sample")
print(line)
chr_fn = lambda s: ClinicalHallucinationRate().calculate_chr(s)
naive = bootstrap_metric(chr_fn, scores, value_key="chr", n_boot=2000, random_state=0)
print("  CHR, ordinary bootstrap :", naive)

# claims grouped by patient/visit -> cluster bootstrap widens the interval
clusters = rng.randint(0, 40, size=len(scores))
clus = bootstrap_metric(chr_fn, scores, value_key="chr", clusters=clusters,
                        n_boot=2000, random_state=0)
print("  CHR, cluster bootstrap  :", clus)

print()
print(line)
print("3. FAIRNESS GAP between two demographic groups — permutation p-value")
print(line)
group_a = rng.normal(0.85, 0.05, 60)       # quality scores, group A
group_b = rng.normal(0.80, 0.05, 60)       # group B
pt = permutation_test(group_a, group_b, n_perm=2000, random_state=1)
print(f"  mean(A) - mean(B) = {pt.estimate:+.3f}   permutation p = {pt.p_value:.3g}"
      f"   (n={pt.n})")
print(f"  {pt}")
