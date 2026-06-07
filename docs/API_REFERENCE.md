# API Reference

Complete API reference for EquiMed_DSS library.

## Domain 1: Reliability and Calibration

### equimed_dss.domain1.DynamicFairnessRatio

```python
class DynamicFairnessRatio()
```

Measures performance consistency across dynamic conditions.

#### Methods

##### calculate_dfr
```python
def calculate_dfr(
    baseline_metric: float,
    dynamic_metric: float
) -> Dict[str, Union[float, str]]
```

Calculate Dynamic Fairness Ratio.

**Parameters:**
- `baseline_metric` (float): Performance metric in baseline condition
- `dynamic_metric` (float): Performance metric in dynamic condition

**Returns:**
- Dictionary with keys:
  - `dfr` (float): The DFR score
  - `interpretation` (str): Human-readable interpretation

**Example:**
```python
dfr = DynamicFairnessRatio()
result = dfr.calculate_dfr(baseline_metric=0.85, dynamic_metric=0.80)
print(result['dfr'])  # 0.941
```

---

### equimed_dss.domain1.ExpectedCalibrationScore

```python
class ExpectedCalibrationScore()
```

Measures calibration quality of predictions.

#### Methods

##### calculate_ecs
```python
def calculate_ecs(
    predictions: List[float],
    actuals: List[int],
    n_bins: int = 10
) -> Dict[str, Union[float, str]]
```

Calculate Expected Calibration Score.

**Parameters:**
- `predictions` (List[float]): Predicted probabilities (0-1)
- `actuals` (List[int]): Actual outcomes (0 or 1)
- `n_bins` (int): Number of calibration bins (default: 10)

**Returns:**
- Dictionary with ECS score and interpretation

---

### equimed_dss.domain1.IntraclassCorrelationCoefficient

```python
class IntraclassCorrelationCoefficient()
```

Assesses reliability across raters or conditions.

#### Methods

##### calculate_icc
```python
def calculate_icc(
    ratings: List[List[float]]
) -> Dict[str, Union[float, str]]
```

Calculate Intraclass Correlation Coefficient.

**Parameters:**
- `ratings` (List[List[float]]): Matrix where each row is a subject and each column is a rater

**Returns:**
- Dictionary with ICC score and interpretation

---

## Domain 2: Fairness, Equity, and Ethics

### equimed_dss.domain2.HierarchicalEquityRatio

```python
class HierarchicalEquityRatio()
```

Calculates equity ratios across demographic groups.

#### Methods

##### calculate_her
```python
def calculate_her(
    group_scores: Dict[str, float],
    reference_group: str = 'White'
) -> Dict[str, Dict[str, Any]]
```

Calculate Hierarchical Equity Ratio for each group.

**Parameters:**
- `group_scores` (Dict[str, float]): Mapping of group names to performance scores
- `reference_group` (str): Name of reference group (default: 'White')

**Returns:**
- Dictionary mapping group names to their HER scores and interpretations

**Raises:**
- `ValueError`: If reference group not found in scores

##### calculate_bias_gini
```python
def calculate_bias_gini(
    scores: List[float]
) -> float
```

Calculate Bias-Gini Index for dispersion measurement.

**Parameters:**
- `scores` (List[float]): List of performance scores

**Returns:**
- Gini coefficient (0-1)

---

### equimed_dss.domain2.HarmAdjustedFairnessGap

```python
class HarmAdjustedFairnessGap()
```

Quantifies fairness gaps weighted by clinical harm.

#### Methods

##### calculate_hafg
```python
def calculate_hafg(
    group1_errors: Dict[str, int],
    group2_errors: Dict[str, int],
    fn_harm_weight: float = 10.0,
    fp_harm_weight: float = 1.0
) -> Dict[str, Union[float, str]]
```

Calculate Harm-Adjusted Fairness Gap.

**Parameters:**
- `group1_errors` (Dict[str, int]): Error counts for group 1 with keys 'fn', 'fp'
- `group2_errors` (Dict[str, int]): Error counts for group 2 with keys 'fn', 'fp'
- `fn_harm_weight` (float): Weight for false negative harm (default: 10.0)
- `fp_harm_weight` (float): Weight for false positive harm (default: 1.0)

**Returns:**
- Dictionary with HAFG score and interpretation

---

### equimed_dss.domain2.EthicalRiskIndex

```python
class EthicalRiskIndex()
```

Aggregates ethical violations weighted by severity.

#### Methods

##### calculate_eri
```python
def calculate_eri(
    violations: List[Dict[str, float]],
    n_total_outputs: int
) -> Dict[str, Union[float, str]]
```

Calculate Ethical Risk Index.

**Parameters:**
- `violations` (List[Dict]): List of violations, each with 'severity' key (1-10 scale)
- `n_total_outputs` (int): Total number of model outputs

**Returns:**
- Dictionary with ERI score and interpretation

---

### equimed_dss.domain2.IntersectionalBiasScore

```python
class IntersectionalBiasScore()
```

Detects bias in intersectional subgroups.

#### Methods

##### calculate_subgroup_similarity
```python
def calculate_subgroup_similarity(
    subgroup_vectors: Dict[str, np.ndarray]
) -> Dict[str, Any]
```

Calculate similarity and detect outlier subgroups.

**Parameters:**
- `subgroup_vectors` (Dict[str, np.ndarray]): Mapping of subgroup names to performance vectors

**Returns:**
- Dictionary with outlier detection results

---

## Domain 3: Governance and Transparency

### equimed_dss.domain3.TemporalFairnessDrift

```python
class TemporalFairnessDrift()
```

Tracks fairness degradation over time.

#### Methods

##### calculate_drift
```python
def calculate_drift(
    time_series_metrics: List[float]
) -> Dict[str, Any]
```

Calculate drift metrics using statistical process control.

**Parameters:**
- `time_series_metrics` (List[float]): Fairness metrics over time

**Returns:**
- Dictionary with drift statistics and control limits

---

### equimed_dss.domain3.AuditTraceabilityScore

```python
class AuditTraceabilityScore()
```

Measures audit trail completeness.

#### Methods

##### calculate_ats
```python
def calculate_ats(
    audit_logs: List[Dict[str, bool]]
) -> Dict[str, Union[float, str]]
```

Calculate Audit Traceability Score.

**Parameters:**
- `audit_logs` (List[Dict]): List of audit entries with required fields as boolean flags

**Returns:**
- Dictionary with ATS score and interpretation

---

### equimed_dss.domain3.GovernanceComplianceIndex

```python
class GovernanceComplianceIndex()
```

Assesses governance and regulatory compliance.

#### Methods

##### calculate_gci
```python
def calculate_gci(
    requirements: List[bool]
) -> Dict[str, Union[float, str]]
```

Calculate Governance Compliance Index.

**Parameters:**
- `requirements` (List[bool]): List of compliance requirements met (True/False)

**Returns:**
- Dictionary with GCI score and interpretation

---

## Geographic Equity (v1.1.0)

### equimed_dss.geographic.BurdenEvidenceMismatch

```python
class BurdenEvidenceMismatch()
```

Computes the total-variation distance between the regional evidence distribution and the
regional disease-burden distribution.

#### Methods

##### calculate_bemi
```python
def calculate_bemi(
    evidence_counts: Dict[str, int],
    burden_shares: Dict[str, float]
) -> Dict[str, Union[float, str]]
```

Calculate Burden-Evidence Mismatch Index (BEMI).

**Parameters:**
- `evidence_counts` (Dict[str, int]): Number of studies (or cases) per WHO region.
- `burden_shares` (Dict[str, float]): Normalized disease-burden share per region; shares
  must sum to 1.0. Use `WHO_REGION_IHD_BURDEN` for IHD DALY shares (Roth GA et al., 2020).

**Returns:**
- Dictionary with keys:
  - `bemi` (float): Total-variation distance in [0, 1]. 0 = evidence mirrors burden;
    1 = distributions are completely disjoint.
  - `evidence_shares` (Dict[str, float]): Computed regional evidence shares.
  - `interpretation` (str): Human-readable verdict.

**Formula:**
```
BEMI = 0.5 * sum_r |evidence_share_r - burden_share_r|
```

**Example:**
```python
from equimed_dss.geographic import BurdenEvidenceMismatch, WHO_REGION_IHD_BURDEN

bemi = BurdenEvidenceMismatch()
result = bemi.calculate_bemi(
    evidence_counts={"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2},
    burden_shares=WHO_REGION_IHD_BURDEN
)
print(result["bemi"])   # e.g., 0.42
```

---

### equimed_dss.geographic.GeographicConcentration

```python
class GeographicConcentration()
```

Quantifies how evenly studies are distributed across regions using a sample-corrected Gini
coefficient and normalized Shannon entropy.

#### Methods

##### calculate_gcc
```python
def calculate_gcc(
    region_counts: Dict[str, int]
) -> Dict[str, Union[float, str]]
```

Calculate Geographic Concentration of Coverage (GCC).

**Parameters:**
- `region_counts` (Dict[str, int]): Number of studies (or cases) per region.

**Returns:**
- Dictionary with keys:
  - `gini_corrected` (float): Sample-corrected Gini G* in [0, 1]. 0 = even; 1 = single region.
  - `entropy_normalized` (float): Normalized Shannon entropy H_norm in [0, 1]. 1 = even;
    0 = single region. Runs opposite to G*.
  - `concentration` (float): 1 - H_norm; higher means more concentrated.
  - `interpretation` (str): Human-readable summary.

**Formulas:**
```
G* = (R / (R - 1)) * G_raw
H_norm = -sum_r p_r * ln(p_r) / ln(R)
concentration = 1 - H_norm
```

**Example:**
```python
from equimed_dss.geographic import GeographicConcentration

gcc = GeographicConcentration()
result = gcc.calculate_gcc({"AFRO": 5, "AMRO": 40, "EURO": 30, "SEARO": 3, "WPRO": 10, "EMRO": 2})
print(result["gini_corrected"])       # e.g., 0.51
print(result["entropy_normalized"])   # e.g., 0.84
```

---

### equimed_dss.geographic.WHO_REGION_IHD_BURDEN

```python
WHO_REGION_IHD_BURDEN: Dict[str, float]
```

Module-level constant containing normalized IHD DALY shares per WHO region, derived from
Roth GA et al., 2020 (GBD Compare for IHD). Shares sum to 1.0. AFRO and SEARO together
carry approximately 36% of global IHD burden.

**Usage:**
```python
from equimed_dss.geographic import WHO_REGION_IHD_BURDEN

print(WHO_REGION_IHD_BURDEN)
# {"AFRO": ..., "AMRO": ..., "EURO": ..., "SEARO": ..., "WPRO": ..., "EMRO": ...}
```

---

## Reporting Tables (v1.1.0)

### equimed_dss.reporting.hierarchical_coefficients_table

```python
def hierarchical_coefficients_table(
    hlm_result: Dict[str, Any],
    decimals: int = 3
) -> pd.DataFrame
```

Convert a `HierarchicalLinearModeling` result dictionary into a tidy DataFrame with one row
per coefficient, including estimate, standard error, t-value, p-value, and confidence interval.

**Parameters:**
- `hlm_result` (Dict): Result dict from `HierarchicalLinearModeling`.
- `decimals` (int): Rounding precision (default: 3).

**Returns:** `pd.DataFrame`

---

### equimed_dss.reporting.mediation_effects_table

```python
def mediation_effects_table(
    mediation_result: Dict[str, Any],
    decimals: int = 3
) -> pd.DataFrame
```

Convert a `MediationAnalysis` result dictionary into a tidy DataFrame with rows for the
direct effect, indirect effect, total effect, and `proportion_mediated`. The column
`outside_bounds` is set to `True` when `proportion_mediated` falls outside [0, 1]
(competitive or unstable mediation); the value is reported unclamped.

**Parameters:**
- `mediation_result` (Dict): Result dict from `MediationAnalysis`.
- `decimals` (int): Rounding precision (default: 3).

**Returns:** `pd.DataFrame`

---

### equimed_dss.reporting.network_centrality_table

```python
def network_centrality_table(
    network_result: Dict[str, Any],
    decimals: int = 3
) -> pd.DataFrame
```

Convert a `NetworkStatistics` result dictionary into a tidy DataFrame with one row per node,
including degree centrality, betweenness centrality, and clustering coefficient.

**Parameters:**
- `network_result` (Dict): Result dict from `NetworkStatistics`.
- `decimals` (int): Rounding precision (default: 3).

**Returns:** `pd.DataFrame`

---

### equimed_dss.reporting.geographic_table

```python
def geographic_table(
    bemi_result: Dict[str, Any],
    gcc_result: Dict[str, Any],
    decimals: int = 3
) -> pd.DataFrame
```

Combine BEMI and GCC result dictionaries into a single tidy summary DataFrame.

**Parameters:**
- `bemi_result` (Dict): Result dict from `BurdenEvidenceMismatch.calculate_bemi`.
- `gcc_result` (Dict): Result dict from `GeographicConcentration.calculate_gcc`.
- `decimals` (int): Rounding precision (default: 3).

**Returns:** `pd.DataFrame`

---

### equimed_dss.reporting.export_table

```python
def export_table(
    df: pd.DataFrame,
    fmt: str,
    path: Optional[str] = None,
    decimals: int = 3
) -> str
```

Render a DataFrame to a string in the specified format, optionally writing to a file.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to export (typically produced by one of the table functions).
- `fmt` (str): Output format. One of `"markdown"`, `"latex"`, `"html"`.
- `path` (Optional[str]): If provided, write the rendered string to this file path.
- `decimals` (int): Rounding precision applied before rendering (default: 3).

**Returns:** Rendered string in the requested format.

**Example:**
```python
from equimed_dss.reporting import geographic_table, export_table

df = geographic_table(bemi_result, gcc_result)
md = export_table(df, fmt="markdown", path="results/geo.md")
export_table(df, fmt="latex", path="results/geo.tex")
export_table(df, fmt="html",  path="results/geo.html")
```

---

## Utilities

### equimed_dss.utils.data_loader

#### generate_synthetic_fairness_data
```python
def generate_synthetic_fairness_data(
    n_groups: int = 4,
    score_range: Tuple[float, float] = (0.5, 0.9)
) -> Dict[str, float]
```

Generate synthetic fairness data for testing.

**Parameters:**
- `n_groups` (int): Number of demographic groups
- `score_range` (Tuple[float, float]): Range of performance scores

**Returns:**
- Dictionary of group scores

---

### equimed_dss.utils.visualization

#### plot_her_heatmap
```python
def plot_her_heatmap(
    her_scores: Dict[str, Dict[str, float]],
    title: str = "Hierarchical Equity Ratio Heatmap",
    save_path: Optional[str] = None
) -> None
```

Plot HER scores as a heatmap.

**Parameters:**
- `her_scores` (Dict): Nested dictionary of HER scores
- `title` (str): Plot title
- `save_path` (Optional[str]): Path to save figure

---

## Common Return Types

### Metric Result Dictionary

Most metrics return a dictionary with the following structure:

```python
{
    'score': float,           # The metric value
    'interpretation': str,    # Human-readable interpretation
    'verdict': str,          # Quick assessment (e.g., "Equitable", "Disparity")
    'metadata': Dict         # Additional context (varies by metric)
}
```

---

## Error Handling

All metrics may raise:
- `ValueError`: Invalid input parameters
- `TypeError`: Incorrect input types
- `ZeroDivisionError`: Division by zero in calculations

Always wrap metric calculations in try-except blocks in production:

```python
try:
    result = metric.calculate(data)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
```

---

## Type Hints

The library uses Python type hints throughout. For full type information, use:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from equimed_dss.domain2 import HierarchicalEquityRatio
```

---

For usage examples, see the [METRICS_GUIDE.md](METRICS_GUIDE.md) or the `examples/` directory.
