# EquiMed_DSS Library Enhancement Roadmap

**Date**: December 3, 2025
**Status**: Under Development (Pre-Publication)
**Manuscript**: EquiMed_DSS_Manuscript.pdf

## Overview

This document outlines the comprehensive enhancement plan to align the EquiMed_DSS Python library with the full scope of the research manuscript and original implementation script.

## Current Status

### ✅ Implemented (10/19 Metrics)

**Domain 1: Reliability**
- ICC (Inter-Rater Reliability)
- ECS (Embedding Consistency Score)
- DFR (Decision Flip Rate)

**Domain 2: Fairness & Ethics**
- HER (Hierarchical Equity Ratio)
- HAFG (Harm-Adjusted Fairness Gap)
- ERI (Ethical Risk Index)
- IBS (Intersectional Bias Score)

**Domain 3: Governance**
- TFD (Temporal Fairness Drift)
- ATS (Audit Traceability Score)
- GCI (Governance Compliance Index)

### ❌ Missing Components

## Phase 1: Complete All 19 Metrics ⏳

### Advanced Reliability Metrics
1. **BCI** (Bootstrap Confidence Intervals) - MISSING
2. **SPA** (Statistical Power Analysis) - MISSING
3. **Bias Concentration Index** - MISSING

### Advanced Information-Theoretic Metrics
4. **MIC** (Mutual Information Content) - MISSING
5. **JSD** (Jensen-Shannon Divergence) - MISSING
6. **WD** (Wasserstein Distance) - MISSING

### Advanced Network & Governance Metrics
7. **NM** (Network Modularity) - MISSING
8. **TS** (Transparency Score) - MISSING
9. **RCS** (Robustness Certification Score) - MISSING

## Phase 2: Advanced Statistical Analyses ⏳

### Critical Statistical Methods (From Manuscript Section 2.4)

1. **Hierarchical Linear Modeling (HLM)**
   - Mixed effects models
   - Variance decomposition
   - ICC calculation
   - Formula: `Yij = γ00 + γ10Xij + u0j + rij`

2. **Mediation Analysis**
   - Direct and indirect effects
   - Proportion mediated calculation
   - Bootstrap confidence intervals
   - Formulas: `M = α0 + α1X + ε1` and `Y = β0 + β1X + β2M + ε2`

3. **Network Analysis**
   - Degree centrality
   - Betweenness centrality
   - Clustering coefficients
   - Community detection

4. **Multi-way ANOVA**
   - Race × Gender × SES × Corpus interactions
   - Effect size calculations (eta-squared)
   - Post-hoc tests

5. **Bootstrap Methods**
   - Confidence interval estimation
   - Bias correction
   - Percentile method

6. **Cross-Validation**
   - K-fold cross-validation
   - Stratified sampling
   - Performance stability assessment

7. **Propensity Score Matching**
   - Covariate balance
   - Treatment effect estimation
   - Sensitivity analysis

8. **Reliability Analysis**
   - Cronbach's Alpha
   - Bland-Altman analysis
   - Test-retest reliability
   - Kappa coefficients

## Phase 3: Comprehensive Visualizations ⏳

### From Manuscript Figures 2-7

1. **Figure 2: Reliability Analysis**
   - ICC visualization
   - Cronbach's Alpha by scale
   - Reliability matrix
   - Bland-Altman plots
   - Test-retest temporal stability

2. **Figure 3: Three-Corpus Comparative Analysis**
   - Bias-Gini Index comparison
   - Temporal drift patterns
   - Clinical harm by group
   - Network stability scores

3. **Figure 4: Temporal and Robustness Analysis**
   - Mediation analysis diagrams
   - Advanced regression pie charts
   - Cross-validation performance
   - Bootstrap CI error bars
   - Propensity score matching
   - Sensitivity analysis plots

4. **Figure 5: Ethics and Governance**
   - Ethical risk vs. safety violations scatter
   - RAMS component analysis
   - Fairness ecosystem radar chart
   - Governance compliance bars
   - Reliability scores comparison

5. **Figure 6: Metric Networks**
   - Peer-reviewed network graph
   - Community network graph
   - MIMIC-IV network graph
   - Combined network (all corpora)
   - Edge weights and node centrality

6. **Figure 7: Performance vs. Fairness Trade-offs**
   - Intersectional heatmaps (3x corpora)
   - Race × Gender fairness matrices
   - Pareto-optimality visualization

### Additional Visualizations

7. **Control Charts** (for TFD monitoring)
8. **Correlation Matrices** (enhanced with hierarchical clustering)
9. **Distribution Comparisons** (JSD, WD visualizations)
10. **Clinical Harm Charts** (weighted by demographic groups)
11. **Transparency Scorecards**
12. **Robustness Certification Plots**

## Phase 4: Data Structure & Utilities ⏳

### Data Input Formats (From Script Lines 125-315)

The library must support these input formats:

1. **MySQL Database Connection**
   ```python
   # Peer-reviewed corpus from MySQL
   config = {
       'host': 'hostname',
       'user': 'username',
       'password': 'password',
       'database': 'dbname'
   }
   ```

2. **CSV Files** (Community corpus)
   - Standard CSV with headers
   - Required columns: ['id', 'content', 'source']
   - Optional: demographic columns

3. **TSV Files** (MIMIC-IV corpus)
   - Tab-separated values
   - Clinical note formats
   - Metadata columns

4. **JSON Format**
   - Structured queries
   - Metadata inclusion
   - Demographic information

### Required Utility Modules

#### `equimed_dss/utils/data_formatters.py`
```python
class CorpusLoader:
    - load_from_mysql()
    - load_from_csv()
    - load_from_tsv()
    - load_from_json()
    - validate_format()
    - standardize_schema()

class DemographicProcessor:
    - extract_demographics()
    - create_intersections()
    - validate_demographic_categories()
    - generate_query_combinations()
```

#### `equimed_dss/utils/validators.py`
```python
class DataValidator:
    - validate_corpus_structure()
    - check_required_fields()
    - validate_demographic_completeness()
    - check_data_quality()
```

### Standard Data Schema

```python
{
    "corpus_type": "peer_reviewed | community | clinical",
    "documents": [
        {
            "id": "unique_id",
            "content": "text_content",
            "demographics": {
                "race": "category",
                "gender": "category",
                "age_group": "category",
                "ses": "low | middle | high"
            },
            "metadata": {
                "source": "source_name",
                "timestamp": "ISO8601",
                "corpus_version": "v1.0"
            }
        }
    ]
}
```

## Phase 5: Statistical Analysis Module ⏳

### `equimed_dss/statistics/` Package Structure

```
statistics/
├── __init__.py
├── hierarchical.py          # HLM, mixed effects
├── mediation.py             # Causal mediation analysis
├── network_analysis.py      # Centrality, modularity
├── bootstrap.py             # Bootstrap methods
├── anova.py                 # Multi-way ANOVA
├── propensity.py            # PSM methods
├── reliability.py           # Cronbach's, ICC, Bland-Altman
└── effect_sizes.py          # Cohen's d, eta-squared
```

## Phase 6: Visualization Module Enhancement ⏳

### `equimed_dss/viz/` Package Expansion

```
viz/
├── __init__.py
├── reliability_plots.py     # ICC, Bland-Altman, reliability dashboard
├── network_viz.py           # Network graphs, correlation networks
├── fairness_viz.py          # Heatmaps, radar charts
├── regression_viz.py        # Mediation diagrams, HLM plots
├── temporal_viz.py          # Control charts, drift analysis
├── comparative_viz.py       # Cross-corpus comparisons
└── publication_ready.py     # High-quality figures for papers
```

## Phase 7: Documentation & Examples ⏳

### Enhanced Documentation

1. **Data Format Guide** (`docs/DATA_FORMATS.md`)
2. **Statistical Methods Guide** (`docs/STATISTICAL_METHODS.md`)
3. **Visualization Gallery** (`docs/VISUALIZATION_GALLERY.md`)
4. **API Reference Updates** (expand existing `docs/API_REFERENCE.md`)

### Example Scripts

```
examples/
├── example_full_pipeline.py              # End-to-end analysis
├── example_data_loading.py               # All data formats
├── example_statistical_analysis.py       # HLM, mediation, etc.
├── example_advanced_viz.py               # All visualizations
├── example_intersectional_analysis.py    # Comprehensive fairness
└── example_manuscript_reproduction.py    # Reproduce manuscript figures
```

## Implementation Priority

### **HIGH PRIORITY** (Pre-Publication Requirements)
1. ✅ Complete all 19 metrics (Phases 1)
2. ✅ Data utilities for format conversion (Phase 4)
3. ✅ Core statistical analyses: HLM, Mediation, Network (Phase 2)
4. ✅ Key visualizations from Figures 2-7 (Phase 3)

### **MEDIUM PRIORITY** (Post-Publication Enhancements)
5. Advanced statistical methods (PSM, IV regression)
6. Interactive visualizations (Plotly integration)
7. Comprehensive documentation

### **LOW PRIORITY** (Future Releases)
8. Real-time monitoring dashboards
9. Automated report generation
10. Web-based interface

## Success Metrics

- [ ] All 19 metrics implemented with unit tests
- [ ] Can reproduce all 7 main figures from manuscript
- [ ] Supports all 4 data input formats
- [ ] 90%+ test coverage
- [ ] Complete API documentation
- [ ] Publication-ready example scripts

## Timeline

- **Week 1-2**: Complete all 19 metrics + data utilities
- **Week 3-4**: Implement core statistical analyses
- **Week 5-6**: Create all visualizations
- **Week 7**: Testing, documentation, examples
- **Week 8**: Final review and publication preparation

## Notes

- Repository is now **PRIVATE** for pre-publication development
- All implementations must align with manuscript formulas
- Code must be publication-ready (clean, documented, tested)
- Consider journal requirements for code/data availability statements

---

**Last Updated**: 2025-12-03
**Next Review**: Weekly during implementation
