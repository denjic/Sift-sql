# EXPERIMENT_DESIGN.md
# Experiment Design: Datasets, Baselines, Ablations, and Statistical Protocol

---

## 1. Research Questions

The experimental evaluation is structured to answer four primary research questions:

- **RQ1:** Does the full four-phase agentic pipeline (Model C) achieve significantly higher Execution Accuracy than the monolithic zero-shot baseline?
- **RQ2:** What is the relative contribution of each individual component (schema linking, CoT planning, AST guard) to overall performance?
- **RQ3:** Does semantic schema pruning via MiniLM cosine retrieval outperform keyword-based pruning (as in MAC-SQL's Selector) in terms of recall at equivalent token budgets?
- **RQ4:** What proportion of errors encountered during evaluation are recoverable by the Self-Correction Agent, and how does recovery rate vary by error type?

---

## 2. Datasets

### 2.1 Spider 1.0 (Primary Evaluation)

**Source:** Yu et al. (EMNLP 2018). Available at: `https://yale-lily.github.io/spider`

**Structure:**
- Training: 7,000 question-SQL pairs across 146 databases
- Development: 1,034 question-SQL pairs across 20 databases
- Test (hidden): 2,147 question-SQL pairs across 40 databases (requires leaderboard submission for EX evaluation)
- Database sizes: 2–20 tables per database; 10–100 columns total
- Query complexity distribution: Easy (24.3%), Medium (37.4%), Hard (22.2%), Extra-Hard (16.1%)

**Evaluation split used:** Development set (1,034 examples). The test set requires leaderboard submission; development EX is the standard reproducible metric.

**Pre-processing:** For each example, the database schema is loaded from the corresponding `.sqlite` file using `sqlite3.PRAGMA` introspection. No additional pre-processing is applied.

**Why Spider 1.0?** Spider 1.0 is the canonical cross-domain Text-to-SQL benchmark with the largest body of comparable prior results. Using it as the primary evaluation surface ensures maximum comparability with the literature.

### 2.2 BIRD (Secondary Evaluation — Future Work)

**Source:** Li et al. (NeurIPS 2023). Available at: `https://bird-bench.github.io/`

**Structure:**
- Training: 9,428 question-SQL pairs across 95 databases
- Development: 1,534 question-SQL pairs across 11 databases
- Test (hidden): 1,790 examples
- Database sizes: Up to 70+ tables, 500+ columns; includes dirty/noisy values and external knowledge requirements

**Planned evaluation:** BIRD development set (1,534 examples). This will serve as an out-of-distribution evaluation to assess whether the system's performance on simpler Spider schemas generalizes to harder, noisier enterprise-representative schemas. BIRD results are not reported in the current paper due to scope constraints but are planned for the follow-up submission.

**Expected performance range:** Given the system uses an 8B model in a zero-shot regime, BIRD EX is expected to fall in the 45–58% range, consistent with DIN-SQL (54.89%) and MAC-SQL (59.59%) as comparable baselines.

### 2.3 Spider 2.0 (Aspirational — Future Work)

**Source:** Available at: `https://spider2-sql.github.io/`

Spider 2.0 extends to enterprise-scale, cross-environment, multi-database settings (Snowflake, BigQuery, DuckDB, PostgreSQL). It requires cross-database query routing and federated execution — capabilities scoped to Future Work Sections V.B and V.C of the paper.

---

## 3. Experimental Configurations (Ablation Study)

### 3.1 Configuration Matrix

| Config | Schema Linker | CoT Planner | SQL Generator | AST Guard | Self-Corrector | Label |
|--------|:-------------:|:-----------:|:-------------:|:---------:|:--------------:|-------|
| Baseline | — | — | LLaMA 3.1-8B (full schema) | — | — | Zero-Shot |
| Model A | MiniLM top-5 | — | LLaMA 3.1-8B | — | — | Linking Only |
| Model B | MiniLM top-5 | — | LLaMA 3.1-8B | AST Guard | Self-Corrector | Linking + Guard |
| **Model C** | **MiniLM top-5** | **CoT Blueprint** | **LLaMA 3.1-8B** | **AST Guard** | **Self-Corrector** | **Full Agentic** |
| Model D | Keyword-match (MAC-SQL style) | CoT Blueprint | LLaMA 3.1-8B | AST Guard | Self-Corrector | Keyword Linker Ablation |
| Model E | MiniLM top-5 | CoT Blueprint | LLaMA 3.1-8B (T=0.7) | AST Guard | Self-Corrector | Stochastic Gen Ablation |

**Configuration notes:**
- **Baseline:** The full schema DDL is injected directly into a single prompt. The LLM is asked to generate SQL directly.
- **Model A:** Only the schema linking phase is active. No planning; generation consumes pruned schema directly.
- **Model B:** Schema linking and the validation+correction loop are active, but no CoT planning. Tests the guard's contribution independent of planning.
- **Model C (Full):** All four phases active. Primary reported result.
- **Model D:** Tests whether keyword-based pruning (MAC-SQL-style) achieves comparable recall to MiniLM-based pruning at equivalent top-$k$. Isolates the embedding contribution.
- **Model E:** Tests the effect of non-deterministic generation (T=0.7) vs. deterministic (T=0.0). Controls for stochasticity in the generation backbone.

### 3.2 Component Contribution Analysis (from Table I)

| Transition | EX Gain | Interpretation |
|------------|---------|----------------|
| Baseline → Model A | +5.8 pp | Semantic pruning reduces noise; model attends to relevant schema |
| Model A → Model B | +2.3 pp | Pre-execution AST guard + correction recovers schema hallucination errors |
| Model B → Model C | +5.6 pp | CoT planning blueprint is the largest individual accuracy driver |
| Baseline → Model C | +13.7 pp | Full system benefit over monolithic baseline |

---

## 4. Baselines for Comparison

| Baseline | Source | Spider 1.0 EX | Notes |
|----------|--------|:-------------:|-------|
| Zero-Shot (GPT-3.5) | Internal | ~65.0 | Full schema injection, gpt-3.5-turbo |
| Zero-Shot (LLaMA-3.1-8B) | **This paper** | 68.4 | Full schema injection, same backbone as ours |
| DIN-SQL (GPT-4) | NeurIPS 2023 | 82.8 | Sequential static decomposition |
| DAIL-SQL (GPT-4) | arXiv 2023 | 86.6 | Few-shot exemplar selection |
| MAC-SQL (GPT-4) | COLING 2025 | 86.2 | Multi-agent; keyword pruning; GPT-4 |
| MARS-SQL (Qwen-7B, SFT+GRPO) | arXiv 2025 | 89.75 | RL-trained; highest Spider 1.0 score |
| **Proposed (LLaMA-3.1-8B, zero-shot)** | **This paper** | **82.1** | Open-weight, training-free |

**Critical framing note:** The proposed system is benchmarked against the monolithic LLaMA-3.1-8B zero-shot baseline (68.4%) as the primary comparison, since both use the same model backbone and zero-shot regime. Comparison to GPT-4-based systems (DIN-SQL, MAC-SQL) contextualizes the result against the broader literature but is not the primary claim — the architectural contribution (the four-phase topology) is evaluated through the ablation, not the cross-system comparison.

---

## 5. Evaluation Metrics

### 5.1 Execution Accuracy (EX) — Primary Metric

$$\text{EX} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\text{execute}(\hat{y}_i, \mathcal{D}_i) = \text{execute}(y_i^*, \mathcal{D}_i)\right]$$

Where $\hat{y}_i$ is the predicted SQL, $y_i^*$ is the gold SQL, and $\mathcal{D}_i$ is the database instance. The comparison is set-equality of result tuples (order-insensitive for SELECT queries without explicit ORDER BY).

**Spider EX evaluation script:** `evaluator/evaluation.py` from the official Spider repository, applied without modification.

### 5.2 Exact Match (EM) — Secondary Metric

$$\text{EM} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\text{normalize}(\hat{y}_i) = \text{normalize}(y_i^*)\right]$$

Where `normalize` applies lowercase conversion, keyword normalization, and clause ordering standardization. EM is reported as a secondary metric; EX is the primary.

### 5.3 Valid Efficiency Score (VES) — Efficiency Metric

VES (Sarker et al., NeurIPS 2024) measures the ratio of valid and efficient SQL generation, penalizing queries that are correct but inefficient (e.g., Cartesian joins that happen to produce the right result). This metric is reported for completeness but is not the primary target.

### 5.4 Latency Metrics

| Metric | Definition |
|--------|-----------|
| Phase 1 latency | Time for MiniLM encoding + cosine retrieval |
| Phase 2 latency | Groq API call time for planning |
| Phase 3 latency | Groq API call time for generation |
| Phase 4 latency | AST parsing time + (conditional) correction call time |
| End-to-end latency | Total pipeline time per query (average over Spider dev set) |

---

## 6. Ablation Study: Schema Linking Recall

To isolate the schema linking contribution (RQ3), we evaluate schema retrieval quality independently:

**Recall@k:** Fraction of examples where all gold-annotated required tables are contained in the top-$k$ retrieved tables (after FK closure).

| Method | $k$ | Recall@$k$ | Avg. Token Count |
|--------|-----|:----------:|:----------------:|
| Full schema (no pruning) | — | 100% | ~4,000 |
| Keyword matching (MAC-SQL style) | 5 | 82.3% | ~1,050 |
| MiniLM cosine (proposed) | 5 | 91.2% | ~1,000 |
| MiniLM cosine (proposed) | 8 | 96.8% | ~1,600 |

The 8.9 percentage-point recall improvement of MiniLM over keyword matching at $k=5$ directly corresponds to a reduction in cases where a required table is excluded from the pruned schema, causing an otherwise-correct query to fail.

---

## 7. Statistical Testing Protocol

To ensure reported improvements are statistically significant and not artifacts of dataset variance:

### 7.1 Significance Testing

- **Test:** McNemar's test (two-tailed) on paired binary outcomes (correct/incorrect per example).
- **Null hypothesis:** The two configurations being compared (e.g., Model B vs. Model C) have the same per-example error rate.
- **Significance threshold:** $p < 0.05$.
- **Reported comparisons:** Baseline vs. Model C (primary); all adjacent ablation transitions.

### 7.2 Confidence Intervals

95% Wilson confidence intervals are reported for all EX percentages:

$$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

Where $\hat{p}$ is the observed accuracy, $n=1034$ (Spider dev set size), and $z=1.96$.

### 7.3 Error Category Analysis

Per-query errors are categorized by:
- Query complexity class (Easy/Medium/Hard/Extra-Hard per Spider annotation)
- Error type (L1 Syntax / L2 Schema Entity / L3 Join Path / L4 Semantic)
- Number of correction attempts required
- Number of tables in the schema (binned: 2–5, 6–10, 11–20, >20)

This enables fine-grained attribution of accuracy gains and losses across system configurations.

---

## 8. Reproducibility Protocol

- All random seeds fixed: `numpy.random.seed(42)`, `torch.manual_seed(42)`
- Temperature = 0.0 for all LLM calls (deterministic generation)
- All experiments run on the same hardware (8-core CPU, 16GB RAM)
- Groq API model version pinned to `llama-3.1-8b-instant` (fixed model version)
- Evaluation scripts, prompts, and configuration files released in the repository
- MLflow experiment logs (run IDs, parameters, metrics) archived alongside the paper
