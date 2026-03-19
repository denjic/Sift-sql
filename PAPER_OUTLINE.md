# PAPER_OUTLINE.md
# Full Section-by-Section Paper Skeleton

**Target venue:** IEEE/ACM conference (COLING, EMNLP, or IEEE TKDE journal track)
**Format:** IEEE double-column, 8-page limit (+ references)
**Current status:** Revised draft based on attached notebook PDF

---

## Title

**Multi-Agent Orchestration for Enhanced Text-to-SQL Generation: A Schema-Aware Approach with Self-Correction**

---

## Author Block

```
[Author Name]
Department of Computer Engineering
[University Name]
[City, Country]
[email@institution.edu]
```

---

## Abstract (~200 words)

**Content to cover:**
- Problem: NL→SQL remains challenging; monolithic LLMs hit a "performance cliff" in enterprise settings due to context overflow and structural hallucination
- Gap: Prior multi-agent work (MAC-SQL, DIN-SQL) uses keyword-based pruning or lacks pre-execution validation; RL-based approaches (MARS-SQL) require training
- Contribution: A training-free four-phase LangGraph pipeline: (1) MiniLM semantic schema pruning, (2) CoT planning agent, (3) deterministic LLaMA-3.1-8B generation, (4) AST Schema Guard + Self-Correction
- Result: 82.1% EX on Spider 1.0 vs. 68.4% zero-shot baseline (+13.7 pp); 75% token reduction on large schemas; 45% error recovery rate
- Future work: Ambiguity resolution, federated execution, schema evolution, RAG value mapping

**Tone:** Active voice, past tense for experiments, present tense for contributions.

---

## Keywords

Text-to-SQL, Multi-Agent Systems, Large Language Models, Schema Linking, Chain-of-Thought Planning, Self-Correction, LangGraph, Abstract Syntax Tree Validation

---

## I. Introduction (~0.75 col)

### A. Opening Hook
- The promise of natural language interfaces to databases for non-technical enterprise stakeholders
- Historical trajectory: rule-based → seq2seq → LLM era

### B. The Performance Cliff Problem
- Academic benchmarks (Spider 1.0: 85%+ EX) vs. enterprise reality (6–21% EX)
- Root cause 1: Context Overflow — schema tokens exceed context window
- Root cause 2: Structural Hallucination — valid SQL referencing non-existent entities
- Cite the "performance cliff" terminology from Li et al. (BIRD, NeurIPS 2023)

### C. Limitations of Prior Solutions
- DIN-SQL: full-schema scanning (26,899 tokens/query); prompt-only correction
- MAC-SQL: keyword pruning (misses idiosyncratic column names); post-execution reactive correction
- MARS-SQL: requires RL fine-tuning; not reproducible without trained weights
- All prior LangGraph implementations: missing one or more of the four phases

### D. Proposed Solution (1-paragraph summary)
- Four-phase LangGraph multi-agent pipeline
- Key novelty: semantic pruning + explicit planning + AST pre-execution validation combined for first time

### E. Contributions List (bulleted)
1. Novel four-phase LangGraph StateGraph architecture with typed state handoff
2. Semantic schema pruning via MiniLM cosine retrieval (75% token reduction, 91.2% recall@5)
3. Dedicated CoT Query Planning Agent emitting a first-class logical blueprint state artifact
4. Pre-execution AST Schema Guard (distinguishes from all prior reactive correction)
5. Comprehensive enterprise gap analysis with concrete architectural roadmap

---

## II. Related Work (~0.75 col)

### A. Foundational Decomposition and Prompt Engineering
- Seq2SQL (Zhong et al., 2017): seq2seq + RL baseline
- DAIL-SQL (Gao et al., 2023): in-context exemplar selection; strong single-prompt baseline
- DIN-SQL (Pourreza & Rafiei, NeurIPS 2023): sequential static decomposition; full-schema injection flaw; prompt-only correction
- Chain-of-Thought prompting (Wei et al., NeurIPS 2022): motivation for planning phase

### B. Collaborative Multi-Agent Architectures
- MAC-SQL (Wang et al., COLING 2025): Selector-Decomposer-Refiner; keyword pruning; open source; 59.59% BIRD EX
- CHESS (Talaei et al., VLDB 2024): LSH + hierarchical vector DB; column-level retrieval; infrastructure-heavy
- Comparison table: schema handling × planning × validation × backbone × training requirement

### C. Reinforcement Learning and Test-Driven Synthesis
- MARS-SQL (Yang et al., arXiv 2025): ReAct RL loop; 77.84% BIRD EX; requires SFT+GRPO fine-tuning
- PExA (Bloomberg, 2025): parallel test-case probing; 70.2% Spider 2.0 EX; addresses value blindness, not hallucination
- Arctic-Text2SQL-R1 (Yao et al., arXiv 2025): reward-grounded reasoning

### D. Enterprise-Oriented Extensions (Future Work Precedents)
- AmbiSQL (Ding et al., 2025): clarification agents for ambiguity
- EvoSchema (Zhang et al., VLDB 2025): dynamic schema discovery
- Saha et al. (IEEE TKDE 2024): federated Text-to-SQL challenges

---

## III. System Architecture (~1.0 col)

### A. Overview and Design Principles
- LangGraph StateGraph abstraction; typed state dictionary (SQLAgentState)
- Zero-shot inference-only design commitment
- Figure 1 reference: the four-phase pipeline diagram (solid = implemented; dashed = future work)

### B. Phase 1: Semantic Schema Linking Agent
- Problem: enterprise schema sizes; context overflow
- Mechanism: all-MiniLM-L6-v2 encoding of serialized table metadata; cosine similarity top-k retrieval
- Foreign key closure algorithm (max 2 hops)
- Key result: 75% token reduction; 91.2% recall@5 on Spider dev

### C. Phase 2: Chain-of-Thought Query Planning Agent
- Problem: planning–generation conflation in all prior systems
- Mechanism: blueprint generation prompt; output enumerates required tables, join columns, filter predicates, aggregations
- Blueprint as a typed intermediate state artifact (key architectural distinction from DIN-SQL, MAC-SQL)
- Effect: largest single-component accuracy driver (+5.6 pp in ablation)

### D. Phase 3: Constrained SQL Generation Agent
- Model: llama-3.1-8b-instant via Groq; temperature = 0.0
- Schema-aware prompting: instructs model to use only pruned_schema entities
- Blueprint-constrained generation: generation agent receives and follows Phase 2 output
- Rationale for open-weight deterministic model vs. stochastic GPT-4

### E. Phase 4: AST Schema Guard and Self-Correction Loop
- Sub-section E.1: AST Guard mechanism (sqlglot parsing; entity membership validation; structured error objects with suggestions)
- Sub-section E.2: Error taxonomy (L1 Syntax, L2 Entity, L3 Join Path, L4 Semantic)
- Sub-section E.3: Self-Correction Agent prompt design; structured error injection; max-N retry policy
- Key distinction from reactive correction: fires before database interaction; provides schema-grounded suggestions

---

## IV. Experimental Evaluation and Results (~1.0 col)

### A. Dataset and Evaluation Metrics
- Spider 1.0 development set (1,034 examples; 20 databases)
- Primary metric: Execution Accuracy (EX); secondary: Exact Match (EM)
- Evaluation script: official Spider evaluator (unmodified)
- Statistical testing: McNemar's test (two-tailed, p < 0.05); 95% Wilson confidence intervals

### B. Baseline Comparison and Ablation Study
- Table I: Pipeline configurations × EX × EM × Avg. Latency
  - Baseline (Zero-Shot): 68.4% EX, 52.1% EM, 1450ms
  - Model A (Linking): 74.2% EX, 61.8% EM, 1200ms
  - Model B (Linking + Guard): 76.5% EX, 62.4% EM, 1350ms
  - Model C (Full Agentic): 82.1% EX, 64.2% EM, 1800ms
- McNemar significance: Baseline vs. Model C: p < 0.001

### C. Analysis and Discussion

**Sub-section C.1: Component Contribution**
- Schema Linking: primary driver of EM improvement; 75% token reduction on >20-table schemas
- CoT Planning: largest single EX driver (+5.6 pp); reduces L2/L3/L4 errors by forcing schema-grounded reasoning
- Self-Correction: 45% error recovery; critical for Extra-Hard queries

**Sub-section C.2: Difficulty-Stratified Results**
- Table: EX by difficulty (Easy/Medium/Hard/Extra-Hard) for all four configurations
- Extra-Hard queries show largest gain (+18.6 pp); validates design rationale

**Sub-section C.3: Schema Size Analysis**
- Table: EX vs. schema size bins (2–5, 6–10, 11–20, >20 tables)
- Demonstrates monotonic improvement with schema size → validates context overflow hypothesis

**Sub-section C.4: Latency–Accuracy Trade-off**
- 400ms overhead for 13.7 pp EX gain; acceptable for enterprise interactive applications
- Correction fires on 31.4% of queries; mean 1.3 attempts per invocation

**Sub-section C.5: Comparison with Prior Work**
- Table: Proposed vs. DIN-SQL vs. MAC-SQL vs. MARS-SQL (with benchmark and backbone context note)
- Emphasize: proposed system achieves parity with DIN-SQL (GPT-4) using an 8B open-weight model in zero-shot regime

---

## V. Future Work and Open Challenges (~0.75 col)

### A. Ambiguity Resolution
- Enterprise query ambiguity; hallucinated constraints; AmbiSQL precedent
- Proposed: Clarification Agent interrupting before Phase 2; targeted multi-choice prompts

### B. Cross-Database Reasoning and Federated Execution
- Heterogeneous data ecosystems (SQL + NoSQL); Saha et al. federated challenges
- Proposed: intermediate AST abstraction layer; federated query engine integration

### C. Schema Evolution and Metadata Monitoring
- Dynamic schema modification; EvoSchema precedent
- Proposed: Dynamic Schema Discovery Agent executing INFORMATION_SCHEMA at runtime

### D. Semantic Content Awareness and Vector-Search Integration
- Value Trap failure mode (structure-correct, value-wrong); PExA evidence-gathering precedent
- Proposed: RAG over high-cardinality cell values; embedding categorical data; mapping colloquial phrases to exact DB strings

---

## VI. Conclusion (~0.25 col)

**Content:**
- Summary of contributions: four-phase LangGraph architecture; semantic pruning; CoT planning; AST pre-execution validation
- Primary result: 82.1% EX on Spider 1.0; 13.7 pp improvement over monolithic baseline
- Key takeaway: the architectural decomposition (not model scale) is the primary driver of improvement — a training-free, open-weight system achieves parity with closed-source GPT-4-based decomposition approaches
- Forward look: four enterprise challenges identified; architectural extensions proposed and motivated
- Closing: the transition from academic benchmark performance to enterprise autonomy requires not just better models, but better system architecture

---

## References

*Ordered by citation number in the paper body.*

| # | Citation |
|---|---------|
| [1] | Zhong et al., "Seq2SQL," CoRR 2017 |
| [2] | Yu et al., "Spider," EMNLP 2018 |
| [3] | Li et al., "BIRD," NeurIPS 2023 |
| [4] | Wei et al., "Chain-of-Thought Prompting," NeurIPS 2022 |
| [5] | Gao et al., "DAIL-SQL," arXiv 2023 |
| [6] | Pourreza & Rafiei, "DIN-SQL," NeurIPS 2023 |
| [7] | Wang et al., "MAC-SQL," COLING 2025 |
| [8] | Talaei et al., "CHESS," VLDB 2024 |
| [9] | Yang et al., "MARS-SQL," arXiv 2025 |
| [10] | Ding et al., "AmbiSQL," arXiv 2025 |
| [11] | Zhang et al., "EvoSchema," VLDB 2025 |
| [12] | Fan et al., "Grounding NL to SQL," ACL 2024 |
| [13] | Saha et al., "Federated Text-to-SQL," IEEE TKDE 2024 |
| [14] | Bloomberg AI, "PExA," 2025 |
| [15] | Yao et al., "Arctic-Text2SQL-R1," arXiv 2025 |
| [16] | Sarker et al., "Valid Efficiency Score," NeurIPS 2024 |
| [17] | Li et al., "Gen-SQL," ACL 2025 |
| [18] | Hogan et al., "Knowledge Graphs," ACM Comput. Surv. 2021 |
| [19] | Chen et al., "Teaching LLMs to Self-Debug," ICLR 2023 |

---

## Appendix (if page limit allows)

### A. Full Prompt Templates
- Planning Agent system prompt (verbatim)
- Generation Agent system prompt (verbatim)
- Self-Correction Agent prompt (verbatim)

### B. Extended Ablation Results
- Model D (Keyword Linker) vs. Model C results
- Model E (Stochastic Generation) vs. Model C results

### C. Case Studies
- 3 example queries: one Easy (baseline sufficient), one Hard (planning crucial), one Extra-Hard (correction required)
- For each: NL query → blueprint → generated SQL → (if applicable) correction → final SQL

### D. LangGraph State Diagram
- Full SQLAgentState TypedDict definition
- State transitions per agent node

---

## Writing Style Guidelines

- **Voice:** Active where possible ("The Schema Guard validates..." not "Validation is performed by...")
- **Tense:** Present for claims and architecture; past for experimental results
- **Abbreviations:** Define on first use: EX (Execution Accuracy), EM (Exact Match), MAS (Multi-Agent System), CoT (Chain-of-Thought), AST (Abstract Syntax Tree)
- **Citations:** IEEE numeric style `[n]`; all claims attributed
- **Tables:** LaTeX `\begin{table}` with `\toprule / \midrule / \bottomrule` (booktabs)
- **Figures:** Figure 1 = pipeline diagram (already in draft); Figure 2 = ablation bar chart; Figure 3 = schema-size vs. EX line chart
- **Page distribution target:** Intro 0.75, Related 0.75, Architecture 1.0, Experiments 1.0, Future 0.75, Conclusion 0.25, References ~1.0, Abstract 0.25 = ~5.75 col = ~8 pages double-column
