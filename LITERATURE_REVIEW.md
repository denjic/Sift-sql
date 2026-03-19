# LITERATURE_REVIEW.md
# Comprehensive Literature Review: Text-to-SQL via Multi-Agent Systems

---

## 1. Overview and Taxonomic Organization

The Text-to-SQL literature has evolved through three generations of increasing architectural complexity: (1) **rule-based and seq2seq approaches** (pre-2022), (2) **monolithic LLM prompting with decomposition** (2022–2023), and (3) **dynamic multi-agent orchestration** with and without reinforcement learning (2024–2026). This review covers the most significant works across all three generations, with emphasis on the multi-agent era most relevant to the proposed system.

We organize the review along five analytical dimensions:
- **Schema handling strategy** — how the system manages schema injection and pruning
- **Planning mechanism** — whether and how intermediate reasoning is structured
- **Generation backbone** — the underlying LLM and its parameterization
- **Validation and correction** — whether and how generated SQL is validated and repaired
- **Benchmark performance** — reported Execution Accuracy (EX) and Exact Match (EM) on Spider 1.0, BIRD, and Spider 2.0

---

## 2. Comparative Review Table

| # | System | Venue / Year | Schema Handling | Planning Mechanism | Generation Backbone | Validation / Correction | Spider 1.0 EX (%) | BIRD EX (%) | Spider 2.0 EX (%) | Open Source |
|---|--------|-------------|-----------------|-------------------|---------------------|------------------------|:-----------------:|:-----------:|:-----------------:|:-----------:|
| 1 | **Seq2SQL** | CoRR 2017 | Full schema injection | None (seq2seq) | LSTM + RL | None | ~48.3 | — | — | Yes |
| 2 | **DAIL-SQL** | arXiv 2023 | Full schema + exemplar selection | In-context few-shot | GPT-4 | None | 86.6 | 54.76 | — | Yes |
| 3 | **DIN-SQL** | NeurIPS 2023 | Full schema (repeated per module) | Sequential static decomposition (4 modules) | GPT-4 | Prompt-only self-correction | 82.8 | 54.89 | — | Partial |
| 4 | **MAC-SQL** | COLING 2025 | Keyword-based Selector pruning | Decomposer: sub-question generation | GPT-4 | Execution-feedback Refiner | 86.2 | 59.59 | — | Yes |
| 5 | **CHESS** | VLDB 2024 | LSH + hierarchical vector DB | Column-level retrieval + CoT | GPT-4 | Execution-feedback | ~87.0 | 65.0 | — | Yes |
| 6 | **MARS-SQL** | arXiv 2025 | RL-trained Grounding Agent | ReAct Think-Act-Observe loop | Qwen2.5-Coder-7B (SFT+GRPO) | Generation-probability Validation Agent | 89.75 | 77.84 | — | Partial |
| 7 | **PExA** | Bloomberg 2025 | Evidence-gathering via DB probing | Parallel test-case generation | Undisclosed (closed) | Test-case verification | — | — | 70.2 | No |
| 8 | **AmbiSQL** | arXiv 2025 | Standard schema injection | Clarification Agent (multi-choice Q) | GPT-4 | Ambiguity-resolved generation | — | ~58.0 | — | No |
| 9 | **EvoSchema** | VLDB 2025 | Dynamic schema discovery queries | Schema-evolution-aware prompting | GPT-4 | None | — | ~55.0 | — | No |
| 10 | **Arctic-Text2SQL-R1** | arXiv 2025 | Standard injection | RL-based chain-of-thought | Arctic-7B (RL-trained) | Reward-grounded self-correction | — | ~72.0 | — | Partial |
| **Ours** | **Proposed** | — | MiniLM cosine top-$k$ pruning | Dedicated CoT Planning Agent (blueprint) | `llama-3.1-8b-instant`, T=0 | Pre-execution AST Schema Guard + N-retry Self-Correction | **82.1** | — | — | Yes (planned) |

> **Note:** Spider 1.0 and BIRD are distinct benchmarks with different difficulty profiles. Spider 1.0 EX > 85% does not imply competitive BIRD performance; DIN-SQL's 82.8% Spider / 54.89% BIRD illustrates the gap. The proposed system's 82.1% on Spider 1.0 is evaluated in a zero-shot, open-weight, inference-only regime, representing a different operating point than GPT-4-based systems.

---

## 3. Generation 1: Rule-Based and Seq2Seq Systems

### 3.1 Seq2SQL (Zhong et al., 2017)

**Seq2SQL** introduced reinforcement learning-guided sequence-to-sequence generation for Text-to-SQL, operating over the WikiSQL dataset. The model trained a policy to maximize query execution reward. While foundational, Seq2SQL was brittle to schema variability and generalised poorly across domains, motivating the shift toward LLM-based approaches.

**Key limitation:** No generalization beyond the training schema domain; no schema linking; query complexity limited to single-table SELECT/WHERE patterns.

---

## 4. Generation 2: Monolithic LLM Prompting

### 4.1 DAIL-SQL (Gao et al., 2023)

DAIL-SQL established a strong in-context learning baseline by carefully engineering the prompt structure: it includes schema DDL, representative question-SQL pairs selected by semantic similarity, and masking rules to prevent data leakage. On Spider 1.0, DAIL-SQL achieved 86.6% EX using GPT-4, setting the first strong single-prompt baseline. However, the system scales poorly to large schemas (full schema injection) and requires access to a curated exemplar bank.

**Key limitation:** No schema pruning, no structured reasoning decomposition, full dependence on GPT-4, exemplar retrieval cost.

### 4.2 DIN-SQL (Pourreza & Rafiei, NeurIPS 2023)

DIN-SQL was the first prominent system to decompose the generation pipeline into distinct sequential modules: schema linking, query classification (easy/medium/hard/nested), sub-question decomposition for complex queries, and self-correction. This static decomposition achieved 82.8% on Spider 1.0. However, the architecture suffers from two critical deficiencies: (1) it uses the full schema in every module, consuming an average of 26,899 tokens per query; and (2) its self-correction is prompt-only — it provides the error message back to the LLM in a new prompt, without any structured validation of schema entity membership.

**Key limitation:** Full schema injection (context overflow), prompt-only correction (no structural validation), GPT-4 dependency, massive token consumption.

**Relevance to proposed system:** DIN-SQL's decomposition structure motivates our agentic separation of phases, but our system replaces full-schema injection with semantic pruning and replaces prompt-only correction with AST-level pre-execution validation.

---

## 5. Generation 3: Collaborative Multi-Agent Systems

### 5.1 MAC-SQL (Wang et al., COLING 2025)

MAC-SQL is the most directly comparable multi-agent framework to the proposed system. It deploys three agents: a **Selector** that prunes irrelevant tables using keyword-based heuristics, a **Decomposer** that breaks complex questions into sub-questions and generates SQL via chain-of-thought, and a **Refiner** that repairs failing queries using execution feedback.

MAC-SQL achieved 59.59% EX on the BIRD holdout test set and 86.2% on Spider 1.0. The architecture is fully open-source (`wbbeyourself/MAC-SQL`) and implemented using LangChain.

**Critical differences from proposed system:**
- The Selector performs keyword matching, not dense embedding retrieval. It cannot handle idiosyncratic column naming (e.g., `cust_lname` ↔ "customer last name").
- The Decomposer combines planning and generation within a single agent; there is no explicit blueprint state artifact.
- The Refiner fires post-execution only; there is no pre-execution AST validation.
- MAC-SQL uses GPT-4 (closed-source); the proposed system uses `llama-3.1-8b-instant` (open-weight).

### 5.2 CHESS (Talaei et al., VLDB 2024)

CHESS (Contextual Harnessing for Efficient SQL Synthesis) addresses context overflow through Locality-Sensitive Hashing (LSH) over column-level descriptions stored in a hierarchical vector database. This enables semantic retrieval of relevant schema elements at the column level of granularity, finer than the table-level pruning of MAC-SQL or the proposed system.

CHESS achieves approximately 65% EX on BIRD, outperforming MAC-SQL on that benchmark. However, the system requires a pre-built vector database populated with column description embeddings — significant infrastructure overhead for deployment.

**Relevance to proposed system:** CHESS validates the semantic retrieval approach for schema pruning. Our system adopts the same conceptual mechanism (cosine similarity over schema metadata embeddings) but implements it in-memory using `all-MiniLM-L6-v2`, eliminating the infrastructure requirement. The trade-off is table-level rather than column-level granularity.

### 5.3 MARS-SQL (Yang et al., arXiv 2025)

MARS-SQL represents the current state-of-the-art on BIRD, achieving 77.84% EX on the development set and 89.75% on Spider 1.0. Its architecture deploys three RL-trained agents: a **Grounding Agent** for schema linking, a **Generation Agent** trained via GRPO reinforcement learning adopting a ReAct Think-Act-Observe loop, and a **Validation Agent** that selects the optimal trajectory by modeling verification as a next-token prediction task (trained via SFT on Qwen2.5-Coder-7B).

**Critical differences from proposed system:**
- MARS-SQL requires RL-based fine-tuning of both the Grounding and Generation agents before inference — substantial compute and data requirements.
- The ReAct loop implicitly learns when to re-plan, rather than using an explicit blueprint phase; planning is emergent, not structured.
- The Validation Agent uses generation probability scoring, not AST-level schema validation.
- MARS-SQL is not fully reproducible without the fine-tuned model weights.
- The proposed system operates in a zero-shot inference-only regime, requiring no training.

### 5.4 PExA (Bloomberg Engineering, 2025)

PExA adopts a software-testing-inspired paradigm: a **Planner** generates semantically meaningful test cases from the NL query, a **Test Case Generator** executes these against the database to accumulate targeted evidence, and a **SQL Proposer** synthesizes the final query grounded in this evidence. PExA achieved 70.2% on Spider 2.0 (Snow leaderboard).

**Critical differences from proposed system:**
- PExA addresses the *value blindness* problem (lack of visibility into row contents) via live database probing, not the structural hallucination problem.
- PExA has no AST validation layer.
- The codebase is not publicly released as of March 2026.
- PExA's evidence-gathering approach complements rather than subsumes the proposed AST guard; a hybrid architecture combining both would address both failure modes.

### 5.5 AmbiSQL (Ding et al., arXiv 2025)

AmbiSQL targets a failure mode orthogonal to structural hallucination: **semantic ambiguity** in enterprise queries where multiple interpretations are equally valid schema-structurally. AmbiSQL integrates a **Clarification Agent** that pauses the generation pipeline to present the user with a targeted multiple-choice question before computing SQL. This is directly relevant to the proposed system's Future Work Section V.A.

### 5.6 EvoSchema (Zhang et al., VLDB 2025)

EvoSchema addresses **schema evolution** — the continuous modification of production database schemas. It proposes a Dynamic Schema Discovery Agent that executes `INFORMATION_SCHEMA` queries at runtime to construct an up-to-date view of the target database, rather than relying on a static schema dictionary. This is directly relevant to Future Work Section V.C.

---

## 6. Synthesis: The Proposed System's Position in the Literature

The proposed system occupies a distinct position in the design space defined by the axes of:

1. **Training requirement:** Zero-shot inference-only vs. fine-tuning (MARS-SQL) vs. exemplar retrieval (DAIL-SQL)
2. **Schema handling:** Dense semantic pruning (ours, CHESS) vs. keyword pruning (MAC-SQL) vs. full injection (DIN-SQL)
3. **Validation mode:** Pre-execution AST (ours) vs. post-execution feedback (all others) vs. generation probability (MARS-SQL)
4. **Orchestration framework:** LangGraph (ours) vs. custom (MAC-SQL) vs. proprietary (Bloomberg)

No reviewed system combines all four of: dense semantic schema pruning, explicit CoT blueprint planning, deterministic open-weight generation, and pre-execution AST entity validation within a single LangGraph orchestration. This combination represents the primary architectural novelty of the proposed work.

---

## 7. Benchmark Context

It is critical to interpret benchmark numbers in context. Spider 1.0 contains 7,000 training and 1,034 development examples across 200 databases, with schemas averaging ~5 tables. BIRD contains 9,428 training and 1,534 development examples across 95 databases with larger, noisier schemas and external knowledge requirements. Spider 2.0 extends further to enterprise-scale, multi-database, and cross-environment settings.

Systems achieving high Spider 1.0 EX (e.g., MARS-SQL at 89.75%) do not automatically transfer to BIRD or Spider 2.0. The proposed system's 82.1% on Spider 1.0 is evaluated in a significantly more constrained operating regime (open-weight 8B model, zero-shot, no fine-tuning) than GPT-4-based systems at comparable or higher accuracy, representing a distinct and valuable contribution to the reproducible, resource-efficient segment of the literature.
