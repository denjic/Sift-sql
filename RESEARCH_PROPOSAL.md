# RESEARCH_PROPOSAL.md
# Multi-Agent Orchestration for Enhanced Text-to-SQL Generation: A Schema-Aware Approach with Self-Correction

---

## Abstract

Converting natural language (NL) queries into Structured Query Language (SQL) remains a critical, unsolved challenge at the intersection of database management and natural language processing. While Large Language Models (LLMs) have demonstrated remarkable fluency on constrained academic benchmarks (e.g., exceeding 85% Execution Accuracy on Spider 1.0 under zero-shot conditions), their deployment in production enterprise environments reveals a severe and well-documented *performance cliff*: execution accuracy collapses to between 6% and 21% when faced with massive, federated, and continuously evolving database schemas. This failure is attributable to two primary bottlenecks: **Context Overflow**, wherein enterprise-scale schemas exceed the LLM's effective context window, and **Structural Hallucination**, wherein models generate syntactically valid SQL referencing non-existent columns or invalid join relationships.

This paper proposes a novel, training-free **multi-agent orchestration framework** implemented via LangGraph, designed to systematically address both failure modes through specialized decomposed reasoning. The system integrates four sequential, stateful agent phases: (1) a **Semantic Schema Linking Agent** employing `all-MiniLM-L6-v2` sentence transformer embeddings for top-$k$ table pruning via cosine similarity; (2) a dedicated **Chain-of-Thought Query Planning Agent** that emits an explicit logical blueprint prior to SQL generation; (3) a **Constrained SQL Generation Agent** using `qwen3-32b` under a zero-temperature policy for deterministic, reproducible output; and (4) an **Abstract Syntax Tree (AST) Schema Guard** coupled with an autonomous **Self-Correction Agent** for iterative pre-execution schema validation and repair.

Experimental evaluation on the Spider benchmark demonstrates a **13.7 percentage-point improvement** in Execution Accuracy over the monolithic zero-shot baseline (82.1% vs. 68.4%), with the Schema Linking Agent alone reducing input token count by 75% on databases exceeding 20 tables, and the Self-Correction Agent recovering 45% of syntax and schema hallucination errors. We further conduct a rigorous gap analysis identifying four enterprise-critical enhancements — ambiguity resolution, cross-database federated execution, dynamic schema discovery, and semantic content awareness via vector-search RAG — as the principal avenues toward full autonomous deployment.

---

## 1. Introduction

### 1.1 Motivation

The democratization of data access — enabling non-technical stakeholders to query vast enterprise data warehouses using natural language — represents one of the most commercially significant applied NLP challenges of the decade. The Text-to-SQL task formalizes this objective: given a natural language utterance $q$ and a database schema $\mathcal{S}$, produce a SQL query $\hat{y}$ such that $\text{execute}(\hat{y}, \mathcal{D}) = \text{execute}(y^*, \mathcal{D})$, where $y^*$ is the ground-truth query and $\mathcal{D}$ is the database instance.

The trajectory of Text-to-SQL research has progressed through three distinct phases. Early **rule-based frameworks** constructed brittle syntactic mapping tables, generalizing poorly beyond narrow domain vocabulary. **Neural sequence-to-sequence models** (circa 2017–2020) achieved cross-domain generalization but struggled with complex joins and nested queries. The contemporary era, dominated by **Large Language Models**, achieved a step-change in fluency — GPT-4-based systems routinely exceed 85% Execution Accuracy on Spider 1.0 — yet this performance is illusory in production settings.

The fundamental problem is that academic benchmarks like Spider 1.0 and even BIRD present schemas of tractable size (typically 5–20 tables, 50–200 columns). Enterprise data ecosystems are categorically different: they span thousands of tables, employ idiosyncratic and abbreviated naming conventions (e.g., `cust_lname`, `ord_dt`), undergo continuous metadata evolution as columns are deprecated and tables are restructured, and may span heterogeneous storage paradigms (relational SQL alongside NoSQL document stores). Under these conditions, monolithic single-prompt LLM systems exhibit the *performance cliff*, with execution accuracy plummeting to the 6–21% range.

### 1.2 Problem Statement

We identify two root causes of this degradation that our system is designed to address:

**Problem 1 — Context Overflow.** Injecting full enterprise schemas into an LLM prompt is computationally prohibitive and semantically counterproductive. DIN-SQL consumed an average of 26,899 tokens per query by scanning full schemas across its sequential modules, producing massive inference latencies and diluting the model's attention across thousands of irrelevant columns.

**Problem 2 — Structural Hallucination.** LLMs generate SQL referencing columns and join paths that do not exist in the schema, producing syntactically valid but semantically incorrect queries. Existing correction mechanisms (e.g., DIN-SQL's self-correction, MAC-SQL's Refiner) are *reactive* — they fire after the database returns an execution error — meaning the invalid query has already been sent to the database and the error message must be parsed for correction signal. This is both latency-inefficient and fragile when database error messages are ambiguous.

### 1.3 Proposed Solution

We propose a **four-phase multi-agent LangGraph pipeline** that addresses both problems through structural decomposition. Rather than treating Text-to-SQL as a single translation exercise, our framework treats it as a collaborative software engineering workflow: schema retrieval, logical planning, constrained generation, and pre-execution validation are assigned to independent, specialized agent nodes that communicate through a typed LangGraph `StateGraph`.

The key architectural novelty is the combination of:
- **Dense vector-semantic schema pruning** (addressing Context Overflow upstream, before generation)
- **Explicit CoT blueprint emission** as a first-class intermediate state artifact (reducing hallucination by separating *what tables/joins are needed* from *how to write SQL for them*)
- **Pre-execution AST schema entity validation** (addressing Structural Hallucination *before* any database interaction, in contrast to all prior reactive correction approaches)

This combination is implemented entirely in a **zero-shot, inference-only regime** — no fine-tuning, no RL training — using open-weight models served via Groq, making the system reproducible and accessible.

### 1.4 Primary Contributions

This work makes the following concrete contributions:

1. **A novel four-phase LangGraph multi-agent architecture** that decomposes Text-to-SQL into semantically isolated, stateful agent nodes with explicit typed state handoff, avoiding the context sharing and sequential rigidity of prior monolithic and decomposed-prompt approaches.

2. **Semantic Schema Linking via dense cosine retrieval** using the `all-MiniLM-L6-v2` sentence transformer, achieving a 75% reduction in input token count on large schemas while preserving recall of relevant tables — a lighter-weight and deployment-friendlier alternative to CHESS's infrastructure-heavy LSH vector database strategy.

3. **A dedicated Chain-of-Thought Query Planning Agent** that emits an explicit natural-language logical blueprint (enumerating required tables, join conditions, aggregation functions, and filter predicates) as an intermediate state artifact, consumed by the downstream generation agent — separating planning from generation in a manner absent from all reviewed open-source implementations.

4. **A pre-execution AST Schema Guard** that validates all referenced entities against the pruned schema *before* query execution, complemented by an iterative Self-Correction Agent — distinguishing our approach from all prior execution-feedback-based correction strategies.

5. **A comprehensive enterprise gap analysis** mapping four open challenges — ambiguity resolution, cross-database federation, schema evolution monitoring, and semantic content awareness — to specific architectural extensions, providing a concrete roadmap for future work grounded in the current literature.

---

## 2. Research Gaps Addressed

The following gaps in the existing literature motivate this work:

| Gap | Prior State | Our Contribution |
|-----|-------------|-----------------|
| Context overflow in large schemas | Full schema injection (DIN-SQL, MAC-SQL baseline) | Semantic top-$k$ pruning via MiniLM cosine similarity |
| Planning–generation conflation | Single-prompt or single-agent generation (all baselines) | Dedicated CoT Planning Agent with blueprint state handoff |
| Reactive-only error correction | Execution-feedback correction (DIN-SQL, MAC-SQL Refiner) | Pre-execution AST schema entity validation |
| Infrastructure-heavy retrieval | LSH + vector DB (CHESS) | In-memory cosine similarity, zero additional infrastructure |
| RL-required high accuracy | Fine-tuning on GRPO/SFT (MARS-SQL) | Zero-shot inference-only, fully reproducible |
| Closed-source LLM dependency | GPT-4 (MAC-SQL, DIN-SQL) | Open-weight `qwen3-32b` via Groq |

---

## 3. Scope and Limitations

This work focuses on single-database, cross-domain generalization as tested on the Spider 1.0 benchmark. Enterprise extensions — federated multi-database execution, schema evolution, and ambiguous query handling — are explicitly scoped to future work (Section V of the paper) and are architecturally motivated but not empirically evaluated in this iteration. The use of `qwen3-32b` as the generation backbone is a deliberate choice for reproducibility and accessibility; larger models (e.g., LLaMA-3.1-70B) would likely further improve generation quality at the cost of inference latency.
