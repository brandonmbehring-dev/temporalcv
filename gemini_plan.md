# temporalcv: Comprehensive Plan & Repository Audit

**Author:** Gemini (on behalf of User)
**Date:** 2026-01-06
**Status:** Proposal

---

## 1. Executive Summary
This document outlines a strategic plan to professionalize the `temporalcv` repository structure. The primary goals are to reduce cognitive load in the root directory, standardize AI context for a platform-agnostic future, and ensure all components (including the Julia package) are fully integrated into the development lifecycle.

## 2. Platform-Agnostic Context Strategy
**Problem:** The current `CLAUDE.md` file contains critical project mandates, architectural patterns (Knowledge Tiers), and style guides. However, its name implies it is specific only to the Claude AI assistant, potentially causing other agents (Gemini, Copilot, local models) to overlook it or treat it as irrelevant configuration.

**Proposal:** Rename `CLAUDE.md` to `AI_CONTEXT.md` (or `.ai-context.md`).

**Reasoning:**
1.  **Universality:** The content within (Code Style, Knowledge Tiers, Hub Integration) defines the *project's reality*, not the *agent's persona*. It applies regardless of which LLM is assisting.
2.  **Explicit Intent:** A file named `AI_CONTEXT.md` acts as a clear signpost for any automated system to "Read this first to understand the rules of the road."
3.  **Future-Proofing:** As users switch between models or use multi-agent systems, a neutral filename prevents confusion about which "instructions" are active.

**Implementation Steps:**
- [ ] Rename `CLAUDE.md` → `AI_CONTEXT.md`.
- [ ] Update `README.md` to reference `AI_CONTEXT.md` as the source of truth for contributing standards.
- [ ] Update `ROADMAP.md` and any scripts referencing `CLAUDE.md`.

## 3. Repository Organization & Archiving
**Problem:** The root directory is cluttered with 12+ audit files (`codex-audit-*`, `gemini-audit-*`, etc.), mixing historical artifacts with active source code. This increases visual noise and makes navigation difficult.

**Proposal:** Create a dedicated archive structure.

**Reasoning:**
- **Clarity:** The root should only contain entry points (`src`, `tests`, `docs`), configuration (`pyproject.toml`), and key documentation (`README`, `ROADMAP`, `AI_CONTEXT`).
- **History:** Audits are valuable historical records of design decisions and critiques but do not need to be top-level citizens.

**Implementation Steps:**
- [ ] Create directory: `docs/audits/`.
- [ ] Move all `codex-audit*`, `gemini-audit*`, `codex-critique*`, and `gemini-critique*` files to `docs/audits/`.
- [ ] (Optional) Create `docs/audits/README.md` indexing these files by date and agent.

## 4. Julia Package Separation
**Problem:** The `julia/` directory exists within this Python-centric repository. While it has ~95% feature parity, keeping two distinct language implementations in one repo complicates versioning, CI/CD configuration, and package management.

**Proposal:** Extract `julia/` to a new standalone repository (e.g., `temporalcv.jl`).

**Reasoning:**
- **Standardization:** Julia packages typically live in their own repositories to play nicely with the Julia General Registry and `Pkg` manager.
- **Focus:** The Python `temporalcv` repo can focus solely on Python standards (PyPI, mypy, pytest) without the noise of Julia infrastructure.
- **Lifecycle:** The Julia package can evolve at its own pace, with its own versioning scheme, independent of the Python release cycle.

**Implementation Steps:**
- [ ] Create a new directory/repo for the Julia package (outside this root).
- [ ] Move the `julia/` directory contents to the new repository root.
- [ ] Remove the `julia/` directory from `temporalcv`.
- [ ] Update `ROADMAP.md` to reference the separate Julia repo (or remove the section if it's considered fully external).
- [ ] Update `README.md` to point users to the new Julia repository location.

## 5. Documentation Architecture
**Status:** The current "Knowledge Tier System" (T1/T2/T3) is excellent and should be preserved.

**Refinement:**
- Ensure the new `AI_CONTEXT.md` explicitly defines these tiers so every agent knows how to treat assumptions vs. proofs.
- Move "Plans" (e.g., `inactive_plan.md` if it existed here) to `docs/plans/` to keep the active workspace focused.

## 6. Summary of Action Items
1.  **Rename Context:** `CLAUDE.md` -> `AI_CONTEXT.md`.
2.  **Archive Audits:** Move 12 files to `docs/audits/`.
3.  **Separate Julia:** Extract `julia/` to a new repository and remove it from here.
4.  **Update Docs:** Point `README` and `ROADMAP` to the new Julia repo location.

This plan provides a cleaner, more robust, and agent-neutral foundation for `temporalcv`'s future.
