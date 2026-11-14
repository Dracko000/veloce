"""Default capability profile for models built with Veloce.

Veloce is an inference compiler and runtime: it does not by itself add
knowledge, reasoning, or agentic skill.  Those come from the trained model
(pretraining data, capacity, post-training).  What Veloce *does* provide is a
default, machine-readable capability profile that every Veloce-built model is
expected to declare in its ``.veloce`` model card, so serving stacks and
evaluators can check what a model claims against measured evidence instead of
guessing from a name.

The four pillars of the default profile:

1. ``specialist_knowledge``   -- deeper knowledge, more accurate answers in
                                specialist topics.
2. ``world_understanding``    -- richer world understanding: real-world context
                                reasoning, not just text-pattern matching.
3. ``coding_agentic``         -- stronger coding and agentic behaviour: plan,
                                call tools, complete multi-step tasks.
4. ``real_world_tasks``       -- real-world task focus: complex coding,
                                front-end builds, professional workflows.

Each pillar carries a ``status``:

* ``declared`` -- the model card says the model targets this pillar, but no
  evidence is attached yet.  This is the default for untrained / throughput
  demonstrators and must never be presented as a capability claim.
* ``trained``  -- the training pipeline included data and objectives for this
  pillar; attach the training recipe in ``evidence``.
* ``verified`` -- measured benchmark results are attached in ``evidence``.

The profile is intentionally conservative.  ``validate_model_card`` rejects
``trained``/``verified`` claims with empty evidence, and rejects unknown
pillars or statuses, so a model card cannot quietly overclaim.
"""

from __future__ import annotations

import copy

PILLARS = {
    "specialist_knowledge": {
        "title": "Deeper knowledge",
        "definition": (
            "More accurate answers in specialist topics: domain terminology, "
            "facts, and procedures that a general text model gets wrong."
        ),
        "delivered_by": (
            "Pretraining data coverage and capacity; domain continued "
            "pretraining or SFT for narrow fields."
        ),
        "metrics": [
            "MMLU-Pro",
            "GPQA-Diamond",
            "domain-held-out QA sets (per vertical)",
        ],
    },
    "world_understanding": {
        "title": "Richer world understanding",
        "definition": (
            "Real-world context reasoning rather than surface text patterns: "
            "grounded entities, events, physical/social context, and "
            "situational inference."
        ),
        "delivered_by": (
            "Broad, diverse, up-to-date pretraining corpora plus multimodal "
            "or tool-grounded context; world-model style training objectives."
        ),
        "metrics": [
            "RealWorldQA",
            "MMMU",
            "Agents' Last Exam (knowledge-work subset)",
        ],
    },
    "coding_agentic": {
        "title": "Stronger coding and agentic",
        "definition": (
            "Plan, call tools, and complete multi-step tasks: repository-scale "
            "code changes, shell/tool use, error recovery, and long-horizon "
            "task completion."
        ),
        "delivered_by": (
            "Code corpora, execution-feedback data, tool-use traces, and RL "
            "in real or simulated environments."
        ),
        "metrics": [
            "SWE-bench Verified",
            "Terminal-Bench",
            "tau-bench (tool use)",
        ],
    },
    "real_world_tasks": {
        "title": "Real-world task focus",
        "definition": (
            "Complex coding, front-end builds, and professional workflows: "
            "end-to-end deliverables rather than isolated benchmark answers."
        ),
        "delivered_by": (
            "Task distributions drawn from real user workflows; environment "
            "scaling; deliverable-level evaluation."
        ),
        "metrics": [
            "SWE-bench Multimodal / front-end tasks",
            "WebArena",
            "OSWorld",
        ],
    },
}

STATUSES = ("declared", "trained", "verified")


def default_capability_profile() -> dict:
    """Return a fresh copy of the default capability profile."""
    return copy.deepcopy(PILLARS)


def default_capabilities() -> dict:
    """Default per-pillar status block for a model card (all ``declared``)."""
    return {
        name: {"status": "declared", "evidence": []}
        for name in PILLARS
    }


def make_model_card(name: str, *, model_type: str = "ternary-transformer",
                    params: dict | None = None, training: dict | None = None,
                    capabilities: dict | None = None,
                    profile: dict | None = None,
                    quality: dict | None = None,
                    source: dict | None = None,
                    quantization: dict | None = None) -> dict:
    """Build a validated model card for a ``.veloce`` manifest.

    Parameters
    ----------
    name : model name (free text, stored verbatim).
    model_type : ``"ternary-transformer"``, ``"toy-classifier"``, ...
    params : arbitrary counts (e.g. ``{"ternary": 275644416}``).
    training : free-form provenance (dataset, method, status).
    capabilities : per-pillar overrides; missing pillars default to
        ``{"status": "declared", "evidence": []}``.
    profile : capability profile to embed; defaults to
        :func:`default_capability_profile`.
    """
    if capabilities is None:
        capabilities = default_capabilities()
    else:
        merged = default_capabilities()
        for key, val in capabilities.items():
            if key not in PILLARS:
                raise ValueError(f"unknown capability pillar {key!r}")
            merged[key] = dict(val)
        capabilities = merged
    card = {
        "name": name,
        "model_type": model_type,
        "capability_profile": "veloce-default",
        "capabilities": capabilities,
        "params": params or {},
        "training": training or {},
        "pillars": profile or default_capability_profile(),
    }
    if source:
        card["source"] = source
    if quantization:
        card["quantization"] = quantization
    if quality:
        card["quality"] = quality
    validate_model_card(card)
    return card


def validate_model_card(card: dict) -> None:
    """Raise ``ValueError`` if a model card is malformed or overclaims."""
    if not isinstance(card, dict):
        raise ValueError("model card must be a dict")
    for key in ("name", "capability_profile", "capabilities"):
        if key not in card:
            raise ValueError(f"model card missing {key!r}")
    caps = card["capabilities"]
    if not isinstance(caps, dict):
        raise ValueError("'capabilities' must be a dict")
    for key, val in caps.items():
        if key not in PILLARS:
            raise ValueError(f"unknown capability pillar {key!r}")
        if not isinstance(val, dict):
            raise ValueError(f"capability {key!r} must be a dict")
        status = val.get("status", "declared")
        if status not in STATUSES:
            raise ValueError(
                f"capability {key!r}: status must be one of {STATUSES}, "
                f"got {status!r}")
        evidence = val.get("evidence", [])
        if status in ("trained", "verified") and not evidence:
            raise ValueError(
                f"capability {key!r}: status {status!r} requires evidence")
    quality = card.get("quality")
    if quality is not None:
        if not isinstance(quality, dict):
            raise ValueError("'quality' must be a dict of metric -> measurement")
        for metric, entry in quality.items():
            if not isinstance(entry, dict):
                raise ValueError(f"quality[{metric!r}] must be a dict")
            for field in ("value", "method", "reproduce"):
                if field not in entry:
                    raise ValueError(
                        f"quality[{metric!r}] missing required field {field!r}")
    source = card.get("source")
    if source is not None:
        if not isinstance(source, dict) or "checkpoint" not in source:
            raise ValueError("'source' must be a dict with a 'checkpoint' field")


def capability_summary(card: dict) -> str:
    """Human-readable one-screen summary of a model card."""
    lines = [
        f"model            : {card.get('name', '?')} "
        f"({card.get('model_type', '?')})",
        f"profile          : {card.get('capability_profile', '?')}",
    ]
    params = card.get("params") or {}
    if params:
        lines.append("params           : " + ", ".join(
            f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
            for k, v in params.items()))
    quality = card.get("quality") or {}
    if quality:
        lines.append("quality          :")
        for metric, entry in quality.items():
            lines.append(f"  {metric:<22} {entry.get('value')}  "
                         f"({entry.get('method', '')})")
    lines.append("capabilities     :")
    for key, spec in (card.get("pillars") or PILLARS).items():
        cap = (card.get("capabilities") or {}).get(key, {})
        status = cap.get("status", "declared")
        evidence = cap.get("evidence", [])
        detail = f" ({len(evidence)} evidence item(s))" if evidence else ""
        lines.append(f"  {key:<22} {status:<9}{detail}  {spec['title']}")
    return "\n".join(lines)


__all__ = [
    "PILLARS", "STATUSES", "default_capability_profile", "default_capabilities",
    "make_model_card", "validate_model_card", "capability_summary",
]
