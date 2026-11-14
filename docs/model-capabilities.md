# Model capability defaults

Veloce is an inference compiler and runtime, not a model. It cannot add
knowledge, world understanding, or agentic skill by itself — those come from
the trained model: pretraining data, parameter count, post-training, and
evaluation. What Veloce standardises is how a model *declares* its intended
capabilities, so a `.veloce` file always travels with an honest, machine-
readable model card instead of a marketing claim.

Every Veloce-built model carries a **default capability profile** with four
pillars:

| Pillar | What it means | Delivered by | Typical eval |
|--------|---------------|--------------|--------------|
| `specialist_knowledge` | **Deeper knowledge** — more accurate answers in specialist topics: domain terminology, facts, procedures. | Pretraining coverage + capacity; domain continued pretraining / SFT. | MMLU-Pro, GPQA-Diamond, domain held-out QA |
| `world_understanding` | **Richer world understanding** — real-world context reasoning, not just text patterns: grounded entities, events, physical/social context, situational inference. | Broad, diverse corpora; grounded/multimodal context; world-model objectives. | RealWorldQA, MMMU, Agents' Last Exam (knowledge work) |
| `coding_agentic` | **Stronger coding and agentic** — plan, call tools, complete multi-step tasks: repo-scale changes, shell/tool use, error recovery. | Code corpora, execution feedback, tool-use traces, RL in environments. | SWE-bench Verified, Terminal-Bench, tau-bench |
| `real_world_tasks` | **Real-world task focus** — complex coding, front-end builds, professional workflows; end-to-end deliverables. | Task distributions from real user workflows; environment scaling; deliverable-level evaluation. | SWE-bench Multimodal, WebArena, OSWorld |

## Status semantics

Each pillar in a model card has a `status`:

| Status | Meaning | Requires |
|--------|---------|----------|
| `declared` | The model targets this pillar. **Not** a capability claim. | nothing |
| `trained` | The training pipeline included this pillar's data/objectives. | non-empty `evidence` |
| `verified` | Measured benchmark results exist. | non-empty `evidence` |

The defaults are conservative: a freshly exported model declares all four
pillars at status `declared`. `veloce.capabilities.validate_model_card`
rejects `trained`/`verified` without evidence, unknown pillars, and unknown
statuses, so a model card cannot silently overclaim. The toy classifier from
`make train` and the untrained 275.6M throughput demonstrator from
`make bench-midi` both remain at `declared` for every pillar.

## Model card in the `.veloce` file

`weights.save_veloce(..., model_card=card)` embeds the card in the `.veloce`
manifest; `weights.VeloceFile(path).model_card` reads it back, and

```bash
./velocec model-card examples/tiny_lm.veloce
```

prints it. Build a card with `veloce.capabilities.make_model_card`:

```python
from veloce.capabilities import make_model_card

card = make_model_card(
    "my-model",
    model_type="ternary-transformer",
    params={"ternary": 1_500_000_000},
    training={"status": "trained", "dataset": "..."},
    capabilities={
        "coding_agentic": {
            "status": "trained",
            "evidence": ["SFT on 120k tool-use trajectories"],
        },
    },
)
```

Pillars not passed stay `declared`. A serving stack can therefore reject a
model whose `coding_agentic` status is `declared` when the deployment needs an
agentic model — or route it to an evaluation queue first.

## What Veloce provides vs. what the model must provide

| Layer | Provided by Veloce | Provided by the model |
|-------|--------------------|------------------------|
| Packed low-bit weights, trit/i4 kernels | ✅ | |
| Zero-copy `.veloce` mmap, static arena | ✅ | |
| Prefix caching / KV cache | ❌ (v1.0 is T=1, no KV cache) | |
| Tool-call markers, chat template | ❌ (runtime-agnostic; host handles) | |
| Specialist knowledge, world grounding | | ✅ data + scale |
| Planning, tool use, multi-step agency | | ✅ post-training + RL |
| Real-world deliverable quality | | ✅ data + evaluation |

The table matters: a 275.6M ternary demonstrator compiled by Veloce is a
*throughput* artifact, not a capable assistant. Claiming the four pillars as
delivered features of the compiler would be dishonest; declaring them as the
default target profile and attaching evidence per checkpoint is the contract
this repository can actually keep.
