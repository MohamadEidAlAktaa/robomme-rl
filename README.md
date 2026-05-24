# robomme-rl

CS224R team project: reinforcement fine-tuning (RFT) loop with GRPO on top of `robomme_policy_learning`.

## Team
- Mohamad
- Maryam
- Kamal

## Layout
```
robomme-rl/
├── modal/                        # Modal image + app definitions
├── src/                          # RFT loop, GRPO, trajectory dumping
├── scripts/                      # Launch scripts
├── third_party/
│   └── robomme_policy_learning/  # cloned here (not committed)
├── .gitignore
└── README.md
```

## Setup

```bash
git clone <this-repo>
cd robomme-rl

# Clone the upstream policy learning repo into third_party/
git clone <robomme_policy_learning-url> third_party/robomme_policy_learning
```

## RoboMME as a dependency
Our code and RoboMME's code stay separate. `robomme_policy_learning` is a dependency we build on — not something we fork into `src/`.

- **Now:** plain clone into `third_party/robomme_policy_learning` (gitignored).
- **Later:** once we know exactly which RoboMME files we need to edit (eval code, trajectory dumping, RFT hooks), convert `third_party/robomme_policy_learning` to a **git submodule pointing at our fork** of upstream. RoboMME edits land on the fork; this repo just bumps the submodule pointer.

Do not copy RoboMME source files into `src/`.

## Running on Modal
See `modal/` for image and app definitions, and `scripts/` for launch entrypoints.
