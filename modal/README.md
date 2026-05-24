# modal/

Modal image and app definitions for `robomme-rl`.

## Files
- `image.py` — image definition. Wraps RoboMME's upstream Dockerfile (CUDA 12.8 + uv + openpi env + robomme micromamba env) and layers our `src/` on top.
- `app.py` — Modal `App` plus a `smoke_test` function that verifies the build, GPU visibility, and that JAX / PyTorch / RoboMME imports work.

## One-time setup
```bash
# Modal CLI in your local env
pip install modal
modal token new   # paste your token

# Init the RoboMME submodule so the Dockerfile build context is complete
cd third_party/robomme_policy_learning
git submodule update --init --depth 1
cd ../..
```

## Run the smoke test
```bash
modal run modal/app.py::main
```

First run builds the image (~20–30 min on Modal). Later runs hit the cache.

## Notes
- `modal/` has no `__init__.py` on purpose — keeping it a flat directory of scripts avoids shadowing the installed `modal` package.
- Edits to RoboMME source need a rebuild (the Dockerfile `COPY`s the cloned dir at build time). Once we convert `third_party/robomme_policy_learning` to a submodule of our fork, the same flow applies.
