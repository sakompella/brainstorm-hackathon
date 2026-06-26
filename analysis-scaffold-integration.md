# Svelte Scaffold Integration Analysis

## 1) Current scaffold state

### What’s in the scaffold
- `frontend/package.json` (lines 6-19) is a fresh Vite/Svelte starter:
  - scripts: `dev`, `build`, `preview`, `check`
  - deps: Svelte `^5.56.3`, Vite `^8.1.0`, `@sveltejs/vite-plugin-svelte ^7.1.2`, TypeScript `~6.0.2`, `svelte-check ^4.6.0`
- `frontend/src/main.ts` (1-9) is the default mount entry.
- `frontend/src/App.svelte` (1-89) is still scaffold content: hero logos, docs/social cards, and the starter `Counter` component.
- `frontend/src/lib/Counter.svelte` (1-10) is the Svelte 5 `$state` counter demo.
- `frontend/src/app.css` (1-260) is the full starter stylesheet, including the hero layout, docs/social sections, responsive starter typography, and the counter styles.
- `frontend/index.html` (1-13) is still the template HTML with `<title>frontend</title>`.
- `frontend/svelte.config.js` (1-2) is empty, so there is no preprocess/kit-specific setup.

### What should be stripped
- Remove the scaffold UI in `src/App.svelte`: logo stack, “Get started”, docs/social blocks, and the `<Counter />` demo.
- Delete or stop importing `src/lib/Counter.svelte`.
- Replace `src/app.css` with app-specific layout/styles for the neural viewer.
- Remove scaffold-only assets if unused: `src/assets/hero.png`, `src/assets/svelte.svg`, `src/assets/vite.svg`, and likely `public/icons.svg`.
- Update `index.html` title/meta for the real app.

## 2) Vite config needs

### Current state
- `frontend/vite.config.ts` (1-7) only enables the Svelte plugin.
- There is no dev-server proxy, no build config, and no explicit output dir.

### Needed for backend integration
- Add a dev proxy so the Svelte app can call the backend at the same origin during development:
  - proxy `/ws` to `ws://localhost:8000/ws` (WebSocket enabled)
  - likely proxy `/health` too if the UI polls status
- Build output:
  - Vite’s default `outDir` is `dist`, which is fine, but the backend must serve that directory explicitly.
  - If the backend keeps its current default `frontend/`, it will serve the source tree instead of the built app.
  - So either:
    1. keep Vite’s default `dist` and run backend with `--static-dir frontend/dist`, or
    2. change backend defaults/resolution to prefer `frontend/dist` when it exists.

## 3) TypeScript config

### Current setup
- `frontend/tsconfig.json` (1-7) is a project-reference root that points at `tsconfig.app.json` and `tsconfig.node.json`.
- `frontend/tsconfig.app.json` (1-20):
  - extends `@tsconfig/svelte/tsconfig.json`
  - `target: es2023`, `module: esnext`
  - types: `svelte`, `vite/client`
  - `noEmit: true`
  - `allowJs: true`, `checkJs: true`
  - includes `src/**/*.ts`, `src/**/*.js`, `src/**/*.svelte`
- `frontend/tsconfig.node.json` (1-23):
  - Node-side config for `vite.config.ts`
  - `module: nodenext`, `types: node`, `noEmit: true`
  - lint-ish strictness (`noUnusedLocals`, `noUnusedParameters`, etc.)

### What may need adjusting
- If the app won’t use plain JS in `src/`, consider turning off `allowJs`/`checkJs` to reduce noise.
- If you add ambient declarations, ensure they’re included (e.g. `src/**/*.d.ts` if needed).
- If you add more Vite-side config files, expand `tsconfig.node.json` includes accordingly.
- The current setup is otherwise fine for Svelte 5 + TS.

## 4) Integration points with backend.py

### Backend serving contract
- `scripts/backend.py` (426-482) creates the browser-facing server:
  - `@app.websocket("/ws")` is the browser socket endpoint.
  - `@app.get("/")` serves `static_dir / "index.html"`.
  - `StaticFiles(directory=static_dir)` is mounted at `/` if the directory exists.
- `scripts/backend.py` (505-560) exposes `--static-dir/-s`, defaulting to `frontend/`, then resolves it via `scripts/static_assets.resolve_static_dir`.
- `scripts/static_assets.py` (5-23) currently prefers:
  1. the explicitly requested path,
  2. `frontend/` in the repo,
  3. packaged `scripts/frontend/`.

### Important implication
- Right now, the backend will serve the source scaffold directory if `frontend/` exists.
- For a built Svelte app, the backend needs to point at the build output (most likely `frontend/dist`).
- `scripts/serve.py` (19-33, 93-103) is just a legacy wrapper around the same backend path, so it needs the same static-dir treatment.
- `scripts/launcher.py` (193-205) already supports `BRAINSTORM_STATIC_DIR`; that’s a useful hook for pointing at `frontend/dist` without changing the CLI flow.

### WebSocket data flow
- `scripts/backend.py` (311-389) emits browser messages:
  - `type: "features"` payloads with `heatmap`, `centroid`, `center_distance`, `confidence`, etc.
  - `type: "status"` payloads for connection state.
- `consume_upstream()` (163-286) still consumes the raw upstream stream from `stream_data.py`; the frontend should only talk to backend `/ws`, not the raw upstream socket.

## 5) Dev workflow

### Recommended development mode
1. Run the Python streamer on `ws://localhost:8765`.
2. Run the backend on `http://localhost:8000`.
3. Run Vite dev server for the frontend.
4. Proxy `/ws` from Vite to the backend so the frontend can use a relative WebSocket URL.

### Practical shape
- Frontend dev server: `frontend/` on Vite’s port (usually 5173).
- Backend: `http://localhost:8000`.
- Browser UI should connect to `new WebSocket('/ws')` or equivalent same-origin logic.
- This keeps dev/prod behavior aligned:
  - dev: Vite serves source, proxying `/ws`
  - prod: backend serves built files and `/ws` on the same origin

### Production-like mode
- Build the frontend to `frontend/dist`.
- Start backend with `--static-dir frontend/dist` (or set `BRAINSTORM_STATIC_DIR=frontend/dist`).
- Open `http://localhost:8000/`.

## 6) `start_all.sh` and `Makefile`

### `start_all.sh`
- Current script (1-48) only starts:
  - `brainstorm-stream` on `:8765`
  - `brainstorm-backend` on `:8000`
- It does **not** build the Svelte app.
- To support built frontend assets, it should:
  - build `frontend/` before starting backend, and
  - pass `--static-dir frontend/dist` (or equivalent env var) to backend.
- If the intended dev flow is Vite HMR instead, `start_all.sh` should probably stay backend-focused and a separate frontend dev command should be documented.

### `Makefile`
- Current targets (1-121) are Python-only: `sync`, `install`, `stream`, `serve`, `format`, `lint`, `type-check`, `test`.
- No frontend install/build/check targets exist.
- Add frontend targets such as:
  - `frontend-install`
  - `frontend-dev`
  - `frontend-build`
  - `frontend-check`
- Update `serve`/`install` docs if the default path becomes “build frontend first, then run backend with `frontend/dist`”.
- `make serve` currently just runs `uv run brainstorm-serve`; after the scaffold lands, it should either:
  - depend on a frontend build step, or
  - be renamed/documented as the dev-only backend launcher.

### `pyproject.toml` packaging note
- `pyproject.toml` (35-45) includes the entire `frontend/**/*` tree in Python build artifacts.
- That is compatible with shipping built assets, but only if the build happens before packaging.
- If you want wheels/sdists to include only built files, the packaging rules should be narrowed to the final asset directory (e.g. `frontend/dist/**/*`).
