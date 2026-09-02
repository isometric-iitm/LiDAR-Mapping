/**
 * Sync vendored runtime assets + baked data into dashboard/public/ at build
 * time so the static export is fully self-contained (no backend, no fs reads
 * in the browser):
 *
 *   1. ONNX Runtime Web dist   -> public/ort/        (runtime import in the
 *                                                    engine worker; bypasses
 *                                                    the bundler, wasm-safe)
 *   2. checkpoints/history.jsonl + latest results/eval_*.json
 *                             -> public/metrics/*.json  (imported directly by
 *                                                    the training/eval pages —
 *                                                    baked into the JS bundle)
 *
 * Run automatically via the "postinstall" and "build" npm scripts. Missing
 * source files are skipped with a warning (fresh checkout without training
 * artifacts still builds).
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const REPO_ROOT = join(__dirname, "..", "..");

const ortPkg = "onnxruntime-web";
// onnxruntime-web's "exports" map only exposes entry points, not the dist
// folder, so resolve the "./webgpu" export and walk up to package root.
let ortDir;
try {
  const webgpuEntry = require.resolve(`${ortPkg}/webgpu`);
  // .../node_modules/onnxruntime-web/dist/... -> package root
  let dir = dirname(webgpuEntry);
  while (basename(dir) !== "onnxruntime-web" && dirname(dir) !== dir) dir = dirname(dir);
  if (basename(dir) !== "onnxruntime-web") throw new Error("package root not found");
  ortDir = dir;
} catch {
  // during `npm install` (postinstall) the package may not be on disk yet on
  // a fresh install; the `build` script runs this again and is authoritative.
  console.warn(`[copy-ort] ${ortPkg} not resolvable yet (fresh install?); skipping (npm run build will copy)`);
  process.exit(0);
}

const outDir = join(__dirname, "..", "public", "ort");
mkdirSync(outDir, { recursive: true });

// ort.webgpu.bundle.min.mjs is the WebGPU bundle entry; at runtime it fetches
// its emscripten loader (ort-wasm-simd-threaded*.mjs) and the wasm binaries
// (.wasm) relative to this file's URL — all of them must be copied.
const WANTED_FILES = ["ort.webgpu.bundle.min.mjs"];
const WANTED_RE = /^ort-wasm-simd-threaded.*\.(mjs|wasm)$/;

let copied = 0;
const dist = join(ortDir, "dist");
for (const f of readdirSync(dist)) {
  if (WANTED_FILES.includes(f) || WANTED_RE.test(f)) {
    copyFileSync(join(dist, f), join(outDir, f));
    copied++;
  }
}

if (!existsSync(join(outDir, "ort.webgpu.bundle.min.mjs"))) {
  console.error("[copy-assets] FAILED: ort.webgpu.bundle.min.mjs missing from onnxruntime-web/dist");
  process.exit(1);
}
console.log(`[copy-assets] ${copied} files -> public/ort/`);

// ---- 2) baked metrics (training history + latest eval) ------------------
// Consumed via direct `import` in src (bundled at build time), also readable
// at /metrics/*.json for debugging. json import attribute keeps bundlers happy.
const metricsDir = join(__dirname, "..", "public", "metrics");
mkdirSync(metricsDir, { recursive: true });

const historySrc = join(REPO_ROOT, "checkpoints", "history.jsonl");
if (existsSync(historySrc)) {
  const lines = readFileSync(historySrc, "utf8").split("\n").filter((l) => l.trim());
  const rows = [];
  for (const line of lines) {
    try {
      rows.push(JSON.parse(line));
    } catch {
      /* skip malformed line */
    }
  }
  writeFileSync(join(metricsDir, "history.json"), JSON.stringify(rows));
  console.log(`[copy-assets] history.json: ${rows.length} validation checkpoints`);
} else {
  writeFileSync(join(metricsDir, "history.json"), "[]");
  console.warn("[copy-assets] checkpoints/history.jsonl not found; history.json -> []");
}

const resultsDir = join(REPO_ROOT, "results");
if (existsSync(resultsDir)) {
  const evals = readdirSync(resultsDir)
    .filter((f) => /^eval_.*\.json$/.test(f))
    .sort();
  if (evals.length > 0) {
    copyFileSync(join(resultsDir, evals[evals.length - 1]), join(metricsDir, "eval.json"));
    console.log(`[copy-assets] eval.json: ${basename(evals[evals.length - 1])}`);
  } else {
    writeFileSync(join(metricsDir, "eval.json"), "{}");
    console.warn("[copy-assets] no results/eval_*.json found; eval.json -> {}");
  }
} else {
  writeFileSync(join(metricsDir, "eval.json"), "{}");
  console.warn("[copy-assets] results/ dir not found; eval.json -> {}");
}
