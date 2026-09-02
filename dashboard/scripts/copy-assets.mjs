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
 * The ORT copy is MINIMAL by design: ort.webgpu.bundle.min.mjs hardcodes its
 * emscripten loader pair (ort-wasm-simd-threaded.asyncify.{mjs,wasm}) — no
 * other variant can ever be fetched — and stale files are PRUNED so an ORT
 * upgrade can never resurrect an oversized file. A 25 MiB deploy guard fails
 * the build loudly if any deployable asset would exceed Cloudflare Pages'
 * per-file limit.
 *
 * Run automatically via the "postinstall" and "build" npm scripts. Missing
 * source files are skipped with a warning (fresh checkout without training
 * artifacts still builds).
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const REPO_ROOT = join(__dirname, "..", "..");

// Baked demo: no onnxruntime-web / ort wasm (was >25 MiB, blocked CF Pages).
// Clean any stale ort dir if present.
const outDir = join(__dirname, "..", "public", "ort");
if (existsSync(outDir)) { rmSync(outDir, { recursive: true, force: true }); console.log(`[copy-assets] removed stale public/ort/`); }

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

// ---- 3) Cloudflare Pages deploy guard ------------------------------------
// CF Pages hard-rejects any file > 25 MiB at upload; fail the build loudly
// here instead of a mysterious deploy failure in CI.
function walkFiles(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    // Node renamed entry.path -> entry.parentPath across versions; accept both.
    const base = e.path ?? e.parentPath;
    const p = join(base, e.name);
    if (e.isDirectory()) out.push(...walkFiles(p));
    else if (e.isFile()) out.push(p);
  }
  return out;
}

const CF_MAX_BYTES = 25 * 1024 * 1024;
const GUARD_DIRS = ["demo"];
let oversized = 0;
for (const dir of GUARD_DIRS) {
  const abs = join(__dirname, "..", "public", dir);
  if (!existsSync(abs)) continue;
  for (const p of walkFiles(abs)) {
    const size = statSync(p).size;
    if (size > CF_MAX_BYTES) {
      console.error(`[copy-assets] DEPLOY BLOCKED: ${p} is ${(size / 1048576).toFixed(2)} MiB (> 25 MiB CF Pages limit)`);
      oversized++;
    }
  }
}
if (oversized > 0) {
  process.exit(1);
}
console.log("[copy-assets] deploy guard OK: every file in public/demo is under the 25 MiB CF Pages limit");
