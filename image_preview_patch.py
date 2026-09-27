#!/usr/bin/env python3
"""Modly image-preview addon - adds a "Preview" node + param sliders to stock Modly.

Two independent features are patched into the desktop app's renderer bundles
(the node UI is compiled into app.asar, so an extension cannot add these):

1. Preview node: a built-in node type (`previewImageNode`, labelled "Preview")
   that renders the incoming image output as a data: URL. It does NOT touch the
   existing "Preview Views" node.
2. Param sliders: a range slider next to every numeric (float) parameter that
   declares `min` + `max`, plus a fix for the stock number box that rewrote
   "0." into a whole number while typing.

Safety: glob-discovered bundles (no hard-coded hashes), unique-anchor checks,
per-feature sentinel guards, stock backup + SHA256 record, restore-by-copy with
verify. Features apply independently - an app that already has the Preview
patch can gain the slider patch without losing it.

CLI (backup/swap/restore need elevation; stage/status work plain):
  python image_preview_patch.py stage      --app <app.asar> --work <dir>
  python image_preview_patch.py status     --app <app.asar>
  python image_preview_patch.py finalize   --app <app.asar> --staged <new.asar>
  python image_preview_patch.py restore    --app <app.asar>
  python image_preview_patch.py apply-wait --app <app.asar> --staged <new.asar> --pid <n> --exe <Modly.exe> --log <file>
  python image_preview_patch.py findapp
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WP_GLOB = "out/renderer/assets/WorkflowsPage-*.js"
PP_GLOB = "out/renderer/assets/paramPicker-*.js"
GP_GLOB = "out/renderer/assets/GeneratePage-*.js"
SENTINEL = "__modlyImgPreview"
SLIDER_SENTINEL = "__modlyFloatSlider"

BACKUP_SUFFIX = ".modlypreview-bak"
RECORD_SUFFIX = ".modlypreview.json"


class PatchError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Injected component (JS, matching the bundle's compiled-JSX style)
# --------------------------------------------------------------------------- #

NEW_COMPONENT = r'''function ImagePreviewNode({ id, selected }) {
  /* __modlyImgPreview */
  const nodeImageOutputs = useWorkflowRunStore((s) => s.nodeImageOutputs);
  const { getEdges } = useReactFlow();
  const incomingEdge = getEdges().find((e) => e.target === id);
  const imageUrl = incomingEdge ? nodeImageOutputs[incomingEdge.source] : void 0;
  const [dataUrl, setDataUrl] = reactExports.useState(void 0);
  reactExports.useEffect(() => {
    let cancelled = false;
    if (!imageUrl) { setDataUrl(void 0); return; }
    (async () => {
      try {
        const settings = await window.electron.settings.get();
        const wsDir = String(settings.workspaceDir).replace(/\\/g, "/").replace(/\/+$/, "");
        const rel = String(imageUrl).replace(/^\/workspace\//, "");
        const absPath = wsDir + "/" + rel;
        const base64 = await window.electron.fs.readFileBase64(absPath);
        if (!cancelled) setDataUrl("data:" + mimeFromPath(absPath) + ";base64," + base64);
      } catch (e) {
        if (!cancelled) setDataUrl(void 0);
      }
    })();
    return () => { cancelled = true; };
  }, [imageUrl]);
  return jsxRuntimeExports.jsxs(BaseNode, {
    id,
    selected,
    title: "Preview",
    minWidth: 200,
    icon: /* @__PURE__ */ jsxRuntimeExports.jsx("svg", { width: "12", height: "12", viewBox: "0 0 24 24", fill: "none", stroke: "#38bdf8", strokeWidth: "2", children: /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, { children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("rect", { x: "3", y: "3", width: "18", height: "18", rx: "2" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("circle", { cx: "8.5", cy: "8.5", r: "1.5" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("polyline", { points: "21 15 16 10 5 21" })
    ] }) }),
    subheader: /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "flex items-center gap-1.5 px-3 py-2", children: [
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-medium border border-sky-500/30 bg-sky-500/10 text-sky-400", children: "image" }),
      /* @__PURE__ */ jsxRuntimeExports.jsx("span", { className: "text-[9px] text-zinc-600", children: "\u2192 preview" })
    ] }),
    handles: /* @__PURE__ */ jsxRuntimeExports.jsx(Handle, { type: "target", position: Position.Left, style: { background: "#38bdf8", width: 14, height: 14, border: "2.5px solid #18181b" } }),
    children: /* @__PURE__ */ jsxRuntimeExports.jsx("div", { className: "px-2 pb-2 pt-1 flex-1 min-h-0", children: dataUrl
      ? /* @__PURE__ */ jsxRuntimeExports.jsx("img", { src: dataUrl, alt: "preview", className: "nodrag w-full h-full object-contain rounded" })
      : /* @__PURE__ */ jsxRuntimeExports.jsx("p", { className: "py-3 text-center text-[10px] text-zinc-600 italic", children: "Connect an image and run to preview." })
    })
  });
}
'''

PANEL_ENTRY = (
    '  { type: "previewImageNode", label: "Preview", color: "#38bdf8", '
    'icon: /* @__PURE__ */ jsxRuntimeExports.jsxs(jsxRuntimeExports.Fragment, '
    '{ children: [/* @__PURE__ */ jsxRuntimeExports.jsx("rect", '
    '{ x: "3", y: "3", width: "18", height: "18", rx: "2" }), '
    '/* @__PURE__ */ jsxRuntimeExports.jsx("circle", { cx: "8.5", cy: "8.5", r: "1.5" }), '
    '/* @__PURE__ */ jsxRuntimeExports.jsx("polyline", { points: "21 15 16 10 5 21" })] }) },'
)

BUILTIN_ENTRY = (
    '  { type: "previewImageNode", label: "Preview", color: "#38bdf8", '
    'description: "Displays the image output of a connected node" },'
)


def _once(text, anchor, what):
    n = text.count(anchor)
    if n != 1:
        raise PatchError(f"anchor '{what}' found {n}x (need exactly 1)")
    return anchor


# --------------------------------------------------------------------------- #
# Patch application
# --------------------------------------------------------------------------- #

def find_bundles(tree_dir):
    tree = Path(tree_dir)
    wp = sorted(tree.glob(WP_GLOB))
    pp = sorted(tree.glob(PP_GLOB))
    gp = sorted(tree.glob(GP_GLOB))
    if len(wp) != 1:
        raise PatchError(f"expected exactly 1 WorkflowsPage bundle, found {len(wp)}")
    if len(pp) != 1:
        raise PatchError(f"expected exactly 1 paramPicker bundle, found {len(pp)}")
    if len(gp) != 1:
        raise PatchError(f"expected exactly 1 GeneratePage bundle, found {len(gp)}")
    return (
        wp[0].relative_to(tree).as_posix(),
        pp[0].relative_to(tree).as_posix(),
        gp[0].relative_to(tree).as_posix(),
    )


WP_ANCHORS = {
    "component": 'function PreviewImageNode({ id, selected }) {',
    "nodeTypes": 'previewNode: PreviewImageNode, waitNode:',
    "panel": '  { type: "waitNode", label: "Wait", color: "#71717a", icon:',
    "builtin": '  { type: "waitNode", label: "Wait", color: "#71717a", description: "Pauses the workflow until you click Continue" },',
    "targetColor": 'targetNode?.type === "previewNode" ? HANDLE_COLOR.image :',
    "tints": 'previewNode: { fill: "rgba(56,189,248,0.22)", stroke: "#38bdf8" }',
    "outType": 'if (node.type === "textNode") return "text";',
    "inType": 'if (node.type === "previewNode") return "image";',
}

PP_ANCHORS = {
    "withoutSource": 'new Set(["outputNode", "previewNode"])',
    "nodeLabel": 'if (node.type === "previewNode") return "Preview Views";',
    "outType": 'if (node.type === "previewNode") return "image";',
}


def apply_workflows(text):
    if SENTINEL in text:
        raise PatchError("WorkflowsPage already patched (sentinel present)")
    for key, anchor in WP_ANCHORS.items():
        _once(text, anchor, f"WP:{key}")

    text = text.replace(
        WP_ANCHORS["component"],
        NEW_COMPONENT.rstrip("\n") + "\n" + WP_ANCHORS["component"],
        1,
    )
    text = text.replace(
        WP_ANCHORS["nodeTypes"],
        'previewNode: PreviewImageNode, previewImageNode: ImagePreviewNode, waitNode:',
        1,
    )
    text = text.replace(
        WP_ANCHORS["panel"],
        PANEL_ENTRY + "\n" + WP_ANCHORS["panel"],
        1,
    )
    text = text.replace(
        WP_ANCHORS["builtin"],
        BUILTIN_ENTRY + "\n" + WP_ANCHORS["builtin"],
        1,
    )
    text = text.replace(
        WP_ANCHORS["targetColor"],
        '(targetNode?.type === "previewNode" || targetNode?.type === "previewImageNode") ? HANDLE_COLOR.image :',
        1,
    )
    text = text.replace(
        WP_ANCHORS["tints"],
        WP_ANCHORS["tints"]
        + ', previewImageNode: { fill: "rgba(56,189,248,0.22)", stroke: "#38bdf8" }',
        1,
    )
    text = text.replace(
        WP_ANCHORS["outType"],
        WP_ANCHORS["outType"] + '\n  if (node.type === "previewImageNode") return "image";',
        1,
    )
    text = text.replace(
        WP_ANCHORS["inType"],
        WP_ANCHORS["inType"] + '\n  if (node.type === "previewImageNode") return "image";',
        1,
    )
    return text


def apply_parampicker(text):
    for key, anchor in PP_ANCHORS.items():
        _once(text, anchor, f"PP:{key}")

    text = text.replace(
        PP_ANCHORS["withoutSource"],
        'new Set(["outputNode", "previewNode", "previewImageNode"])',
        1,
    )
    text = text.replace(
        PP_ANCHORS["nodeLabel"],
        PP_ANCHORS["nodeLabel"] + '\n  if (node.type === "previewImageNode") return "Preview";',
        1,
    )
    text = text.replace(
        PP_ANCHORS["outType"],
        PP_ANCHORS["outType"] + '\n  if (node.type === "previewImageNode") return "image";',
        1,
    )
    return text


# --------------------------------------------------------------------------- #
# Param slider patch - range slider for float params (min + max) + typing fix
# --------------------------------------------------------------------------- #

# Stock FloatInput re-synced its text from the committed number whenever the
# prop lagged a keystroke, so "0." was rewritten to "0" mid-typing. Only reset
# when the incoming value differs from BOTH what we last emitted and what the
# text currently reads.
SLIDER_SYNC_OLD = (
    '  const [text, setText] = reactExports.useState(String(value));\n'
    '  const prevValue = reactExports.useRef(value);\n'
    '  if (prevValue.current !== value && parseFloat(text.replace(",", ".")) !== value) {\n'
    '    prevValue.current = value;\n'
    '    setText(String(value));\n'
    '  }'
)

SLIDER_SYNC_NEW = (
    f'  /* {SLIDER_SENTINEL} */\n'
    '  const [text, setText] = reactExports.useState(String(value));\n'
    '  const prevValue = reactExports.useRef(value);\n'
    '  const parsedText = parseFloat(text.replace(",", "."));\n'
    '  if (prevValue.current !== value && value !== parsedText) {\n'
    '    prevValue.current = value;\n'
    '    setText(String(value));\n'
    '  }'
)

SLIDER_BRANCH_OLD = (
    '  if (param.type === "float") {\n'
    '    return /* @__PURE__ */ jsxRuntimeExports.jsx(FloatInput, '
    '{ value, onChange: (v) => onChange(v), className: inputCls });\n'
    '  }'
)

SLIDER_BRANCH_NEW = (
    '  if (param.type === "float") {\n'
    '    const typedNum = typeof value === "number" ? value : parseFloat(value);\n'
    '    const curNum = isNaN(typedNum) ? (typeof param.default === "number" ? param.default : 0) : typedNum;\n'
    '    if (typeof param.min === "number" && typeof param.max === "number") {\n'
    '      const sliderStep = typeof param.step === "number" && param.step > 0 ? param.step : (param.max - param.min) / 100;\n'
    '      return /* @__PURE__ */ jsxRuntimeExports.jsxs("div", { className: "flex items-center gap-1.5 w-full", children: [\n'
    '        /* @__PURE__ */ jsxRuntimeExports.jsx("input", {\n'
    '          type: "range", min: param.min, max: param.max, step: sliderStep,\n'
    '          value: Math.min(param.max, Math.max(param.min, curNum)),\n'
    '          onChange: (e) => { const n = parseFloat(e.target.value); if (!isNaN(n)) onChange(n); },\n'
    '          style: { accentColor: "#38bdf8", cursor: "pointer" },\n'
    '          className: "nodrag flex-1"\n'
    '        }),\n'
    '        /* @__PURE__ */ jsxRuntimeExports.jsx(FloatInput, {\n'
    '          value, onChange: (v) => onChange(v),\n'
    '          className: inputCls.replace("w-full", "w-16 shrink-0 text-center")\n'
    '        })\n'
    '      ] });\n'
    '    }\n'
    '    return /* @__PURE__ */ jsxRuntimeExports.jsx(FloatInput, { value, onChange: (v) => onChange(v), className: inputCls });\n'
    '  }'
)

SLIDER_ANCHORS = {
    "floatSync": SLIDER_SYNC_OLD,
    "floatBranch": SLIDER_BRANCH_OLD,
}


def apply_slider(text, bundle_name):
    if SLIDER_SENTINEL in text:
        raise PatchError(f"{bundle_name} slider already patched (sentinel present)")
    for key, anchor in SLIDER_ANCHORS.items():
        _once(text, anchor, f"{bundle_name}:{key}")
    text = text.replace(SLIDER_SYNC_OLD, SLIDER_SYNC_NEW, 1)
    text = text.replace(SLIDER_BRANCH_OLD, SLIDER_BRANCH_NEW, 1)
    return text


def _count_problems(text, anchors, label, problems):
    for key, anchor in anchors.items():
        n = text.count(anchor)
        if n != 1:
            problems.append(f"{label} anchor {key}: found {n}x (need 1x)")


def gate(tree_dir):
    """Return (wp_rel, pp_rel, gp_rel, needs_preview, needs_slider) or raise.

    Features are independent: an app that already carries the Preview patch can
    still receive the slider patch (and vice versa), so anchors for a feature
    are only required when that feature is not applied yet.
    """
    try:
        wp, pp, gp = find_bundles(tree_dir)
    except PatchError:
        raise
    tree = Path(tree_dir)
    gtext = (tree / wp).read_text(encoding="utf-8", errors="replace")
    ptext = (tree / pp).read_text(encoding="utf-8", errors="replace")
    gp_text = (tree / gp).read_text(encoding="utf-8", errors="replace")

    needs_preview = SENTINEL not in gtext
    needs_slider = SLIDER_SENTINEL not in gtext
    needs_slider_gp = SLIDER_SENTINEL not in gp_text

    problems = []
    if needs_preview:
        _count_problems(gtext, WP_ANCHORS, "WP", problems)
        _count_problems(ptext, PP_ANCHORS, "PP", problems)
    if needs_slider:
        _count_problems(gtext, SLIDER_ANCHORS, "WP-slider", problems)
    if needs_slider_gp:
        _count_problems(gp_text, SLIDER_ANCHORS, "GP-slider", problems)
    if not (needs_preview or needs_slider or needs_slider_gp):
        raise PatchError("already patched (all sentinels present) - restore first")
    if problems:
        raise PatchError("; ".join(problems))
    return wp, pp, gp, needs_preview, needs_slider


def apply_patch(tree_dir):
    wp, pp, gp, needs_preview, needs_slider = gate(tree_dir)
    tree = Path(tree_dir)

    gtext = (tree / wp).read_text(encoding="utf-8", errors="replace")
    if needs_preview:
        gtext = apply_workflows(gtext)
    if needs_slider:
        gtext = apply_slider(gtext, "WorkflowsPage")
    (tree / wp).write_text(gtext, encoding="utf-8", newline="")

    if needs_preview:
        (tree / pp).write_text(
            apply_parampicker((tree / pp).read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8", newline="",
        )

    gp_text = (tree / gp).read_text(encoding="utf-8", errors="replace")
    if SLIDER_SENTINEL not in gp_text:
        gp_text = apply_slider(gp_text, "GeneratePage")
    (tree / gp).write_text(gp_text, encoding="utf-8", newline="")

    final = (tree / wp).read_text(encoding="utf-8", errors="replace")
    if needs_preview and SENTINEL not in final:
        raise PatchError("preview patch applied but sentinel missing - aborting")
    if needs_slider and SLIDER_SENTINEL not in final:
        raise PatchError("slider patch applied but sentinel missing - aborting")
    if SLIDER_SENTINEL not in (tree / gp).read_text(encoding="utf-8", errors="replace"):
        raise PatchError("slider patch applied to GeneratePage but sentinel missing - aborting")
    return wp, pp, gp


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _esm_check(path):
    node = shutil.which("node")
    if not node:
        return "node not found - syntax check skipped"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "check.mjs"
        shutil.copyfile(path, target)
        runner = (
            "import(process.argv[1]).then("
            "()=>console.log('EXEC'),"
            "e=>console.log((e instanceof SyntaxError?'SYNTAX:':'OTHER:')"
            "+String((e&&e.message)||e).slice(0,200)))"
        )
        r = subprocess.run(
            [node, "--input-type=module", "-e", runner, target.as_uri()],
            capture_output=True, text=True,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if "SYNTAX:" in out:
            raise PatchError(f"ESM parse failed for {path}:\n{out[-2000:]}")
    return "ESM parse clean"


def _record_path(app_asar):
    return Path(str(app_asar) + RECORD_SUFFIX)


def _backup_path(app_asar):
    return Path(str(app_asar) + BACKUP_SUFFIX)


def patch_state(app_asar):
    """Which features are present in the app bundle: preview + slider."""
    import asar_min as A
    with tempfile.TemporaryDirectory() as tmp:
        A.extract(str(app_asar), tmp)
        tree = Path(tmp)

        def _read(glob):
            hits = sorted(tree.glob(glob))
            if not hits:
                return ""
            return hits[0].read_text(encoding="utf-8", errors="replace")

        wp_text = _read(WP_GLOB)
        gp_text = _read(GP_GLOB)
        return {
            "preview": bool(wp_text) and SENTINEL in wp_text,
            "slider": bool(wp_text) and SLIDER_SENTINEL in wp_text,
            "slider_generate": bool(gp_text) and SLIDER_SENTINEL in gp_text,
        }


def is_patched(app_asar):
    """True if the Preview node patch is present in the WorkflowsPage bundle."""
    return patch_state(app_asar)["preview"]


def needs_patch(app_asar):
    """True if any feature (Preview node, param sliders) is not yet applied."""
    state = patch_state(app_asar)
    return not (state["preview"] and state["slider"] and state["slider_generate"])


def stage(app_asar, work_dir):
    import asar_min as A
    app_asar = Path(app_asar)
    work_dir = Path(work_dir)
    tree = work_dir / "tree"
    if tree.exists():
        shutil.rmtree(tree)
    work_dir.mkdir(parents=True, exist_ok=True)
    header, _ = A.read_header(str(app_asar))
    A.extract(str(app_asar), str(tree))
    wp, pp, gp = apply_patch(tree)
    checks = [_esm_check(tree / wp), _esm_check(tree / pp), _esm_check(tree / gp)]
    staged = work_dir / "app.patched.asar"
    A.repack(header, str(tree), str(staged))
    return staged, checks


def find_app_asar():
    try:
        ppid = os.getppid()
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {ppid} -ErrorAction SilentlyContinue).Path"],
            capture_output=True, text=True, timeout=30)
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.lower().endswith("modly.exe"):
                cand = Path(line).parent / "resources" / "app.asar"
                if cand.is_file():
                    return str(cand)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    for fallback in (r"C:\Program Files\Modly\resources\app.asar",):
        if Path(fallback).is_file():
            return fallback
    return None


def modly_exe_for_pid(pid):
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).Path"],
            capture_output=True, text=True, timeout=30)
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.lower().endswith("modly.exe"):
                return line
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def modly_pid_and_exe():
    """Return (pid, exe_path) of the running Modly, or (None, None)."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process -Name Modly -ErrorAction SilentlyContinue | "
             "Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True, text=True, timeout=30)
        txt = (r.stdout or "").strip()
        if txt.isdigit():
            pid = int(txt)
            return pid, modly_exe_for_pid(pid)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None, None


def _pid_alive(pid):
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | "
             f"Select-Object -First 1 -ExpandProperty Id"],
            capture_output=True, text=True, timeout=30)
        return bool((r.stdout or "").strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def _swap_verified(staged, app_asar):
    staged_hash = sha256(staged)
    shutil.copyfile(Path(staged), Path(app_asar))
    if sha256(app_asar) != staged_hash:
        raise PatchError("swap verify failed")
    return staged_hash


def cmd_backup(app_asar):
    app_asar = Path(app_asar)
    bak = _backup_path(app_asar)
    rec = _record_path(app_asar)
    stock_hash = sha256(app_asar)
    shutil.copyfile(app_asar, bak)
    if sha256(bak) != stock_hash:
        raise PatchError("backup hash mismatch - aborting")
    rec.write_text(json.dumps({"stock_sha256": stock_hash}, indent=1), encoding="utf-8")
    return {"backup": str(bak), "stock_sha256": stock_hash}


def cmd_finalize(app_asar, staged):
    app_asar = Path(app_asar)
    rec = _record_path(app_asar)
    record = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else None
    cur = sha256(app_asar)
    staged_hash = sha256(Path(staged))
    if cur == staged_hash:
        return {"already_patched": str(app_asar)}
    # Re-baseline when the app.asar is a version we have not snapshotted yet
    # (e.g. Modly auto-updated after the addon was last applied).
    known = {record.get("stock_sha256"), record.get("patched_sha256")} if record else set()
    if cur not in known:
        cmd_backup(app_asar)
        record = json.loads(rec.read_text(encoding="utf-8"))
    patched_hash = _swap_verified(staged, app_asar)
    record["patched_sha256"] = patched_hash
    rec.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return {"patched": str(app_asar), "patched_sha256": patched_hash}


def cmd_restore(app_asar):
    app_asar = Path(app_asar)
    bak = _backup_path(app_asar)
    rec = _record_path(app_asar)
    if not bak.is_file() or not rec.is_file():
        return {"nothing": "no backup found"}
    record = json.loads(rec.read_text(encoding="utf-8"))
    cur = sha256(app_asar)
    if cur == record.get("stock_sha256"):
        return {"already_stock": str(app_asar)}
    if cur != record.get("patched_sha256"):
        return {"stale": "app.asar changed version since the addon was applied; "
                         "removing stale snapshot only"}
    if sha256(bak) != record.get("stock_sha256"):
        raise PatchError("backup differs from record - refusing to restore")
    _swap_verified(bak, app_asar)
    return {"restored": str(app_asar)}


def run_elevated_detached(exe, args, log_path):
    log_path = Path(log_path)
    bat = Path(str(log_path) + ".run.bat")
    cmd = subprocess.list2cmdline([str(exe)] + [str(a) for a in args])
    bat.write_text(f'@{cmd} > "{log_path}" 2>&1\n', encoding="utf-8")
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Start-Process", "-FilePath", str(bat), "-Verb", "RunAs"],
        capture_output=True, text=True, timeout=120)


def run_elevated_wait(exe, args, log_path):
    log_path = Path(log_path)
    bat = Path(str(log_path) + ".run.bat")
    cmd = subprocess.list2cmdline([str(exe)] + [str(a) for a in args])
    bat.write_text(f'@{cmd} > "{log_path}" 2>&1\n', encoding="utf-8")
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Start-Process", "-FilePath", str(bat), "-Verb", "RunAs", "-Wait"],
        capture_output=True, text=True, timeout=3600)
    return r.returncode


def cmd_apply_wait(app_asar, staged, pid, exe, log_path):
    """Elevated: wait for Modly to exit, swap the asar, relaunch Modly."""
    log = Path(log_path)
    deadline = time.time() + 2 * 60 * 60
    while time.time() < deadline and _pid_alive(int(pid)):
        time.sleep(2)
    if _pid_alive(int(pid)):
        log.write_text("timed out waiting for Modly to close\n", encoding="utf-8")
        return 1
    time.sleep(1.5)  # let file handles release
    result = cmd_finalize(app_asar, staged)
    if "patched" in result or "already_patched" in result:
        shutil.rmtree(Path(staged).parent, ignore_errors=True)
    if exe and Path(exe).is_file():
        try:
            subprocess.Popen(["explorer.exe", exe])
        except OSError:
            pass
    log.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return 0


def cmd_status(app_asar):
    app_asar = Path(app_asar)
    out = {"app": str(app_asar), "sha256": sha256(app_asar)}
    try:
        state = patch_state(app_asar)
        out["features"] = state
        out["needs_patch"] = not (
            state["preview"] and state["slider"] and state["slider_generate"]
        )
        out["patched"] = state["preview"]
    except Exception as exc:  # noqa: BLE001
        out["patched"] = f"unknown ({exc})"
    rec = _record_path(app_asar)
    if rec.is_file():
        try:
            out["record"] = json.loads(rec.read_text(encoding="utf-8"))
        except ValueError:
            out["record"] = "unreadable"
    out["backup_present"] = _backup_path(app_asar).is_file()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Modly image-preview addon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage"); p.add_argument("--app", required=True); p.add_argument("--work", required=True)
    p = sub.add_parser("status"); p.add_argument("--app", required=True)
    p = sub.add_parser("backup"); p.add_argument("--app", required=True)
    p = sub.add_parser("finalize"); p.add_argument("--app", required=True); p.add_argument("--staged", required=True)
    p = sub.add_parser("restore"); p.add_argument("--app", required=True)
    p = sub.add_parser("apply-wait"); p.add_argument("--app", required=True); p.add_argument("--staged", required=True)
    p.add_argument("--pid", required=True); p.add_argument("--exe", default=""); p.add_argument("--log", required=True)
    sub.add_parser("findapp")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "stage":
            staged, checks = stage(args.app, args.work)
            print(json.dumps({"staged": str(staged), "sha256": sha256(staged), "checks": checks}))
        elif args.cmd == "status":
            print(json.dumps(cmd_status(args.app)))
        elif args.cmd == "backup":
            print(json.dumps(cmd_backup(args.app)))
        elif args.cmd == "finalize":
            print(json.dumps(cmd_finalize(args.app, args.staged)))
        elif args.cmd == "restore":
            print(json.dumps(cmd_restore(args.app)))
        elif args.cmd == "apply-wait":
            return cmd_apply_wait(args.app, args.staged, args.pid, args.exe, args.log)
        elif args.cmd == "findapp":
            found = find_app_asar()
            print(json.dumps({"app_asar": found}))
            return 0 if found else 2
    except PatchError as exc:
        print(json.dumps({"error": str(exc)[:800]}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
