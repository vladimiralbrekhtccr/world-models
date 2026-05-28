# Prompt — Ryōtenkai WACV 2027 tracker (paste this into Claude)

Paste the **entire block below** into a fresh Claude conversation. Claude
will respond with a self-contained `tracker.html` file you can save and
open in any browser. Tasks are saved in `localStorage`, so progress
persists across visits.

If your Claude has Artifacts / Canvas, it'll render the page inline
right in the chat — no save needed.

---

```
Build me a single self-contained HTML file (no build step, no external deps,
inline CSS + vanilla JS) that I can open in a browser as my personal WACV 2027
paper-submission tracker.

REQUIREMENTS
- File name: tracker.html
- Title (centered, large): "Ryōtenkai — WACV 2027 Tracker"
- Subtitle: "Personal stage plan. WACV registration Aug 20 · submission Aug 27 · my internal target Jul 20."
- Header strip showing 3 live countdowns side-by-side:
    · "Internal sub Jul 20, 2026" → days remaining
    · "Registration Aug 20, 2026" → days remaining
    · "Final sub Aug 27, 2026 (AoE)" → days remaining
  (countdowns refresh on page load, computed from today)
- Sticky thin progress bar at top showing percentage of tasks checked
- One collapsible section per stage (default expanded), with stage header
  and a stage-level mini progress bar
- Each task is one row containing:
    · checkbox  (persists in localStorage)
    · task title (bold)
    · short description (one line, dim text)
    · target-date pill on the right (small, monospaced)
    · expandable notes textarea per task (also persists to localStorage)
- Strikethrough + 0.5 opacity on a row when its checkbox is checked
- "Reset all" button bottom-right (with confirm dialog)
- "Export progress as JSON" button bottom-right (downloads a snapshot)
- Theme: warm-dark (background #14110d, text #e8ddc8, accent #e07a50, panels
  #1f1a14, borders #2e2820). Match the aesthetic of kitan-a.com/3dgs/learn/.
- Typography: sans-serif body (system stack), monospace for dates and progress
  numbers. Generous spacing, comfortable to read for an hour.
- All localStorage keys prefixed with "rt-wacv-" so they don't collide with
  anything else.
- Mobile-friendly (collapses gracefully on iPad/phone).

STAGES AND TASKS (use exactly this content, in this order)

— Stage 0 — Kickoff (May 27 – Jun 3)
  · Read WACV 2027 Call-for-Papers + Author Guidelines end to end.
    target: Jun 1
  · Set up paper repo (Overleaf or local LaTeX) + Git tracking.
    target: Jun 2
  · Download WACV / IEEE conference template; build empty paper skeleton
    with section headers.
    target: Jun 2
  · Install reference manager (Zotero) + create empty bibliography file.
    target: Jun 3
  · Block 2 hours/day in calendar for research, 7 days/week.
    target: Jun 3

— Stage 1 — Foundations (May 27 – Jun 10, in parallel with Stage 0)
  · Re-read 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) cover to
    cover. Take physical notes on §3 (representation) and §4 (renderer).
    target: Jun 4
  · Read NeRF (Mildenhall et al., ECCV 2020). Note the implicit/explicit
    contrast.
    target: Jun 6
  · Read Mip-NeRF 360 (Barron et al., CVPR 2022). Note the unbounded-scene
    contraction.
    target: Jun 7
  · Read DUSt3R + MASt3R + VGGT (the feed-forward camera+depth lineage).
    target: Jun 9
  · Read SuGaR (Guédon, CVPR 2024) and 2D Gaussian Splatting (Huang, SIGGRAPH
    2024) — direct splat-to-mesh competitors.
    target: Jun 10

— Stage 2 — Literature deep dive + angle lock (Jun 3 – Jun 17)
  · Read 8–10 WACV 2026 splat papers (ForestSplats, 3D Superquadric Splatting,
    DARB-Splatting, MagicDrive3D, Gaussian Swaying, GDoFS, STRinGS, RapidMV).
    target: Jun 12
  · Read 5–8 recent generative scene-synthesis papers (Wonder3D, WonderWorld,
    CAT3D, Bolt3D, Marble, HunyuanWorld 2.0).
    target: Jun 14
  · Pick paper angle: A1 (splat→mesh for web), A2 (comparative benchmark), or
    A3 (end-to-end Ryōtenkai system).
    target: Jun 15
  · Write 1-page problem statement + paper outline (5 sections, 2 figures
    planned).
    target: Jun 16
  · Lock paper title + tentative abstract (200 words).
    target: Jun 17

— Stage 3 — Baselines + evaluation harness (Jun 17 – Jul 1)
  · Pick benchmark scenes (Mip-NeRF 360 = 7 scenes; optional Tanks & Temples).
    target: Jun 19
  · Set up evaluation pipeline: PSNR, SSIM, LPIPS, Chamfer (if mesh), file
    size, load time. Document each metric's implementation source.
    target: Jun 23
  · Reproduce baseline #1 on 1 scene. Verify your numbers within ±0.5 of
    published numbers.
    target: Jun 26
  · Reproduce baselines #2 and #3 on 1 scene.
    target: Jul 1

— Stage 4 — Method + experiments (Jul 1 – Jul 8)
  · Implement the core contribution on top of baselines. Get it working on
    1 scene.
    target: Jul 3
  · Run method on full benchmark (7 Mip-NeRF 360 scenes).
    target: Jul 6
  · Collect raw numbers into a single CSV. Don't analyze yet — just collect.
    target: Jul 8

— Stage 5 — Ablations + figures (Jul 8 – Jul 15)
  · Ablate each pipeline component (3–5 ablations).
    target: Jul 11
  · Build qualitative comparison figure (your method vs 3 baselines on 2
    scenes).
    target: Jul 13
  · Build main quantitative results table (LaTeX-ready).
    target: Jul 14
  · Build teaser figure for paper page 1 (one striking image + one-line
    caption).
    target: Jul 15

— Stage 6 — First full draft (Jul 8 – Jul 20)
  · Write Introduction (1.5 pages, includes the JJK "domain expansion"
    metaphor opener).
    target: Jul 12
  · Write Related Work (1 page, 25–30 citations).
    target: Jul 14
  · Write Method (2 pages, with method overview figure).
    target: Jul 16
  · Write Experiments + Results (2 pages + tables).
    target: Jul 18
  · Write Conclusion (0.25 page) + Limitations + Broader Impact.
    target: Jul 19
  · INTERNAL DEADLINE — full first draft circulated to readers.
    target: Jul 20

— Stage 7 — Internal review + polish (Jul 20 – Aug 19)
  · Get 2–3 trusted readers to review the draft. Collect comments.
    target: Jul 27
  · Iterate on figures based on feedback (revise teaser, methods figure,
    qual comparison).
    target: Aug 3
  · Run any final ablations that reviewers flagged as missing.
    target: Aug 10
  · Polish abstract to its final form.
    target: Aug 14
  · Final proofread (read aloud, every sentence).
    target: Aug 17
  · Prepare supplementary material (additional results, video, code link if
    public).
    target: Aug 19

— Stage 8 — Registration + submission (Aug 19 – Aug 27)
  · WACV REGISTRATION DEADLINE — title, abstract, authors, keywords locked
    in the submission system.
    target: Aug 20
  · Run paper through plagiarism / overlap checker.
    target: Aug 23
  · Format check: page count, margins, font, blind-review compliance
    (no author info anywhere).
    target: Aug 25
  · Build final submission ZIP (paper PDF + supplementary).
    target: Aug 26
  · WACV PAPER SUBMISSION DEADLINE — upload PDF + supp by 23:59 AoE.
    target: Aug 27

OUTPUT
Write the complete tracker.html as one code block, ready to save and open.
Include sensible defaults for all interactivity (smooth scrolling, focus
states, hover states on buttons). Keep total file under 25 KB.
```

---

## Notes when reading later

- **Internal deadline = Jul 20** (already in your Google Sheet). Keep the
  Aug 20 / Aug 27 row of dates as the HARD line that cannot move.
- **The angle decision lands on Jun 15.** That's the highest-leverage moment
  in the whole plan — if Jun 15 slips, every later stage compresses.
  Don't slip Jun 15.
- **Stage 0 + Stage 1 run in parallel** in Week 1. Don't wait until the repo
  is set up to start reading; reading IS the work of Week 1.
- **CSV version** of the same task list is in `tracker.csv` (same folder) —
  drop it into Google Sheets / Numbers / Airtable if you'd rather track
  there than in a self-contained HTML page.
