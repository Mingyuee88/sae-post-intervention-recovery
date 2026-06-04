# SAE Project Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a light academic project page for "SAE Interventions are Unreliable" on the repository's `gh-pages` branch.

**Architecture:** The site is a static GitHub Pages artifact with one `index.html`, one stylesheet, local paper/result images, and a small Node-based content check. The page follows a clean academic project-page layout inspired by DeepCache and the Academic Project Page Template.

**Tech Stack:** Static HTML, CSS, local image/PDF assets, Node.js for verification.

---

### Task 1: Static Page Checks

**Files:**
- Create: `tests/page-check.mjs`

- [x] **Step 1: Write the failing test**
  Create a Node script that verifies the expected project-page files, required sections, required buttons, footer language, and absence of the Responsible Release section.

- [x] **Step 2: Run test to verify it fails**
  Run: `node tests/page-check.mjs`
  Expected: FAIL because `index.html` does not exist yet.

### Task 2: Page Implementation

**Files:**
- Create: `index.html`
- Create: `static/css/index.css`
- Create: `static/images/*`
- Create: `static/pdf/SAE_preprint.pdf`

- [ ] **Step 1: Add HTML structure**
  Implement hero, abstract, core idea, key results, attribution, citation, and template footer.

- [ ] **Step 2: Add light academic styling**
  Use a white/light gray surface, centered paper title, rounded buttons, readable sections, and responsive image grids.

- [ ] **Step 3: Add assets**
  Copy selected paper/result figures and the preprint PDF into static asset folders.

- [ ] **Step 4: Run verification**
  Run: `node tests/page-check.mjs`
  Expected: PASS.

### Task 3: Deploy

**Files:**
- Modify: Git branch `gh-pages`

- [ ] **Step 1: Commit page artifacts**
  Run: `git add . && git commit -m "Add academic project page"`.

- [ ] **Step 2: Push to GitHub Pages branch**
  Run: `git push -u origin HEAD:gh-pages`.

- [ ] **Step 3: Verify deployment target**
  Confirm the expected URL is `https://mingyuee88.github.io/sae-post-intervention-recovery/`.
