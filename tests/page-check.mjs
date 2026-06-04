import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const requiredFiles = [
  "index.html",
  "static/css/index.css",
  "static/images/defense_large.png",
  "static/images/tpp_tradeoff_scatter.png",
  "static/images/unlearning_oabd_comparison.png",
  "static/images/ioi.png",
  "static/images/refusal_summary.png",
  "static/images/refusal_attribution.png",
  "static/pdf/SAE_preprint.pdf",
];

const failures = [];

for (const file of requiredFiles) {
  if (!existsSync(join(root, file))) {
    failures.push(`Missing required file: ${file}`);
  }
}

if (existsSync(join(root, "index.html"))) {
  const html = readFileSync(join(root, "index.html"), "utf8");
  const requiredText = [
    "SAE Interventions are Unreliable",
    "Post-Intervention Recovery of Suppressed Behavior",
    "Mingyue Cui",
    "Linghui Shen",
    "Xingyi Yang",
    "Abstract",
    "Core Idea",
    "Key Results",
    "Recovery-Path Attribution",
    "BibTeX",
    "Academic Project Page Template",
    "Creative Commons Attribution-ShareAlike 4.0 International License",
  ];

  for (const text of requiredText) {
    if (!html.includes(text)) {
      failures.push(`Missing required page text: ${text}`);
    }
  }

  const forbiddenText = ["Responsible Release"];
  for (const text of forbiddenText) {
    if (html.includes(text)) {
      failures.push(`Forbidden page text present: ${text}`);
    }
  }

  const requiredLinks = [
    "static/pdf/SAE_preprint.pdf",
    "https://github.com/Mingyuee88/sae-post-intervention-recovery",
    "https://arxiv.org/abs/xxxx.xxxxx",
  ];

  for (const link of requiredLinks) {
    if (!html.includes(link)) {
      failures.push(`Missing required link: ${link}`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}

console.log("Page check passed.");
