// Sevrage IA, démonstration web : formulaire, appel au modèle et affichage des résultats.
// Rien n'est conservé : les valeurs saisies ne quittent la page que pour /api/predict.
"use strict";

const CONFIG_URL = "/config.json";
const PREDICT_URL = "/api/predict";
const SVG_NS = "http://www.w3.org/2000/svg";
const STATUS_ICONS = { met: "✓", moderate: "!", low: "✕", info: "i" };
const numberFormat = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

const state = { config: null, inputs: new Map() };

function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(options)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "className") node.className = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of [].concat(children)) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

function svgEl(tag, attributes = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  if (text !== null) node.textContent = text;
  return node;
}

// ------------------------------------------------------------------ chargement

async function init() {
  try {
    const response = await fetch(CONFIG_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    state.config = await response.json();
  } catch (error) {
    document.getElementById("intro").textContent = "";
    showError("La page n'a pas pu se charger. Réessayez dans un instant.");
    document.getElementById("submit").disabled = true;
    return;
  }
  const config = state.config;
  document.getElementById("intro").textContent = config.intro;
  renderNotices(config);
  renderForm(config);
  renderGuide(config);
  renderPerformance(config);
  document.getElementById("example").addEventListener("click", () => fill(config.example));
  document.getElementById("clear").addEventListener("click", () => fill({}));
  document.getElementById("form").addEventListener("submit", onSubmit);
}

function renderNotices(config) {
  const box = document.getElementById("notices");
  const [publicWarning, ...others] = config.warnings;
  box.append(
    el("div", { className: "notice notice-warning", role: "note" }, [
      el("strong", { text: "À lire avant de commencer" }),
      el("p", { text: publicWarning }),
      ...others.map((text) => el("p", { text })),
    ]),
  );
  const synthetic = config.model.synthetic;
  box.append(
    el("div", { className: synthetic ? "notice notice-synthetic" : "notice notice-info", role: "note" }, [
      el("strong", { text: synthetic ? "Modèle de démonstration" : "D'où vient ce modèle ?" }),
      el("p", { text: config.model.note }),
    ]),
  );
}

function renderForm(config) {
  const container = document.getElementById("groups");
  for (const group of config.groups) {
    const fields = el("div", { className: "fields" });
    for (const field of group.fields) fields.append(renderField(field));
    container.append(el("fieldset", { className: "group" }, [el("legend", { text: group.label }), fields]));
  }
}

function renderField(field) {
  const id = `champ-${field.name}`;
  const hintId = field.hint ? `${id}-aide` : null;
  let control;
  if (field.kind === "numeric") {
    control = el("input", {
      id,
      type: "number",
      inputmode: "decimal",
      step: "any",
      min: field.min,
      max: field.max,
      placeholder: "Je ne sais pas",
      "aria-describedby": hintId,
    });
  } else {
    control = el("select", { id });
    field.choices.forEach((choice, index) => control.append(el("option", { value: String(index), text: choice.label })));
  }
  state.inputs.set(field.name, { field, control });
  return el("div", { className: "field" }, [
    el("label", { for: id, text: field.label }),
    control,
    hintId ? el("small", { id: hintId, className: "hint", text: field.hint }) : null,
  ]);
}

function fill(values) {
  for (const [name, { field, control }] of state.inputs) {
    const value = values[name];
    if (field.kind === "numeric") {
      control.value = value === null || value === undefined ? "" : String(value);
    } else {
      const index = field.choices.findIndex((choice) => choice.value !== null && choice.value === value);
      control.value = String(index >= 0 ? index : 0);
    }
  }
}

function collect() {
  const values = {};
  for (const [name, { field, control }] of state.inputs) {
    if (field.kind === "numeric") {
      const number = control.valueAsNumber;
      values[name] = Number.isFinite(number) ? number : null;
    } else {
      const choice = field.choices[Number(control.value)];
      values[name] = choice ? choice.value : null;
    }
  }
  return values;
}

function renderGuide(config) {
  const body = document.getElementById("guide-body");
  for (const item of config.guide) {
    body.append(el("div", {}, [el("h4", { text: item.title }), el("p", { text: item.text })]));
  }
}

function renderPerformance(config) {
  const rows = config.model.performance;
  if (!rows.length) return;
  const table = el("table", {}, [
    el("thead", {}, el("tr", {}, ["Question", "Mesure", "Valeur", "Comment la lire"].map((text) => el("th", { scope: "col", text })))),
    el("tbody", {}, rows.map((row) => el("tr", {}, [row.output, row.measure, row.value, row.hint].map((text) => el("td", { text }))))),
  ]);
  const body = document.getElementById("perf-body");
  body.append(
    el("p", { className: "hint", text: "Mesuré sur des patients que le modèle n'avait jamais vus pendant son apprentissage." }),
    el("div", { className: "table-wrap" }, table),
  );
  if (config.model.protocol) body.append(el("p", { className: "hint", text: config.model.protocol }));
  document.getElementById("perf").hidden = false;
}

// ------------------------------------------------------------------ prédiction

async function onSubmit(event) {
  event.preventDefault();
  const button = document.getElementById("submit");
  const results = document.getElementById("results");
  const label = button.textContent;
  button.disabled = true;
  button.textContent = "Le modèle réfléchit…";
  results.setAttribute("aria-busy", "true");
  try {
    const response = await fetch(PREDICT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: collect() }),
    });
    const data = await response.json().catch(() => ({ error: "Réponse illisible du serveur." }));
    if (response.ok) renderResults(data);
    else showError(data.error || "Le calcul n'a pas abouti.", data.details || []);
  } catch (error) {
    showError("Impossible de joindre le modèle. Vérifiez votre connexion puis réessayez.");
  } finally {
    button.disabled = false;
    button.textContent = label;
    results.removeAttribute("aria-busy");
    if (window.matchMedia("(max-width: 900px)").matches) results.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function showError(message, details = []) {
  const root = document.getElementById("results");
  root.replaceChildren(
    el("div", { className: "error-box", role: "alert" }, [
      el("strong", { text: message }),
      details.length ? el("ul", {}, details.map((text) => el("li", { text }))) : null,
    ]),
  );
}

function renderResults(data) {
  const root = document.getElementById("results");
  const plural = data.filled > 1 ? "s" : "";
  const grid = el("div", { className: "outputs" }, data.outputs.map((output) => (output.kind === "class" ? classCard(output) : numberCard(output))));
  const blocks = [
    el("p", { className: "filled", text: `Calculé avec ${data.filled} information${plural} sur ${data.total}.` }),
    summaryCard(data.summary),
    data.alerts.length ? alertsCard(data.alerts) : null,
    el("h3", { className: "section-title", text: "Le détail, question par question" }),
    grid,
    data.weights.length ? weightsCard(data.weights) : null,
    downloadButton(data.raw),
  ];
  root.replaceChildren(...blocks.filter(Boolean));
}

function trustChip(trust) {
  return el("span", { className: `trust trust-${trust.level}`, text: trust.label });
}

function whyBlock(items, title) {
  if (!items.length) return null;
  return el("details", { className: "why" }, [
    el("summary", { text: title }),
    el("ul", {}, items.map((item) => el("li", { className: `status-${item.status}` }, [
      el("span", { className: "icon", "aria-hidden": "true", text: STATUS_ICONS[item.status] || "•" }),
      el("span", { text: item.text }),
    ]))),
  ]);
}

function summaryCard(summary) {
  return el("article", { className: "card summary" }, [
    el("div", { className: "card-head" }, [el("h3", { text: "En résumé" }), trustChip(summary.trust)]),
    el("p", { className: "headline", text: summary.headline }),
    summary.figures.length
      ? el("div", { className: "figures" }, summary.figures.map((figure) => el("div", { className: "figure", title: figure.help }, [
        el("span", { className: "figure-label", text: figure.label }),
        el("strong", { className: "figure-value", text: figure.value }),
      ])))
      : null,
    summary.note ? el("p", { className: "hint", text: summary.note }) : null,
    el("p", { className: "trust-sentence", text: summary.trust.sentence }),
    whyBlock(summary.why, "Pourquoi cette confiance ?"),
  ]);
}

function alertsCard(alerts) {
  return el("article", { className: "card alerts", role: "note" }, alerts.map((group) => el("div", { className: `alert-${group.level}` }, [
    el("h4", { text: group.title }),
    el("ul", {}, group.items.map((text) => el("li", { text }))),
  ])));
}

function outputHead(output) {
  return [
    el("div", { className: "card-head" }, [el("h4", { text: output.title }), trustChip(output.trust)]),
    el("p", { className: "question", text: output.question }),
  ];
}

function outputFoot(output) {
  return [
    output.notes.length ? el("ul", { className: "notes" }, output.notes.map((text) => el("li", { text }))) : null,
    el("p", { className: "trust-sentence", text: output.trust.sentence }),
    whyBlock(output.why, "Pourquoi cette confiance ?"),
  ];
}

function classCard(output) {
  const bars = el("div", { className: "bars" }, output.rows.map((row) => {
    const fill = el("span", { className: "bar-fill" });
    fill.style.width = `${Math.max(0, Math.min(100, row.percent))}%`;
    return el("div", { className: row.answer ? "bar-row is-answer" : "bar-row" }, [
      el("span", { className: "bar-label" }, [row.label, row.answer ? el("span", { className: "bar-tag", text: output.tag }) : null]),
      el("span", { className: "bar-track", "aria-hidden": "true" }, fill),
      el("span", { className: "bar-value", text: `${row.percent} %` }),
    ]);
  }));
  return el("article", { className: "card output" }, [
    ...outputHead(output),
    el("p", { className: "answer" }, [`${output.lead} : `, el("strong", { text: output.answer })]),
    bars,
    el("p", { className: "possible", text: output.possible }),
    ...outputFoot(output),
  ]);
}

function numberCard(output) {
  return el("article", { className: "card output" }, [
    ...outputHead(output),
    el("p", { className: "answer" }, [
      "Environ ",
      el("strong", { text: output.answer }),
      output.answer_note ? el("span", { className: "answer-note", text: output.answer_note }) : null,
    ]),
    rangeChart(output),
    el("p", { className: "range-text", text: output.range }),
    ...outputFoot(output),
  ]);
}

function niceMax(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  const step = [1, 2, 2.5, 5, 10].find((s) => s * power >= value);
  return step * power;
}

function rangeChart(output) {
  const width = 300;
  const left = 14;
  const right = width - 14;
  const open = output.upper === null;
  const top = open ? output.point * 2 : output.upper;
  const max = niceMax(Math.max(top, output.point, output.lower) * 1.1);
  const x = (value) => left + (Math.max(0, Math.min(value, max)) / max) * (right - left);
  const unit = output.unit ? ` ${output.unit}` : "";
  const chart = svgEl("svg", { viewBox: "0 0 300 70", role: "img", "aria-label": output.range });
  const bandEnd = open ? right : x(output.upper);
  chart.append(
    svgEl("line", { x1: left, y1: 42, x2: right, y2: 42, class: "axis" }),
    svgEl("rect", { x: x(output.lower), y: 33, width: Math.max(4, bandEnd - x(output.lower)), height: 18, rx: 9, class: open ? "band-open" : "band" }),
    svgEl("circle", { cx: x(output.point), cy: 42, r: 7, class: "dot" }),
    svgEl("text", { x: x(output.point), y: 22, "text-anchor": "middle", class: "point-label" }, `${numberFormat.format(output.point)}${unit}`),
  );
  if (open) {
    chart.append(svgEl("text", { x: right, y: 66, "text-anchor": "end" }, "sans limite haute"));
  } else if (bandEnd - x(output.lower) < 70) {
    chart.append(svgEl("text", { x: (x(output.lower) + bandEnd) / 2, y: 66, "text-anchor": "middle" }, `${numberFormat.format(output.lower)} à ${numberFormat.format(output.upper)}`));
  } else {
    chart.append(
      svgEl("text", { x: x(output.lower), y: 66, "text-anchor": "start" }, numberFormat.format(output.lower)),
      svgEl("text", { x: bandEnd, y: 66, "text-anchor": "end" }, numberFormat.format(output.upper)),
    );
  }
  return el("div", { className: "range" }, chart);
}

function weightsCard(weights) {
  const panel = el("div", { id: "poids-panneau", role: "tabpanel" });
  const buttons = weights.map((weight, index) => {
    const button = el("button", { type: "button", role: "tab", "aria-selected": index === 0 ? "true" : "false", "aria-controls": "poids-panneau", text: weight.title });
    button.addEventListener("click", () => {
      buttons.forEach((other) => other.setAttribute("aria-selected", other === button ? "true" : "false"));
      drawWeights(panel, weight);
    });
    return button;
  });
  drawWeights(panel, weights[0]);
  return el("article", { className: "card weights" }, [
    el("h3", { text: "Ce qui a pesé dans la balance" }),
    el("p", { className: "hint", text: state.config.weights_hint }),
    el("div", { className: "tabs", role: "tablist", "aria-label": "Question" }, buttons),
    panel,
    el("div", { className: "legend" }, [
      el("span", {}, [el("span", { className: "swatch up", "aria-hidden": "true" }), "fait monter"]),
      el("span", {}, [el("span", { className: "swatch down", "aria-hidden": "true" }), "fait baisser"]),
    ]),
  ]);
}

function drawWeights(panel, weight) {
  const biggest = Math.max(...weight.items.map((item) => Math.abs(item.delta)), 1e-9);
  const rows = weight.items.map((item) => {
    const fill = el("span", { className: item.delta > 0 ? "w-fill w-up" : "w-fill w-down" });
    fill.style.width = `${(Math.abs(item.delta) / biggest) * 100}%`;
    const negative = el("span", { className: "w-neg" }, item.delta < 0 ? fill : null);
    const positive = el("span", {}, item.delta > 0 ? fill : null);
    return el("div", { className: "w-row" }, [
      el("span", { text: `${item.label} : ${item.value}` }),
      el("span", { className: "w-track", "aria-hidden": "true" }, [negative, positive]),
      el("span", { className: "w-text", text: item.text }),
    ]);
  });
  panel.replaceChildren(
    el("p", { className: "hint", text: weight.axis }),
    ...(rows.length ? rows : [el("p", { className: "hint", text: "Aucune information n'a pesé de façon notable." })]),
  );
}

function downloadButton(raw) {
  const button = el("button", { type: "button", className: "ghost", text: "Télécharger le résultat (JSON)" });
  button.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(raw, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = el("a", { href: url, download: "prediction.json" });
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });
  return button;
}

document.addEventListener("DOMContentLoaded", init);
