// Build the P2 thesis Word files from data/document_model.json (produced by build_document.py).
// Outputs: Thesis_Paper/01_Full_Thesis_P2_RAHC-LoRA.docx, chapters/*.docx, 00_README_Index.docx
"use strict";
const fs = require("fs");
const path = require("path");
const DOCX_PATH = process.env.DOCX_MODULE || "docx";
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, TabStopType, BorderStyle, PageNumber, NumberFormat, Footer, SectionType,
  TableOfContents, Bookmark, PageReference, PositionalTab, PositionalTabAlignment, PositionalTabRelativeTo,
  PositionalTabLeader, LevelFormat, VerticalAlign, PageBreak,
} = require(DOCX_PATH);

const ROOT = path.resolve(__dirname, "..");
const model = JSON.parse(fs.readFileSync(path.join(ROOT, "data", "document_model.json"), "utf8"));
const FONT = "Times New Roman";
const MONO = "Consolas";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1440;
const TEXT_W = PAGE_W - 2 * MARGIN; // 9026 DXA = 6.27 in

// ------------------------------------------------------------------ helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function bm(id) { return ("bm_" + id).replace(/[^A-Za-z0-9_]/g, "_").slice(0, 40); }

function textRuns(runs, base = {}) {
  return runs.map((r) => new TextRun({
    text: r.text, bold: r.bold || base.bold, italics: r.italic || base.italics,
    font: r.code ? MONO : (base.font || FONT), size: r.code ? 21 : base.size,
    subScript: r.sub, superScript: r.sup, color: base.color,
  }));
}

let numberingConfigs = [];
let numberedCounter = 0;
function newNumberedRef() {
  numberedCounter += 1;
  const ref = `num${numberedCounter}`;
  numberingConfigs.push({
    reference: ref,
    levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
  });
  return ref;
}

function body(runs, opts = {}) {
  return new Paragraph({
    children: textRuns(runs, opts.run || {}), alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { after: 140, line: 360 }, indent: opts.indent, keepNext: opts.keepNext,
  });
}

function captionPara(label, captionRuns, id) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 276 },
    children: [new Bookmark({ id: bm(id), children: [new TextRun({ text: `${label}: `, bold: true, font: FONT, size: 21 })] }),
      ...textRuns(captionRuns, { size: 21 })],
  });
}

function figureBlock(b) {
  const img = pngSize(b.path);
  const maxWIn = 6.2, maxHIn = 7.8;
  let wIn = maxWIn, hIn = (img.h / img.w) * wIn;
  if (hIn > maxHIn) { hIn = maxHIn; wIn = (img.w / img.h) * hIn; }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: img.data, transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
        altText: { title: b.label, description: b.caption_runs.map((r) => r.text).join(""), name: b.id } })] }),
    captionPara(b.label, b.caption_runs, b.id),
  ];
}

function equationBlock(b) {
  const img = pngSize(b.path);
  let wIn = img.w / 300, hIn = img.h / 300;
  const maxW = 5.5;
  if (wIn > maxW) { hIn = hIn * (maxW / wIn); wIn = maxW; }
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const numW = 1000, eqW = TEXT_W - 2 * numW;
  const cell = (w, children, align) => new TableCell({ width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    borders: { top: none, bottom: none, left: none, right: none }, margins: { top: 40, bottom: 40, left: 0, right: 0 },
    children: [new Paragraph({ alignment: align, spacing: { before: 0, after: 0 }, children })] });
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [numW, eqW, numW],
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: [new TableRow({ cantSplit: true, children: [
      cell(numW, [new TextRun({ text: "", font: FONT })], AlignmentType.LEFT),
      cell(eqW, [new ImageRun({ type: "png", data: img.data, transformation: { width: Math.round(wIn * 96), height: Math.round(hIn * 96) },
        altText: { title: `Equation ${b.number}`, description: b.id, name: b.id } })], AlignmentType.CENTER),
      cell(numW, [new TextRun({ text: b.number, font: FONT, size: 24 })], AlignmentType.RIGHT)] })] }),
    new Paragraph({ spacing: { before: 0, after: 80 }, children: [] })];
}

const thin = { style: BorderStyle.SINGLE, size: 4, color: "C3C2B7" };
const hair = { style: BorderStyle.SINGLE, size: 2, color: "E1E0D9" };
function tableBlock(b) {
  const widths = b.widths.map((w) => Math.round(w * TEXT_W));
  const diff = TEXT_W - widths.reduce((a, c) => a + c, 0);
  widths[widths.length - 1] += diff;
  const cellPara = (text, bold) => new Paragraph({ spacing: { before: 0, after: 0, line: 240 },
    children: [new TextRun({ text: String(text), bold, font: FONT, size: 18 })] });
  const header = new TableRow({ tableHeader: true, cantSplit: true, children: b.columns.map((c, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, shading: { fill: "E3EEFB", type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 60, bottom: 60, left: 80, right: 80 }, verticalAlign: VerticalAlign.CENTER, children: [cellPara(c, true)] })) });
  const rows = b.rows.map((row, r) => new TableRow({ cantSplit: true, children: row.map((c, i) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: r % 2 === 1 ? { fill: "F7F7F5", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 }, children: [cellPara(c, i === 0 && false)] })) }));
  const out = [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100, line: 276 },
      children: [new Bookmark({ id: bm(b.id), children: [new TextRun({ text: `${b.label}: `, bold: true, font: FONT, size: 21 })] }),
        new TextRun({ text: b.caption, font: FONT, size: 21 })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: [header, ...rows],
      borders: { top: thin, bottom: thin, left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
        right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, insideHorizontal: hair,
        insideVertical: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } } }),
  ];
  if (b.note) out.push(new Paragraph({ spacing: { before: 60, after: 240, line: 260 },
    children: [new TextRun({ text: `Note: ${b.note}`, italics: true, font: FONT, size: 18 })] }));
  else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

function codeBlock(b) {
  const out = [new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 },
    children: [new TextRun({ text: b.title, bold: true, font: FONT, size: 22 })] })];
  b.lines.forEach((line, i) => out.push(new Paragraph({
    keepNext: i < b.lines.length - 1, keepLines: true, spacing: { before: 0, after: 0, line: 240 },
    shading: { fill: "F4F4F2", type: ShadingType.CLEAR, color: "auto" }, indent: { left: 120, right: 120 },
    children: [new TextRun({ text: line.length ? line : " ", font: MONO, size: 17 })] })));
  out.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return out;
}

function headingBlock(b) {
  if (b.level === 1) {
    let text = b.text;
    if (b.kind === "chapter") text = `Chapter ${b.number}: ${b.text}`;
    return [new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun({ text, font: FONT })] })];
  }
  const text = b.number ? `${b.number}  ${b.text}` : b.text;
  const lvl = { 2: HeadingLevel.HEADING_2, 3: HeadingLevel.HEADING_3, 4: HeadingLevel.HEADING_4 }[b.level];
  return [new Paragraph({ heading: lvl, keepNext: true, children: [new TextRun({ text, font: FONT })] })];
}

// ------------------------------------------------------------------ special pages
const M = model.meta;
function centered(text, size = 24, bold = false, after = 120, italics = false) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after },
    children: [new TextRun({ text, size, bold, italics, font: FONT })] });
}
function titlePage() {
  const out = [new Paragraph({ spacing: { before: 1400 }, children: [] }),
    centered(M.title, 36, true, 600), centered("by", 24, false, 300)];
  M.authors.forEach(([n, id]) => out.push(centered(`${n} (${id})`, 24, false, 60)));
  out.push(new Paragraph({ spacing: { before: 600 }, children: [] }),
    centered(`A thesis submitted to the ${M.department}`, 24, false, 40),
    centered("in partial fulfillment of the requirements for the degree of", 24, false, 40),
    centered(M.degree, 24, false, 800),
    centered(M.department, 24, false, 40), centered(M.university, 24, false, 40), centered(M.date, 24, false, 1000),
    centered(`© 2026. ${M.university}`, 22, false, 40), centered("All rights reserved.", 22, false, 40));
  return out;
}
function plainHeading(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun({ text, font: FONT })] });
}
function signatureTable(entries) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const rows = [];
  for (let i = 0; i < entries.length; i += 2) {
    const pair = entries.slice(i, i + 2);
    rows.push(new TableRow({ children: [0, 1].map((k) => new TableCell({ width: { size: TEXT_W / 2, type: WidthType.DXA },
      borders: { top: none, bottom: none, left: none, right: none },
      children: pair[k] ? [new Paragraph({ spacing: { before: 480 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "______________________", font: FONT })] }),
        centered(pair[k][0], 24, false, 0), centered(pair[k][1], 22, false, 120)] : [new Paragraph({ children: [] })] })) }));
  }
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W / 2, TEXT_W / 2], rows,
    borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none } });
}
function declarationPage() {
  const items = [
    "The thesis submitted is our own original work while completing degree at BRAC University.",
    "The thesis does not contain material previously published or written by a third party, except where this is appropriately cited through full and accurate referencing.",
    "The thesis does not contain material which has been accepted, or submitted, for any other degree or diploma at a university or other institution.",
    "We have acknowledged all main sources of help.",
  ];
  const ref = newNumberedRef();
  return [plainHeading("Declaration"), body([{ text: "It is hereby declared that" }]),
    ...items.map((t) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 100, line: 340 }, alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: t, font: FONT })] })),
    body([{ text: "Student's Full Name & Signature:", bold: true }]), signatureTable(M.authors)];
}
function approvalPage() {
  const ref = newNumberedRef();
  const out = [plainHeading("Approval"),
    body([{ text: `The thesis titled "${M.title}" submitted by` }]),
    ...M.authors.map(([n, id]) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60 }, children: [new TextRun({ text: `${n} (${id})`, font: FONT })] })),
    body([{ text: `of ${M.semester} has been accepted as satisfactory in partial fulfillment of the requirement for the degree of ${M.degree} on ${M.approval_date}.` }]),
    new Paragraph({ spacing: { before: 240, after: 120 }, children: [new TextRun({ text: "Examining Committee:", bold: true, font: FONT })] })];
  const member = (role, [name, title]) => [
    new Paragraph({ spacing: { before: 240 }, children: [new TextRun({ text: role, bold: true, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { before: 360 }, children: [new TextRun({ text: "______________________________", font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: name, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: title, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: M.department, font: FONT })] }),
    new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: M.university, font: FONT })] })];
  out.push(...member("Supervisor: (Member)", M.supervisor), ...member("Thesis Coordinator: (Member)", M.coordinator),
    ...member("Head of Department: (Chair)", M.head));
  return out;
}
let LIST_WITH_PAGES = true;
function shortCaption(c) {
  const first = c.split(/(?<=\.)\s/)[0];
  return first.length > 150 ? first.slice(0, 147).replace(/\s+\S*$/, "") + " ..." : first;
}
function listOf(kind) {
  const items = kind === "fig" ? model.figures : model.tables;
  const word = kind === "fig" ? "Figure" : "Table";
  const out = [plainHeading(kind === "fig" ? "List of Figures" : "List of Tables")];
  items.forEach((it) => {
    const children = [new TextRun({ text: `${word} ${it.number}\t`, font: FONT, size: 22 }),
      new TextRun({ text: shortCaption(it.caption), font: FONT, size: 22 })];
    if (LIST_WITH_PAGES) children.push(new TextRun({ children: [new PositionalTab({ alignment: PositionalTabAlignment.RIGHT,
      relativeTo: PositionalTabRelativeTo.MARGIN, leader: PositionalTabLeader.DOT })], font: FONT, size: 22 }), new PageReference(bm(it.id)));
    out.push(new Paragraph({ spacing: { after: 80, line: 276 }, indent: { left: 1100, hanging: 1100 }, children,
      tabStops: [{ type: TabStopType.LEFT, position: 1100 }] }));
  });
  return out;
}
function nomenclature() {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const w1 = 2400, w2 = TEXT_W - 2400;
  return [plainHeading("Nomenclature"),
    body([{ text: "The following symbols and abbreviations are used in the body of this document." }]),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [w1, w2],
      borders: { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none },
      rows: model.nomenclature.map(([a, b]) => new TableRow({ children: [
        new TableCell({ width: { size: w1, type: WidthType.DXA }, children: [new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: a, bold: true, font: FONT, size: 22 })] })] }),
        new TableCell({ width: { size: w2, type: WidthType.DXA }, children: [new Paragraph({ spacing: { after: 40 }, children: [new TextRun({ text: b, font: FONT, size: 22 })] })] })] })) })];
}
function tocBlock() {
  return [plainHeading("Table of Contents"),
    new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
    new Paragraph({ spacing: { before: 120 }, children: [new TextRun({ text: "(Right-click the table and choose Update Field if page numbers are not shown.)", italics: true, size: 18, font: FONT, color: "898781" })] })];
}
function referencesBlock() {
  const out = [plainHeading("Bibliography")];
  model.references.forEach((r) => out.push(new Paragraph({ spacing: { after: 100, line: 276 }, alignment: AlignmentType.LEFT,
    indent: { left: 640, hanging: 640 }, tabStops: [{ type: TabStopType.LEFT, position: 640 }],
    children: [new TextRun({ text: `[${r.n}]\t`, font: FONT, size: 22 }), new TextRun({ text: r.text, font: FONT, size: 22 })] })));
  return out;
}

// ------------------------------------------------------------------ assemble
function renderBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    switch (b.type) {
      case "heading": out.push(...headingBlock(b)); break;
      case "para": out.push(body(b.runs)); break;
      case "quote": out.push(body(b.runs, { indent: { left: 720, right: 720 }, run: { italics: true } })); break;
      case "bullets": b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 340 }, children: textRuns(it) }))); break;
      case "numbered": { const ref = newNumberedRef(); b.items.forEach((it) => out.push(new Paragraph({ numbering: { reference: ref, level: 0 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 340 }, children: textRuns(it) }))); break; }
      case "figure": out.push(...figureBlock(b)); break;
      case "table": out.push(...tableBlock(b)); break;
      case "equation": out.push(...equationBlock(b)); break;
      case "code": out.push(...codeBlock(b)); break;
      case "special":
        if (b.name === "TITLEPAGE") out.push(...titlePage());
        else if (b.name === "DECLARATION") out.push(...declarationPage());
        else if (b.name === "APPROVAL") out.push(...approvalPage());
        else if (b.name === "TOC") out.push(...tocBlock());
        else if (b.name === "LOF") out.push(...listOf("fig"));
        else if (b.name === "LOT") out.push(...listOf("tab"));
        else if (b.name === "NOMENCLATURE") out.push(...nomenclature());
        break;
      default: throw new Error("unknown block " + b.type);
    }
  }
  return out;
}

function footer(format) {
  return new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });
}
function sectionProps(format, start) {
  return { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
    pageNumbers: start ? { start, formatType: format } : { formatType: format } }, type: SectionType.NEXT_PAGE };
}
const styles = {
  default: { document: { run: { font: FONT, size: 24 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 34, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 0, after: 360 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 28, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 25, bold: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { size: 24, bold: true, italics: true, font: FONT, color: "000000" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 3 } },
  ],
};
function numberingConfig() {
  return { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }, ...numberingConfigs] };
}
async function write(doc, file) {
  const buf = await Packer.toBuffer(doc);
  fs.writeFileSync(file, buf);
  console.log("wrote", path.relative(ROOT, file), (buf.length / 1024).toFixed(0) + " KB");
}

async function main() {
  const parts = Object.fromEntries(model.parts.map((p) => [p.part, p]));
  const front = parts["00_front_matter"].blocks;
  const titleIdx = front.findIndex((b) => b.type === "special" && b.name === "TITLEPAGE");
  const titleBlocks = front.slice(0, titleIdx + 1);
  const frontRest = front.slice(titleIdx + 1);
  const chapterKeys = ["01_introduction", "02_literature_review", "03_requirements", "04_methodology", "05_implementation", "06_results", "07_conclusion"];

  // ---- full thesis
  numberingConfigs = []; numberedCounter = 0;
  const secTitle = { properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN } } }, children: renderBlocks(titleBlocks) };
  const secFront = { properties: sectionProps(NumberFormat.LOWER_ROMAN, 1), footers: { default: footer() }, children: renderBlocks(frontRest) };
  const mainChildren = [];
  chapterKeys.forEach((k) => mainChildren.push(...renderBlocks(parts[k].blocks)));
  mainChildren.push(...referencesBlock());
  mainChildren.push(...renderBlocks(parts["08_appendices"].blocks));
  const secMain = { properties: sectionProps(NumberFormat.DECIMAL, 1), footers: { default: footer() }, children: mainChildren };
  const full = new Document({ creator: "RAHC-LoRA thesis team", title: M.title, description: "P2 thesis, BRAC University",
    styles, numbering: numberingConfig(), features: { updateFields: true }, sections: [secTitle, secFront, secMain] });
  await write(full, path.join(ROOT, "01_Full_Thesis_P2_RAHC-LoRA.docx"));

  // ---- per-part documents
  const chapDir = path.join(ROOT, "chapters");
  fs.mkdirSync(chapDir, { recursive: true });
  const single = async (name, blocks, extra = []) => {
    numberingConfigs = []; numberedCounter = 0;
    const children = [...renderBlocks(blocks), ...extra];
    const doc = new Document({ creator: "RAHC-LoRA thesis team", title: `${M.title} - ${name}`, styles, numbering: numberingConfig(),
      features: { updateFields: true }, sections: [{ properties: sectionProps(NumberFormat.DECIMAL, 1), footers: { default: footer() }, children }] });
    await write(doc, path.join(chapDir, `${name}.docx`));
  };
  LIST_WITH_PAGES = false; // page numbers exist only in the combined document
  await single("00_Front_Matter", front);
  LIST_WITH_PAGES = true;
  const names = { "01_introduction": "01_Chapter1_Introduction", "02_literature_review": "02_Chapter2_Literature_Review",
    "03_requirements": "03_Chapter3_Requirements_Impacts_Constraints", "04_methodology": "04_Chapter4_Proposed_Methodology",
    "05_implementation": "05_Chapter5_Implementation_and_Setup", "06_results": "06_Chapter6_Results_and_Discussion",
    "07_conclusion": "07_Chapter7_Conclusion_and_Future_Work" };
  for (const k of chapterKeys) await single(names[k], parts[k].blocks);
  numberingConfigs = []; numberedCounter = 0;
  await single("08_Bibliography", [], referencesBlock());
  await single("09_Appendices", parts["08_appendices"].blocks);
  await buildIndex();
}

// ------------------------------------------------------------------ read-me / index document
function simpleTable(columns, rows, fracs) {
  return tableBlock({ id: "idx_" + Math.random().toString(36).slice(2, 8), label: "", caption: "", columns, rows, widths: fracs }).slice(1, 2);
}
function h(text, level) {
  return new Paragraph({ heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2, keepNext: true,
    children: [new TextRun({ text, font: FONT })] });
}
function p(text, opts = {}) { return body([{ text, ...opts }]); }
function bullets(items) {
  return items.map((t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60, line: 300 },
    children: [new TextRun({ text: t, font: FONT })] }));
}
async function buildIndex() {
  numberingConfigs = []; numberedCounter = 0;
  const rel = (f) => path.relative(ROOT, f).split(path.sep).join("/");
  const c = [];
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "P2 Thesis Package: Read Me First", bold: true, size: 36, font: FONT })] }));
  c.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 360 }, children: [new TextRun({ text: M.title, italics: true, size: 24, font: FONT })] }));
  c.push(h("1. What is in this folder", 2));
  c.push(...simpleTable(["Path", "Contents"], [
    ["01_Full_Thesis_P2_RAHC-LoRA.docx", "Complete P2 thesis (front matter, 7 chapters, bibliography, appendices) with embedded figures, native Word tables, numbered equations, table of contents, and lists of figures and tables."],
    ["chapters/*.docx", "The same text split per chapter (00 front matter to 09 appendices) for parallel editing or chapter-by-chapter LaTeX conversion."],
    ["figures/fig_X_Y_*.png", `${model.figures.length} figures at 300 dpi; X_Y equals the figure number in the thesis.`],
    ["tables/tab_X_Y_*.png", `${model.tables.length} tables rendered as images; the editable rows are the native Word tables and data/tables.json.`],
    ["latex_support/references.bib", `${model.references.length} BibTeX entries in citation order; citation_key_map.txt maps [n] to keys.`],
    ["latex_support/equations.tex", `${model.equations.length} display equations with \\label{eq:...}, numbered as in the Word file.`],
    ["latex_support/figures_and_tables.tex", "Ready-made figure and table environments with captions and labels."],
    ["data/results_summary.json, master_run_table.csv", "Every number in the thesis, derived from the raw run artifacts by source/analyze_runs.py."],
    ["source/", "Scripts that regenerate everything: analyze_runs.py, make_diagrams.py, make_charts.py, make_tables.py, build_document.py, build_docx.js, and the chapter text (content/*.txt)."],
  ], [0.34, 0.66]));
  c.push(h("2. Research status at a glance", 2));
  c.push(...bullets([
    "Complete: specification; Phases 0-2 (platform, tasks, rewards, sandbox, evaluation, standard RL-LoRA); 59 tests pass.",
    "Partial: Phase 3 (direct HLoRA-to-RL baseline implemented and piloted; supervised HLoRA reproduction not run).",
    "Not started: Phases 4-7 (the RAHC-LoRA components). The thesis presents RAHC-LoRA as the proposed method with its full design; results in Chapter 6 are baseline and pilot evidence only.",
    "Key results: no stream reached the 10-point forgetting threshold (-1.30 to +3.12 pp); HLoRA-RL did not reduce forgetting but learned GSM8K better in one seed (32.8% vs 3.1%, n = 64); its softmax layer weights collapsed onto one module.",
  ]));
  c.push(h("3. Items the team must fill in or confirm", 2));
  c.push(...bullets([
    "Title page and approval page: submission month ([Month] 2026), semester ([Semester], 2026), and approval date ([Date]).",
    "Declaration: signatures of all five members.",
    "Section 3.8: the assignment of subsystems to team members (marked [Team to complete ...]).",
    "Figure 3.1: the proposed schedule for the final-thesis phases (confirm with the supervisor).",
    "Examining committee names and titles (copied from P1; confirm they are current).",
    "Update all fields in Word (Ctrl+A, then F9) after any edit so that the table of contents, list pages, and cross-reference page numbers refresh.",
  ]));
  c.push(h("4. Figure index", 2));
  c.push(...simpleTable(["No.", "File", "Caption"], model.figures.map((f) => [f.number, rel(f.path), f.caption]), [0.07, 0.36, 0.57]));
  c.push(h("5. Table index", 2));
  c.push(...simpleTable(["No.", "File", "Caption"], model.tables.map((t) => [t.number, rel(t.path), t.caption]), [0.07, 0.36, 0.57]));
  c.push(h("6. Equation index", 2));
  c.push(...simpleTable(["No.", "LaTeX label", "No.", "LaTeX label"], (() => {
    const e = model.equations; const rows = [];
    for (let i = 0; i < e.length; i += 2) rows.push([`(${e[i].number})`, `eq:${e[i].id}`, e[i + 1] ? `(${e[i + 1].number})` : "", e[i + 1] ? `eq:${e[i + 1].id}` : ""]);
    return rows; })(), [0.1, 0.4, 0.1, 0.4]));
  c.push(h("7. Converting to LaTeX", 2));
  c.push(...bullets([
    "Use the BRAC thesis LaTeX template; copy figures/ and tables/ next to it and set \\graphicspath{{figures/}{tables/}}.",
    "Add latex_support/references.bib and cite with \\cite{key}; citation_key_map.txt maps each [n] in the Word text to its key.",
    "Paste equations from latex_support/equations.tex (they need \\usepackage{amsmath,amssymb}).",
    "Figure and table environments with the same captions and labels are in latex_support/figures_and_tables.tex. For editable LaTeX tables, convert the native Word tables (or data/tables.json) with a tool such as pandoc or tablesgenerator.",
    "Cross-references in the Word text (Figure 4.1, Table 5.2, Section 4.3) correspond to \\ref{fig:...}, \\ref{tab:...}, and \\ref{sec:...} labels.",
  ]));
  c.push(h("8. Verification notes", 2));
  c.push(p("All numbers were re-derived from the raw run artifacts. Earlier internal reports contained several discrepancies (for example, LoRA rank 16 and G = 8 instead of the actual rank 8 and G = 4, and 'retention' or 'breakthrough' claims that the data do not support). They are listed in Appendix D (Table D.1). The thesis uses only the verified values."));
  c.push(p("Bibliography: all entries were checked, and the most recent ones were verified online (for example HLoRA/HLER arXiv:2501.13669 by Song et al.). Several entries in the P1 bibliography appear to have incorrect author lists and are worth correcting if P1 is reused: HLER is by S. Song et al. (not Zhao et al.); I-LoRA (arXiv:2402.18865) is by W. Ren et al.; MbPA++ is by C. de Masson d'Autume et al. (NeurIPS 2019); the implicit-inference paper is by S. Kotha et al. (ICLR 2024); spurious forgetting (arXiv:2501.13453) is by J. Zheng et al.; the COLING 2020 continual-learning survey is by M. Biesialska et al. (the Parisi et al. review appeared in Neural Networks, 2019); and P1 entry [23] reuses the arXiv ID of entry [2]."));
  c.push(p("Two implementation issues surfaced during the analysis and are reported in the thesis: the captured effective gradient applies the LoRA scale twice (a constant factor of 2), and the multiple-choice parser accepts hedged answers such as 'Answer: A or B'. Both are scheduled for correction before Phase 4."));
  c.push(h("9. Regenerating the package", 2));
  c.push(...codeBlock({ title: "From the repository root", lines: [
    "python Thesis_Paper/source/analyze_runs.py",
    "python Thesis_Paper/source/make_diagrams.py",
    "python Thesis_Paper/source/make_charts.py",
    "python Thesis_Paper/source/build_document.py",
    "node   Thesis_Paper/source/build_docx.js      # needs the npm package 'docx' (v9)"] }));
  const doc = new Document({ creator: "RAHC-LoRA thesis team", title: "P2 thesis package index", styles, numbering: numberingConfig(),
    sections: [{ properties: sectionProps(NumberFormat.DECIMAL, 1), footers: { default: footer() }, children: c }] });
  await write(doc, path.join(ROOT, "00_README_Index.docx"));
}
main().catch((e) => { console.error(e); process.exit(1); });
