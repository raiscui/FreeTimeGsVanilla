#!/usr/bin/env node
/**
 * beautiful-mermaid-rs (compat wrapper)
 *
 * 目标:
 * - 从 stdin 读取 Mermaid 文本.
 * - 默认输出 SVG 到 stdout.
 * - --ascii 输出 Unicode/ASCII 文字图(便于终端阅读).
 *
 * 设计约束(对齐本仓库 AGENTS.md 里的约定):
 * - 不接受文件路径参数,只从 stdin 读.
 * - 退出码:
 *   - 0: 成功(或 BrokenPipe).
 *   - 1: 渲染失败/读取 stdin 失败.
 *   - 2: 参数或用法错误.
 */

import fs from "node:fs";

import { renderMermaid, renderMermaidAscii } from "beautiful-mermaid";

function printHelp() {
  console.error("Usage:");
  console.error("  beautiful-mermaid-rs < diagram.mmd > diagram.svg");
  console.error("  cat diagram.mmd | beautiful-mermaid-rs --ascii");
  console.error("");
  console.error("Options:");
  console.error("  --ascii       Output ASCII/Unicode text (default: SVG)");
  console.error("  --use-ascii   (Only with --ascii) Force pure ASCII characters");
  console.error("  -h, --help    Show help");
  console.error("  -V, --version Show version");
}

function readStdinUtf8() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

function parseArgs(argv) {
  const flags = new Set(argv);

  // 非 flag 参数一律认为是用法错误(不接受文件路径).
  for (const a of argv) {
    if (!a.startsWith("-")) {
      return { ok: false, exitCode: 2, error: `Unexpected argument: ${a}` };
    }
  }

  const help = flags.has("-h") || flags.has("--help");
  const version = flags.has("-V") || flags.has("--version");
  const ascii = flags.has("--ascii");
  const useAscii = flags.has("--use-ascii");

  if (useAscii && !ascii) {
    return { ok: false, exitCode: 2, error: "--use-ascii requires --ascii" };
  }

  const knownFlags = new Set(["-h", "--help", "-V", "--version", "--ascii", "--use-ascii"]);
  const unknown = argv.filter((a) => a.startsWith("-") && !knownFlags.has(a));
  if (unknown.length > 0) {
    return { ok: false, exitCode: 2, error: `Unknown flag(s): ${unknown.join(" ")}` };
  }

  return { ok: true, help, version, ascii, useAscii };
}

async function main() {
  // BrokenPipe 视为成功(符合 Unix 习惯).
  process.stdout.on("error", (err) => {
    if (err?.code === "EPIPE") {
      process.exit(0);
    }
  });

  const parsed = parseArgs(process.argv.slice(2));
  if (!parsed.ok) {
    console.error(parsed.error);
    printHelp();
    process.exit(parsed.exitCode);
  }

  if (parsed.help) {
    printHelp();
    process.exit(0);
  }

  if (parsed.version) {
    // 这里不引入 package.json 读取,直接输出一个固定标记即可.
    // 版本不影响功能,避免在 CLI 层做额外 IO.
    console.log("beautiful-mermaid-rs (js wrapper)");
    process.exit(0);
  }

  const mermaidText = readStdinUtf8();
  if (!mermaidText.trim()) {
    console.error("stdin is empty.");
    printHelp();
    process.exit(2);
  }

  try {
    if (parsed.ascii) {
      const out = renderMermaidAscii(mermaidText, { useAscii: parsed.useAscii });
      process.stdout.write(out.endsWith("\n") ? out : `${out}\n`);
    } else {
      const svg = await renderMermaid(mermaidText);
      process.stdout.write(svg.endsWith("\n") ? svg : `${svg}\n`);
    }
  } catch (err) {
    console.error(err?.message ?? String(err));
    process.exit(1);
  }
}

main();

