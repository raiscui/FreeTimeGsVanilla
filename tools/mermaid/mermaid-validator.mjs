#!/usr/bin/env node
/**
 * mermaid-validator
 *
 * 目标:
 * - 校验 Markdown 文件里的 ```mermaid 代码块是否能被解析.
 * - 也支持从 stdin 读取内容(自动识别 markdown 或纯 mermaid).
 *
 * 说明:
 * - 这里用 beautiful-mermaid 的 renderer 作为"语法是否正确"的判据.
 * - 因为它不依赖浏览器/DOM,并且支持 flowchart/sequence 等常见图类型.
 */

import fs from "node:fs";
import path from "node:path";

import { renderMermaid } from "beautiful-mermaid";

function printUsage() {
  // 约定: 只输出简单用法,避免噪音.
  console.error("Usage:");
  console.error("  tools/mermaid-validator <file.md> [more files]");
  console.error("  cat file.md | tools/mermaid-validator");
}

function readStdinUtf8() {
  try {
    return fs.readFileSync(0, "utf8");
  } catch {
    return "";
  }
}

function extractMermaidBlocksFromMarkdown(markdownText) {
  const blocks = [];
  // 说明:
  // - 用 [\\s\\S] 替代 dotAll,避免依赖运行时 flags.
  // - mermaid fence 要求形如:
  //   ```mermaid
  //   ...
  //   ```
  const fenceRegex = /```mermaid\s*\n([\s\S]*?)\n```/g;
  let match = null;
  while ((match = fenceRegex.exec(markdownText)) !== null) {
    blocks.push(match[1]);
  }
  return blocks;
}

async function validateMermaidText(mermaidText, contextLabel) {
  const trimmed = mermaidText.trim();
  if (!trimmed) {
    throw new Error(`${contextLabel}: mermaid 内容为空`);
  }

  // 只要 renderer 不抛异常,就认为语法有效.
  // theme 不影响语法,这里用最小默认即可.
  await renderMermaid(trimmed);
}

async function validateOneInput(content, label) {
  const hasFence = content.includes("```mermaid");
  if (!hasFence) {
    // 认为是纯 mermaid.
    await validateMermaidText(content, label);
    return { blocks: 1 };
  }

  const blocks = extractMermaidBlocksFromMarkdown(content);
  if (blocks.length === 0) {
    // 有 fence 标记但没匹配到块,说明 fence 可能写错了.
    throw new Error(`${label}: 未找到可解析的 mermaid code block(请检查 Markdown 的 mermaid code fence 写法)`);
  }

  for (let i = 0; i < blocks.length; i += 1) {
    await validateMermaidText(blocks[i], `${label}:block#${i + 1}`);
  }
  return { blocks: blocks.length };
}

async function main() {
  const args = process.argv.slice(2);
  const hasHelp = args.includes("-h") || args.includes("--help");
  if (hasHelp) {
    printUsage();
    process.exit(0);
  }

  const filePaths = args.filter((a) => !a.startsWith("-"));
  const flagArgs = args.filter((a) => a.startsWith("-"));
  if (flagArgs.length > 0) {
    console.error(`Unknown flags: ${flagArgs.join(" ")}`);
    printUsage();
    process.exit(2);
  }

  const inputs = [];
  if (filePaths.length === 0) {
    const stdin = readStdinUtf8();
    if (!stdin.trim()) {
      console.error("stdin is empty.");
      printUsage();
      process.exit(2);
    }
    inputs.push({ label: "stdin", content: stdin });
  } else {
    for (const fp of filePaths) {
      const abs = path.resolve(process.cwd(), fp);
      if (!fs.existsSync(abs)) {
        console.error(`File not found: ${fp}`);
        process.exit(2);
      }
      inputs.push({ label: fp, content: fs.readFileSync(abs, "utf8") });
    }
  }

  let totalBlocks = 0;
  for (const input of inputs) {
    const { blocks } = await validateOneInput(input.content, input.label);
    totalBlocks += blocks;
    console.error(`[mermaid-validator] OK: ${input.label} (${blocks} block(s))`);
  }
  console.error(`[mermaid-validator] All OK. Total blocks: ${totalBlocks}`);
}

main().catch((err) => {
  console.error(`[mermaid-validator] ERROR: ${err?.message ?? String(err)}`);
  process.exit(1);
});
