import { spawnSync } from "child_process";
import path from "path";

const BRIDGE = path.join(process.cwd(), "scripts", "bridge.py");
const PYTHON = path.join(process.cwd(), "..", ".venv", "bin", "python");

interface BridgeCallOptions {
  timeout?: number;
}

export function callBridge(action: string, data: Record<string, unknown> = {}, opts: BridgeCallOptions = {}) {
  const input = JSON.stringify({ action, ...data });
  const result = spawnSync(PYTHON, [BRIDGE], {
    input,
    timeout: opts.timeout ?? 30_000,
    encoding: "utf-8",
  });

  if (result.error) {
    throw new Error(`Bridge process error: ${result.error.message}`);
  }

  const output = result.stdout.trim();
  if (!output) {
    throw new Error(result.stderr || "Bridge returned empty output");
  }

  const parsed = JSON.parse(output);
  if (parsed.error) {
    throw new Error(parsed.error);
  }
  return parsed;
}
