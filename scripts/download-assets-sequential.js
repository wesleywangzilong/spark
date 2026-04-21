import { execSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  statSync,
  unlinkSync,
} from "node:fs";
import { join } from "node:path";

const assetsDir = "examples/assets/";
const urls = JSON.parse(readFileSync("examples/assets.json", "utf8"));
const entries = Object.entries(urls);

let ok = 0;
const failed = [];
let skipped = 0;
for (let i = 0; i < entries.length; i++) {
  const [key, data] = entries[i];
  const dir = join(assetsDir, data.directory);
  const file = join(dir, key);
  mkdirSync(dir, { recursive: true });
  if (existsSync(file)) {
    if (statSync(file).size === 0) unlinkSync(file);
    else {
      skipped++;
      console.log(`[${i + 1}/${entries.length}] skip ${key} (exists)`);
      continue;
    }
  }
  process.stdout.write(`[${i + 1}/${entries.length}] ${key} ... `);
  try {
    execSync(
      `curl -fsSL --max-time 600 --retry 3 --retry-delay 2 -o "${file}" "${data.url}"`,
      { stdio: ["ignore", "ignore", "pipe"] },
    );
    const sz = statSync(file).size;
    console.log(`ok (${(sz / 1024 / 1024).toFixed(2)} MB)`);
    ok++;
  } catch (e) {
    console.log("FAIL");
    failed.push(key);
    if (existsSync(file)) unlinkSync(file);
  }
}
console.log(`\nDone: ok=${ok}, skipped=${skipped}, failed=${failed.length}`);
if (failed.length) console.log("Failed:", failed.join(", "));
