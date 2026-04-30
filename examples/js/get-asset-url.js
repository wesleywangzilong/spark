export async function getAssetFileURL(assetFile) {
  try {
    const response = await fetch("../assets.json");
    const assetsInfo = await response.json();
    const entry = assetsInfo[assetFile];
    if (!entry) return null;
    // Relative path from the example page (at .../examples/<name>/index.html)
    // to .../examples/assets/<directory>/<file>. Works in both spark Vite dev
    // and sparkapp WebViewAssetLoader (mounted at /assets/). The old upstream
    // behavior defaulted to the remote sparkjs.dev CDN URL via `entry.url` and
    // only switched to local when `window.sparkLocalAssets` was set — a flag
    // Vite's `define` never actually wires up because the source uses
    // `window.sparkLocalAssets` (member expression, not bare identifier).
    return `../assets/${entry.directory}/${assetFile}`;
  } catch (error) {
    console.error("Failed to load asset file URL:", error);
    return null;
  }
}
