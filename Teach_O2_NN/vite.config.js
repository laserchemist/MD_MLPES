import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps every asset path relative, so the built site works from any folder
// (GitHub Pages, a USB stick, or opened from a laptop during a talk).
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "site", emptyOutDir: true },
});
