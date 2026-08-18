import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    port: 5173,
    allowedHosts: ["terminal.local"],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
    warmup: {
      clientFiles: ["./src/main.tsx"],
    },
  },
  preview: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        /*
         * Every lazy workstation page imports `EChart`, so Rollup used to fold
         * the whole echarts + zrender vendor tree into one shared 651 kB chunk
         * named after that component. Splitting the vendors out gives each a
         * stable filename that survives app edits (long-lived browser cache)
         * and drops the largest chunk under the 500 kB warning threshold.
         *
         * zrender is echarts' rendering engine and is a separate package, so
         * it splits cleanly; the charts/components registered in EChart.tsx
         * stay tree-shaken exactly as before.
         */
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("node_modules/zrender") || id.includes("node_modules/echarts")) {
            // echarts and zrender must stay in ONE chunk. Splitting them (either
            // package-wise or lib/chart vs core) makes Rollup report
            // "Circular chunk: A -> B -> A", because echarts' internals are
            // genuinely cyclic. Trading a size warning for a module-init-order
            // hazard is not an improvement, so this chunk stays over the
            // 500 kB threshold and the warning is left standing rather than
            // silenced with chunkSizeWarningLimit.
            return "vendor-echarts";
          }
          if (
            id.includes("node_modules/react-router") ||
            id.includes("node_modules/react-dom") ||
            id.includes("node_modules/scheduler") ||
            /node_modules\/react\//.test(id)
          ) {
            return "vendor-react";
          }
          if (id.includes("node_modules/@tanstack")) return "vendor-query";
          return undefined;
        },
      },
    },
  },
  plugins: [react()],
});
