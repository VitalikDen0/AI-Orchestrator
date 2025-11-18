import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    server: {
      host: "0.0.0.0",
      port: Number(env.VITE_PORT ?? 5173),
      proxy: {
        "/api": {
          target: env.VITE_API_BASE ?? "http://127.0.0.1:8001",
          changeOrigin: true
        }
      }
    },
    build: {
      outDir: resolve(__dirname, "dist"),
      emptyOutDir: true
    }
  };
});
