import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    allowedHosts: [
      "localhost", // Keep localhost for local development
      "3c578b7fe391.ngrok-free.app",
    ],
    hmr: {
      overlay: false,
    },
  },
});
