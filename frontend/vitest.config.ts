import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    include: ["src/**/*.test.{ts,tsx}"],
    // pure pose modules run in node; component tests opt into jsdom with a docblock
    environment: "node",
    setupFiles: [],
  },
});
