import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
  },
  publicDir: 'public',
  server: {
    port: 3000,
    open: true,
  },
});

