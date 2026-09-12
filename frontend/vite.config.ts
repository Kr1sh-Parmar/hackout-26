import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The app calls same-origin /api; this forwards it to FastAPI with the prefix stripped, exactly as
// nginx does in docker. No CORS, no hardcoded host in the bundle. VITE_PROXY_TARGET points it elsewhere.
const api = {
  '/api': {
    target: process.env.VITE_PROXY_TARGET ?? 'http://localhost:8000',
    changeOrigin: true,
    rewrite: (path: string) => path.replace(/^\/api/, ''),
  },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: api },
  preview: { port: 4173, proxy: api },
});
