import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: [
        'haarcascades/*.xml',
        'sounds/*.mp3',
        'sounds/*.m4a',
        'models/**/*',
      ],
      manifest: {
        name: 'Drowsiness Detector',
        short_name: 'Drowsy',
        description: 'On-device driver drowsiness detection',
        theme_color: '#0B0F14',
        background_color: '#0B0F14',
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        icons: [
          {
            src: '/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: '/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
          },
        ],
      },
      workbox: {
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
      },
    }),
  ],
  server: {
    port: 5173,
    host: true,
  },
});
