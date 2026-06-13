/** @type {import('next').NextConfig} */
const nextConfig = {
  // onnxruntime-web ships its own .wasm; we load those from the jsDelivr CDN at
  // runtime (see lib/detector.ts), so nothing extra needs bundling here.
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: false },
  webpack: (config) => {
    // onnxruntime-web references `fs`/`path` in its node entry; stub them out
    // for the browser bundle.
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
};

export default nextConfig;
