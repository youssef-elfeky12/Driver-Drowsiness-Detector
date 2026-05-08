/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0B0F14',
        surface: '#141B23',
        surface2: '#1C2530',
        primary: '#3B82F6',
        amber: '#F59E0B',
        danger: '#EF4444',
        ok: '#22C55E',
        muted: '#9CA3AF',
        text: '#E5E7EB',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-red': 'pulse-red 0.66s ease-in-out infinite',
        'slide-up': 'slide-up 0.4s ease-out',
        'slide-in-right': 'slide-in-right 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
      },
      keyframes: {
        'pulse-red': {
          '0%, 100%': { backgroundColor: 'rgba(239, 68, 68, 0.0)' },
          '50%': { backgroundColor: 'rgba(239, 68, 68, 0.55)' },
        },
        'slide-up': {
          from: { transform: 'translateY(100%)', opacity: '0' },
          to: { transform: 'translateY(0)', opacity: '1' },
        },
        'slide-in-right': {
          from: { transform: 'translateX(120%)', opacity: '0' },
          to: { transform: 'translateX(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
};
