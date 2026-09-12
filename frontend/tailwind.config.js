/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#f8fafc',
        primary: '#2563eb', // bright blue accent
        success: '#16a34a',
        warning: '#ea580c',
        danger: '#dc2626',
        surface: 'rgba(255, 255, 255, 0.7)',
        'surface-solid': '#ffffff',
        muted: '#64748b',
        text: '#0f172a',
        // dev-02 validated data palette. Colour follows the entity, never its position in a list.
        series: {
          wind: 'var(--series-wind)',
          solar: 'var(--series-solar)',
          net: 'var(--series-net)',
          persistence: 'var(--series-persistence)',
        },
        status: {
          good: 'var(--status-good)',
          warning: 'var(--status-warning)',
          serious: 'var(--status-serious)',
          critical: 'var(--status-critical)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 4px 40px -2px rgba(37, 99, 235, 0.05)',
      }
    },
  },
  plugins: [],
}
