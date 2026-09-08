/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        abyss: '#040a10',
        mist: '#d9e7ec',
        dim: '#8fa6b0',
        line: 'rgba(126, 196, 207, 0.14)',
        accent: '#56d4e2',
        subsurface: '#0d4257',
      },
      fontFamily: {
        disp: ['"Space Grotesk"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
