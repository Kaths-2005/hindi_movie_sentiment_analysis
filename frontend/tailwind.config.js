/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        cinema: {
          bg: "#12090C",
          card: "#1F1418",
          border: "#3A2530",
          gold: "#E8A33D",
          rani: "#C2185B",
          text: "#F5E9DA",
          muted: "#C9B8AE",
        },
      },
    },
  },
  plugins: [],
};
