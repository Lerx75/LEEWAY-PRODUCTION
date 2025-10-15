// Temporarily disable Tailwind's PostCSS plugin to avoid pulling in
// lightningcss native bindings during Docker builds on slim images.
// We rely on our own CSS and inline styles for now.
export default {
  plugins: [],
};
