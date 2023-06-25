/** @type {import('tailwindcss').Config} */
module.exports = {
  mode: 'jit',
  content: [
    "./templates/**/*.{html,htm}",
    "./node_modules/flowbite/**/*.js"
    ],
  theme: {
    extend: {
      maxWidth: {
        50: '50%',
        30: '30%',
        // 100% is not required as max-w-full will be present by default
      }
    },
  },
  plugins: [
    require('flowbite/plugin')
  ],
}

