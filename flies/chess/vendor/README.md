# Vendored neural viewer for the chess arena

The chess cinematic arena (`arena.html`, served by `cinematic.py`) was built on the starter repo's brain viewer,
`web/matrix.js`, and took its DOM and styles from the starter `web/index.html`. The shared world has since
replaced both (`web/brain.js`, the station world in `web/index.html`), so the two original files are kept
here unchanged, from commit 83c87c3:

- `matrix.js` = starter `web/matrix.js`
- `matrix_host.html` = starter `web/index.html` (only its `<style>` and `#matrix` element are used)
