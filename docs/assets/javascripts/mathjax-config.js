// MathJax 3 config for pymdownx.arithmatex (generic mode). Must load before tex-svg.js.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
    enableMenu: false, // the menu lazy-loads a11y extensions from a CDN
  },
  svg: { fontCache: "global" },
};

// Re-typeset after Material's instant navigation swaps page content.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
