/// <reference types="vite/client" />
// Gives TypeScript `import.meta.glob`, which the hook-order test uses to read
// component sources through Vite rather than node:fs -- the tsconfig carries no
// node types, and this project should not grow a dependency to run one check.
