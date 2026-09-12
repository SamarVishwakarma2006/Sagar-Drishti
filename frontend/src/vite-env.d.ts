/// <reference types="vite/client" />

interface ImportMetaEnv {
  // No external API keys required — SagarBot runs in deterministic offline mode.
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
