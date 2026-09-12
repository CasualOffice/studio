import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { AssistResult, EngineProgress, GenerateArgs, HostInfo, Lora, ModelStatus, Orphans, Panel, RepairReport, ResolvedLora, ResolvedModel, SetupProgress, SetupState, StorageInfo, VaultItem, VaultStatus } from "./types";

/** Tauri rejects with our structured error; normalise it to a message. */
export function errText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e && typeof e === "object" && "message" in e) {
    return String((e as { message: unknown }).message);
  }
  return String(e);
}

/**
 * The engine's original wording, when there is one.
 *
 * Failures are restated in plain language for the toast, but the raw text is
 * what you need when the restatement is not enough, so it travels alongside.
 */
export function errDetail(e: unknown): string | null {
  if (e && typeof e === "object" && "detail" in e) {
    const d = String((e as { detail: unknown }).detail);
    const m = "message" in e ? String((e as { message: unknown }).message) : "";
    return d && d !== m ? d : null;
  }
  return null;
}

export const api = {
  hostInfo: () => invoke<HostInfo>("host_info"),
  setupState: () => invoke<SetupState>("setup_state"),
  runSetup: (force = false) => invoke<void>("run_setup", { force }),

  listModels: () => invoke<ModelStatus[]>("list_models"),
  storageInfo: () => invoke<StorageInfo>("storage_info"),
  listLoras: () => invoke<Lora[]>("list_loras"),
  resolveLora: (repo: string) => invoke<ResolvedLora>("resolve_lora", { repo }),
  addLora: (jobId: string, repo: string, file: string, name: string, bytes: number) =>
    invoke<Lora[]>("add_lora", { jobId, repo, file, name, bytes }),
  removeLora: (handle: string) => invoke<Lora[]>("remove_lora", { handle }),
  setModelsLocation: (path: string, moveExisting: boolean) =>
    invoke<StorageInfo>("set_models_location", { path, moveExisting }),
  downloadModel: (modelId: string, jobId: string) =>
    invoke<unknown>("download_model", { modelId, jobId }),
  deleteModel: (modelId: string) => invoke<number>("delete_model", { modelId }),
  resolveModel: (repo: string) => invoke<ResolvedModel>("resolve_model", { repo }),
  addCustomModel: (
    repo: string, name: string, tasks: string[], bytes: number,
    quantize: number | null, family: string | null, backend: string | null
  ) => invoke<void>("add_custom_model",
                    { spec: { repo, name, tasks, bytes, quantize, family, backend } }),
  /** Who is in the story and where it happens. Free of picture time. */
  storyCast: (jobId: string, story: string) =>
    invoke<{
      people: { name: string; description: string; mentions: number; tier: number }[];
      places: { name: string; description: string; mentions: number }[];
      words: number;
    }>("story_cast", { jobId, story }),

  shotList: (jobId: string, story: string, panels: number | null) =>
    invoke<{
      panels: Panel[];
      asked: number | null;
      derived: boolean;
      words: number;
      expected_low?: number;
      expected_high?: number;
      out_of_range?: boolean;
      too_long?: boolean;
      coverage?: { covered: number; total: number; percent: number; missing: string[] };
    }>("shot_list", { jobId, story, panels }),
  enrichPanels: (jobId: string, panels: Panel[], style: string) =>
    invoke<{ panels: Panel[] }>("enrich_panels", { jobId, panels, style }),
  composeBoard: (
    jobId: string, panels: string[], captions: string[], shots: string[],
    dialogue: { speaker: string; text: string }[][], scenes: number[],
    layout: string, project: string | null, projectName: string
  ) => invoke<string[]>("compose_board",
                        { jobId, board: { panels, captions, shots, dialogue,
                                          scenes, layout, project,
                                          project_name: projectName } }),
  boardStateGet: () => invoke<string | null>("board_state_get"),
  boardStateSet: (value: string) => invoke<void>("board_state_set", { value }),
  findOrphans: () => invoke<Orphans>("find_orphans"),
  sweepOrphans: () => invoke<number>("sweep_orphans"),
  hfTokenStatus: () =>
    invoke<{ present: boolean; hint: string | null }>("hf_token_status"),
  setHfToken: (token: string) => invoke<void>("set_hf_token", { token }),
  removeCustomModel: (modelId: string) =>
    invoke<void>("remove_custom_model", { modelId }),
  unloadModel: () => invoke<void>("unload_model"),
  enginePing: () => invoke<Record<string, unknown>>("engine_ping"),
  cancelJob: (jobId: string) => invoke<void>("cancel_job", { jobId }),

  assistPrompt: (jobId: string, prompt: string, mode: string, images: string[]) =>
    invoke<AssistResult>(
      "assist_prompt", { jobId, prompt, mode, images }
    ),

  generate: (args: GenerateArgs) => invoke<string[]>("generate", { args }),
  editImage: (args: GenerateArgs) => invoke<string[]>("edit_image", { args }),
  upscale: (jobId: string, modelId: string, image: string, resolution: string, lowRam: boolean) =>
    invoke<string[]>("upscale", { jobId, modelId, image, resolution, lowRam }),
  generateVideo: (args: {
    jobId: string; modelId: string; prompt: string; negativePrompt: string | null;
    width: number; height: number; frames: number; fps: number; steps: number;
    guidance: number | null; seed: number; firstFrame: string | null;
  }) => invoke<string[]>("generate_video", args),

  vaultStatus: () => invoke<VaultStatus>("vault_status"),
  vaultCreate: (passphrase: string, enableBiometry: boolean) =>
    invoke<VaultStatus>("vault_create", { passphrase, enableBiometry }),
  vaultUnlockPassphrase: (passphrase: string) =>
    invoke<VaultStatus>("vault_unlock_passphrase", { passphrase }),
  vaultUnlockBiometry: () => invoke<VaultStatus>("vault_unlock_biometry"),
  vaultLock: () => invoke<VaultStatus>("vault_lock"),
  vaultEnableBiometry: () => invoke<VaultStatus>("vault_enable_biometry"),
  vaultDisableBiometry: () => invoke<VaultStatus>("vault_disable_biometry"),
  vaultChangePassphrase: (current: string, next: string) =>
    invoke<void>("vault_change_passphrase", { current, next }),
  vaultList: () => invoke<VaultItem[]>("vault_list"),
  vaultRepair: () => invoke<RepairReport>("vault_repair"),
  vaultDelete: (id: string) => invoke<void>("vault_delete", { id }),
  vaultExport: (id: string, dest: string, overwrite = false) =>
    invoke<number>("vault_export", { id, dest, overwrite }),
  vaultImport: (source: string, kind: "image" | "any") =>
    invoke<string>("vault_import", { source, kind }),
  vaultImportBytes: (data: Uint8Array, name: string, mime: string, kind: string) =>
    invoke<string>("vault_import_bytes", {
      data: Array.from(data), name, mime, kind,
    }),
};

export const onSetupProgress = (cb: (p: SetupProgress) => void): Promise<UnlistenFn> =>
  listen<SetupProgress>("setup://progress", (e) => cb(e.payload));

export const onEngineProgress = (cb: (p: EngineProgress) => void): Promise<UnlistenFn> =>
  listen<EngineProgress>("engine://progress", (e) => cb(e.payload));

export interface PreviewFrame {
  jobId: string;
  /** Data URL, ready to drop straight into an <img>. */
  src: string;
  step: number | null;
  totalSteps: number | null;
}

/**
 * Partial frames during a run.
 *
 * A separate channel from progress so that a listener interested only in
 * "how far along" is not woken for every decoded image.
 */
export const onEnginePreview = (cb: (f: PreviewFrame) => void): Promise<UnlistenFn> =>
  listen<{ id: string; jpeg: string; step: number | null; total_steps: number | null }>(
    "engine://preview",
    (e) => cb({
      jobId: e.payload.id,
      src: `data:image/jpeg;base64,${e.payload.jpeg}`,
      step: e.payload.step,
      totalSteps: e.payload.total_steps,
    })
  );

export const onEngineExit = (cb: () => void): Promise<UnlistenFn> =>
  listen("engine://exit", () => cb());

export interface EngineLog {
  level: "info" | "warn" | "error";
  message: string;
  at: number;
}

/**
 * Engine diagnostics. The worker reports when it silently degrades — dropping
 * a parameter a route rejected, reusing a resident model, sealing an output —
 * and without somewhere to show it that information is simply lost.
 */
export const onEngineLog = (cb: (l: EngineLog) => void): Promise<UnlistenFn> =>
  listen<{ level?: string; message?: string }>("engine://log", (e) =>
    cb({
      level: (e.payload.level as EngineLog["level"]) ?? "info",
      message: e.payload.message ?? "",
      at: Date.now(),
    })
  );

export const onEngineStderr = (cb: (line: string) => void): Promise<UnlistenFn> =>
  listen<string>("engine://stderr", (e) => cb(e.payload));

/**
 * URL for a vault item. The Rust handler decrypts into memory and streams the
 * bytes back, so nothing is ever written to disk in readable form.
 * Requests are refused while the vault is locked.
 */
export const vaultUrl = (id: string) => `vault://localhost/${encodeURIComponent(id)}`;

export const newJobId = () =>
  (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2));

export function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), u.length - 1);
  return `${(n / 1024 ** i).toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
}

export function fmtDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}
