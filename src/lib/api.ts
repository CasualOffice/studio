import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  EngineProgress, GenerateArgs, HostInfo, ModelStatus,
  RepairReport, ResolvedModel, SetupProgress, SetupState, StorageInfo, VaultItem, VaultStatus,
} from "./types";

/** Tauri rejects with our structured error; normalise it to a message. */
export function errText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e && typeof e === "object" && "message" in e) {
    return String((e as { message: unknown }).message);
  }
  return String(e);
}

export const api = {
  hostInfo: () => invoke<HostInfo>("host_info"),
  setupState: () => invoke<SetupState>("setup_state"),
  runSetup: (force = false) => invoke<void>("run_setup", { force }),

  listModels: () => invoke<ModelStatus[]>("list_models"),
  storageInfo: () => invoke<StorageInfo>("storage_info"),
  setModelsLocation: (path: string, moveExisting: boolean) =>
    invoke<StorageInfo>("set_models_location", { path, moveExisting }),
  downloadModel: (modelId: string, jobId: string) =>
    invoke<unknown>("download_model", { modelId, jobId }),
  deleteModel: (modelId: string) => invoke<number>("delete_model", { modelId }),
  resolveModel: (repo: string) => invoke<ResolvedModel>("resolve_model", { repo }),
  addCustomModel: (
    repo: string, name: string, tasks: string[], bytes: number, quantize: number | null
  ) => invoke<void>("add_custom_model", { repo, name, tasks, bytes, quantize }),
  removeCustomModel: (modelId: string) =>
    invoke<void>("remove_custom_model", { modelId }),
  unloadModel: () => invoke<void>("unload_model"),
  enginePing: () => invoke<Record<string, unknown>>("engine_ping"),
  cancelJob: (jobId: string) => invoke<void>("cancel_job", { jobId }),

  assistPrompt: (jobId: string, prompt: string, mode: string, images: string[]) =>
    invoke<{ prompt: string; original: string; saw_image: boolean }>(
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
  vaultExport: (id: string, dest: string) =>
    invoke<number>("vault_export", { id, dest }),
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
