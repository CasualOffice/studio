export type Task = "text_to_image" | "edit" | "upscale" | "video" | "assist";
export type Fit = "good" | "tight" | "too_much_memory" | "too_much_disk" | "broken";

export interface HostInfo {
  chip: string;
  arch: string;
  total_ram_gib: number;
  usable_ram_gib: number;
  free_disk_gib: number;
  total_disk_gib: number;
  disk_headroom_gib: number;
  apple_silicon: boolean;
}

export interface InstallStamp {
  python_version: string;
  mlxgen_version: string;
  installed_at: string;
}

export interface SetupState {
  ready: boolean;
  python_present: boolean;
  venv_present: boolean;
  engine_present: boolean;
  stamp: InstallStamp | null;
  root: string;
}

export interface SetupProgress {
  step: string;
  detail: string;
  progress: number | null;
  done: boolean;
  error: string | null;
}

export interface ModelStatus {
  id: string;
  repo: string;
  name: string;
  family: string | null;
  tasks: Task[];
  tasks_str: string[];
  quantize: number | null;
  package_gib: number;
  peak_gib: number;
  peak_estimated: boolean;
  steps_default: number;
  /** How many source images an edit may take on this model. */
  max_edit_images: number;
  guidance_default: number;
  /** Distilled/Turbo checkpoints reject guidance above 1.0. */
  guidance_max: number;
  notes: string;
  /** Set when the model cannot run for reasons unrelated to this machine. */
  broken: string | null;
  /** True for models the user added by repo id. */
  custom: boolean;
  installed: boolean;
  installed_bytes: number;
  fit: Fit;
  fit_reason: string;
  /** Peak exceeds the ceiling but the resident weight floor does not. */
  low_ram_may_help: boolean;
  /** Measured ms per denoise step per megapixel, once a run has completed. */
  measured_ms_per_step_mpx: number | null;
  measured_runs: number;
  /** Measured milliseconds to bring the weights into memory. */
  measured_load_ms: number;
  /** Weights alone exceed memory; no cache trimming can recover this. */
  hopeless: boolean;
}

export interface EngineProgress {
  job_id: string;
  phase: string;
  progress: number | null;
  step: number | null;
  total_steps: number | null;
  message: string | null;
  done_bytes: number | null;
  total_bytes: number | null;
  seed: number | null;
  item_index: number | null;
  item_count: number | null;
}

export interface VaultStatus {
  exists: boolean;
  unlocked: boolean;
  biometry_available: boolean;
  biometry_enrolled: boolean;
  via_biometry: boolean;
  item_count: number;
  unlocked_seconds: number;
  root: string;
}

export interface VaultItem {
  id: string;
  kind: string;
  name: string;
  mime: string;
  bytes: number;
  model: string;
  prompt: string;
  seed: number;
  width: number | null;
  height: number | null;
  steps: number | null;
  guidance: number | null;
  /** Ids of other vault items, never filesystem paths. */
  inputs: string[];
  created_at: string;
  duration_ms: number;
}

export interface GenerateArgs {
  job_id: string;
  model_id: string;
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number | null;
  seed: number;
  count: number;
  images: string[];
  image_strength: number | null;
  i2i_mode: string | null;
  low_ram: boolean;
  /** Show the picture forming, step by step. */
  preview: boolean;
  cache_limit_gb: number | null;
  allow_over_budget: boolean;
  /** Adapter handles paired with their strengths. */
  loras: [string, number][];
  /** Vault id of a painted mask: white where the model may change things. */
  mask: string | null;
  /** CSS-like padding for outpainting, e.g. "10%,25%,10%,25%". */
  outpaint_padding: string | null;
  outpaint_fill: string | null;
}

export interface AppErrorShape {
  kind: string;
  message: string;
}

export interface ResolvedModel {
  model: string;
  modes: string[];
  tasks: string[];
  bytes: number;
  private: boolean;
  gated: boolean;
  routable: boolean;
  error: string | null;
  /** Verdict computed against this machine before anything is downloaded. */
  package_gib: number;
  peak_gib: number;
  fit: Fit;
  fit_reason: string;
  /** Smallest Mac memory configuration that would run it, if not this one. */
  required_ram_gib: number | null;
}

export interface StorageInfo {
  models_root: string;
  models_bytes: number;
  volume_free_gib: number;
  is_external: boolean;
}

export interface RepairReport {
  recovered: number;
  dropped: number;
  unreadable: number;
}

/**
 * Enough of a past run to reproduce or vary it.
 *
 * Everything here was already recorded when the item was made, so recreating
 * costs nothing extra to store.
 */
export interface Recipe {
  kind: string;
  prompt: string;
  modelName: string;
  seed: number;
  width: number | null;
  height: number | null;
  steps: number | null;
  guidance: number | null;
  inputs: string[];
  /** Same seed reproduces it; a new seed varies it. */
  reuseSeed: boolean;
}

export interface Lora {
  handle: string;
  repo: string;
  name: string;
  bytes: number;
  scale: number;
  installed: boolean;
}

export interface ResolvedLora {
  repo: string;
  files: { name: string; bytes: number }[];
  bytes: number;
  gated: boolean;
  base_model: string | null;
}
