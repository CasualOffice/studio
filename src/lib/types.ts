export type Task = "text_to_image" | "edit" | "upscale" | "video" | "assist";
export type Fit = "good" | "tight" | "too_much_memory" | "too_much_disk" | "broken";

export interface HostInfo {
  chip: string;
  arch: string;
  total_ram_gib: number;
  usable_ram_gib: number;
  /** Largest model this Mac can run, in billions of parameters. */
  max_params_4bit: number;
  max_params_8bit: number;
  max_params_bf16: number;
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
  /** Needs a Hugging Face token: the repository is behind a licence. */
  gated: boolean;
  /** Whether a video model can start from a still picture. */
  video_from_image: boolean;
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
  /** Downloads only. Measured engine-side, where the bytes actually land. */
  bytes_per_second?: number | null;
  eta_seconds?: number | null;
  /** How long since the byte count last moved. A download that quietly
   *  stopped is otherwise indistinguishable from a slow one. */
  stalled_seconds?: number | null;
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
  /** The board this belongs to. Members are one entry in the library. */
  project?: string | null;
  project_name?: string | null;
  project_index?: number | null;
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
  /** The board this output belongs to, if any. */
  project?: string | null;
  project_name?: string | null;
  project_index?: number | null;
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
  /** Router family, detected if the resolver could not infer it. */
  family: string | null;
  /** Names an mflux backend for FLUX.1, which the unified router cannot place. */
  backend: string | null;
  /** "lora" when the repo is an adapter rather than a whole model. */
  kind: string;
  /** Gated, and no access token is stored: the download cannot start. */
  needs_token: boolean;
  error: string | null;
  /** Verdict computed against this machine before anything is downloaded. */
  package_gib: number;
  peak_gib: number;
  fit: Fit;
  fit_reason: string;
  /** Smallest Mac memory configuration that would run it, if not this one. */
  required_ram_gib: number | null;
  /** Parameter count, when the repository reports one. */
  params: number;
  max_params_4bit: number;
  max_params_8bit: number;
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

/** One frame of a picture board, as the writer divided the story. */
export interface Panel {
  shot: "wide" | "medium" | "close-up";
  subject: string;
  action: string;
  setting: string;
  /** Whether the person we follow is visible in this panel. */
  character_in_frame: boolean;
  /** Story names of every person visible in this frame. */
  characters?: string[];
  /** Exact prose fragment this panel covers, used to audit story coverage. */
  source?: string;
  /** One line of narration under the panel. Empty when it needs no words. */
  caption: string;
  /** Which continuous stretch of story this belongs to. Pages break here. */
  scene?: number;
  scene_title?: string;
  /** What is spoken aloud in this panel. Usually empty. */
  dialogue?: { speaker: string; text: string }[];
  /** The scene's location, settled once and shared by every panel in it.
   *  Without this each panel invented its own version of the same room. */
  place?: string;
  /** The worked-up description: surface, background, light. Empty until the
   *  panel has been through the enrich stage, and the prompt falls back to
   *  the terse fields when it is. Distinct from `scene`, which is a number. */
  description?: string;
}

/** Cached weight files nothing points at, left by interrupted downloads. */
export interface Orphans {
  bytes: number;
  files: number;
  /** Which repositories hold them, largest first. */
  repos: [string, number][];
}

/** Who is in a story and where it happens, read out of the prose itself. */
export interface CastMember {
  name: string;
  description: string;
  /** How often the story refers to them. Drives the tier. */
  mentions: number;
  /** 1 gets a full character sheet, 2 a lighter reference, 3 a name only. */
  tier: number;
}

export interface CastPlace {
  name: string;
  description: string;
  mentions: number;
}

export interface Cast {
  people: CastMember[];
  places: CastPlace[];
  words: number;
}
