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

  /**
   * How this was actually made.
   *
   * `model` is a display name and can be renamed out from under a saved run;
   * everything below is the run itself, as the engine received it. All of it
   * is optional because every item written before these were recorded is still
   * in the vault and must still load -- a missing field means "not known",
   * never "was off".
   */
  model_repo?: string | null;
  quantize?: number | null;
  negative_prompt?: string | null;
  /** Adapter handles paired with the strengths they ran at. */
  loras?: [string, number][];
  /** The engine's two image-to-image routes: "edit" or "latent". */
  i2i_mode?: string | null;
  image_strength?: number | null;
  /** CSS-like padding for outpainting, e.g. "10%,25%,10%,25%". */
  outpaint_padding?: string | null;
  outpaint_fill?: string | null;
  low_ram?: boolean;
  /** Which engine produced it, for runs the router could have sent elsewhere. */
  engine?: string | null;
  /** Working material that was never meant to outlive the run that made it. */
  transient?: boolean;
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
  /** For display. A model can be renamed; this is only what it was called. */
  modelName: string;
  /**
   * The repository the picture actually came from.
   *
   * Names are not identity. Two installs can show the same name, and a custom
   * model can be renamed after the fact, either of which would quietly
   * reproduce a run against different weights. Optional: items recorded before
   * the repo was stored have only the name.
   */
  modelRepo?: string | null;
  seed: number;
  width: number | null;
  height: number | null;
  steps: number | null;
  guidance: number | null;
  inputs: string[];
  /** Same seed reproduces it; a new seed varies it. */
  reuseSeed: boolean;

  /**
   * The rest of what decided the picture.
   *
   * These were dropped on the floor until now: the recipe carried ten fields
   * and "use these settings again" then ran with whatever the controls
   * happened to be showing, so a reproduction of a reinterpret came out as an
   * instruct edit and a negative prompt vanished without a word. Undefined
   * means the item predates the recording and the control must be left alone.
   */
  negativePrompt?: string | null;
  loras?: [string, number][];
  /** Engine-side: "edit" or "latent". See `editKindFor` for the UI mapping. */
  i2iMode?: string | null;
  imageStrength?: number | null;
  outpaintPadding?: string | null;
  outpaintFill?: string | null;
  lowRam?: boolean;
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
  /**
   * Why this repository holds no adapter, when it does not.
   *
   * Read from the file's own header before the download is offered. Nothing
   * checked before: a VAE repository was indistinguishable from an adapter
   * repository right up to the point where the adapter changed nothing.
   */
  not_an_adapter?: string;
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
  /**
   * What this panel is for, which is not the same question as how far away
   * the camera is. A close-up of a face registering bad news and a close-up of
   * a dripping tap want opposite treatment: one needs the character held
   * exactly, the other must not have her anywhere near it.
   */
  purpose?: "establishing" | "action" | "reaction" | "detail" | "dialogue";
  /** Which continuous stretch of story this belongs to. Pages break here. */
  scene?: number;
  scene_title?: string;
  /** What is spoken aloud in this panel. Usually empty. */
  dialogue?: { speaker: string; text: string }[];
  /** The scene's location, settled once and shared by every panel in it.
   *  Without this each panel invented its own version of the same room. */
  place?: string;
  /** Which place this is, by identity. Panels sharing a key share one drawn
   *  room, so returning to the kitchen looks like returning to it -- and a
   *  story the writer left in one scene still changes rooms when it moves. */
  place_key?: string;
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
  /**
   * The story is told by someone it never names.
   *
   * A first-person story has no name in the prose for the person it is about,
   * so the cast reader has nothing to return for the lead -- and with no lead
   * there is no reference sheet, which is the thing that keeps a face the same
   * face across panels. The interface asks for a name instead of inventing
   * one. Optional: a draft saved before this does not carry it.
   */
  first_person?: boolean;
}

/** What the prompt enhancer did, said plainly rather than inferred. */
export type AssistOutcome =
  | "proposal"          // it has a suggestion for you
  | "already_specific"  // your words were fine as they were
  | "rewrite_rejected"  // it tried, and what it wrote was worse
  | "unclear";          // the request names nothing that could be drawn

export interface AssistResult {
  prompt: string;
  original: string;
  outcome?: AssistOutcome;
  changed?: boolean;
  /** Significant words the rewrite introduced. */
  added?: string[];
  /** Words the rewrite no longer says, having rephrased around them. */
  dropped?: string[];
  /** Spellings the rewrite fixed, as [typed, written] pairs. */
  corrected?: [string, string][];
  /** Empty quality labels it stripped out. */
  removed?: string[];
  /** What it wrote, when that was rejected — shown so you can judge it. */
  attempt?: string;
  rejected_because?: string;
  saw_image?: boolean;
  description?: string;
  unclear?: boolean;
  note?: string;
}
