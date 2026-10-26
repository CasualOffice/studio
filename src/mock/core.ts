/**
 * A stand-in Tauri host for `vite --mode mock`: the real UI in a browser with
 * sample data, so screens can be looked at and screenshotted without an
 * engine, a vault or any private content. Never part of the app build.
 */
const now = Date.now();
const iso = (daysAgo: number) => new Date(now - daysAgo * 864e5).toISOString();

const model = (id: string, name: string, tasks: string[], extra: Record<string, unknown> = {}) => ({
  id, repo: `mock/${id}`, name, family: null, installed: true, installed_bytes: 4e9, hopeless: false,
  fit: "good", fit_reason: "fits comfortably", tasks, tasks_str: tasks, quantize: 4,
  steps_default: 4, guidance_max: 1, guidance_default: 1, max_edit_images: 3,
  package_gib: 4.2, peak_gib: 9.1, peak_estimated: true, gated: false, video_from_image: false,
  notes: "A sample model for the mock host.", broken: null, custom: false, low_ram_may_help: false,
  measured_ms_per_step_mpx: 47000, measured_runs: 12, measured_load_ms: 600, ...extra,
});

const models = [
  model("flux2-klein-4b-4bit", "FLUX.2 Klein 4B", ["text_to_image", "edit"]),
  model("flux2-klein-9b-4bit", "FLUX.2 Klein 9B", ["text_to_image", "edit"], { fit: "tight", fit_reason: "tight on 16 GB" }),
  model("seedvr2-3b-4bit", "SeedVR2 3B", ["upscale"]),
  model("ltx-video", "LTX Video", ["video"], { installed: false, installed_bytes: 0, video_from_image: true }),
  model("qwen3-4b-instruct-4bit", "Story analyst (Qwen3 4B)", ["assist"]),
  model("kokoro-82m", "Fast audiobook voices (Kokoro 82M)", ["speech"], { package_gib: 0.45, peak_gib: 1.5 }),
  model("qwen3-tts-1.7b-custom-6bit", "Audiobook voices (Qwen3-TTS 1.7B)", ["speech"], { installed: false, installed_bytes: 0 }),
];

const prompts = [
  "A lighthouse at dusk over a stormy sea, oil painting", "Portrait of an old harbourmaster, ink and watercolour",
  "A green dory pulled up on a pebble beach at dawn", "Kitchen interior with tar and oranges, warm lamplight",
  "A boy mending a fishing net, graphic novel panel", "Ferry pier in heavy rain, cinematic",
];
const items = [
  ...prompts.map((prompt, i) => ({
    id: `img-${i}`, kind: i % 3 === 1 ? "edit" : "generate", name: `image-${i}.png`, mime: "image/png",
    bytes: 1.2e6, model: "FLUX.2 Klein 4B", prompt, seed: 1000 + i, width: i % 2 ? 768 : 1024, height: 768,
    steps: 4, guidance: 1, inputs: [], created_at: iso(i), duration_ms: 41000,
  })),
  { id: "up-1", kind: "upscale", name: "upscaled.png", mime: "image/png", bytes: 6e6, model: "SeedVR2 3B",
    prompt: "", seed: 0, width: 2048, height: 1536, steps: null, guidance: null, inputs: ["img-0"], created_at: iso(2), duration_ms: 90000 },
  { id: "aud-1", kind: "audiobook", name: "The Ferry.wav", mime: "audio/wav", bytes: 4e7, model: "Kokoro 82M",
    prompt: "The Ferry", seed: 0, width: null, height: null, steps: null, guidance: null, inputs: [], created_at: iso(0),
    duration_ms: 60000, markers: [{ time: 0, label: "The pier" }, { time: 95, label: "The kitchen" }] },
  ...[0, 1, 2].map((p) => ({ id: `panel-${p}`, kind: "edit", name: `panel-${p}.png`, mime: "image/png", bytes: 9e5,
    model: "FLUX.2 Klein 4B", prompt: `Panel ${p + 1}`, seed: p, width: 768, height: 768, steps: 4, guidance: 1,
    inputs: [], created_at: iso(1), duration_ms: 70000, project: "board-1", project_name: "The Ferry", project_index: p })),
];

const source = "The ferry to Hollin Island ran twice a week, and I had missed both.\n\n“You’re the Pell girl,” she said.\n\n“I’m not late,” I said. “The ferry was early.”";
const library = {
  version: 1, activeBookId: "book-1",
  books: [{ version: 1, id: "book-1", characters: [
    { id: "n", name: "Narrator", aliases: [], visual: "", voice: "" },
    { id: "a", name: "Agnes Thorne", aliases: ["Agnes"], visual: "broad, weathered", voice: "gruff, warm" },
    { id: "t", name: "Tam", aliases: [], visual: "a boy of twelve", voice: "quiet" }],
    chapters: [{ id: "ch-1", title: "The Ferry", source, source_hash: "mock", scenes: [{ id: "s1", title: "The pier",
      summary: "", start: 0, end: source.length, beats: [{ id: "b1", start: 0, end: source.length, source,
        importance: 3, tension: 3, action: "Arrival", setting: "pier", mood: "wry", visible: [],
        dialogue: [{ speaker_id: "a", text: "You’re the Pell girl,", offscreen: false, start: 70, end: 91 },
                   { speaker_id: "n", text: "I’m not late,", offscreen: false, start: 104, end: 117 }] }] }] }] }],
  audio: { "book-1": { model_id: "kokoro-82m", narrator: "bf_emma", voices: { a: "hf_alpha", t: "am_puck" },
    rate: 160, outputs: { "ch-1": ["aud-1"] } } },
};

// A board part-way through: cast approved, page one drawn and composed.
const actor = (id: string, name: string, expression: string) =>
  ({ character_id: id, name, visual: "", expression, body_language: "" });
const boardPanel = (n: number, over: Record<string, unknown>) => ({
  id: `bp-${n}`, shot: "MS", angle: "eye", framing: "single", subject: "", action: "", setting: "pier",
  place_key: "pier", caption: "", character_in_frame: true, scene: 1, scene_title: "The pier", mood: "wry",
  chapter_id: "ch-1", beat_id: "b1", weight: 2, dialogue: [], covers: [], folded: [], ...over,
});
const boardDraft = {
  version: 1, story: source, projectId: "board-1", character: "", sheet: null, ownSheet: [], placeSheets: {},
  storyBook: library.books[0], chapterId: "ch-1", format: "comic", pageCount: 2, castApproved: true,
  characterSheets: { a: "sheet-agnes", n: "sheet-narrator" }, pages: ["page-1"], overflow: [],
  cast: { places: [], words: 34, people: [
    { id: "n", name: "Narrator", aliases: [], description: "", outfit: "a yellow oilskin", mentions: 3, tier: 1, recurring: true },
    { id: "a", name: "Agnes Thorne", aliases: ["Agnes"], description: "broad, weathered", outfit: "a navy fisherman's jumper",
      mentions: 2, tier: 2, recurring: true },
    { id: "t", name: "Tam", aliases: [], description: "a boy of twelve", outfit: "", mentions: 0, tier: 3, recurring: false }] },
  panels: [
    boardPanel(0, { shot: "EWS", angle: "high", framing: "establishing", weight: 3, character_in_frame: false, cast: [],
      description: "The ferry pier in the rain, the boat already leaving. Nobody in view.",
      covers: ["The ferry to Hollin Island ran twice a week, and I had missed both."],
      caption: "The ferry to Hollin Island ran twice a week, and I had missed both." }),
    boardPanel(1, { shot: "MCU", framing: "ots", focus: "Agnes Thorne", cast: [actor("a", "Agnes Thorne", "amused"), actor("n", "Narrator", "stung")],
      characters: ["Agnes Thorne", "Narrator"], positions: { "Agnes Thorne": "left third", Narrator: "right third" },
      description: "Agnes Thorne sizes up the narrator, at the pier.",
      dialogue: [{ speaker: "Agnes Thorne", speaker_id: "a", text: "You’re the Pell girl," }] }),
    boardPanel(2, { shot: "MS", framing: "two", cast: [actor("n", "Narrator", "defiant"), actor("a", "Agnes Thorne", "")],
      characters: ["Narrator", "Agnes Thorne"], positions: { "Agnes Thorne": "left third", Narrator: "right third" },
      description: "The narrator squares up to Agnes, at the pier.",
      dialogue: [{ speaker: "Narrator", speaker_id: "n", text: "I’m not late," }, { speaker: "Narrator", speaker_id: "n", text: "The ferry was early." }] }),
    boardPanel(3, { shot: "CU", cast: [actor("a", "Agnes Thorne", "laughing")], characters: ["Agnes Thorne"],
      positions: { "Agnes Thorne": "left third" }, description: "Agnes laughs, at the pier." }),
    boardPanel(4, { scene: 2, shot: "WS", weight: 3, cast: [actor("n", "Narrator", "tired")], characters: ["Narrator"],
      positions: { Narrator: "right third" }, description: "The narrator alone on the pier as night falls." }),
  ],
  drawn: ["art-1", "art-2", "art-3", "art-4", null], redraws: [0, 0, 0, 0, 0], notes: {}, variants: { 1: ["art-old"] },
  coverage: { covered: 4, total: 5, percent: 80, missing: ["“The ferry was early.”"], missing_total: 1,
    sentences: ["The ferry to Hollin Island ran twice a week, and I had missed both.", "You’re the Pell girl,",
                "I’m not late,", "The ferry was early.", "She laughed."] },
  style: { id: "ink", version: 1, words: "black ink illustration with grey wash, visible brushwork, high contrast",
           framing: "cropped composition, full bleed artwork, edge to edge" },
};

const answers: Record<string, unknown> = {
  board_state_get: JSON.stringify(boardDraft),
  setup_state: { ready: true, python_present: true, venv_present: true, engine_present: true,
    stamp: { python_version: "3.12.14", mlxgen_version: "0.36.0", installed_at: iso(30) }, root: "/mock" },
  host_info: { chip: "Apple M4", arch: "arm64", total_ram_gib: 16, usable_ram_gib: 12, max_params_4bit: 14,
    max_params_8bit: 7, max_params_bf16: 3, free_disk_gib: 180, total_disk_gib: 460, disk_headroom_gib: 20, apple_silicon: true },
  vault_status: { exists: true, unlocked: true, biometry_available: false, biometry_enrolled: false,
    via_biometry: false, item_count: items.length, unlocked_seconds: 60, root: "/mock/vault" },
  list_models: models,
  vault_list: items,
  story_library_get: JSON.stringify(library),
  audio_voices: { voices: ["hf_alpha", "hf_beta", "hm_omega", "hm_psi", "af_heart", "am_puck", "bf_emma", "bm_george"]
    .map((id) => ({ id, language: { h: "Hindi / Indian English", a: "English (US)", b: "English (UK)" }[id[0]] })) },
  hf_token_status: { present: false, hint: null },
  storage_info: { models_root: "/mock/models", models_bytes: 2.4e10, volume_free_gib: 180, is_external: false },
  // Prompt help, whatever was typed: enough to look at the card.
  assist_prompt: {
    version: 2, outcome: "proposal", changed: true, goal: "enhance", mode: "generate",
    original: "2 red cars with no sunroof, plain white background",
    prompt: "Two red cars side by side, neither with a sunroof, on a plain white background, lit by soft even studio light from above, seen at eye level.",
    target_label: "FLUX.2 Klein 4B", released: "FLUX.2 Klein 4B",
    locked: ["2 red cars", "no sunroof", "red cars", "white background"],
    variants: [
      { kind: "faithful", label: "Faithful", prompt: "Two red cars without a sunroof on a plain white background.",
        additions: [], missing: [], reworded: [] },
      { kind: "enriched", label: "Enriched",
        prompt: "Two red cars side by side, neither with a sunroof, on a plain white background, lit by soft even studio light from above, seen at eye level.",
        additions: ["side by side", "lit by soft even studio light from above", "seen at eye level"], missing: [], reworded: [] },
      { kind: "stylistic", label: "Stylistic",
        prompt: "Two red cars side by side, neither with a sunroof, on a plain white background, lit by soft even studio light from above, seen at eye level. Gouache illustration with flat colour.",
        additions: ["side by side", "lit by soft even studio light from above", "seen at eye level", "Gouache illustration with flat colour"],
        missing: [], reworded: [] },
    ],
    added: [], dropped: [], corrected: [], removed: [], saw_image: false, unclear: false,
    description: "", note: "", attempt: "", rejected_because: "", regions: [], target_region: "",
  },
};

export async function invoke(cmd: string): Promise<unknown> {
  if (cmd in answers) return structuredClone(answers[cmd]);
  if (cmd.startsWith("list_") || cmd.endsWith("_list")) return [];
  return null;
}
export const convertFileSrc = (p: string) => p;
