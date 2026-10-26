/** @vitest-environment jsdom */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { hostAnswers } from "../test/setup";
import Storyboard from "./Storyboard";
import type { ModelStatus, Panel } from "../lib/types";
import type { StoryBook, StoryLibrary } from "../lib/story";
import { lockStyle, STYLES } from "../lib/board";

vi.mock("@tauri-apps/api/webview", () => ({
  getCurrentWebview: () => ({ onDragDropEvent: async () => () => {} }),
}));

const source = "Mira finds the platform empty. Jonas holds the key. “Last train!” a guard called.";
const beat = (id: string, visible: string[], appearance = "") => ({
  id, start: 0, end: source.length, source, importance: 3, tension: 3, action: "a", setting: "station", mood: "",
  visible: visible.map((c) => ({ character_id: c, expression: "", body_language: "", appearance })), dialogue: [],
});
const book: StoryBook = { version: 1, id: "book", characters: [
  { id: "mira", name: "Mira", aliases: ["Mi"], visual: "short black hair", voice: "quiet" },
  { id: "jonas", name: "Jonas", aliases: [], visual: "", voice: "" },
  { id: "guard", name: "The guard", aliases: ["guard"], visual: "", voice: "" },
], chapters: [{ id: "chapter", title: "The reveal", source, source_hash: "hash",
  scenes: [{ id: "scene", title: "Station", summary: "", start: 0, end: source.length,
    beats: [beat("b1", ["mira", "jonas"], "red coat"), beat("b2", ["mira", "jonas"])] }] }] };
const mira = { character_id: "mira", name: "Mira", visual: "short black hair", expression: "grief", body_language: "still" };
const jonas = { character_id: "jonas", name: "Jonas", visual: "", expression: "", body_language: "" };
const panel = (over: Partial<Panel> = {}): Panel => ({
  shot: "MS", subject: "Mira", action: "", setting: "station", caption: "", character_in_frame: true,
  characters: ["Mira"], chapter_id: "chapter", beat_id: "b1", mood: "loss", scene: 1, cast: [mira],
  positions: { Mira: "left third" }, moment: "Mira finds the platform empty.",
  description: "Mira stands on the empty platform", covers: ["Mira finds the platform empty."], ...over,
});
const coverage = { covered: 1, total: 3, percent: 33, missing: [], missing_total: 0,
  sentences: ["Mira finds the platform empty.", "Jonas holds the key.", "Last train!"] };
const baseModel: ModelStatus = {
  id: "image", repo: "test/image", name: "Image model", family: null,
  installed: true, installed_bytes: 1, hopeless: false, fit: "good", fit_reason: "fits",
  tasks: ["edit", "text_to_image"], tasks_str: [], quantize: 4,
  steps_default: 4, guidance_max: 1, guidance_default: 1, max_edit_images: 1,
  package_gib: 4, peak_gib: 6, peak_estimated: true, gated: false, video_from_image: false,
  notes: "", broken: null, custom: false, low_ram_may_help: false,
  measured_ms_per_step_mpx: null, measured_runs: 0, measured_load_ms: 0,
};
const models: ModelStatus[] = [baseModel,
  { ...baseModel, id: "qwen3.5-9b-story-4bit", name: "Analyst", tasks: ["assist"] }];
const notify = vi.fn();
const save = vi.fn(async (_book: StoryBook) => {});
type Args = { args: Record<string, unknown> };
afterEach(cleanup);
beforeEach(() => {
  vi.clearAllMocks(); localStorage.clear();
  hostAnswers({ board_state_get: null, board_state_previous: null, board_state_set: null, vault_delete: null });
});
const show = (library: StoryLibrary = { version: 1, books: [], activeBookId: null }) =>
  render(<Storyboard models={models} notify={notify} onProduced={() => {}} storyLibrary={library}
                     onStoryAnalyzed={save} onCharacterReference={async () => {}} />);

/** A board saved with its cast approved and some panels drawn. */
function draft(panels: Panel[], drawn: (string | null)[], extra: Record<string, unknown> = {}) {
  return JSON.stringify({
    version: 1, story: source, cast: null, projectId: "project", style: lockStyle(STYLES[0].id), character: "",
    panels: panels.map((p, i) => ({ ...p, id: `p${i}` })), sheet: null, drawn, redraws: [], notes: {}, pages: [],
    ownSheet: [], placeSheets: {}, coverage, storyBook: book, chapterId: "chapter",
    characterSheets: { mira: "sheet-mira", jonas: "sheet-jonas" }, castApproved: true, format: "manga",
    pageCount: 4, ...extra,
  });
}

describe("Make comic: one pass, then the cast", () => {
  it("analyses the story with the page count, then draws one full-body sheet per recurring person and stops for approval", async () => {
    const analyze = vi.fn(() => ({ book, chapter_id: "chapter", coverage,
      panels: [panel(), panel({ cast: [mira, jonas], characters: ["Mira", "Jonas"], beat_id: "b2" })] }));
    const sheets: Record<string, unknown>[] = [];
    const generate = vi.fn((a: Args) => { sheets.push(a.args); return [`sheet-${sheets.length}`]; });
    const edit = vi.fn(() => ["panel"]);
    hostAnswers({ analyze_story: analyze, generate, edit_image: edit });
    show();
    fireEvent.change(await screen.findByRole("textbox", { name: "Manuscript prose" }), { target: { value: source } });
    fireEvent.click(screen.getByRole("button", { name: "Make comic" }));
    await waitFor(() => expect(save).toHaveBeenCalledWith(book));
    expect(analyze).toHaveBeenCalledWith(expect.objectContaining({ args: expect.objectContaining({
      source, pages: 4, format: "manga" }) }));
    await waitFor(() => expect(generate).toHaveBeenCalledTimes(2));
    const cast = await screen.findByRole("region", { name: "Cast" });
    expect(within(cast).getByRole("article", { name: "Mira" })).toBeTruthy();
    expect(within(cast).getByRole("article", { name: "Jonas" })).toBeTruthy();
    // Heard once, never seen: no sheet.
    expect(within(cast).queryByRole("article", { name: "The guard" })).toBeNull();
    const prompts = sheets.map((a) => a.prompt as string);
    expect(prompts[0]).toContain("full body, head to toe");
    expect(prompts[0]).toContain("Wearing red coat");
    // A standing figure: taller than wide, in multiples of 16.
    expect((sheets[0].height as number) > (sheets[0].width as number)).toBe(true);
    expect((sheets[0].width as number) % 16).toBe(0);
    // Seeded by who they are.
    expect(sheets[0].seed).not.toBe(sheets[1].seed);
    // Nothing is drawn before the cast is approved.
    expect(edit).not.toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: /Approve the cast and draw/ })).toBeTruthy();
  });

  it("loads an analysed chapter without analysing it again", async () => {
    const direct = vi.fn(() => ({ panels: [panel()], coverage }));
    hostAnswers({ direct_story: direct });
    show({ version: 1, books: [book], activeBookId: book.id });
    fireEvent.change(await screen.findByLabelText("Use an analysed chapter"), { target: { value: "book:chapter" } });
    expect(await screen.findByText("Loaded “The reveal” — 1 panel planned.")).toBeTruthy();
    expect(direct).toHaveBeenCalledWith(expect.objectContaining({
      args: expect.objectContaining({ book, chapter_id: "chapter", format: "manga", refine: false, pages: 4 }) }));
    expect(save).not.toHaveBeenCalled();
    expect(await screen.findByRole("button", { name: /Draw the cast sheets \(2\)/ })).toBeTruthy();
  });

  it("continues a stopped analysis, keeping what was done", async () => {
    const stub = { id: "stub", title: "Passage 2 (not analysed)", summary: "", start: source.length, end: source.length,
                   beats: [], unanalysed: true };
    const stopped = { ...book, chapters: [{ ...book.chapters[0], scenes: [...book.chapters[0].scenes, stub] }] };
    const resume = vi.fn(() => ({ book, chapter_id: "chapter", panels: [panel(), panel({ beat_id: "b2" })], unanalysed: 0 }));
    hostAnswers({ direct_story: () => ({ panels: [panel()] }), resume_story: resume });
    show({ version: 1, books: [stopped], activeBookId: stopped.id });
    fireEvent.change(await screen.findByLabelText("Use an analysed chapter"), { target: { value: "book:chapter" } });
    expect(await screen.findByText("1 passage not analysed yet")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Continue analysis" }));
    await waitFor(() => expect(save).toHaveBeenCalledWith(book));
    expect(resume).toHaveBeenCalledWith(expect.objectContaining({ args: expect.objectContaining({
      book: stopped, chapter_id: "chapter", again: false }) }));
    expect(await screen.findByText("Analysis complete: 2 panels planned. Drawn panels were kept.")).toBeTruthy();
  });
});

describe("drawing, page by page", () => {
  it("draws page one first at each cell's shape, composes it at once, then carries on", async () => {
    const panels = Array.from({ length: 6 }, (_, i) => panel({ beat_id: `b${i}`, weight: i === 0 ? 3 : 2,
                                                            shot: i === 0 ? "EWS" : "MS" }));
    hostAnswers({ board_state_get: draft(panels, [], { castApproved: false }) });
    const edits: Record<string, unknown>[] = [];
    const composes: Record<string, unknown>[] = [];
    hostAnswers({
      edit_image: (a: Args) => { edits.push(a.args); return [`art-${edits.length}`]; },
      compose_board: (a: { board: Record<string, unknown> }) => {
        composes.push({ ...a.board, after: edits.length });
        return { pages: [`page-${composes.length}`], composition: "c", overflow: [] };
      },
    });
    show();
    fireEvent.click(await screen.findByRole("button", { name: /Approve the cast and draw/ }));
    await waitFor(() => expect(composes).toHaveLength(2));
    // Page one (four panels in manga) was composed before page two was drawn.
    expect(composes[0].after).toBe(4);
    expect(composes[0].pages).toEqual([0, 0, 0, 0]);
    expect(composes[0].direction).toBe("rtl");
    expect((composes[0].cells as number[][]).length).toBe(4);
    expect(composes[1].pages).toEqual([0, 0, 0, 0, 1, 1]);
    // The establishing panel takes a wide cell and is drawn wide, not square.
    expect((edits[0].width as number) > (edits[0].height as number)).toBe(true);
    for (const e of edits) {
      expect((e.width as number) % 16).toBe(0);
      expect((e.height as number) % 16).toBe(0);
      expect(e.mask).toBe(null);
      expect(e.images).toEqual(["sheet-mira"]);
    }
    // Balloons go toward their speaker: no "speaker label" reaches the page.
    expect(await screen.findByText(/Drew 6 panels — 6 of 6 drawn\./)).toBeTruthy();
  });

  it("stops between panels and continues where it stopped", async () => {
    const panels = Array.from({ length: 3 }, (_, i) => panel({ beat_id: `b${i}` }));
    hostAnswers({ board_state_get: draft(panels, []), cancel_job: null,
                  compose_board: () => ({ pages: ["page"], composition: "c", overflow: [] }) });
    let release: (ids: string[]) => void = () => {};
    const edit = vi.fn(() => (edit.mock.calls.length === 1 ? new Promise<string[]>((r) => { release = r; }) : ["art"]));
    hostAnswers({ edit_image: edit });
    show();
    const run = await screen.findByRole("button", { name: /Draw the comic/ });
    run.focus();
    fireEvent.click(run);
    const stop = await screen.findByRole("button", { name: "Stop drawing" });
    expect(stop).toBe(run);
    expect(document.activeElement).toBe(run);
    fireEvent.click(stop);
    release(["art-1"]);
    expect(await screen.findByText(/Stopped — 1 of 3 drawn/)).toBeTruthy();
    expect(edit).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: /Continue drawing · 2 left/ }));
    await waitFor(() => expect(edit).toHaveBeenCalledTimes(3));
  });

  it("draws a cast sheet again when the saved one was deleted from the Vault", async () => {
    const saved = { ...book, references: { mira: { image: "deleted", style_words: lockStyle(STYLES[0].id).words,
      visual: "short black hair", outfit: "red coat", kind: "cast-sheet-v2" as const } } };
    const generate = vi.fn(() => ["new-sheet"]);
    hostAnswers({ direct_story: () => ({ panels: [panel()] }), vault_list: () => [], generate });
    show({ version: 1, books: [saved], activeBookId: saved.id });
    fireEvent.change(await screen.findByLabelText("Use an analysed chapter"), { target: { value: "book:chapter" } });
    fireEvent.click(await screen.findByRole("button", { name: /Draw the cast sheets/ }));
    await waitFor(() => expect(generate).toHaveBeenCalledTimes(2));
  });
});

describe("fixing in place", () => {
  const two = [panel({ dialogue: [{ speaker: "Mira", text: "Gone." }] }), panel({ beat_id: "b2" })];

  it("offers three versions from new seeds and keeps the one picked", async () => {
    hostAnswers({ board_state_get: draft(two, ["art-a", "art-b"]) });
    const seeds: unknown[] = [];
    const edit = vi.fn((a: Args) => { seeds.push(a.args.seed); return [`v${seeds.length}`]; });
    const compose = vi.fn(() => ({ pages: ["page"], composition: "c", overflow: [] }));
    hostAnswers({ edit_image: edit, compose_board: compose });
    show();
    fireEvent.click(await screen.findByRole("button", { name: "Try 3 other versions" }));
    await waitFor(() => expect(edit).toHaveBeenCalledTimes(3));
    expect(new Set(seeds).size).toBe(3);
    fireEvent.click(await screen.findByRole("button", { name: "Use version 2" }));
    await waitFor(() => expect(compose).toHaveBeenCalled());
    expect((screen.getAllByAltText("Panel 1")[0] as HTMLImageElement).src).toContain("v2");
  });

  it("never redraws a picture for a change to its words", async () => {
    hostAnswers({ board_state_get: draft(two, ["art-a", "art-b"], { pages: ["page"] }) });
    const edit = vi.fn(() => ["x"]);
    hostAnswers({ edit_image: edit });
    show();
    fireEvent.change(await screen.findByLabelText("Balloon 1 text"), { target: { value: "It's gone." } });
    fireEvent.change(screen.getByLabelText("Caption"), { target: { value: "At the station." } });
    expect((await screen.findAllByRole("button", { name: "Update the pages" })).length).toBeGreaterThan(0);
    expect(edit).not.toHaveBeenCalled();
  });

  it("recasts one person: only their panels are drawn again", async () => {
    const mixed = [panel(), panel({ beat_id: "b2", cast: [jonas], characters: ["Jonas"], positions: { Jonas: "right third" } })];
    hostAnswers({ board_state_get: draft(mixed, ["art-a", "art-b"]),
                  generate: () => ["sheet-jonas-2"],
                  compose_board: () => ({ pages: ["page"], composition: "c", overflow: [] }) });
    const edit = vi.fn(() => ["redrawn"]);
    hostAnswers({ edit_image: edit });
    show();
    fireEvent.click(await screen.findByRole("button", { name: /^Cast/ }));
    const card = await screen.findByRole("article", { name: "Jonas" });
    fireEvent.click(within(card).getByRole("button", { name: "Draw the sheet again" }));
    fireEvent.click(await screen.findByRole("button", { name: "Approve and redraw 1 panels" }));
    await waitFor(() => expect(edit).toHaveBeenCalledTimes(1));
    expect((edit.mock.calls[0] as unknown as [Args])[0].args.images).toEqual(["sheet-jonas-2"]);
  });

  it("says honestly how much of the story is on the page, and what is not", async () => {
    hostAnswers({ board_state_get: draft([panel()], ["art-a"]) });
    show();
    expect(await screen.findByText("33% of the story is on the page")).toBeTruthy();
    expect(screen.getByText("Jonas holds the key.")).toBeTruthy();
  });
});

describe("Board accessibility", () => {
  it("names the story, the panel editor and the board settings", async () => {
    show();
    expect(await screen.findByRole("textbox", { name: "Manuscript prose" })).toBeTruthy();
    expect(screen.getByLabelText("Model")).toBeTruthy();
    expect(screen.getByLabelText("Pages")).toBeTruthy();
    expect(screen.getAllByRole("button", { pressed: true }).length).toBeGreaterThan(0);
    cleanup();
    hostAnswers({ board_state_get: draft([panel()], [null]) });
    show();
    for (const name of [/^Brief/, "Caption", "Shot", "Mood", "Mira expression"]) {
      expect(await screen.findByLabelText(name)).toBeTruthy();
    }
  });

  it("says why the run button cannot run, and keeps focus on it while it is Stop", async () => {
    const cancel = vi.fn();
    hostAnswers({ analyze_story: () => new Promise(() => {}), cancel_job: cancel });
    show();
    const prose = await screen.findByRole("textbox", { name: "Manuscript prose" });
    const run = screen.getByRole("button", { name: "Make comic" });
    expect(run.getAttribute("aria-disabled")).toBe("true");
    expect(screen.getByText("Write or paste the story first.")).toBeTruthy();
    fireEvent.change(prose, { target: { value: source } });
    expect(run.getAttribute("aria-disabled")).toBeNull();
    run.focus();
    fireEvent.click(run);
    const stop = await screen.findByRole("button", { name: "Stop the analysis" });
    expect(stop).toBe(run);
    expect(document.activeElement).toBe(run);
    fireEvent.click(stop);
    await waitFor(() => expect(cancel).toHaveBeenCalled());
  });

  it("marks the current step and goes back without discarding anything", async () => {
    hostAnswers({ board_state_get: draft([panel()], [null]) });
    show();
    const steps = await screen.findByRole("list", { name: "Board steps" });
    const current = () => steps.querySelector("[aria-current='step']")!;
    expect(current().textContent).toMatch(/Draw/);
    expect(screen.queryByRole("button", { name: /^Pages/ })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^Story/ }));
    expect(current().textContent).toMatch(/Story/);
    expect(screen.getByRole("button", { name: /Draw the comic/ })).toBeTruthy();
  });

  it("opens a board from an older version read-only", async () => {
    const old = { shot: "medium", subject: "Mira", action: "waits", setting: "station", caption: "", character_in_frame: true } as Panel;
    hostAnswers({ board_state_get: JSON.stringify({ version: 1, story: source, cast: null, projectId: "p",
      style: null, character: "", panels: [old], sheet: "s", drawn: ["art"], redraws: [], notes: {}, pages: [],
      ownSheet: [], placeSheets: {}, coverage: null }) });
    show();
    expect(await screen.findByText("This board was made by an older version")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Make the comic again" })).toBeTruthy();
    expect((screen.getByLabelText(/^Brief/) as HTMLTextAreaElement).disabled).toBe(true);
  });
});
