import { ANALYST_IDS, castFromBook, type CharacterReference, type StoryBook, type StoryLibrary } from "../lib/story";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api, errText, isCancelled, newJobId, onEngineProgress, vaultUrl,
         type TextOverflow } from "../lib/api";
import type { Cast, EngineProgress, ModelStatus, Panel } from "../lib/types";
import { ImageDrop, JobProgress } from "./shared";
import MaskCanvas from "./MaskCanvas";
import { loadPref, savePref } from "../lib/prefs";
import { anchorOverflow, artOf, fromArrays, identify, setArt, toArrays,
         edit as editPanel, insert as insertPanel, merge as mergePanelIn,
         move as movePanelIn, remove as removePanel, split as splitPanelIn,
         type BoardDoc, type PanelArt } from "../lib/boardDoc";
import { loadBoardDraft, loadSavedBoard, type BoardDraft, type Coverage }
  from "../lib/boardDraft";
import { CUSTOM_STYLE, STYLES, isKnownStyle, lockCustomStyle, lockStyle,
         panelPrompt, panelReferences, panelSeed, placeKey, placePrompt, identityPrompt, storyContinuityReference,
         placeSeed, previousPanel, reviveLock, sheetPrompt, styleHasMoved,
         styleLabel } from "../lib/board";
import type { StyleLock } from "../lib/board";

/**
 * A story, divided into panels, drawn as one consistent character.
 *
 * The pieces here are proven separately: the writer divides prose reliably,
 * and a character sheet fed back as a reference holds the character across
 * shots. This puts them in order and keeps the one thing that makes the
 * result read as a story rather than a set of unrelated pictures -- the same
 * sheet references every panel, and the same style words lead every prompt.
 */

/**
 * How large each panel is drawn.
 *
 * 512 was chosen because a board is read at a glance, and it was wrong.
 * Measured on FLUX.2 Klein 4-bit, the same prompt and seed down the same
 * reference-conditioned route:
 *
 *     512x512     36s    ~4 GiB    the face came back as a blank shadow
 *     768x768     77s    9.11 GiB  the face came back as a face
 *    1024x1024   164s   13.22 GiB  past the 12 GiB budget
 *
 * A wide shot of a person at 512 does not have the pixels to put a face in,
 * so the panel that establishes who we are following is the one that fails.
 * Twice the time for a panel you can use is not a cost, and 1024 is not
 * available on this machine at all.
 *
 * Memory does not move with the number of references: the same panel drawn
 * against both a character sheet and a room sheet also peaked at 9.11 GiB,
 * and took 130 seconds instead of 77. References cost time, not memory --
 * the same shape the video sizes turned out to have.
 *
 * 512 stays on offer because 9.11 GiB is most of what this Mac has: with a
 * browser open the run can be refused for want of memory, and a smaller panel
 * you can finish beats a larger one you cannot start.
 */
const PANEL_SIZES: [string, number][] = [
  ["Detailed (768px)", 768],
  ["Quick (512px)", 512],
];

type Stage = "idle" | "dividing" | "enriching" | "casting" | "building" | "drawing";

export default function Storyboard({
  models, notify, onProduced, openProject, onOpened, storyLibrary, onStoryAnalyzed, onCharacterReference,
}: {
  models: ModelStatus[];
  storyLibrary: StoryLibrary | null;
  onStoryAnalyzed: (book: StoryBook) => Promise<void>;
  onCharacterReference: (bookId: string, characterId: string, reference: CharacterReference) => Promise<void>;
  notify: (m: string, bad?: boolean) => void;
  onProduced: () => void;
  /** A saved board the Vault asked to reopen, by project id. */
  openProject?: string | null;
  onOpened?: () => void;
}) {
  const usable = useMemo(
    () => models.filter(
      (m) => m.installed && !m.hopeless && m.fit !== "broken"
        && m.tasks.includes("edit" as never)
        && m.tasks.includes("text_to_image" as never)
    ),
    [models]
  );
  const writerReady = useMemo(
    () => models.some((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless),
    [models]
  );

  const [modelId, setModelId] = useState("");
  const model = usable.find((m) => m.id === modelId) ?? usable[0];

  const [story, setStory] = useState("");
  const [storyBook, setStoryBook] = useState<StoryBook | null>(null);
  const storyBookRef = useRef<StoryBook | null>(null);
  useEffect(() => { storyBookRef.current = storyBook; }, [storyBook]);
  const ignoreSavedReferences = useRef(false);
  const [chapterId, setChapterId] = useState<string | null>(null);
  const [chapterTitle, setChapterTitle] = useState("Chapter 1");
  const [appendChapter, setAppendChapter] = useState(false);
  const [continuationBookId, setContinuationBookId] = useState("");
  const [analystId, setAnalystId] = useState("");
  const [characterSheets, setCharacterSheets] = useState<Record<string, string>>({});
  const characterSheetsRef = useRef<Record<string, string>>({});
  const referenceCache = useRef<Record<string, string>>({});
  const clearCharacterSheets = () => {
    characterSheetsRef.current = {}; setCharacterSheets({});
    referenceCache.current = {};
  };

  // The cast, read out of the story rather than typed in. `null` means the
  // story has not been read yet.
  const [cast, setCast] = useState<Cast | null>(null);
  /**
   * How many panels one press draws.
   *
   * Drawing a whole board is twenty to sixty minutes, and the style is either
   * right or wrong within the first two panels. Committing to all of it before
   * seeing any of it is the expensive way to find out. A batch is the unit of
   * commitment: draw four, look, fix what is wrong, draw four more.
   */
  const [batch, setBatch] = useState(() => loadPref("boardBatch", 4));
  /** Panel size, remembered. See PANEL_SIZES for what the choice costs. */
  const [panelPx, setPanelPx] = useState(() => {
    const saved = loadPref("boardPanelPx", 768);
    return PANEL_SIZES.some(([, px]) => px === saved) ? saved : 768;
  });
  /**
   * The id every panel of this board is stamped with.
   *
   * It is what makes the board one entry in the library instead of a pile of
   * loose pictures sharing a timestamp. Kept across restarts with the rest of
   * the board, and replaced when a new story is divided.
   */
  const [projectId, setProjectId] = useState<string | null>(null);


  /**
   * The words this board is drawn from, not the name of a preset.
   *
   * The remembered id is only the starting point for a board that has not
   * pinned one yet. Once pinned it travels in the encrypted draft, so a board
   * half-drawn under one build is finished under the same words -- and chapter
   * eleven can be made to match chapter one however the presets move.
   */
  const [style, setStyle] = useState<StyleLock>(
    () => lockStyle(loadPref("boardStyle", STYLES[0].id)));
  const [character, setCharacter] = useState("");

  /**
   * The board's panels and the work attached to them, by identity.
   *
   * This used to be five arrays lined up by position -- panels, drawn,
   * redraws, notes, variants -- and position is not identity. Nothing could
   * move: reordering two panels swapped their pictures and notes, inserting
   * one at the top shifted every note down by one, and deleting one moved the
   * whole board up. The arrays stayed the right length, so nothing was
   * detectably wrong until someone read the pages.
   *
   * The arrays below are derived from it, so every place that reads
   * `drawn[i]` is unchanged. Nothing is written by position any more.
   */
  const [doc, setDoc] = useState<BoardDoc>({ panels: [], art: {} });
  /**
   * Whether this story has been divided at all.
   *
   * Distinct from having no panels. `null` panels is the state before the
   * board has been planned, and it is what gates the whole workflow; an empty
   * division would be a different and much stranger thing.
   */
  const [divided, setDivided] = useState(false);
  const panels = useMemo(
    () => (divided ? doc.panels : null), [divided, doc.panels]);
  const { drawn, redraws, notes, variants } = useMemo(
    () => toArrays(doc), [doc]);
  /** One panel's work, by the position the interface is showing it at. */
  const artAt = (i: number) => artOf(doc, doc.panels[i]?.id ?? "");
  /** Change one panel's work, found by position and stored by identity. */
  const putArt = (i: number, patch: Partial<PanelArt>) =>
    setDoc((d) => (d.panels[i] ? setArt(d, d.panels[i].id, patch) : d));
  const [sheet, setSheet] = useState<string | null>(null);
  /**
   * The same list, readable inside a running batch.
   *
   * Panels are drawn in a loop and each one asks what the panel before it
   * looked like. React state does not update until the loop yields, so every
   * panel in a batch would see the list as it was when the batch started and
   * the continuity reference would always be missing.
   */
  const drawnRef = useRef<(string | null)[]>([]);
  useEffect(() => { drawnRef.current = drawn; }, [drawn]);
  /** The distinct scenes the panels fall into, in order. */
  const scenes = useMemo(() => {
    const seen: number[] = [];
    for (const p of panels ?? []) {
      const n = Number(p.scene ?? 1) || 1;
      if (!seen.includes(n)) seen.push(n);
    }
    return seen;
  }, [panels]);

  /**
   * Who the panels are drawn against.
   *
   * What you typed wins, because it is the most direct statement of intent.
   * Failing that, the story's own lead -- the most-mentioned person, with the
   * description the story gave them. Failing that, nothing, and the panels are
   * drawn from their own subjects. Shared by the run and by redraw, so a
   * redrawn panel is cast exactly like the one it replaces.
   */
  const lead = useMemo(() => cast?.people.find((p) => p.tier === 1), [cast]);
  /**
   * One person, as the prompt should name them.
   *
   * A person the story never describes is named and nothing more. This used to
   * append "appearance not specified", which is not a fact about the story --
   * it is the tool's own bookkeeping, in English, in the prompt, and the
   * generator drew it as though it had been asked for something. The name on
   * its own is the truthful instruction: the drawer invents a face either way,
   * and the reference sheet is what keeps it the same face.
   */
  const personBrief = (name: string) => {
    const person = cast?.people.find((p) => p.name.toLowerCase() === name.toLowerCase());
    if (!person) return name;
    if (person === lead && character.trim()) return `${person.name}: ${character.trim()}`;
    return person.description.trim()
      ? `${person.name}: ${person.description.trim()}`
      : person.name;
  };
  const castForSheet = useMemo(() => {
    const major = cast?.people.filter((p) => p.tier <= 2).slice(0, 4) ?? [];
    if (major.length === 0) return character.trim();
    return major.map((p) => {
      if (p === lead && character.trim()) return `${p.name}: ${character.trim()}`;
      return p.description.trim() ? `${p.name}: ${p.description.trim()}` : p.name;
    }).join("\n");
  }, [cast, character, lead]);
  /**
   * How many people get a full description in one panel's prompt.
   *
   * Every named character's whole appearance brief was joined with "; ", so a
   * panel with four people carried four wardrobes. The drawer blends them or
   * drops some, and with one reference slot there is no way to anchor four
   * faces anyway. Beyond this many, the rest are named and not described --
   * which is what a script does when a room is full.
   */
  const DESCRIBED_PER_PANEL = 2;

  const panelWho = (panel: Panel) => {
    const all = panel.characters ?? [];
    const named = all.slice(0, DESCRIBED_PER_PANEL)
      .map(personBrief).filter(Boolean);
    const rest = all.slice(DESCRIBED_PER_PANEL)
      .map((n) => n.trim()).filter(Boolean);
    if (named.length > 0) {
      return rest.length ? `${named.join("; ")}; also ${rest.join(", ")}`
                         : named.join("; ");
    }
    return panel.character_in_frame
      ? (character.trim() || lead?.description || lead?.name || "") : "";
  };

  /** What to call this board in the library: the story's own opening. */
  const projectName = useMemo(() => {
    const first = story.trim().split(/\s+/).slice(0, 7).join(" ");
    return first ? (story.trim().length > first.length ? first + "\u2026" : first)
                 : "Picture board";
  }, [story]);

  /** Which panel the notes rail is showing. The board owns the selection. */
  const [selected, setSelected] = useState(0);
  /**
   * Which half of the rail is showing: the selected panel, or the board.
   *
   * The rail used to swap to per-panel controls the instant a division
   * existed, which put the three board-wide decisions -- the look, the cast
   * sheet, the rooms -- out of reach for the whole rest of the job. The only
   * way back to them was `Back to the prose`, which throws away the division,
   * the briefs and every note. So the two failures this tool calls fatal were
   * the two a person could not fix without losing everything else.
   */
  const [railTab, setRailTab] = useState<"panel" | "board">("panel");
  /** A style change that would discard drawn work, waiting on a second press. */
  const [confirmStyle, setConfirmStyle] = useState<string | null>(null);
  /** The look the user is describing for themselves, and whether it is open. */
  const [customOpen, setCustomOpen] = useState(false);
  const [customWords, setCustomWords] = useState("");
  /** A name for a narrator the story never names. */
  const [narratorName, setNarratorName] = useState("");
  const [narratorLook, setNarratorLook] = useState("");
  /** Which cast member's appearance is open for correcting. */
  const [editingPerson, setEditingPerson] = useState<string | null>(null);

  /**
   * Put someone into the cast by hand.
   *
   * The cast is read out of the prose, which is right -- asking someone to
   * type what chapter one already says is the tool failing at its job. But a
   * story told in the first person names nobody, and a story that calls
   * someone "her sister" gives the drawer nothing to hold on to, and there was
   * no way to say who they are. So the read is a proposal that can be
   * corrected, which is what it was always supposed to be.
   *
   * `mentions` is set above the current top so the named person leads and gets
   * the full sheet: they are the one the story is about, which is precisely
   * why the prose never stopped to name them.
   */
  /**
   * Correct what someone looks like.
   *
   * The cast reader is asked for appearance and told to leave it empty when
   * the story never says. It does neither: for a story that does not describe
   * its people it returns their part in the plot -- "She is the one who finds
   * the letter" -- and that goes into every panel as drawing instructions,
   * where it means nothing. A 4B model will not reliably tell appearance from
   * narration, and no code-side test can either: a correct description is
   * drawn from the prose too, so anything that rejects narration for
   * overlapping with the story would reject the right answer just as often.
   *
   * So it is editable. The read stays a proposal, which is what it was always
   * meant to be, and the one judgement a person can make instantly -- is this
   * what they look like? -- is the one the tool stops guessing at.
   */
  const describePerson = (name: string, look: string) => {
    clearCharacterSheets();
    const currentBook = storyBookRef.current;
    const personId = cast?.people.find((p) => p.name === name)?.id;
    if (currentBook && personId) {
      const updated = { ...currentBook, characters: currentBook.characters.map((c) =>
        c.id === personId ? { ...c, visual: look } : c) };
      storyBookRef.current = updated; setStoryBook(updated);
      void onStoryAnalyzed(updated).catch((error) => notify(`Could not save the character bible: ${errText(error)}`, true));
    }
    setCast((c) => c && ({
      ...c,
      people: c.people.map((p) => p.name === name
        ? { ...p, description: look } : p),
    }));
    setSheet(null); setPages([]); setOverflow([]);
  };

  const addPerson = (name: string, look: string, lead = false) => {
    clearCharacterSheets();
    const clean = name.trim();
    if (!clean) return;
    setCast((c) => {
      const base: Cast = c ?? { people: [], places: [], words: 0 };
      if (base.people.some((p) => p.name.toLowerCase() === clean.toLowerCase())) {
        notify(`${clean} is already in the cast.`, true);
        return base;
      }
      const top = base.people.reduce((n, p) => Math.max(n, p.mentions), 0);
      const person = {
        name: clean,
        description: look.trim(),
        mentions: lead ? top + 1 : Math.max(1, Math.round(top / 2)),
        tier: lead ? 1 : 2,
      };
      const people = [person, ...base.people]
        .sort((a, b) => b.mentions - a.mentions);
      return { ...base, people };
    });
    // The sheet was drawn against the old cast, so it is not this cast.
    setSheet(null); setPages([]); setOverflow([]);
    setNarratorName(""); setNarratorLook("");
    notify(`${clean} added to the cast. The cast sheet will be drawn again.`);
  };

  const [stage, setStage] = useState<Stage>("idle");
  const [prog, setProg] = useState<EngineProgress | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [done, setDone] = useState(0);
  /** Set when the user cancels, checked between panels. Cancelling the job in
   *  flight only ever stopped one panel; the loop then started the next. */
  const stop = useRef(false);
  const [elapsed, setElapsed] = useState(0);
  const [composing, setComposing] = useState(false);
  const [pages, setPages] = useState<string[]>([]);
  /**
   * Approved text the composed pages could not letter, by board panel index.
   *
   * A cell has a real limit: a 244px cell in a three-panel tier cannot hold
   * three balloons at a readable size, and something has to give. What must
   * not give is the user's knowledge of it -- the composer used to draw what
   * fitted and return a page that looked finished, so a panel arrived in the
   * bound PDF with a speaker missing and nothing anywhere said which one.
   *
   * Cleared wherever `pages` is, because it describes those pages and nothing
   * else. Kept in the draft, because they are kept too.
   */
  const [overflow, setOverflow] =
    useState<(TextOverflow & { panelId?: string })[]>([]);
  /**
   * The overflow report, pointing at where its panels are now.
   *
   * Stored with the identity of the panel that lost text, because a stored
   * position stops being true the moment the board is reordered -- and a
   * warning that names the wrong panel is worse than none.
   */
  const overflowNow = useMemo(
    () => overflow.flatMap((o) => {
      if (!o.panelId) return [o];
      const at = doc.panels.findIndex((x) => x.id === o.panelId);
      return at < 0 ? [] : [{ ...o, panel: at }];
    }),
    [overflow, doc.panels]);
  /** The panel whose region is being painted, and the painted mask itself. */
  /**
   * Why the board is not being written down, when it is not.
   *
   * The whole board -- manuscript, division, notes, character binding -- is
   * one JSON string in one slot. A failed save used to raise a toast and
   * nothing else, so it was gone in a few seconds while the board carried on
   * looking exactly as it had, and the user kept working on something that was
   * not being saved. This stays until a save succeeds.
   */
  const [unsaved, setUnsaved] = useState<string | null>(null);
  /** An earlier state of this board, offered when one is worth offering. */
  const [previous, setPrevious] = useState<Partial<BoardDraft> | null>(null);
  /** Which panel's Remove button is asking to be pressed again. */
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [repairing, setRepairing] = useState<number | null>(null);
  const [maskBytes, setMaskBytes] = useState<Uint8Array | null>(null);
  /** A page sets panels in rows with gutters; a strip stacks them at one
   *  width for scrolling. Different things, and people want both. */
  const [layout, setLayout] = useState(() => loadPref("boardLayout", "page"));
  /** A character picture the user brought, used instead of casting one.
   *  Their own face, an earlier board's sheet, a photograph -- whatever it
   *  is, it anchors the character better than a description can. */
  const [ownSheet, setOwnSheet] = useState<string[]>([]);
  /** One location sheet per scene, keyed by scene number. The character sheet
   *  holds the person; these hold the rooms. */
  /**
   * One drawn room per place, keyed by the place itself.
   *
   * Keyed by scene number before, which meant a story the writer put entirely
   * in scene 1 got one room however far it travelled -- the background never
   * changed -- while the same kitchen in two scenes got two different
   * kitchens. The engine now names each place and every panel carries its
   * key, so a room is drawn once and reused wherever the story returns to it.
   */
  const [placeSheets, setPlaceSheets] = useState<Record<string, string>>({});
  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const timer = useRef<number | null>(null);
  const [draftLoaded, setDraftLoaded] = useState(false);
  /**
   * Which reset is armed, if any.
   *
   * Both resets discard the board, and neither used to ask. "Back to the
   * prose" is a small button directly under the coverage panel -- exactly
   * where someone goes after being told a passage needs review -- and it threw
   * away every drawn panel, every note, the composed pages and the project
   * identity in one click. The pictures survive in the Vault, but nothing
   * loads a Vault project back into a board, and the notes exist nowhere else.
   */
  const [confirmReset, setConfirmReset] = useState<"prose" | "new" | null>(null);


  // A Board contains the manuscript and creative history. Load it from the
  // encrypted vault, migrating old plaintext preferences exactly once.
  /**
   * Put a stored board on screen.
   *
   * Shared by the draft restored at launch and by a comic reopened from the
   * Vault, so the two cannot drift into applying different subsets of it --
   * the sort of difference that surfaces as a board that opens with its notes
   * missing.
   */
  const applyDraft = (draft: Partial<BoardDraft>) => {
        setStory(typeof draft.story === "string" ? draft.story : "");
        setCast(draft.cast ?? null);
        setStoryBook(draft.storyBook ?? null);
        storyBookRef.current = draft.storyBook ?? null;
        ignoreSavedReferences.current = false;
        setChapterId(draft.chapterId ?? null);
        characterSheetsRef.current = draft.characterSheets ?? {};
        setCharacterSheets(draft.characterSheets ?? {});
        referenceCache.current = {};
        setProjectId(draft.projectId ?? null);
        setCharacter(typeof draft.character === "string" ? draft.character : "");
        // A v1 draft holds the parallel arrays; a v2 one holds the same
        // work already keyed by panel id. `fromArrays` reads the arrays by
        // position exactly once -- here, where the positions are still the
        // truth -- and identify() keeps ids a v2 draft already has, so a
        // reopen does not renumber the board and detach every picture.
        setDivided(Array.isArray(draft.panels));
        setDoc(fromArrays(
          Array.isArray(draft.panels) ? draft.panels : null,
          Array.isArray(draft.drawn) ? draft.drawn : [],
          Array.isArray(draft.redraws) ? draft.redraws : [],
          draft.notes ?? {},
          draft.variants ?? {}));
        setSheet(typeof draft.sheet === "string" ? draft.sheet : null);
        setPages(Array.isArray(draft.pages) ? draft.pages : []);
        setOverflow(Array.isArray(draft.overflow) ? draft.overflow : []);
        setOwnSheet(Array.isArray(draft.ownSheet) ? draft.ownSheet : []);
        setPlaceSheets(draft.placeSheets ?? {});
        setCoverage(draft.coverage ?? null);
        // The style the board was drawn under wins over the remembered
        // preset. A draft written before the lock existed has none, so it
        // takes the preset it named -- which is the last moment that
        // substitution is invisible, and the notice below is why it is not.
        const locked = reviveLock(draft.style);
        if (locked) {
          setStyle(locked);
          if (locked.id === CUSTOM_STYLE) setCustomWords(locked.words);
          if (styleHasMoved(locked) && (draft.drawn ?? []).some(Boolean)) {
            notify(
              `This board is drawn in ${styleLabel(locked.id)} as it was when `
              + "it was started. The preset has changed since, so a new board "
              + "will not match it.",
            );
          }
        } else {
          // The raw remembered id, not `style.id` -- `lockStyle` has already
          // substituted the first preset by the time it is state, so asking
          // the state whether the id was known can only ever answer yes.
          const remembered = loadPref("boardStyle", STYLES[0].id);
          if (!isKnownStyle(remembered)) {
            notify(
              "This board was made in a style this build no longer has "
              + `("${remembered}"), so it is now set to ${STYLES[0].label}. `
              + "Anything already drawn was drawn in the old one.",
              true,
            );
          }
        }
  };

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const draft: Partial<BoardDraft> | null = await loadBoardDraft();
        if (!live) return;
        if (draft) applyDraft(draft);
        // A board that restored empty while an earlier state of it still
        // exists is the shape the destroyed-manuscript bug had. It is not
        // proof of anything -- a board can legitimately be emptied -- so the
        // earlier state is offered rather than applied.
        if (!draft?.story?.trim() && !draft?.panels?.length) {
          try {
            const back = await api.boardStatePrevious();
            if (live && back) {
              const parsed = JSON.parse(back) as Partial<BoardDraft>;
              if (parsed.story?.trim() || parsed.panels?.length) {
                setPrevious(parsed);
              }
            }
          } catch { /* no step back is not an error */ }
        }
        // Arm the autosave only once the restore has actually succeeded. A
        // draft of `null` counts: that is a good read of a board nobody has
        // started yet, and a first-time user still needs saving to work.
        setDraftLoaded(true);
      } catch (e) {
        // Deliberately leaves the autosave disarmed, and says so.
        //
        // This used to be a `finally`, so a failed restore armed the save
        // effect below with every piece of state at its default -- and 350 ms
        // later wrote `{story: "", panels: null, notes: {}}` over the stored
        // draft. A transient read error was enough to destroy the manuscript
        // and every hand-typed note, and the legacy migration then cleared the
        // plaintext copies too, so there was nowhere left to recover from.
        // Losing the session is recoverable; overwriting the only copy is not.
        if (live) {
          notify(
            `Could not restore the encrypted Board draft: ${errText(e)}. `
            + "Nothing will be saved over it -- reopen the app to try again.",
            true,
          );
        }
      }
    })();
    return () => { live = false; };
  }, [notify]);

  useEffect(() => {
    if (!draftLoaded) return;
    const draft: BoardDraft = {
      version: 1, story, cast, projectId, style, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets,
      coverage, overflow, variants, storyBook, chapterId, characterSheets,
    };
    const timeout = window.setTimeout(() => {
      const body = JSON.stringify(draft);
      void api.boardStateSet(body).then(
        () => setUnsaved(null),
        (e) => {
          // A toast is not enough for this one. It is gone in a few seconds
          // and the board goes on looking exactly as it did, so the user keeps
          // working on something that is not being written down -- and finds
          // out when they reopen the app. This stays on screen until a save
          // succeeds.
          const why = errText(e);
          setUnsaved(why);
          notify(`Could not save the encrypted Board draft: ${why}`, true);
        });
      // A second copy under this board's own id, so the comic survives the next
      // story. The draft slot holds the board in front of you and nothing else:
      // dividing a new story overwrote the division, the notes and the
      // character binding of the last one, while its pictures sat in the vault
      // with no way left to reach them.
      if (projectId) {
        void api.boardStateSet(body, `board-${projectId}`).catch(() => {
          /* the live draft saved; the archive can wait for the next edit */
        });
      }
    }, 350);
    return () => window.clearTimeout(timeout);
  }, [draftLoaded, overflow, variants, storyBook, chapterId, characterSheets,
      story, cast, projectId, style, character, panels, sheet, drawn,
      redraws, notes, pages, ownSheet, placeSheets, coverage, notify]);

  /**
   * Open a comic the Vault asked for.
   *
   * Waits for the launch restore, so a board opened from the library is not
   * overwritten a moment later by the draft that was already on screen. The
   * autosave then writes it into the draft slot as the board in front of you,
   * which is what makes it the one you are working on.
   */
  useEffect(() => {
    if (!openProject || !draftLoaded) return;
    if (openProject === projectId) { onOpened?.(); return; }
    let live = true;
    void (async () => {
      try {
        const saved = await loadSavedBoard(openProject);
        if (!live) return;
        if (!saved) {
          notify(
            "That comic was made before boards were saved individually, so "
            + "its panels are in the Vault but its story and notes are not.",
            true,
          );
        } else {
          applyDraft(saved);
          notify("Opened in the Board.");
        }
      } catch (e) {
        if (live) notify(`Could not open that board: ${errText(e)}`, true);
      } finally {
        if (live) onOpened?.();
      }
    })();
    return () => { live = false; };
  // Runs when the library names a different board, not on every render.
  }, [openProject, draftLoaded]); // eslint-disable-line react-hooks/exhaustive-deps

  // An armed reset is disarmed by anything else happening: pressing Escape,
  // or the board changing under it. A destructive button left loaded is how a
  // later, unrelated click deletes something.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { setConfirmReset(null); setConfirmStyle(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    setConfirmReset(null); setConfirmStyle(null);
  }, [stage, drawn, panels]);

  // These are interface choices, not project content, so they remain prefs.
  useEffect(() => { savePref("boardBatch", batch); }, [batch]);
  useEffect(() => { savePref("boardPanelPx", panelPx); }, [panelPx]);
  useEffect(() => { savePref("boardStyle", style.id); }, [style.id]);
  useEffect(() => { savePref("boardLayout", layout); }, [layout]);

  const busy = stage !== "idle";

  const changeStory = (next: string) => {
    if (next === story) return;
    setStory(next);
    setStoryBook(null); setChapterId(null); clearCharacterSheets();
    // Every downstream decision was made from the previous manuscript. Keeping
    // any of it would quietly mix two different stories.
    setCast(null); setProjectId(null); setSheet(null);
    setDoc({ panels: [], art: {} }); setDivided(false);
    setPages([]); setOverflow([]);
    setPlaceSheets({}); setCoverage(null); setSelected(0);
  };

  /** Whether anything has been drawn in the current style. */
  const drawnInStyle = drawn.some(Boolean) || !!sheet
    || Object.keys(placeSheets).length > 0;

  /**
   * Change the look of a board that is already under way.
   *
   * The style leads every prompt, so everything already drawn was drawn in the
   * old one and a half-restyled board is not a board -- the pictures go, and
   * they are still in the Vault if they were wanted. What stays is everything
   * that is a reading of the manuscript rather than of the style: the
   * division, the briefs, the notes and the coverage report. Having to retype
   * a note to change a look is what made the look uncorrectable in practice.
   */
  const changeStyle = (id: string) => {
    const next = id === CUSTOM_STYLE
      ? lockCustomStyle(customWords)
      : lockStyle(id);
    if (!next.words.trim()) {
      notify("Describe the look first, then it can be used.", true);
      return;
    }
    // Same words already in force: nothing to invalidate.
    if (next.id === style.id && next.words === style.words) return;
    if (drawnInStyle && confirmStyle !== id) { setConfirmStyle(id); return; }
    setStyle(next);
    clearCharacterSheets();
    setSheet(null);
    setDoc((d) => ({ ...d, art: {} }));
    setPlaceSheets({}); setPages([]); setOverflow([]);
    setConfirmStyle(null);
  };

  /**
   * Throw away the cast sheet so the next draw builds a new one.
   *
   * Panels already drawn were drawn against the old sheet and are left alone:
   * which of them to redraw is a judgement about the pictures, and the board
   * already has a per-panel redraw for making it. The composed pages go,
   * because a page is only as current as the panels in it.
   */
  const recastSheet = () => {
    ignoreSavedReferences.current = true;
    clearCharacterSheets();
    setSheet(null); setPages([]); setOverflow([]);
    notify("The cast sheet will be drawn again on the next run. Panels already "
      + "drawn were drawn against the old one -- redraw the ones that no "
      + "longer match.");
  };

  /** The same, for one scene's room. */
  const rebuildPlace = (key: string) => {
    setPlaceSheets((prev) => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
    setPages([]); setOverflow([]);
    notify(`${key} will be drawn again on the next run.`);
  };

  /** The places this board actually has, in the order the story reaches them. */
  const places = useMemo(() => {
    const seen: string[] = [];
    for (const p of panels ?? []) {
      const key = placeKey(p);
      if (!seen.includes(key)) seen.push(key);
    }
    return seen;
  }, [panels]);

  /** Step one: divide the prose. Nothing is drawn yet. */
  /** Validate the chapter's story model, save it, then replace the panel plan. */
  const divide = async () => {
    if (!story.trim()) { notify("Write the story first.", true); return; }
    const id = newJobId();
    if (!storyLibrary) { notify("The story library has not loaded yet.", true); return; }
    setStage("dividing"); setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const previous = appendChapter
        ? storyLibrary.books.find((b) => b.id === (continuationBookId || storyLibrary.activeBookId)) ?? null
        : null;
      const r = await api.analyzeStory(id, story, chapterTitle, previous, analystId || null);
      // Save the shared model before replacing the current Board's work.
      await onStoryAnalyzed(r.book);
      setStoryBook(r.book); storyBookRef.current = r.book; setChapterId(r.chapter_id);
      ignoreSavedReferences.current = false;
      setCast(castFromBook(r.book));
      setDoc(fromArrays(r.panels)); setDivided(true);
      setSheet(null); clearCharacterSheets();
      setSelected(0); setPages([]); setOverflow([]); setPlaceSheets({});
      setProjectId(newJobId());
      const beats = r.book.chapters.at(-1)!.scenes.flatMap((s) => s.beats);
      setCoverage({ covered: beats.length, total: beats.length, percent: 100, missing: [] });
      setAppendChapter(false);
      notify(`Analyzed ${beats.length} beats and directed ${r.panels.length} panels. The chapter is also available in Audiobook.`);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setStage("idle"); setProg(null); setJobId(null);
    }
  };

  /** Which panels have a picture, and which are still waiting. */
  const drawnCount = useMemo(
    () => drawn.filter(Boolean).length, [drawn]);
  const remaining = Math.max(0, (panels?.length ?? 0) - drawnCount);
  /**
   * How long the next press will take.
   *
   * Measured warm on this machine: 36 seconds a panel at 512, 77 at 768. The
   * first panel of a run also loads the model, which is most of the spread.
   * The old figure was a flat 0.6-2 minutes regardless of size, written when
   * every panel was 512.
   */
  const estimate = useMemo(() => {
    // Every image the run will draw, not just the panels in the batch.
    // `draw()` calls buildPlaces() first, which draws a room sheet for each
    // scene that has a place and has not got one yet, and the cast sheet if it
    // is missing. Counting the panels alone understated a first run by a sheet
    // per scene -- so the one figure quoted before the expensive part was the
    // one number the user could not rely on.
    const rooms = new Set(
      (panels ?? [])
        .filter((p) => (p.place ?? "").trim() && !placeSheets[placeKey(p)])
        .map((p) => placeKey(p))
    ).size;
    const drawing = Math.min(batch, Math.max(1, remaining));
    const sheets = (ownSheet.length ? 0 : 1) + rooms;

    /**
     * A panel costs more than a picture, because a panel has references.
     *
     * 77s at 768 was measured on a bare text-to-image run, and no panel a
     * board draws is one: every one of them is conditioned on the cast sheet,
     * the room, or both, which is the whole mechanism that keeps a face and a
     * room the same. Timed on this machine driving the real engine, low-RAM
     * on: the sheet and the rooms took 41-59s with no references, and the
     * panels took 159s, 168s and 257s with two. So the figure quoted before
     * the expensive part -- the one number the tool asks to be trusted --
     * understated a board by about two and a half times.
     *
     * The sheets keep the unreferenced rate, since that is what they are.
     */
    const perSheet = panelPx >= 768 ? 60 : 36;
    const perPanel = panelPx >= 768 ? 170 : 90;
    const low = sheets * perSheet + drawing * perPanel;
    return { low: Math.ceil(low / 60), high: Math.ceil(low * 1.6 / 60) };
  }, [batch, remaining, ownSheet.length, panelPx, panels, placeSheets]);

  /** What a reset would throw away, named so the question can be answered. */
  const atRisk = useMemo(() => {
    const bits: string[] = [];
    const noted = Object.values(notes).filter((n) => n.trim()).length;
    const plural = (n: number, word: string) =>
      `${n} ${word}${n === 1 ? "" : "s"}`;
    if (drawnCount) bits.push(plural(drawnCount, "drawn panel"));
    if (noted) bits.push(plural(noted, "note"));
    if (pages.length) bits.push(plural(pages.length, "composed page"));
    return bits;
  }, [drawnCount, notes, pages]);
  const preparedCount = panels?.filter((p) => p.cast !== undefined || (p.description ?? "").trim()).length ?? 0;
  // Every panel still waiting for a picture has to be worked up first. Gating
  // on "at least one" let a single prepared panel unlock drawing all of them,
  // and the rest were drawn from the terse fields -- which is the difference
  // between a scene and a caption, and costs a minute of drawing each to find
  // out.
  const unpreparedRemaining = panels?.filter(
    (p, i) => !drawn[i] && p.cast === undefined && !(p.description ?? "").trim()).length ?? 0;
  const workflow = !cast ? "read" : !panels ? "plan" : unpreparedRemaining > 0
    ? "prepare" : remaining > 0 ? "draw" : "compose";

  /**
   * The indices this press will draw: the next `batch` panels with no picture.
   *
   * Skipping ones that already have a picture is what makes a second press
   * continue rather than start over, and what lets a redrawn panel keep the
   * version you accepted.
   */
  const nextBatch = (): number[] => {
    const want = Math.max(1, Math.min(batch, panels?.length ?? 1));
    const out: number[] = [];
    for (let i = 0; i < (panels?.length ?? 0) && out.length < want; i++) {
      if (!drawn[i]) out.push(i);
    }
    return out;
  };

  /** Step two: cast the character once, then draw every panel against it. */
  const draw = async () => {
    if (!model) { notify("No model that can both generate and edit is installed.", true); return; }
    if (!panels) return;
    // No hard stop for a missing character. The cast is read out of the story,
    // so the common case is that the person is already known; and a story with
    // no one in it -- a place, a mood, a sequence of weather -- is still a
    // board worth drawing. Refusing to start until you had typed a character
    // in was the single thing that made pasting a story feel like it did
    // nothing at all.


    stop.current = false;
    setDone(0);
    // The two blocks that used to sit here resized `redraws` and `drawn` to
    // the panel count and copied the old values across by position. Art is
    // keyed by panel now, so there is no length to keep in step and nothing
    // to copy: a later batch already adds to what is there, because what is
    // there is attached to the panels rather than to their positions.
    setElapsed(0);
    const started = Date.now();
    if (timer.current) window.clearInterval(timer.current);
    timer.current = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    // The sheet is the anchor. Every panel references it, which is the whole
    // reason the character survives from one shot to the next -- so if the
    // user brought their own, there is nothing to cast.
    const existingSheet = ownSheet[0] ?? sheet;
    if (existingSheet || !castForSheet.trim() || panels.some((p) => p.cast !== undefined)) {
      setSheet(existingSheet ?? null);
      const places = await buildPlaces();
      setStage("drawing");
      for (const i of nextBatch()) {
        if (stop.current) { notify("Stopped."); break; }
        const ok = await drawOne(i, existingSheet ?? null, places);
        if (!ok && stop.current) break;
      }
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      setStage("idle"); setProg(null); setJobId(null);
      return;
    }

    const castId = newJobId();
    setStage("casting"); setJobId(castId);
    let un = await onEngineProgress((p) => { if (p.job_id === castId) setProg(p); });
    let sheetId: string;
    try {
      const res = await api.generate({
        job_id: castId, model_id: model.id,
        prompt: sheetPrompt(style, castForSheet),
        negative_prompt: null,
        width: panelPx, height: panelPx,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        seed: 7, count: 1, images: [], image_strength: null, i2i_mode: null,
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
        // Reference material for this board, not a loose picture:
        // the sheets live inside the comic they were drawn for.
        project: projectId, project_name: projectName,
      });
      if (!res.length) throw new Error("the character sheet came back empty");
      sheetId = res[0];
      setSheet(sheetId);
      onProduced();
    } catch (e) {
      notify(errText(e), true);
      if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
      un(); setStage("idle"); setProg(null); setJobId(null);
      return;
    }
    un();

    // Panels run one at a time. Two models never share this machine's memory,
    // and a board of six is minutes of work either way.
    const places = await buildPlaces();
    setStage("drawing");
    for (const i of nextBatch()) {
      if (stop.current) { notify("Stopped."); break; }
      const ok = await drawOne(i, sheetId, places);
      if (!ok && stop.current) break;
    }
    if (timer.current) { window.clearInterval(timer.current); timer.current = null; }
    setStage("idle"); setProg(null); setJobId(null);
  };

  /** Draw each scene's room once, empty, for its panels to be set inside.
   *
   *  The same mechanism that holds the character, applied to the place. A
   *  described room is drawn differently every time; a referenced one is not.
   */
  const buildPlaces = async (): Promise<Record<string, string>> => {
    if (!model || !panels) return {};
    const wanted = new Map<string, string>();
    for (const p of panels) {
      // Structured panels use story state and compatible finished frames, not a fixed room sheet.
      if (p.cast !== undefined) continue;
      const key = placeKey(p);
      const place = (p.place ?? "").trim();
      if (place && !wanted.has(key) && !placeSheets[key]) wanted.set(key, place);
    }
    if (wanted.size === 0) return placeSheets;

    setStage("building");
    const made: Record<string, string> = { ...placeSheets };
    for (const [key, place] of wanted) {
      if (stop.current) break;
      const id = newJobId();
      setJobId(id);
      const un = await onEngineProgress((e) => { if (e.job_id === id) setProg(e); });
      try {
        const res = await api.generate({
          job_id: id, model_id: model.id,
          prompt: placePrompt(style, place),
          negative_prompt: null,
          width: panelPx, height: panelPx,
          steps: model.steps_default || 4,
          guidance: Math.min(1.0, model.guidance_max),
          seed: 21 + placeSeed(key), count: 1, images: [], image_strength: null,
          i2i_mode: null, low_ram: true, preview: false, cache_limit_gb: null,
          allow_over_budget: false, loras: [], mask: null,
          outpaint_padding: null, outpaint_fill: null,
          // Reference material for this board, not a loose picture:
          // the sheets live inside the comic they were drawn for.
          project: projectId, project_name: projectName,
        });
        if (res[0]) made[key] = res[0];
        onProduced();
      } catch (e) {
        // A missing room is survivable: the panel falls back to its written
        // description, which is what it used before this existed.
        notify(`The room for ${key}: ${errText(e)}`, true);
      } finally {
        un();
      }
    }
    setPlaceSheets(made);
    return made;
  };

  // A painted mask belongs to one picture. Selecting a different panel, or
  // that panel's picture being replaced under it, leaves a mask that lines up
  // with nothing -- so the repair closes rather than being applied to the
  // wrong frame.
  useEffect(() => {
    setRepairing(null);
    setMaskBytes(null);
    setConfirmDelete(null);
  }, [selected, drawn]);

  /**
   * File the picture a panel is about to lose.
   *
   * Called before anything overwrites `drawn[i]`. The picture itself is not
   * touched -- it is sealed in the vault and stays there -- this only keeps
   * the board's own way back to it.
   */
  const supersede = (i: number, previous: string | null | undefined) => {
    if (!previous) return;
    const had = artAt(i).variants;
    if (had.includes(previous)) return;
    putArt(i, { variants: [...had, previous] });
  };

  /**
   * Put a previous picture of a panel back.
   *
   * The current one takes its place in the history rather than being dropped,
   * so this is a swap and going back again is always possible.
   */
  const restoreVariant = (i: number, id: string) => {
    const current = drawnRef.current[i] ?? drawn[i];
    const rest = artAt(i).variants.filter((x) => x !== id);
    putArt(i, {
      drawn: id,
      variants: current ? [...rest, current] : rest,
    });
    drawnRef.current = (() => {
      const n = [...drawnRef.current]; n[i] = id; return n;
    })();
    setPages([]); setOverflow([]);
    notify(`Panel ${i + 1} put back to its previous picture.`);
  };

  /** Draw a single panel against the sheet. Shared by the run and by redraw,
   *  so a board with one bad panel costs one panel to fix rather than six. */
  const drawOne = async (i: number, sheetId: string | null,
                         places: Record<string, string> = placeSheets,
                         redrawCount = redraws[i] ?? 0): Promise<boolean> => {
    if (!model || !panels) return false;
    const id = newJobId();
    setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    // What the last moment looked like, when there is one in this scene. The
    // board had no continuity at all: every panel was drawn from the sheet and
    // the room, so the light, the weather and the state of the character's
    // clothes reset between one panel and the next inside a single scene.
    try {
      let identity = sheetId;
      if (panels[i].cast !== undefined) {
        const actors = panels[i].cast!;
        const ids: string[] = [];
        for (const actor of actors) {
          if (stop.current) return false;
          const person = cast?.people.find((p) => p.id === actor.character_id);
          let ref: string | undefined = characterSheetsRef.current[actor.character_id];
          const currentBook = storyBookRef.current;
          const saved = storyLibrary?.books.find((b) => b.id === currentBook?.id)?.references?.[actor.character_id]
            ?? currentBook?.references?.[actor.character_id];
          if (saved?.kind !== "identity-face-v1") ref = undefined;
          const visual = (person === lead && character.trim()) || person?.description || actor.visual;
          if (!ref && !ignoreSavedReferences.current && saved?.kind === "identity-face-v1" && saved.style_words === style.words && saved.visual === visual) {
            ref = saved.image;
          }
          if (person === lead && ownSheet[0]) ref = ownSheet[0];
          if (!ref) {
            const generated = await api.generate({
              job_id: id, model_id: model.id,
              prompt: identityPrompt(style, `${actor.name}: ${visual}`),
              negative_prompt: null, width: panelPx, height: panelPx,
              steps: model.steps_default || 4, guidance: Math.min(1.0, model.guidance_max),
              seed: panelSeed(actors.indexOf(actor), 0), count: 1,
              images: [], image_strength: null, i2i_mode: null,
              low_ram: true, preview: false, cache_limit_gb: null,
              allow_over_budget: false, loras: [], mask: null,
              outpaint_padding: null, outpaint_fill: null,
              project: projectId, project_name: projectName,
            });
            if (!generated[0]) throw new Error(`No character sheet returned for ${actor.name}`);
            ref = generated[0]; onProduced();
          }
          if (currentBook && (currentBook.references?.[actor.character_id]?.image !== ref
              || currentBook.references?.[actor.character_id]?.visual !== visual
              || currentBook.references?.[actor.character_id]?.style_words !== style.words)) {
            const updated = { ...currentBook, references: { ...currentBook.references,
              [actor.character_id]: { image: ref, style_words: style.words, visual, kind: "identity-face-v1" as const } } };
            await onCharacterReference(currentBook.id, actor.character_id, updated.references[actor.character_id]);
            storyBookRef.current = updated; setStoryBook(updated);
          }
          characterSheetsRef.current = { ...characterSheetsRef.current, [actor.character_id]: ref };
          setCharacterSheets(characterSheetsRef.current);
          ids.push(ref);
        }
        identity = ids[0] ?? null;
        if (ids.length > 1) {
          const key = ids.join(":");
          identity = referenceCache.current[key] ?? await api.characterReference(id, ids);
          referenceCache.current[key] = identity;
        }
      }
      if (stop.current) return false;
      const continuation = panels[i].cast !== undefined
        ? storyContinuityReference(i, panels, drawnRef.current) : null;
      const refs = panelReferences(
        panels[i], continuation ?? identity, places[placeKey(panels[i])] ?? null,
        panels[i].cast !== undefined ? null : previousPanel(i, panels, drawnRef.current),
        model.max_edit_images || 1);
      const params = {
        job_id: id, model_id: model.id,
        prompt: panelPrompt({ ...panels[i], cast: panels[i].cast?.map((actor) => ({
          ...actor, visual: (cast?.people.find((p) => p.id === actor.character_id) === lead && character.trim())
            || cast?.people.find((p) => p.id === actor.character_id)?.description || actor.visual,
        })) }, style, panelWho(panels[i]), notes[i]),
        negative_prompt: null,
        width: panelPx, height: panelPx,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        // Redrawing the same panel with the same seed reproduces the picture
        // that was already rejected, so a redraw moves the seed on.
        seed: panelSeed(i, redrawCount),
        count: 1,
        images: refs,
        // "edit" is the reference-conditioned route -- the model is handed
        // the sheet and the panel description together. "latent" would
        // instead redraw the sheet itself, which is not what a panel is.
        image_strength: null, i2i_mode: refs.length ? "edit" : null,
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: null,
        outpaint_padding: null, outpaint_fill: null,
        // Every panel is stamped with the board it belongs to, so the library
        // shows one comic instead of a pile of pictures that happen to share
        // a timestamp. The index keeps them in reading order.
        project: projectId, project_name: projectName, project_index: i,
      };
      // A panel with no reference is a plain drawing. Sending it down the edit
      // route fails outright -- an edit needs an input image -- which is what
      // would happen to every panel in a board whose character is not in them
      // and whose rooms did not draw.
      const res = refs.length
        ? await api.editImage(params)
        : await api.generate(params);
      // Before the id is overwritten, not after: this is the only moment the
      // board still knows what the panel used to be.
      if (res[0]) supersede(i, drawnRef.current[i]);
      putArt(i, { drawn: res[0] ?? null });
      // Immediately, not on the next render: the next panel in this batch asks
      // for it before React has committed the state above.
      drawnRef.current = (() => {
        const n = [...drawnRef.current];
        n[i] = res[0] ?? null;
        return n;
      })();
      // A composed page is only as current as the panels in it, so any panel
      // arriving retires the pages. `edit` already did this and `redraw` did
      // not, which left the one hole: correcting a frame and drawing it again
      // kept the old pages on screen, still showing the frame that had just
      // been replaced -- and those are the two pictures the whole board is for.
      setPages([]); setOverflow([]);
      setDone((v) => Math.max(v, i + 1));
      onProduced();
      return true;
    } catch (e) {
      const msg = errText(e);
      notify(`Panel ${i + 1}: ${msg}`, true);
      if (isCancelled(e)) stop.current = true;
      return false;
    } finally {
      un();
    }
  };

  /** Work each panel up into a scene: surface, background, light.
   *
   *  Its own step, and readable before anything is drawn. The division is
   *  terse on purpose -- that is the right level for judging whether the
   *  story was cut correctly -- and far too thin to draw from.
   */
  const enrich = async () => {
    if (!panels) return;
    const id = newJobId();
    setStage("enriching"); setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    try {
      const r = await api.enrichPanels(id, panels, style.words, cast?.places ?? []);
      // The same panels, worked up. The engine returns them in order and does
      // not carry ids, so the ids are put back by position -- which is safe
      // here and only here, because this is the round trip that cannot
      // reorder or drop one. Everything that can goes through boardDoc.
      setDoc((d) => ({
        ...d,
        panels: identify(r.panels.map(
          (p, i) => ({ ...p, id: d.panels[i]?.id }) as Panel)),
      }));
      const filled = r.panels.filter((p) => (p.description ?? "").trim()).length;
      notify(filled === r.panels.length
        ? "Every panel worked up. Read them before drawing."
        : `${filled} of ${r.panels.length} worked up; the rest stay as written.`);
    } catch (e) {
      notify(errText(e), true);
    } finally {
      un(); setStage("idle"); setProg(null); setJobId(null);
    }
  };

  /**
   * Change one field of one panel. The division is a draft, not a verdict.
   *
   * What the change invalidates is decided in one place -- `staleness` in
   * boardDoc -- rather than here. Every edit used to retire the picture,
   * including a caption fix, so correcting a comma cost a five-minute render
   * and came back with a different face. A picture is a drawing of an action,
   * in a place, of these people; change one of those and it is a picture of
   * something else, and change the lettering and it is not.
   */
  const edit = (i: number, patch: Partial<Panel>) => {
    const panel = doc.panels[i];
    if (!panel) return;
    if (patch.characters && panel.cast !== undefined) {
      const actors = patch.characters.map((name) => {
        const person = cast?.people.find((c) => c.name.toLowerCase() === name.toLowerCase()
          || c.aliases?.some((a) => a.toLowerCase() === name.toLowerCase()));
        const old = panel.cast?.find((a) => a.character_id === person?.id);
        return person?.id ? { character_id: person.id, name: person.name, visual: person.description,
          expression: old?.expression ?? "", body_language: old?.body_language ?? "",
          appearance: old?.appearance ?? "" } : null;
      });
      // An unknown hand-entered name still uses the legacy free-form route.
      patch = { ...patch, cast: actors.every((a) => a !== null) ? actors.filter((a) => a !== null) : undefined };
    }
    const { doc: next, stale } = editPanel(doc, panel.id, patch);
    setDoc(next);
    if (stale.art) {
      // Retired, not deleted: it goes to the panel's history, where it is one
      // click away if the edit turns out to be the mistake.
      drawnRef.current = (() => {
        const n = [...drawnRef.current]; n[i] = null; return n;
      })();
    }
    if (stale.pages) { setPages([]); setOverflow([]); }
  };

  /**
   * Move, add, remove, split or join a panel.
   *
   * None of this was possible while the board's work lived in arrays lined up
   * by position: moving a panel left its picture and its note behind for
   * whatever landed in that slot. The operations are in boardDoc, by identity,
   * and tested there -- this keeps the selection pointing at the panel the
   * user was looking at rather than at the position it used to occupy.
   */
  const structural = (
    op: (d: BoardDoc) => BoardDoc, keep: string | null, at?: number,
  ) => {
    setDoc((d) => {
      const next = op(d);
      const to = keep ? next.panels.findIndex((x) => x.id === keep) : at ?? 0;
      setSelected(Math.max(0, Math.min(to === -1 ? 0 : to,
                                       next.panels.length - 1)));
      return next;
    });
    // The pages were composed from an order that no longer exists.
    setPages([]); setOverflow([]);
  };

  const movePanel = (i: number, by: number) => {
    const panel = doc.panels[i];
    if (!panel) return;
    const to = i + by;
    if (to < 0 || to >= doc.panels.length) return;
    structural((d) => movePanelIn(d, panel.id, to), panel.id);
  };

  const addPanelAfter = (i: number) => {
    structural((d) => insertPanel(d, i + 1, {
      shot: "medium", subject: "", action: "", setting: "",
      character_in_frame: false, caption: "",
      scene: doc.panels[i]?.scene ?? 1,
    }), null, i + 1);
    notify("An empty panel. Describe it, then draw it on the next run.");
  };

  const deletePanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel) return;
    const had = artOf(doc, panel.id).drawn;
    structural((d) => removePanel(d, panel.id).doc, null, Math.max(0, i - 1));
    notify(had
      ? `Panel ${i + 1} removed. Its picture stays in the Vault.`
      : `Panel ${i + 1} removed.`);
  };

  const splitPanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel) return;
    structural((d) => splitPanelIn(d, panel.id, { action: "" }), panel.id);
    notify("Split in two. The picture stays with the first half — which half "
           + "it really belongs to is a judgement about the picture.");
  };

  const mergePanel = (i: number) => {
    const panel = doc.panels[i];
    if (!panel || i + 1 >= doc.panels.length) return;
    structural((d) => mergePanelIn(d, panel.id), panel.id);
    notify("Joined. The second panel's picture is kept in this one's history.");
  };

  /**
   * Repair one painted region of a drawn panel.
   *
   * A note and a redraw are a different repair from this one, and the board
   * only had the first. A note changes what was asked for and the whole frame
   * is drawn again from it -- which is the right move when the panel is wrong,
   * and the wrong one when it is nearly right: fixing a hand costs the face,
   * because the seed moves and everything is reconsidered. The model already
   * takes a mask and Studio already paints one; the board was the only surface
   * sending `mask: null`.
   *
   * The panel itself is the input image here, not the cast sheet. A masked
   * edit regenerates inside the painted region of the picture it is given, so
   * the picture it is given has to be the one being corrected.
   */
  const repair = async (i: number) => {
    const current = drawn[i];
    if (!model || !panels || !current) return;
    if (!maskBytes) { notify("Paint over the part to redraw first.", true); return; }
    const id = newJobId();
    stop.current = false;
    setStage("drawing"); setJobId(id);
    const un = await onEngineProgress((p) => { if (p.job_id === id) setProg(p); });
    const nextRedraw = (redraws[i] ?? 0) + 1;
    let maskId: string | null = null;
    try {
      // The mask is content like anything else: sealed into the vault, handed
      // over by id, and removed once the engine has read it. It is working
      // material and has no business outliving the repair.
      maskId = await api.vaultImportBytes(
        maskBytes, "mask.png", "image/png", "mask");
      const note = notes[i]?.trim();
      const res = await api.editImage({
        job_id: id, model_id: model.id,
        // What should be in the painted region. A note is written about a
        // specific fault -- "older, grey at the temples" -- which is exactly
        // the instruction a masked repair wants; without one there is nothing
        // to say but the panel's own brief.
        prompt: note ? `${note}. ${style.words}`
                     : panelPrompt(panels[i], style, panelWho(panels[i])),
        negative_prompt: null,
        width: panelPx, height: panelPx,
        steps: model.steps_default || 4,
        guidance: Math.min(1.0, model.guidance_max),
        seed: panelSeed(i, nextRedraw),
        count: 1,
        images: [current],
        image_strength: null, i2i_mode: "edit",
        low_ram: true, preview: false, cache_limit_gb: null,
        allow_over_budget: false, loras: [], mask: maskId,
        outpaint_padding: null, outpaint_fill: null,
        project: projectId, project_name: projectName, project_index: i,
      });
      if (!res[0]) { notify("The engine returned no picture.", true); return; }
      supersede(i, current);
      putArt(i, { drawn: res[0], redraws: nextRedraw });
      drawnRef.current = (() => {
        const n = [...drawnRef.current]; n[i] = res[0]; return n;
      })();
      setPages([]); setOverflow([]);
      setRepairing(null); setMaskBytes(null);
      onProduced();
      notify(`Panel ${i + 1} repaired. The picture it replaces is kept — `
             + "put it back from the panel's history if this is worse.");
    } catch (e) {
      // A model that cannot mask says so here rather than being asked to
      // guess. Refusing is better than silently redrawing the whole frame,
      // which is what an unmasked fallback would do behind the user's back.
      notify(isCancelled(e) ? "Cancelled." : `Panel ${i + 1}: ${errText(e)}`,
             !isCancelled(e));
    } finally {
      if (maskId) {
        try { await api.vaultDelete(maskId); } catch { /* lock/cleanup race */ }
      }
      un(); setStage("idle"); setProg(null); setJobId(null);
    }
  };

  /** Redraw one panel, keeping the rest of the board. */
  const redraw = async (i: number) => {
    if (!drawn[i]) { notify("Draw this panel first.", true); return; }
    stop.current = false;
    const nextRedraw = (redraws[i] ?? 0) + 1;
    putArt(i, { redraws: nextRedraw });
    setStage("drawing");
    await drawOne(i, sheet, placeSheets, nextRedraw);
    setStage("idle"); setProg(null); setJobId(null);
  };

  /** Stack the drawn panels into one page, captions underneath.
   *
   *  Only the panels that actually exist go in: a half-drawn board still
   *  makes a readable page, and a gap for a missing one would not.
   */
  const compose = async () => {
    if (!panels) return;
    type Ready = { id: string; caption: string; shot: Panel["shot"];
                   dialogue: { speaker: string; text: string }[]; scene: number;
                   at: number };
    const pairs = panels
      .map((p, i) => ({ id: drawn[i], caption: p.caption, shot: p.shot,
                        dialogue: p.dialogue ?? [], scene: p.scene ?? 1,
                        at: i }))
      .filter((x): x is Ready => Boolean(x.id));
    /** Panel identity for each panel the engine will be handed, in order. */
    const sentIds = pairs.map((x) => doc.panels[x.at]?.id ?? "");
    if (pairs.length === 0) { notify("Draw at least one panel first.", true); return; }

    setComposing(true);
    try {
      const made = await api.composeBoard(
        newJobId(),
        pairs.map((p) => p.id),
        pairs.map((p) => p.caption),
        pairs.map((p) => p.shot),
        pairs.map((p) => p.dialogue ?? []),
        pairs.map((p) => p.scene),
        layout, projectId, projectName);
      setPages(made.pages);
      // The engine counts the panels it was given, which are the drawn ones
      // in the order they were sent. The user counts panels on the board, and
      // the two differ by every gap in it -- so a report about "panel 3" would
      // name the wrong panel on any board that is not finished. Anchored to
      // identity rather than position, so it also survives the board being
      // reordered afterwards, and a report about a panel that has since been
      // deleted is dropped rather than left naming whatever is there now.
      const lost = anchorOverflow(made.overflow, sentIds, doc);
      setOverflow(lost);
      onProduced();
      notify(made.pages.length === 1
        ? `Composed one page from ${pairs.length} panels. It is in the Vault.`
        : `Composed ${made.pages.length} pages from ${pairs.length} panels, `
          + "broken where the scenes change. They are in the Vault.");
      if (lost.length) {
        notify(
          `${lost.length} panel${lost.length === 1 ? "" : "s"} could not fit `
          + "all their text. The pages are composed; see what was left off "
          + "before binding them.", true);
      }
    } catch (e) {
      notify(errText(e), true);
    } finally {
      setComposing(false);
    }
  };

  const cancel = async () => {
    stop.current = true;
    if (jobId) { try { await api.cancelJob(jobId); } catch { /* already gone */ } }
  };

  if (usable.length === 0) {
    return (
      <div className="panel">
        <h2>Picture board</h2>
        <div className="notice warn">
          <strong>Needs a model that can both generate and edit</strong>
          A board generates individual character references and uses them to draw
          the panels, so the image model has to do both. FLUX.2 Klein is the
          smallest that does — add it from the Models tab.
        </div>
      </div>
    );
  }

  // Clamp rather than trust: any path that shortens the board would otherwise
  // leave the inspector pointing at a panel that no longer exists.
  const selIndex = panels ? Math.min(selected, panels.length - 1) : 0;
  const sel = panels?.[selIndex] ?? null;

  if (!draftLoaded) {
    return <div className="empty-state">Opening your encrypted Board draft…</div>;
  }

  return (
    <div style={{ padding: 14 }}>
      <div className="board-flow" aria-label="Board workflow">
        {["read", "plan", "prepare", "draw", "compose"].map((name) => (
          <span key={name} className={name === workflow ? "current" : ""}>
            {name === "read" ? "Read story" : name === "plan" ? "Plan panels"
              : name === "prepare" ? "Prepare briefs" : name === "draw" ? "Draw"
                : "Compose"}
          </span>
        ))}
        <span className="board-save">
          {unsaved ? "Not being saved" : "Encrypted draft saved automatically"}
        </span>
      </div>

      {unsaved && (
        // Stays until a save succeeds. The board is still usable and still
        // holds everything; what it is not doing is writing any of it down.
        <div className="notice warn" style={{ margin: "10px 0" }}>
          <strong>This board is not being saved</strong>
          <div style={{ fontSize: 11.5, lineHeight: 1.6, marginTop: 4 }}>
            {unsaved}
          </div>
          <div style={{ fontSize: 11.5, lineHeight: 1.6, marginTop: 4 }}>
            Everything on screen is still here. Nothing since the last
            successful save has been written to the vault, so copy the
            manuscript out before closing the app if this does not clear.
          </div>
        </div>
      )}

      {previous && (
        // Offered, never applied. Which of two versions of a manuscript is the
        // right one is not a question this can answer.
        <div className="notice warn" style={{ margin: "10px 0" }}>
          <strong>This board opened empty, and an earlier version of it exists</strong>
          <div style={{ fontSize: 11.5, lineHeight: 1.6, marginTop: 4 }}>
            {previous.story?.trim()
              ? `It begins "${previous.story.trim().slice(0, 90)}…"`
              : "It has a division but no manuscript."}
            {previous.panels?.length
              ? ` ${previous.panels.length} panels.`
              : ""}
          </div>
          <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
            <button className="btn small primary" disabled={busy}
                    onClick={() => { applyDraft(previous); setPrevious(null); }}>
              Put it back
            </button>
            <button className="btn small" onClick={() => setPrevious(null)}>
              No, this board is meant to be empty
            </button>
          </div>
        </div>
      )}

      <div className="board-work">

        {/* ---- the manuscript: your prose, never rewritten ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Manuscript
            <span className="n">
              {story.trim() ? `${story.trim().split(/\s+/).length} words` : "empty"}
            </span>
          </div>
          <div className="inner">
            {!panels ? (
              <>
                {!!storyLibrary?.books.length && <label>Use an analyzed chapter<select
                  value="" disabled={busy} onChange={(e) => {
                    const [bid, cid] = e.target.value.split(":");
                    const book = storyLibrary.books.find((b) => b.id === bid);
                    const chapter = book?.chapters.find((c) => c.id === cid);
                    if (!book || !chapter) return;
                    void (async () => {
                      setStage("dividing");
                      try {
                        const result = await api.directStory(newJobId(), book, cid);
                        setStory(chapter.source); setChapterTitle(chapter.title);
                        setStoryBook(book); storyBookRef.current = book; setChapterId(cid);
                        ignoreSavedReferences.current = false; setCast(castFromBook(book));
                        setDoc(fromArrays(result.panels)); setDivided(true); setProjectId(newJobId());
                        setSheet(null); clearCharacterSheets(); setPages([]); setOverflow([]); setPlaceSheets({});
                        setSelected(0); setCoverage(null);
                      } catch (error) { notify(errText(error), true); }
                      finally { setStage("idle"); }
                    })();
                  }}>
                  <option value="">Choose a saved chapter…</option>
                  {storyLibrary.books.flatMap((book) => book.chapters.map((chapter) =>
                    <option key={chapter.id} value={`${book.id}:${chapter.id}`}>{book.chapters[0]?.title} · {chapter.title}</option>))}
                </select></label>}
                <label>Chapter title<input value={chapterTitle} disabled={busy}
                  onChange={(e) => setChapterTitle(e.target.value)} /></label>
                <label>Story analyst<select value={analystId} disabled={busy}
                  onChange={(e) => setAnalystId(e.target.value)}>
                  <option value="">Best installed model for this Mac</option>
                  {models.filter((m) => ANALYST_IDS.includes(m.id) && m.installed && !m.hopeless)
                    .map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                </select></label>
                {!!storyLibrary?.books.length && <label>
                  <input type="checkbox" checked={appendChapter} disabled={busy}
                    onChange={(e) => setAppendChapter(e.target.checked)} />
                  Add to the current book and reuse its character bible
                </label>}
                {appendChapter && storyLibrary && <label>Continue book<select
                  value={continuationBookId || storyLibrary.activeBookId || storyLibrary.books.at(-1)?.id}
                  disabled={busy} onChange={(e) => setContinuationBookId(e.target.value)}>
                  {storyLibrary.books.map((book) => <option key={book.id} value={book.id}>
                    {book.chapters[0]?.title} · {book.chapters.length} chapters
                  </option>)}
                </select></label>}
                <textarea
                  value={story}
                  rows={14}
                  disabled={busy}
                  onChange={(e) => changeStory(e.target.value)}
                  placeholder={"Mira comes home and finds the door already open. "
                    + "Nothing is taken, but every photograph has been turned to "
                    + "face the wall."}
                />
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 6,
                              lineHeight: 1.55 }}>
                  Prose, not prompts. It is divided into panels for you, and
                  review its interpretation before drawing.
                </div>
              </>
            ) : (
              <div className="ms-prose">
                {/* Divided: the prose stays on screen, and the sentences behind
                    the selected panel are lit. Clicking one selects its panel,
                    so the story and the board point at each other. */}
                {panels.map((p, i) => (
                  <span
                    key={p.id}
                    className={"ms-beat" + (i === selected ? " lit" : "")}
                    onClick={() => setSelected(i)}
                  >
                    {p.source || [p.subject, p.action].filter(Boolean).join(", ")
                      || p.caption || `Panel ${i + 1}`}
                  </span>
                ))}
                {coverage && coverage.total > 0 && (() => {
                  // The engine caps the listed passages at twenty. Counting
                  // the list meant the worst-covered boards reported the
                  // smallest shortfall, so count what it actually found and
                  // say when the list below it is only part of that.
                  const lost = coverage.missing_total ?? coverage.missing.length;
                  const shown = coverage.missing.length;
                  return (
                    <div className={"notice" + (lost ? " warn" : " good")}
                         style={{ marginTop: 10 }}>
                      <strong>{coverage.percent}% of the prose is anchored</strong>
                      {lost > 0 ? (
                        <details>
                          <summary>
                            {lost} passage{lost === 1 ? "" : "s"} need review
                            {shown < lost ? ` (first ${shown} shown)` : ""}
                          </summary>
                          {coverage.missing.map((text, i) => <p key={i}>{text}</p>)}
                        </details>
                      ) : "Every passage has a panel source quote."}
                    </div>
                  );
                })()}
                <button
                  className={"btn small" + (confirmReset === "prose" ? " danger" : "")}
                  style={{ marginTop: 8 }} disabled={busy}
                  onClick={() => {
                    if (atRisk.length && confirmReset !== "prose") {
                      setConfirmReset("prose"); return;
                    }
                    setConfirmReset(null);
                    setProjectId(null); setSheet(null);
                    setDoc({ panels: [], art: {} }); setDivided(false);
                    setPages([]); setOverflow([]);
                    setPlaceSheets({}); setCoverage(null); setSelected(0);
                  }}>
                  {confirmReset === "prose"
                    ? `Discard ${atRisk.join(", ")}? The pictures stay in the Vault`
                    : "Back to the prose"}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* ---- the board: the only place panels exist ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Board
            <span className="n">
              {panels ? `${drawnCount}/${panels.length} drawn · ${scenes.length} scenes`
                      : "nothing yet"}
            </span>
          </div>
          <div className="inner">
            {!panels ? (
              <div style={{ fontSize: 12, color: "var(--text-faint)", lineHeight: 1.7 }}>
                Paste a story on the left, then divide it. The number of panels
                is worked out from the story's own beats — there is no number to
                choose, because a number chosen before the story is read is a
                guess.
              </div>
            ) : (
              <div className="board-cards">
                {panels.map((p, i) => (
                  <Fragment key={i}>
                    {(i === 0 || panels[i - 1].scene !== p.scene) && (
                      <div className="board-scene">
                        {p.scene_title || `Scene ${p.scene ?? 1}`}
                      </div>
                    )}
                    <button
                      className="board-card"
                      aria-current={i === selected ? "true" : "false"}
                      onClick={() => setSelected(i)}
                    >
                      <span className="shot">
                        {drawn[i]
                          ? <img src={vaultUrl(drawn[i]!)} alt={`Panel ${i + 1}`} />
                          : stage === "drawing" && done === i ? "drawing…" : "—"}
                        <span className="n">{String(i + 1).padStart(2, "0")}</span>
                      </span>
                      <span className="cap">
                        <b>{p.shot}</b>
                        {(p.caption || p.action || p.subject || "").slice(0, 54)}
                      </span>
                    </button>
                  </Fragment>
                ))}
              </div>
            )}

            {Object.entries(characterSheets).length > 0 && <div className="story-character-sheets">
              <div className="sub-head">Individual character references</div>
              {Object.entries(characterSheets).map(([cid, image]) => <figure key={cid}>
                <img src={vaultUrl(image)} alt={cast?.people.find((p) => p.id === cid)?.name ?? "Character"} />
                <figcaption>{cast?.people.find((p) => p.id === cid)?.name}</figcaption>
              </figure>)}
            </div>}
            {storyBook && chapterId && <details className="notice">
              <summary>Story direction · {storyBook.chapters.find((c) => c.id === chapterId)?.title}</summary>
              {storyBook.chapters.find((c) => c.id === chapterId)?.scenes.map((scene) => <div key={scene.id}>
                <strong>{scene.title}</strong><p>{scene.summary}</p>
                {scene.beats.filter((b) => b.importance >= 4).map((b) =>
                  <p key={b.id}>{b.importance === 5 ? "Climax" : "Turn"}: {b.action} · {b.mood}</p>)}
              </div>)}
            </details>}
            {sheet && (
              <div style={{ marginTop: 18 }}>
                <div className="sub-head">Character sheet</div>
                <img src={vaultUrl(sheet)} alt="Character sheet"
                     style={{ width: "100%", maxWidth: 260, borderRadius: 6 }} />
                <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                              lineHeight: 1.5 }}>
                  Panels share this cast reference, so recurring people keep the
                  same appearance from one shot to the next.
                </div>
              </div>
            )}

            {pages.length > 0 && (
              <div style={{ marginTop: 18 }}>
                <div className="sub-head">
                  {pages.length === 1 ? "The page" : `${pages.length} pages`}
                </div>
                {overflowNow.length > 0 && (
                  <div className="notice warn" style={{ marginBottom: 10 }}>
                    <strong>
                      {overflowNow.length} panel{overflowNow.length === 1 ? "" : "s"}
                      {" "}lost text on the page
                    </strong>
                    <details>
                      <summary>
                        These pages are composed and readable, but they are not
                        the whole script. Shorten the text or give the panel a
                        wider cell, then compose again.
                      </summary>
                      {overflowNow.map((o) => {
                        const missed = o.spoken - o.lettered;
                        const parts = [
                          missed > 0
                            ? `${missed} of ${o.spoken} spoken `
                              + `line${o.spoken === 1 ? "" : "s"} not drawn`
                            : "",
                          o.caption_clipped ? "caption cut short" : "",
                        ].filter(Boolean);
                        return (
                          <p key={`${o.panel}-${o.page}`}>
                            <button className="btn small"
                                    style={{ marginRight: 8 }}
                                    onClick={() => {
                                      setSelected(o.panel);
                                      setRailTab("panel");
                                    }}>
                              Panel {o.panel + 1}
                            </button>
                            {parts.join(", ")}
                            {o.page ? ` (page ${o.page})` : ""}
                          </p>
                        );
                      })}
                    </details>
                  </div>
                )}
                {pages.map((id, i) => (
                  <img key={id} src={vaultUrl(id)} alt={`Page ${i + 1}`}
                       style={{ width: "100%", borderRadius: 6, marginBottom: 10 }} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ---- notes: what produced the selected panel, and how to correct it ---- */}
        <div className="board-pane">
          <div className="board-pane-head">
            Notes
            {panels ? (
              <span className="n" style={{ display: "flex", gap: 4 }}>
                <button className={"pill" + (railTab === "panel" ? " installed" : "")}
                        style={{ cursor: "pointer" }}
                        onClick={() => setRailTab("panel")}>
                  Panel {selIndex + 1}
                </button>
                <button className={"pill" + (railTab === "board" ? " installed" : "")}
                        style={{ cursor: "pointer" }}
                        onClick={() => setRailTab("board")}>
                  Board
                </button>
              </span>
            ) : <span className="n">settings</span>}
          </div>
          <div className="inner">
            {!panels || !sel || railTab === "board" ? (
              <>
                <div className="field">
                  <label>Lead appearance override <em>optional</em></label>
                  <textarea
                    value={character}
                    rows={3}
                    disabled={busy}
                    onChange={(e) => { setCharacter(e.target.value); clearCharacterSheets(); setPages([]); setOverflow([]); }}
                    placeholder="short black hair, red scarf, narrow face"
                  />
                </div>
                <div className="field">
                  <label>Or bring a picture of them</label>
                  <ImageDrop images={ownSheet} onChange={(images) => { setOwnSheet(images); clearCharacterSheets(); setPages([]); setOverflow([]); }} max={1}
                             onError={(m) => notify(m, true)} />
                  <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                lineHeight: 1.5 }}>
                    {ownSheet.length > 0
                      ? "This picture anchors the lead character. Other visible characters keep their individual references."
                      : "A photo, drawing, or reference from an earlier chapter for the lead character."}
                  </div>
                </div>
                <div className="field">
                  <label>Style</label>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
                    {STYLES.map(({ id, label, words }) => (
                      <button key={id}
                              className={"pill" + (style.id === id ? " installed" : "")
                                + (confirmStyle === id ? " danger" : "")}
                              // What the pill will actually ask for. The only
                              // thing the tool ever said about a style used to
                              // appear after the board was drawn.
                              title={words}
                              style={{ cursor: "pointer" }} disabled={busy}
                              onClick={() => changeStyle(id)}>
                        {confirmStyle === id ? `Restyle? ${label}` : label}
                      </button>
                    ))}
                    <button className={"pill" + (style.id === CUSTOM_STYLE ? " installed" : "")}
                            title="Describe a look in your own words"
                            style={{ cursor: "pointer" }} disabled={busy}
                            onClick={() => setCustomOpen((v) => !v)}>
                      Describe it…
                    </button>
                  </div>
                  {(customOpen || style.id === CUSTOM_STYLE) && (
                    <div style={{ marginTop: 7 }}>
                      <textarea
                        rows={2} disabled={busy}
                        value={customWords}
                        onChange={(e) => setCustomWords(e.target.value)}
                        placeholder="1950s newspaper strip, coarse halftone, three colours"
                      />
                      <div style={{ display: "flex", gap: 6, alignItems: "center",
                                    flexWrap: "wrap", marginTop: 5 }}>
                        <button className="btn small" disabled={busy || !customWords.trim()
                                  || customWords.trim() === style.words}
                                onClick={() => changeStyle(CUSTOM_STYLE)}>
                          {drawnInStyle && confirmStyle === CUSTOM_STYLE
                            ? "Press again to restyle"
                            : "Use this look"}
                        </button>
                        <span style={{ fontSize: 10.5, color: "var(--text-faint)",
                                       lineHeight: 1.5 }}>
                          These words lead every panel prompt. The four presets
                          are a starting point, not the range.
                        </span>
                      </div>
                    </div>
                  )}
                  {confirmStyle && (
                    <div style={{ fontSize: 10.5, color: "var(--bad)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      Press again to restyle. The {drawnCount} drawn panel
                      {drawnCount === 1 ? "" : "s"}, the cast sheet and the rooms
                      go, because they were drawn in {styleLabel(style.id)}. The
                      division, the briefs and your notes stay. The old pictures
                      remain in the Vault.
                    </div>
                  )}
                  {styleHasMoved(style) && (
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      This board keeps the {styleLabel(style.id)} it was started in.
                      The preset has changed since, so a new board will look
                      different -- picking {styleLabel(style.id)} again here takes
                      the current version.
                    </div>
                  )}
                </div>
                {panels && (
                  <div className="field">
                    <label>References <em>drawn once, reused by every panel</em></label>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 5,
                                  alignItems: "center" }}>
                      <button className="btn small" disabled={busy || !sheet}
                              onClick={recastSheet}>
                        {sheet ? "Draw the cast sheet again" : "No cast sheet yet"}
                      </button>
                      {places.filter((k) => placeSheets[k]).map((k) => (
                        <button key={k} className="btn small" disabled={busy}
                                onClick={() => rebuildPlace(k)}>
                          Redraw {k}
                        </button>
                      ))}
                    </div>
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)", marginTop: 5,
                                  lineHeight: 1.5 }}>
                      A reference is drawn once and every panel that needs it is
                      drawn against the same one -- which is what keeps a face and
                      a room the same from panel to panel. Throwing one away here
                      rebuilds it on the next run and leaves the panels alone, so
                      you choose which of them no longer match.
                    </div>
                  </div>
                )}
                <div className="field">
                  <label>Model</label>
                  <select value={model?.id ?? ""} disabled={busy}
                          onChange={(e) => setModelId(e.target.value)}>
                    {usable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                </div>
              </>
            ) : (
              <>
                <div className="field">
                  <label>Shot</label>
                  <select value={sel.shot} disabled={busy}
                          onChange={(e) => edit(selIndex, { shot: e.target.value as Panel["shot"] })}>
                    <option value="wide">wide</option>
                    <option value="medium">medium</option>
                    <option value="close-up">close-up</option>
                  </select>
                </div>
                <div className="field">
                  <label>Story source <em>coverage anchor</em></label>
                  <div style={{ fontSize: 11, lineHeight: 1.55, color: "var(--text-dim)",
                                background: "var(--bg-sunk)", borderRadius: 5,
                                padding: "6px 8px" }}>
                    {sel.source || "No exact source quote was returned. Review this panel against the manuscript."}
                  </div>
                </div>
                <div className="field">
                  <label style={{ display: "flex", alignItems: "center", gap: 6,
                                  cursor: "pointer" }}>
                    <input type="checkbox" checked={sel.character_in_frame} disabled={busy}
                           style={{ width: "auto", margin: 0 }}
                           onChange={(e) => edit(selIndex, { character_in_frame: e.target.checked })} />
                    the one we follow is in this frame
                  </label>
                </div>
                <div className="field">
                  <label>People in frame <em>names from the story</em></label>
                  <input type="text" value={(sel.characters ?? []).join(", ")} disabled={busy}
                         onChange={(e) => edit(selIndex, {
                           characters: e.target.value.split(",").map((v) => v.trim()).filter(Boolean),
                           character_in_frame: Boolean(e.target.value.trim()),
                         })} />
                </div>
                <div className="field">
                  <label>Subject</label>
                  <input type="text" value={sel.subject} disabled={busy}
                         onChange={(e) => edit(selIndex, { subject: e.target.value })} />
                </div>
                <div className="field">
                  <label>Action</label>
                  <input type="text" value={sel.action} disabled={busy}
                         onChange={(e) => edit(selIndex, { action: e.target.value })} />
                </div>
                <div className="field">
                  <label>Setting</label>
                  <input type="text" value={sel.setting} disabled={busy}
                         onChange={(e) => edit(selIndex, { setting: e.target.value })} />
                </div>
                {sel.cast !== undefined && <>
                  <div className="field"><label>Mood</label><input value={sel.mood ?? ""} disabled={busy}
                    onChange={(e) => edit(selIndex, { mood: e.target.value })} /></div>
                  <div className="field"><label>Current location state</label><input value={sel.location_state ?? ""} disabled={busy}
                    onChange={(e) => edit(selIndex, { location_state: e.target.value })} /></div>
                  {sel.cast.map((actor, index) => <div className="field" key={actor.character_id}>
                    <label>{actor.name} · expression and body language</label>
                    <input aria-label={`${actor.name} expression`} value={actor.expression} disabled={busy}
                      onChange={(e) => edit(selIndex, { cast: sel.cast!.map((a, at) => at === index
                        ? { ...a, expression: e.target.value } : a) })} />
                    <input aria-label={`${actor.name} current outfit and condition`} placeholder="Current outfit and condition (replaces earlier appearance)"
                      value={actor.appearance ?? ""} disabled={busy}
                      onChange={(e) => edit(selIndex, { cast: sel.cast!.map((a, at) => at === index
                        ? { ...a, appearance: e.target.value } : a) })} />
                    <input aria-label={`${actor.name} body language`} value={actor.body_language} disabled={busy}
                      onChange={(e) => edit(selIndex, { cast: sel.cast!.map((a, at) => at === index
                        ? { ...a, body_language: e.target.value } : a) })} />
                  </div>)}
                </>}
                <div className="field">
                  {sel.dialogue?.some((line) => line.attribution_review) && <p className="notice warn">Some dialogue speakers need review. Original words were preserved; check the speaker names before drawing or lettering.</p>}
                  <label>Caption <em>what the picture cannot say</em></label>
                  <input type="text" value={sel.caption} disabled={busy}
                         onChange={(e) => edit(selIndex, { caption: e.target.value })} />
                </div>
                <div className="field">
                  <label>Dialogue <em>spoken aloud in this frame</em></label>
                  <input
                    type="text"
                    value={(sel.dialogue ?? []).map(
                      (d) => (d.speaker ? `${d.speaker}: ` : "") + d.text).join(" / ")}
                    disabled={busy}
                    placeholder="Name: what they say / Name: what they say"
                    onChange={(e) => edit(selIndex, {
                      // "Name: line / Name: line" is quicker to correct than a
                      // pair of fields per speaker, and matches how the writer
                      // returns it.
                      dialogue: e.target.value.split("/").map((raw) => {
                        const [a, ...rest] = raw.split(":");
                        return rest.length
                          ? { speaker: a.trim(), text: rest.join(":").trim() }
                          : { speaker: "", text: a.trim() };
                      }).filter((d) => d.text),
                    })}
                  />
                </div>
                {(sel.description ?? "").trim() && (
                  <div className="field">
                    <label>The worked-up scene <em>replaces the terse fields</em></label>
                    <textarea
                      value={sel.description} disabled={busy} rows={3}
                      style={{ fontSize: 11.5, color: "var(--text-dim)" }}
                      onChange={(e) => edit(selIndex, { description: e.target.value })}
                    />
                  </div>
                )}
                {(sel.place ?? "").trim() && (
                  <div className="field">
                    <label>This scene's place <em>every panel here shares it</em></label>
                    <div style={{ fontSize: 11, lineHeight: 1.55, color: "var(--text-dim)",
                                  background: "var(--bg-sunk)", borderRadius: 5,
                                  padding: "6px 8px" }}>
                      {sel.place}
                    </div>
                  </div>
                )}
                <div className="field">
                  <label>
                    This panel's place in the board
                    <em>the division is a draft</em>
                  </label>
                  {/* Every one of these was unsafe until the board stopped
                      keeping its work in arrays lined up by position. Moving a
                      panel left its picture and its note behind for whatever
                      landed in the slot, and the arrays stayed the right
                      length, so nothing looked wrong until the pages were
                      read. */}
                  <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                    <button className="btn small" disabled={busy || selIndex === 0}
                            title="Move this panel earlier"
                            onClick={() => movePanel(selIndex, -1)}>↑ Earlier</button>
                    <button className="btn small"
                            disabled={busy || selIndex >= panels.length - 1}
                            title="Move this panel later"
                            onClick={() => movePanel(selIndex, 1)}>↓ Later</button>
                    <button className="btn small" disabled={busy}
                            title="Add an empty panel after this one"
                            onClick={() => addPanelAfter(selIndex)}>+ After</button>
                    <button className="btn small" disabled={busy}
                            title="Split this beat into two panels"
                            onClick={() => splitPanel(selIndex)}>Split</button>
                    <button className="btn small"
                            disabled={busy || selIndex >= panels.length - 1}
                            title="Join this panel with the one after it"
                            onClick={() => mergePanel(selIndex)}>Join next</button>
                    <button className={"btn small"
                            + (confirmDelete === selIndex ? " danger" : "")}
                            disabled={busy || panels.length < 2}
                            onClick={() => {
                              if (confirmDelete !== selIndex) {
                                setConfirmDelete(selIndex); return;
                              }
                              setConfirmDelete(null);
                              deletePanel(selIndex);
                            }}>
                      {confirmDelete === selIndex
                        ? (drawn[selIndex]
                            ? "Remove it? The picture stays in the Vault"
                            : "Remove it?")
                        : "Remove"}
                    </button>
                  </div>
                </div>
                <div className="field">
                  <label>The brief <em>the exact words asked for</em></label>
                  <div style={{ fontSize: 11, lineHeight: 1.6, color: "var(--text-dim)",
                                background: "var(--bg-sunk)", borderRadius: 5,
                                padding: "7px 9px", userSelect: "text" }}>
                    {panelPrompt(sel, style, panelWho(sel), notes[selIndex])}
                  </div>
                </div>
                {drawn[selIndex] && (
                  <div className="field">
                    <label>Your note <em>redraws just this frame</em></label>
                    <input
                      type="text"
                      value={notes[selIndex] ?? ""}
                      disabled={busy}
                      placeholder="older, grey at the temples"
                      style={{ borderStyle: notes[selIndex]?.trim() ? "solid" : "dashed" }}
                      onChange={(e) =>
                        putArt(selIndex, { note: e.target.value })}
                    />
                    <button className="btn small full" style={{ marginTop: 6 }}
                            disabled={busy} onClick={() => void redraw(selIndex)}>
                      {notes[selIndex]?.trim() ? "Redraw with note" : "Redraw this panel"}
                    </button>
                    {/* The second kind of repair. A redraw reconsiders the
                        whole frame, which is right when the panel is wrong and
                        wasteful when it is nearly right -- correcting a hand
                        should not cost the face. */}
                    <button className="btn small full" style={{ marginTop: 6 }}
                            disabled={busy}
                            onClick={() => {
                              setMaskBytes(null);
                              setRepairing(repairing === selIndex ? null : selIndex);
                            }}>
                      {repairing === selIndex
                        ? "Cancel the painted repair"
                        : "Repair one part of it"}
                    </button>
                    {repairing === selIndex && (
                      <div style={{ marginTop: 8 }}>
                        <div style={{ fontSize: 10.5, color: "var(--text-faint)",
                                      lineHeight: 1.55, marginBottom: 6 }}>
                          Paint over what is wrong. Only the painted part is
                          drawn again; the rest of the panel is kept exactly as
                          it is. Your note above says what should be there.
                        </div>
                        <MaskCanvas
                          sourceId={drawn[selIndex]!}
                          onMaskChange={setMaskBytes}
                          disabled={busy}
                        />
                        <button className="btn small full primary"
                                style={{ marginTop: 6 }}
                                disabled={busy || !maskBytes}
                                onClick={() => void repair(selIndex)}>
                          {maskBytes
                            ? "Redraw the painted region"
                            : "Paint a region first"}
                        </button>
                      </div>
                    )}
                  </div>
                )}
                {(variants[selIndex]?.length ?? 0) > 0 && (
                  <div className="field">
                    <label>
                      Earlier pictures <em>this panel has been drawn before</em>
                    </label>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {variants[selIndex].map((id) => (
                        <button key={id} className="btn small"
                                disabled={busy}
                                title="Put this picture back"
                                style={{ padding: 3, lineHeight: 0 }}
                                onClick={() => restoreVariant(selIndex, id)}>
                          <img src={vaultUrl(id)} alt="An earlier version"
                               style={{ width: 64, height: 64, objectFit: "cover",
                                        borderRadius: 4, display: "block" }} />
                        </button>
                      ))}
                    </div>
                    <div style={{ fontSize: 10.5, color: "var(--text-faint)",
                                  marginTop: 5, lineHeight: 1.55 }}>
                      Click one to put it back. The current picture takes its
                      place here, so nothing is lost either way.
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* ---- the cast, along the bottom: it belongs to the whole board ---- */}
      {/* A story told in the first person never names the person it is about,
          so there is nothing for the cast reader to return for the lead -- and
          with no lead there is no reference sheet, which is what makes a face
          the same face from panel to panel. Asked for once, here, rather than
          left for someone to work out from the pictures. */}
      {cast?.first_person && !cast.people.some((p) => p.tier === 1) && (
        <div className="notice warn" style={{ margin: "10px 0" }}>
          <strong>This story is told by someone it never names.</strong>
          <div style={{ fontSize: 11.5, lineHeight: 1.6, marginTop: 4 }}>
            Give them a name and, if you like, a description. Without one there
            is no cast sheet, and a face drawn from nothing is a different face
            in every panel.
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 8 }}>
            <input value={narratorName} disabled={busy}
                   style={{ maxWidth: 160 }}
                   onChange={(e) => setNarratorName(e.target.value)}
                   placeholder="Name" />
            <input value={narratorLook} disabled={busy}
                   style={{ flex: 1, minWidth: 180 }}
                   onChange={(e) => setNarratorLook(e.target.value)}
                   placeholder="short black hair, red scarf (optional)" />
            <button className="btn small primary"
                    disabled={busy || !narratorName.trim()}
                    onClick={() => addPerson(narratorName, narratorLook, true)}>
              Add them
            </button>
          </div>
        </div>
      )}

      {cast && (cast.people.length > 0 || cast.places.length > 0) && (
        <div className="board-shelf">
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: ".13em",
                         textTransform: "uppercase", color: "var(--text-faint)" }}>
            Cast
          </span>
          {cast.people.map((p) => (
            <span key={p.name} className="pill" style={{
                     opacity: p.tier === 1 ? 1 : p.tier === 2 ? 0.8 : 0.62,
                     cursor: "pointer" }}
                    title={(p.description || "The story never says what they look like.")
                           + ` · ${p.mentions} mentions · click to describe`}
                    onClick={() => setEditingPerson(
                      editingPerson === p.name ? null : p.name)}>
              {p.name}
            </span>
          ))}
          {editingPerson && (() => {
            const p = cast.people.find((q) => q.name === editingPerson);
            if (!p) return null;
            return (
              <span style={{ display: "flex", gap: 6, alignItems: "center",
                             flexBasis: "100%", marginTop: 6 }}>
                <input
                  autoFocus defaultValue={p.description} disabled={busy}
                  style={{ flex: 1, minWidth: 200 }}
                  placeholder={`What does ${p.name} look like? The story does not say.`}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      describePerson(p.name, (e.target as HTMLInputElement).value);
                      setEditingPerson(null);
                    }
                    if (e.key === "Escape") setEditingPerson(null);
                  }}
                  onBlur={(e) => {
                    if (e.target.value !== p.description) {
                      describePerson(p.name, e.target.value);
                    }
                    setEditingPerson(null);
                  }} />
                <span style={{ fontSize: 10, color: "var(--text-faint)" }}>
                  appearance only &mdash; it leads every panel she is in
                </span>
              </span>
            );
          })()}
          {cast.places.length > 0 && <span className="divider" />}
          {cast.places.map((pl) => (
            <span key={pl.name} className="pill" style={{ opacity: 0.62 }}
                  title={pl.description || "The story never describes it."}>
              {pl.name}
            </span>
          ))}
        </div>
      )}

      {/* ---- the run bar: what it costs, said before you commit ---- */}
      <div className="board-run">
        {!panels ? (
          <>
            <button className="btn primary small"
              disabled={busy || !story.trim() || !writerReady || !storyLibrary}
              onClick={() => void divide()}>
              {stage === "dividing" ? "Understanding and directing…" : "Analyze story and plan panels"}
            </button>
            {!writerReady && (
              // Reading and dividing are both the writer's work. Saying so here
              // beats a failure after the press.
              <span style={{ fontSize: 10.5, color: "var(--warn)" }}>
                Install a story analyst or prompt writer from Models.
              </span>
            )}
          </>
        ) : (
          <>
            <span style={{ fontSize: 11.5, color: "var(--text-dim)",
                           fontVariantNumeric: "tabular-nums" }}>
              <b style={{ color: "var(--text)" }}>{drawnCount}</b> of {panels.length} drawn
            </span>
            {remaining > 0 && preparedCount > 0 && !busy && (
              <span className="stepper" title="Panels per press">
                <button type="button" onClick={() => setBatch((n) => Math.max(1, n - 1))}>
                  &minus;
                </button>
                <span>{Math.min(batch, remaining)} at a time</span>
                <button type="button" onClick={() => setBatch((n) => Math.min(24, n + 1))}>
                  +
                </button>
              </span>
            )}
            {remaining > 0 && !busy && (
              <select
                value={panelPx}
                onChange={(e) => setPanelPx(Number(e.target.value))}
                title="How large each panel is drawn"
                style={{ width: "auto", fontSize: 11, padding: "3px 6px" }}
              >
                {PANEL_SIZES.map(([label, px]) => (
                  <option key={px} value={px}>{label}</option>
                ))}
              </select>
            )}
            {/* Gated on the panels still waiting for a picture, which is what
                `unpreparedRemaining` was written for. `preparedCount > 0` let
                one prepared panel unlock drawing all of them, and the rest went
                down the terse-field path -- a minute of drawing each to find
                out. Preparing comes first, as the stage order promises. */}
            {unpreparedRemaining > 0 && (
              <button className="btn primary small" disabled={busy}
                      onClick={() => void enrich()}>
                {stage === "enriching" ? "Preparing briefs…"
                  : preparedCount === 0 ? "Prepare drawing briefs"
                  : `Prepare ${unpreparedRemaining} remaining brief${unpreparedRemaining === 1 ? "" : "s"}`}
              </button>
            )}
            {remaining > 0 && unpreparedRemaining === 0 && (
              <button className="btn primary small" disabled={busy} onClick={draw}>
                {stage === "drawing" ? `Drawing ${done}/${panels.length}…`
                  : stage === "building" ? "Building the rooms…"
                  : stage === "casting" ? "Casting…"
                  : drawnCount === 0
                    ? `Draw ${Math.min(batch, remaining)} of ${panels.length} · about ${estimate.low}–${estimate.high} min`
                    : `Draw next ${Math.min(batch, remaining)}`}
              </button>
            )}
            {remaining === 0 && !busy && (
              <span style={{ fontSize: 11, color: "var(--good)" }}>
                All {panels.length} drawn
              </span>
            )}
            {!busy && preparedCount > 0 && drawnCount === 0 && (
              <button className="btn small" disabled={busy} onClick={() => void enrich()}>
                Prepare again
              </button>
            )}
            {busy && <button className="btn small" onClick={cancel}>Cancel</button>}
            {/* Available from the first drawn panel, not the last.
                `compose` has always worked on whatever is drawn -- it filters
                to the panels that have a picture -- and this gate was the only
                thing insisting on a finished board. A board is twenty minutes
                to an hour, and seeing the first page laid out is how you find
                out the style is wrong while it still costs two panels to fix
                rather than twelve. */}
            {!busy && drawn.some(Boolean) && (
              <>
                <select value={layout} style={{ width: "auto", fontSize: 11, padding: "2px 6px" }}
                        onChange={(e) => setLayout(e.target.value)}>
                  <option value="page">Page</option>
                  <option value="strip">Scrolling strip</option>
                </select>
                <button className="btn small" disabled={composing}
                        onClick={() => void compose()}>
                  {composing ? "Composing…"
                    : remaining === 0 ? "Make a page"
                    : `Page from the ${drawnCount} drawn`}
                </button>
              </>
            )}
          </>
        )}
        <div style={{ flex: 1 }} />
        {!busy && (story || panels) && (
          <button
            className={"btn small" + (confirmReset === "new" ? " danger" : "")}
            onClick={() => {
              if (atRisk.length && confirmReset !== "new") {
                setConfirmReset("new"); return;
              }
              setConfirmReset(null);
              setStory(""); setCast(null); setProjectId(null);
              setCharacter(""); setSheet(null); clearCharacterSheets(); setStoryBook(null); setChapterId(null);
              setDoc({ panels: [], art: {} }); setDivided(false);
              setPages([]); setOverflow([]); setOwnSheet([]); setPlaceSheets({});
              setCoverage(null);
              setSelected(0);
            }}>
            {confirmReset === "new"
              ? `Start over, discarding ${atRisk.join(", ")}?`
              : "New board"}
          </button>
        )}
        {elapsed > 0 && busy && (
          <span style={{ fontSize: 10.5, color: "var(--text-faint)",
                         fontVariantNumeric: "tabular-nums" }}>
            {Math.floor(elapsed / 60)}m {elapsed % 60}s
          </span>
        )}
      </div>

      {prog && (
        <div style={{ marginTop: 8 }}>
          <JobProgress p={prog} label={
            stage === "dividing" ? "Dividing the story"
              : stage === "casting" ? "Casting the character"
              : `Panel ${done + 1} of ${panels?.length ?? 0}`} />
        </div>
      )}
    </div>
  );
}
