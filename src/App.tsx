import Audiobook from "./components/Audiobook";
import { EMPTY_LIBRARY, keepBook, type CharacterReference, type StoryBook, type StoryLibrary } from "./lib/story";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, errText, onEngineExit, onEngineProgress } from "./lib/api";
import { NowPlaying, type Track } from "./lib/nowPlaying";
import AudioPlayer from "./components/library/AudioPlayer";
import LivePlayer from "./components/library/LivePlayer";
import { Bar } from "./components/shared";
import type { HostInfo, ModelStatus, Recipe, SetupState, VaultItem, VaultStatus } from "./lib/types";
import Setup from "./components/Setup";
import VaultGate from "./components/VaultGate";
import Models from "./components/Models";
import Studio from "./components/Studio";
import Gallery from "./components/Gallery";
import Upscale from "./components/Upscale";
import Video from "./components/Video";
import Storyboard from "./components/Storyboard";
import Welcome from "./components/Welcome";
import { loadPref, savePref } from "./lib/prefs";
import { migrateBoardDraft } from "./lib/boardDraft";
import Security from "./components/Security";
import Activity from "./components/Activity";
import { Toast } from "./components/shared";

type Tab = "generate" | "edit" | "board" | "audiobook" | "video" | "upscale" | "models" | "gallery" | "activity" | "security";

const TABS: [Tab, string, string][] = [
  ["generate", "✦", "Generate"],
  ["edit", "✎", "Edit"],
  ["upscale", "⤢", "Upscale"],
  ["video", "▷", "Video"],
  ["board", "▤", "Board"],
  ["audiobook", "♫", "Audiobook"],
  ["gallery", "▦", "Vault"],
  ["models", "◍", "Models"],
  ["activity", "◷", "Activity"],
  ["security", "⚿", "Security"],
];
/** Sidebar sections, macOS-style, so ten flat tabs read as four kinds of work. */
const SECTIONS: [string, Tab[]][] = [
  ["Create", ["generate", "edit", "upscale", "video"]],
  ["Story", ["board", "audiobook"]],
  ["Library", ["gallery"]],
  ["Manage", ["models", "activity", "security"]],
];

/** Lock the vault after this long without interaction. */
const IDLE_LOCK_MS = 15 * 60 * 1000;

export default function App() {
  const [host, setHost] = useState<HostInfo | null>(null);
  const [setup, setSetup] = useState<SetupState | null>(null);
  const [vault, setVault] = useState<VaultStatus | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [items, setItems] = useState<VaultItem[]>([]);
  const [tab, setTab] = useState<Tab>(() => (import.meta.env.MODE === "mock"
    && new URLSearchParams(location.search).get("tab") as Tab) || "generate");
  useEffect(() => { tabRef.current = tab; }, [tab]);
  const [toast, setToast] = useState<{ msg: string; bad?: boolean } | null>(null);

  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  // Every job the engine is running, with the screen it was started from, so
  // it can be watched and stopped from anywhere rather than only its own tab.
  const [jobs, setJobs] = useState<Record<string, { label: string; progress: number | null; tab: Tab; at: number }>>({});
  const tabRef = useRef<Tab>("generate");
  const [track, setTrack] = useState<Track | null>(null);
  const nowPlaying = useMemo(() => ({ track, play: setTrack, stop: () => setTrack(null) }), [track]);
  // A narration heard while it is read plays from its own audio queue, which
  // would keep sounding with no player on screen: it stops with its track.
  const liveTrack = useRef<Track["live"]>(undefined);
  useEffect(() => {
    if (liveTrack.current && liveTrack.current !== track?.live) liveTrack.current.close();
    liveTrack.current = track?.live;
  }, [track]);
  // Owned here so Generate and the Vault can both hand images to the Edit tab.
  const [editImages, setEditImages] = useState<string[]>([]);
  const [videoFrame, setVideoFrame] = useState<string[]>([]);
  const [recipe, setRecipe] = useState<Recipe | null>(null);
  /** Dismissed once, per install. Someone who has chosen their own models
   *  should not be met by the starter list every launch. */
  const [welcomed, setWelcomed] = useState(() => loadPref("welcomed", false));
  const [upscaleImage, setUpscaleImage] = useState<string[]>([]);
  /**
   * A comic the Vault asked to reopen, by project id.
   *
   * Owned here because the request crosses tabs: the card is in the Vault and
   * the board it opens is in Board.
   */
  const [openBoard, setOpenBoard] = useState<string | null>(null);

  const [storyLibrary, setStoryLibrary] = useState<StoryLibrary | null>(null);
  const libraryRef = useRef<StoryLibrary | null>(null);
  const libraryWrites = useRef<Promise<void>>(Promise.resolve());
  const saveLibrary = useCallback((update: (current: StoryLibrary) => StoryLibrary) => {
    const write = libraryWrites.current.catch(() => {}).then(async () => {
      if (!libraryRef.current) throw new Error("The story library has not loaded; nothing was overwritten.");
      const next = update(libraryRef.current);
      await api.storyLibrarySet(JSON.stringify(next));
      libraryRef.current = next;
      setStoryLibrary(next);
    });
    libraryWrites.current = write;
    return write;
  }, []);
  const keepStory = useCallback((book: StoryBook) =>
    saveLibrary((current) => keepBook(current, book)), [saveLibrary]);

  const saveCharacterReference = useCallback((bookId: string, characterId: string, reference: CharacterReference) =>
    saveLibrary((current) => ({ ...current, books: current.books.map((book) => book.id === bookId
      ? { ...book, references: { ...book.references, [characterId]: reference } } : book) })), [saveLibrary]);

  // An error waits to be read; good news arriving after it must not replace
  // it and then fade, taking the error with it.
  const notify = useCallback((msg: string, bad?: boolean) =>
    setToast((current) => (current?.bad && !bad ? current : { msg, bad })), []);

  /**
   * Anything that got away.
   *
   * A rejected promise nobody caught produces no message at all, which on
   * screen is indistinguishable from the button doing nothing -- the single
   * most common way this app has wasted someone's time. The error boundary
   * only sees failures during render; these happen after it. Surfacing them
   * costs nothing and turns silence into something a person can act on and
   * copy.
   */
  useEffect(() => {
    const onReject = (e: PromiseRejectionEvent) => {
      const r: unknown = e.reason;
      const msg = typeof r === "string" ? r
        : r instanceof Error ? r.message
        : (() => { try { return JSON.stringify(r); } catch { return String(r); } })();
      notify(msg || "Something failed without saying what.", true);
    };
    const onError = (e: ErrorEvent) => {
      notify(e.message || "Something failed without saying what.", true);
    };
    window.addEventListener("unhandledrejection", onReject);
    window.addEventListener("error", onError);
    return () => {
      window.removeEventListener("unhandledrejection", onReject);
      window.removeEventListener("error", onError);
    };
  }, [notify]);

  /**
   * Carry work between tabs by id.
   *
   * Results are already vault items, so moving one onward costs nothing and
   * never touches the filesystem. This is what turns four separate tools into
   * a pipeline: generate, edit that, animate the result.
   */
  const send = useCallback((dest: "edit" | "video" | "upscale", ids: string[]) => {
    if (dest === "edit") { setEditImages(ids); setTab("edit"); }
    if (dest === "video") { setVideoFrame(ids.slice(0, 1)); setTab("video"); }
    if (dest === "upscale") { setUpscaleImage(ids.slice(0, 1)); setTab("upscale"); }
    const where = dest === "edit" ? "Edit" : dest === "video" ? "Video" : "Upscale";
    notify(`Carried over to ${where}.`);
  }, [notify]);

  const sendToEdit = useCallback((ids: string[]) => send("edit", ids), [send]);

  /**
   * Load a past run's settings back into the tab that produced it.
   *
   * Reproducing an image needs nothing new: the prompt, seed, size and steps
   * were all recorded when it was made. Keeping the seed repeats it exactly;
   * dropping it explores around it, which is what people actually want most of
   * the time.
   *
   * Everything the run sent is copied across, not just the handful the first
   * version carried. "Same seed" promises the same picture, and it cannot keep
   * that promise while the negative prompt, the adapters, the edit route and
   * the strength are all quietly taken from whatever the controls were last
   * left on. Fields the item does not have stay undefined and the Studio then
   * leaves the matching control where it is, so old items behave as before.
   */
  const reuse = useCallback((item: VaultItem, reuseSeed: boolean) => {
    setRecipe({
      kind: item.kind,
      prompt: item.prompt,
      modelName: item.model,
      modelRepo: item.model_repo,
      seed: item.seed,
      width: item.width,
      height: item.height,
      steps: item.steps,
      guidance: item.guidance,
      inputs: item.inputs,
      reuseSeed,
      negativePrompt: item.negative_prompt,
      loras: item.loras,
      i2iMode: item.i2i_mode,
      imageStrength: item.image_strength,
      outpaintPadding: item.outpaint_padding,
      outpaintFill: item.outpaint_fill,
      lowRam: item.low_ram,
    });
    if (item.kind === "video") setTab("video");
    else if (item.kind === "edit") { setEditImages(item.inputs); setTab("edit"); }
    else setTab("generate");
    notify(reuseSeed ? "Settings restored — same seed." : "Settings restored — new seed each run.");
  }, [notify]);

  // Any tab can show that the engine is working, so leaving Generate mid-run
  // does not look like the job vanished.
  useEffect(() => {
    let un: (() => void) | undefined;
    let idle: number | undefined;
    onEngineProgress((p) => {
      // Engine work is activity too. A person waiting for a long local render
      // must not lose the result merely because they are not moving the mouse.
      window.dispatchEvent(new Event("modelstudio:activity"));
      const phase = p.phase === "download" ? "Downloading" :
        p.phase === "load" ? "Loading model" : "Working";
      // A download is the one job long enough that you will leave the tab it
      // started on, so the shell carries the same numbers the Models tab
      // shows rather than a bare "Downloading" that tells you nothing.
      if (p.phase === "download" && p.total_bytes) {
        const gb = (n: number) => (n / 1024 ** 3).toFixed(1);
        const pct = Math.round((p.progress ?? 0) * 100);
        const stalled = (p.stalled_seconds ?? 0) >= 45;
        setBusyLabel(stalled
          ? `Downloading — stalled ${Math.round(p.stalled_seconds ?? 0)}s`
          : `Downloading ${pct}% · ${gb(p.done_bytes ?? 0)}/${gb(p.total_bytes)} GB`);
      } else {
        setBusyLabel(p.step && p.total_steps ? `${phase} ${p.step}/${p.total_steps}` : phase);
      }
      setJobs((all) => ({ ...all, [p.job_id]: {
        label: p.message || phase, progress: p.progress,
        tab: all[p.job_id]?.tab ?? tabRef.current, at: Date.now() } }));
      window.clearTimeout(idle);
      // Progress events stop when a job ends; treat silence as done.
      idle = window.setTimeout(() => setBusyLabel(null), 4000);
    }).then((u) => { un = u; });
    // A job silent for 20 seconds has finished (or failed, which its own
    // screen reports); model loads go quiet for less than that.
    const sweep = window.setInterval(() => setJobs((all) => {
      const live = Object.entries(all).filter(([, j]) => Date.now() - j.at < 20000);
      return live.length === Object.keys(all).length ? all : Object.fromEntries(live);
    }), 3000);
    return () => { un?.(); window.clearTimeout(idle); window.clearInterval(sweep); };
  }, []);
  // Recordings are served from the vault, so locking it ends playback.
  useEffect(() => { if (!vault?.unlocked) setTrack(null); }, [vault?.unlocked]);

  useEffect(() => {
    if (!vault?.unlocked) {
      libraryRef.current = null;
      setStoryLibrary(null);
      return;
    }
    let live = true;
    void api.storyLibraryGet().then((raw) => {
      if (!live) return;
      const loaded = raw ? JSON.parse(raw) as StoryLibrary : { ...EMPTY_LIBRARY };
      if (loaded.version !== 1 || !Array.isArray(loaded.books)) throw new Error("Unsupported story library");
      libraryRef.current = loaded;
      setStoryLibrary(loaded);
    }).catch((e) => notify(`Could not load the story library: ${errText(e)}`, true));
    return () => { live = false; };
  }, [vault?.unlocked, notify]);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
      setHost(await api.hostInfo());
    } catch (e) { notify(errText(e), true); }
  }, [notify]);

  // Whether the vault has been listed yet, and why it could not be. Without
  // these the library said "The vault is empty" while the first listing was
  // still on its way, and said it again when the listing failed.
  const [itemsLoaded, setItemsLoaded] = useState(false);
  const [itemsError, setItemsError] = useState<string | null>(null);
  const refreshItems = useCallback(async () => {
    try {
      setItems(await api.vaultList());
      setItemsError(null);
    } catch (e) {
      // A locked vault fails here too, but then the lock screen is what is
      // on show; this only ever reaches the library while it is unlocked.
      setItemsError(errText(e));
    } finally {
      setItemsLoaded(true);
    }
  }, []);

  useEffect(() => {
    (async () => {
      try {
        setHost(await api.hostInfo());
        setSetup(await api.setupState());
        setVault(await api.vaultStatus());
      } catch (e) { notify(errText(e), true); }
    })();
  }, [notify]);

  const ready = setup?.ready && vault?.unlocked;

  /**
   * Notice when the runtime becomes usable without being told.
   *
   * The setup state is read once when the window opens. A window opened while
   * the runtime was broken therefore holds "not ready" forever, and the only
   * way out is to run the install again -- even when the runtime has since
   * been repaired, or was repaired by hand, or finished in another window.
   * That is how someone ends up installing the runtime three times and being
   * shown the same screen after each one.
   *
   * So while it is not ready, ask again: on a timer, and whenever the window
   * comes back to the front. The check is a few filesystem stats, and it stops
   * the moment the answer changes.
   */
  useEffect(() => {
    if (setup?.ready) return;
    let live = true;
    const recheck = async () => {
      try {
        const next = await api.setupState();
        if (live && next.ready) setSetup(next);
      } catch { /* the host is busy; the next tick will ask again */ }
    };
    const t = window.setInterval(recheck, 3000);
    window.addEventListener("focus", recheck);
    return () => {
      live = false;
      window.clearInterval(t);
      window.removeEventListener("focus", recheck);
    };
  }, [setup?.ready]);

  useEffect(() => {
    if (!ready) return;
    void migrateBoardDraft().catch((e) =>
      notify(`Could not secure the old Board draft: ${errText(e)}`, true));
    void refreshModels();
    void refreshItems();
  }, [ready, refreshModels, refreshItems, notify]);

  useEffect(() => {
    let un: (() => void) | undefined;
    onEngineExit(() => notify("The engine stopped. It will restart on the next run.", true))
      .then((u) => { un = u; });
    return () => un?.();
  }, [notify]);

  // Idle auto-lock. Leaving a machine unattended is the failure mode an
  // at-rest-encrypted vault is otherwise powerless against.
  const timer = useRef<number | undefined>(undefined);
  const warnTimer = useRef<number | undefined>(undefined);
  // A minute's notice: the lock unmounts every screen, so an unsaved prompt
  // or a half-edited panel would otherwise vanish without warning.
  const [lockSoon, setLockSoon] = useState(false);
  useEffect(() => {
    if (!ready) return;
    const reset = () => {
      window.clearTimeout(timer.current);
      window.clearTimeout(warnTimer.current);
      setLockSoon(false);
      warnTimer.current = window.setTimeout(() => setLockSoon(true), IDLE_LOCK_MS - 60_000);
      timer.current = window.setTimeout(async () => {
        try {
          setVault(await api.vaultLock());
          notify("Vault locked after 15 minutes of inactivity.");
        } catch {
          // Active jobs deliberately refuse a lock. Re-arm the timer so the
          // vault still locks once the work has gone quiet.
          reset();
        }
      }, IDLE_LOCK_MS);
    };
    const events = ["mousemove", "keydown", "mousedown", "wheel", "modelstudio:activity"];
    events.forEach((e) => window.addEventListener(e, reset, { passive: true }));
    reset();
    return () => {
      window.clearTimeout(timer.current);
      window.clearTimeout(warnTimer.current);
      events.forEach((e) => window.removeEventListener(e, reset));
    };
  }, [ready, notify]);

  // Above every early return, because hooks cannot be conditional. This sat
  // below them: while setup and the vault were still loading, the component
  // returned before reaching it, so the first render ran one fewer hook than
  // the next -- and React tore the tree down with "rendered more hooks than
  // during the previous render". On launch, every time.
  /**
   * What the library holds, counted the way it is shown.
   *
   * This was `items.length` -- every row in the vault. The Vault groups a
   * board into one card, so a person who had just composed a twelve-panel
   * board over two scenes saw "Composed 2 pages" and then a badge reading 17:
   * the cast sheet, two room sheets, twelve panels and two pages, counted as
   * though they were seventeen separate things they had made. They made one
   * comic. Masks are working data and are not in the library at all.
   */
  const vaultCount = useMemo(() => {
    const loose = items.filter((i) => i.kind !== "mask");
    const projects = new Set<string>();
    let singles = 0;
    for (const item of loose) {
      if (item.project) projects.add(item.project);
      else singles++;
    }
    return projects.size + singles;
  }, [items]);

  if (!setup || !vault) {
    // Draggable while it loads, like every other screen.
    return <><div className="drag-region" data-tauri-drag-region />
      <div className="empty-state" role="status" style={{ paddingTop: 160 }}>Loading…</div></>;
  }

  // The runtime has to exist before a vault is worth creating.
  if (!setup.ready) {
    return (
      <>
        <div className="drag-region" data-tauri-drag-region />
        <Setup
          host={host}
          state={setup}
          onDone={async () => {
            setSetup(await api.setupState());
            notify("Runtime installed.");
          }}
        />
        {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
      </>
    );
  }

  if (!vault.unlocked) {
    return (
      <>
        <div className="drag-region" data-tauri-drag-region />
        <VaultGate status={vault} onOpen={setVault} />
        {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
      </>
    );
  }

  const installedCount = models.filter((m) => m.installed).length;

  // Nothing installed means no tab can do anything, so the first screen asks
  // one question instead of presenting twenty-six models and a memory budget.
  if (installedCount === 0 && !welcomed && models.length > 0) {
    return (
      <>
        <div className="drag-region" data-tauri-drag-region />
        <div className="shell">
          <main className="main">
            <div className="topbar" data-tauri-drag-region>
              <h1 data-tauri-drag-region>Model Studio</h1>
            </div>
            <Welcome
              models={models}
              notify={notify}
              onChanged={refreshModels}
              onSkip={() => { setWelcomed(true); savePref("welcomed", true); setTab("models"); }}
            />
          </main>
        </div>
        {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
      </>
    );
  }
  const diskUsedFrac = host
    ? Math.min(1, 1 - host.free_disk_gib / Math.max(host.total_disk_gib, 1))
    : 0;

  return (
    <NowPlaying.Provider value={nowPlaying}>
    <div className="shell">
      <div className="drag-region" data-tauri-drag-region />
      <aside className="sidebar">
        <div className="brand">
          Model Studio
          <span>Local · Encrypted · Apple Silicon</span>
        </div>
        <nav className="nav" aria-label="Sections">
          {SECTIONS.map(([heading, ids]) => (
            <div key={heading} className="nav-section" role="group" aria-labelledby={`nav-${heading}`}>
              <div className="nav-heading" id={`nav-${heading}`}>{heading}</div>
              {TABS.filter(([id]) => ids.includes(id)).map(([id, glyph, label]) => (
                <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}
                  aria-current={tab === id ? "page" : undefined} title={label}>
                  <span className="glyph" aria-hidden="true">{glyph}</span>
                  <span className="nav-label">{label}</span>
                  {id === "models" && installedCount > 0 && (
                    <span className="badge" aria-label={`${installedCount} installed`}>{installedCount}</span>
                  )}
                  {id === "gallery" && vaultCount > 0 && (
                    <span className="badge" aria-label={`${vaultCount} items`}>{vaultCount}</span>
                  )}
                </button>
              ))}
            </div>
          ))}
        </nav>

        {Object.keys(jobs).length > 0 && (
          <section className="jobs" aria-label="Running jobs">
            <div className="nav-heading">Running</div>
            {Object.entries(jobs).map(([id, j]) => (
              <div className="job" key={id}>
                <button type="button" className="job-name" onClick={() => setTab(j.tab)}
                  title={`Go to ${TABS.find(([t]) => t === j.tab)?.[2]}`}>
                  <span className="t-meta">{TABS.find(([t]) => t === j.tab)?.[2]}</span> {j.label}
                </button>
                <div className="job-row">
                  <Bar value={j.progress} label={j.label} />
                  <button type="button" className="btn small" aria-label={`Stop ${j.label}`}
                    onClick={() => void api.cancelJob(id).catch((e) => notify(errText(e), true))}>Stop</button>
                </div>
              </div>
            ))}
          </section>
        )}

        <div className="sidebar-foot">
          {host && (
            <>
              {/* A label, not a meter: it was a bar that was always full. */}
              <div className="meter">
                <div className="meter-label">
                  <span>Memory for models</span>
                  <span>{host.usable_ram_gib.toFixed(1)} GB</span>
                </div>
              </div>
              <div className="meter">
                <div className="meter-label">
                  <span>Disk</span>
                  <span>{host.free_disk_gib.toFixed(0)} GB free</span>
                </div>
                <div className="meter-track">
                  <div
                    className={
                      "meter-fill" +
                      (host.free_disk_gib < 10 ? " bad" : host.free_disk_gib < 25 ? " warn" : "")
                    }
                    style={{ width: `${diskUsedFrac * 100}%` }}
                  />
                </div>
              </div>
              <div style={{ fontSize: 11, color: "var(--text-faint)" }}>
                {host.chip} · {host.total_ram_gib.toFixed(0)} GB
              </div>
            </>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="topbar" data-tauri-drag-region>
          <h1 data-tauri-drag-region>{TABS.find(([id]) => id === tab)?.[2]}</h1>
          <div className="spacer" data-tauri-drag-region />
          {busyLabel && (
            <span
              className={"pill" + (busyLabel.includes("stalled") ? " warn" : "")}
              title={busyLabel.includes("stalled")
                ? "No data has arrived for a while. It may need cancelling and retrying."
                : "A job is running in the engine"}
            >
              <span className="spin" aria-hidden="true">◐</span> {busyLabel}
              {/* Said, not only hovered: a stall needs action. */}
              {busyLabel.includes("stalled") && <span> — may need cancelling and retrying</span>}
            </span>
          )}
          <span className="pill good" title="Content is encrypted at rest">
            <span className="dot" aria-hidden="true" />Unlocked<span className="visually-hidden"> — content is encrypted at rest</span>
          </span>
          {(tab === "generate" || tab === "edit" || tab === "upscale") && (
            <button
              className="btn small"
              title="Cancel any running job and release the model from memory"
              onClick={async () => {
                try {
                  await api.unloadModel();
                  setBusyLabel(null);
                  notify("Released model memory. Any running job was cancelled.");
                } catch (e) { notify(errText(e), true); }
              }}
            >
              Free memory
            </button>
          )}
          {tab === "models" && <button className="btn small" onClick={refreshModels}>Refresh</button>}
          {tab === "gallery" && <button className="btn small" onClick={refreshItems}>Refresh</button>}
          <button
            className="btn small"
            onClick={async () => {
              try { setVault(await api.vaultLock()); }
              catch (e) { notify(errText(e), true); }
            }}
          >
            Lock
          </button>
        </div>

        {/* Positioned, so the Vault's docked viewer can cover the content
            area alone -- sidebar, top bar and player stay usable. */}
        <div className="content-wrap">
        <div className="content">
          {/*
            Generate and Edit stay mounted and are hidden with CSS rather than
            unmounted. Switching tabs during a run would otherwise discard the
            job's progress, elapsed time and result -- the work continues in the
            engine, but the UI would forget it was ever started.
          */}
          <div style={{ display: tab === "generate" ? "block" : "none" }}>
            <Studio
              mode="generate"
              models={models}
              notify={notify}
              onProduced={refreshItems}
              onSendToEdit={sendToEdit}
              send={send}
              recipe={recipe?.kind !== "video" && recipe?.kind !== "edit" ? recipe : null}
              onRecipeUsed={() => setRecipe(null)}
            />
          </div>
          <div style={{ display: tab === "edit" ? "block" : "none" }}>
            <Studio
              mode="edit"
              models={models}
              notify={notify}
              onProduced={refreshItems}
              images={editImages}
              onImagesChange={setEditImages}
              send={send}
              recipe={recipe?.kind === "edit" ? recipe : null}
              onRecipeUsed={() => setRecipe(null)}
            />
          </div>
          <div style={{ display: tab === "board" ? "block" : "none" }}>
            <Storyboard
              models={models}
              notify={notify}
              onProduced={refreshItems}
              storyLibrary={storyLibrary}
              onStoryAnalyzed={keepStory}
              onCharacterReference={saveCharacterReference}
              openProject={openBoard}
              onOpened={() => setOpenBoard(null)}
            />
          </div>
          <div style={{ display: tab === "audiobook" ? "block" : "none" }}>
            <Audiobook library={storyLibrary} models={models} notify={notify}
              onLibraryChange={saveLibrary} onProduced={refreshItems} />
          </div>
          <div style={{ display: tab === "video" ? "block" : "none" }}>
            <Video
              models={models}
              notify={notify}
              onProduced={refreshItems}
              firstFrame={videoFrame}
              onFirstFrameChange={setVideoFrame}
            />
          </div>
          <div style={{ display: tab === "upscale" ? "block" : "none" }}>
            <Upscale
              models={models}
              notify={notify}
              onProduced={refreshItems}
              images={upscaleImage}
              onImagesChange={setUpscaleImage}
              send={send}
            />
          </div>
          {tab === "models" && (
            <Models models={models} host={host} onChanged={refreshModels} notify={notify} />
          )}
          {/* Kept mounted like Generate and Edit: leaving the Vault and coming
              back finds the same file open at the same place. */}
          <div style={{ display: tab === "gallery" ? "block" : "none" }}>
            <Gallery
              active={tab === "gallery"}
              items={items}
              onChanged={refreshItems}
              notify={notify}
              send={send}
              models={models}
              onReuse={reuse}
              onOpenBoard={(project) => { setOpenBoard(project); setTab("board"); }}
              loading={!itemsLoaded}
              loadError={itemsError}
            />
          </div>
          {tab === "activity" && <Activity notify={notify} />}
          {tab === "security" && (
            <Security
              status={vault}
              onStatus={setVault}
              notify={notify}
              onImported={refreshItems}
              onStorageMoved={refreshModels}
            />
          )}
        </div>
        </div>
        {lockSoon && (
          <div className="notice warn lock-soon" role="alert">
            The vault locks in a minute to protect your work while you are away.
            <button type="button" className="btn small primary"
              onClick={() => window.dispatchEvent(new Event("modelstudio:activity"))}>Stay unlocked</button>
          </div>
        )}
        {track && (
          <section className="now-playing" aria-label="Now playing">
            {track.live
              ? <LivePlayer key={track.src} queue={track.live} title={track.title} />
              : <AudioPlayer key={`${track.src}@${track.start ?? 0}`} src={track.src} title={track.title}
                  markers={track.markers ?? []} start={track.start} autoPlay follow />}
            <button type="button" className="btn small" aria-label="Close player" onClick={() => setTrack(null)}>
              <span aria-hidden>×</span>
            </button>
          </section>
        )}
      </main>

      {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
    </div>
    </NowPlaying.Provider>
  );
}
