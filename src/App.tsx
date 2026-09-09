import { useCallback, useEffect, useRef, useState } from "react";
import { api, errText, onEngineExit, onEngineProgress } from "./lib/api";
import type { HostInfo, ModelStatus, SetupState, VaultItem, VaultStatus } from "./lib/types";
import Setup from "./components/Setup";
import VaultGate from "./components/VaultGate";
import Models from "./components/Models";
import Studio from "./components/Studio";
import Gallery from "./components/Gallery";
import Upscale from "./components/Upscale";
import Security from "./components/Security";
import Activity from "./components/Activity";
import { Toast } from "./components/shared";

type Tab = "generate" | "edit" | "upscale" | "models" | "gallery" | "activity" | "security";

const TABS: [Tab, string, string][] = [
  ["generate", "✦", "Generate"],
  ["edit", "✎", "Edit"],
  ["upscale", "⤢", "Upscale"],
  ["models", "◍", "Models"],
  ["gallery", "▦", "Vault"],
  ["activity", "◷", "Activity"],
  ["security", "⚿", "Security"],
];

/** Lock the vault after this long without interaction. */
const IDLE_LOCK_MS = 15 * 60 * 1000;

export default function App() {
  const [host, setHost] = useState<HostInfo | null>(null);
  const [setup, setSetup] = useState<SetupState | null>(null);
  const [vault, setVault] = useState<VaultStatus | null>(null);
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [items, setItems] = useState<VaultItem[]>([]);
  const [tab, setTab] = useState<Tab>("generate");
  const [toast, setToast] = useState<{ msg: string; bad?: boolean } | null>(null);

  const [busyLabel, setBusyLabel] = useState<string | null>(null);
  // Owned here so Generate and the Vault can both hand images to the Edit tab.
  const [editImages, setEditImages] = useState<string[]>([]);

  const notify = useCallback((msg: string, bad?: boolean) => setToast({ msg, bad }), []);

  const sendToEdit = useCallback((ids: string[]) => {
    setEditImages(ids);
    setTab("edit");
    notify(ids.length === 1 ? "Opened in Edit." : `Opened ${ids.length} images in Edit.`);
  }, [notify]);

  // Any tab can show that the engine is working, so leaving Generate mid-run
  // does not look like the job vanished.
  useEffect(() => {
    let un: (() => void) | undefined;
    let idle: number | undefined;
    onEngineProgress((p) => {
      const phase = p.phase === "download" ? "Downloading" :
        p.phase === "load" ? "Loading model" : "Working";
      setBusyLabel(p.step && p.total_steps ? `${phase} ${p.step}/${p.total_steps}` : phase);
      window.clearTimeout(idle);
      // Progress events stop when a job ends; treat silence as done.
      idle = window.setTimeout(() => setBusyLabel(null), 4000);
    }).then((u) => { un = u; });
    return () => { un?.(); window.clearTimeout(idle); };
  }, []);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
      setHost(await api.hostInfo());
    } catch (e) { notify(errText(e), true); }
  }, [notify]);

  const refreshItems = useCallback(async () => {
    try { setItems(await api.vaultList()); } catch { /* locked or empty */ }
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

  useEffect(() => {
    if (!ready) return;
    void refreshModels();
    void refreshItems();
  }, [ready, refreshModels, refreshItems]);

  useEffect(() => {
    let un: (() => void) | undefined;
    onEngineExit(() => notify("The engine stopped. It will restart on the next run.", true))
      .then((u) => { un = u; });
    return () => un?.();
  }, [notify]);

  // Idle auto-lock. Leaving a machine unattended is the failure mode an
  // at-rest-encrypted vault is otherwise powerless against.
  const timer = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!ready) return;
    const reset = () => {
      window.clearTimeout(timer.current);
      timer.current = window.setTimeout(async () => {
        try {
          setVault(await api.vaultLock());
          notify("Vault locked after 15 minutes of inactivity.");
        } catch { /* already locked */ }
      }, IDLE_LOCK_MS);
    };
    const events: (keyof WindowEventMap)[] = ["mousemove", "keydown", "mousedown", "wheel"];
    events.forEach((e) => window.addEventListener(e, reset, { passive: true }));
    reset();
    return () => {
      window.clearTimeout(timer.current);
      events.forEach((e) => window.removeEventListener(e, reset));
    };
  }, [ready, notify]);

  if (!setup || !vault) {
    return <div className="empty-state" style={{ paddingTop: 160 }}>Loading…</div>;
  }

  // The runtime has to exist before a vault is worth creating.
  if (!setup.ready) {
    return (
      <>
        <div className="drag-region" />
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
        <div className="drag-region" />
        <VaultGate status={vault} onOpen={setVault} />
        {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
      </>
    );
  }

  const installedCount = models.filter((m) => m.installed).length;
  const diskUsedFrac = host
    ? Math.min(1, 1 - host.free_disk_gib / Math.max(host.total_disk_gib, 1))
    : 0;

  return (
    <div className="shell">
      <div className="drag-region" />
      <aside className="sidebar">
        <div className="brand">
          Model Studio
          <span>Local · Encrypted · Apple Silicon</span>
        </div>
        <nav className="nav">
          {TABS.map(([id, glyph, label]) => (
            <button key={id} className={tab === id ? "active" : ""} onClick={() => setTab(id)}>
              <span className="glyph">{glyph}</span>
              {label}
              {id === "models" && installedCount > 0 && (
                <span className="badge">{installedCount}</span>
              )}
              {id === "gallery" && items.length > 0 && (
                <span className="badge">{items.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          {host && (
            <>
              <div className="meter">
                <div className="meter-label">
                  <span>Memory for models</span>
                  <span>{host.usable_ram_gib.toFixed(1)} GB</span>
                </div>
                <div className="meter-track">
                  <div className="meter-fill" style={{ width: "100%" }} />
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
              <div style={{ fontSize: 10, color: "var(--text-faint)" }}>
                {host.chip} · {host.total_ram_gib.toFixed(0)} GB
              </div>
            </>
          )}
        </div>
      </aside>

      <main className="main">
        <div className="topbar">
          <h1>{TABS.find(([id]) => id === tab)?.[2]}</h1>
          <div className="spacer" />
          {busyLabel && (
            <span className="pill" title="A job is running in the engine">
              <span className="spin">◐</span> {busyLabel}
            </span>
          )}
          <span className="pill good" title="Content is encrypted at rest">
            <span className="dot" />Unlocked
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
            />
          </div>
          <div style={{ display: tab === "upscale" ? "block" : "none" }}>
            <Upscale models={models} notify={notify} onProduced={refreshItems} />
          </div>
          {tab === "models" && (
            <Models models={models} host={host} onChanged={refreshModels} notify={notify} />
          )}
          {tab === "gallery" && (
            <Gallery
              items={items}
              onChanged={refreshItems}
              notify={notify}
              onSendToEdit={sendToEdit}
            />
          )}
          {tab === "activity" && <Activity notify={notify} />}
          {tab === "security" && (
            <Security
              status={vault}
              onStatus={setVault}
              notify={notify}
              onImported={refreshItems}
            />
          )}
        </div>
      </main>

      {toast && <Toast msg={toast.msg} bad={toast.bad} onDone={() => setToast(null)} />}
    </div>
  );
}
