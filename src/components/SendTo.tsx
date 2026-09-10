import type { ModelStatus } from "../lib/types";

export type Destination = "edit" | "video" | "upscale";

export interface SendToTargets {
  /** Where a result can be carried next, given what is installed. */
  send: (dest: Destination, ids: string[]) => void;
  models: ModelStatus[];
}

/**
 * Carry a result into the next step without going through the filesystem.
 *
 * Everything here is already a vault item, so handing work onward is just
 * passing an id. The alternative -- export, then re-import -- would write
 * plaintext to disk purely to move between two tabs of the same app.
 *
 * Destinations that cannot run are hidden rather than shown failing: there is
 * no point offering "Animate" on a machine with no video model installed.
 */
export default function SendTo({
  ids, send, models, exclude = [], compact,
}: SendToTargets & {
  ids: string[];
  /** Destinations to leave out, usually the tab you are already on. */
  exclude?: Destination[];
  compact?: boolean;
}) {
  const canRun = (task: string) =>
    models.some((m) => m.installed && !m.hopeless && m.fit !== "broken"
      && m.tasks.includes(task as never));

  const options: [Destination, string, boolean][] = [
    ["edit", "Edit this", canRun("edit")],
    ["video", "Animate", canRun("video")],
    ["upscale", "Enlarge", canRun("upscale")],
  ];

  const available = options.filter(
    ([dest, , ok]) => ok && !exclude.includes(dest)
  );
  if (available.length === 0 || ids.length === 0) return null;

  return (
    <>
      {available.map(([dest, label]) => (
        <button
          key={dest}
          className="btn small"
          style={compact ? { padding: "3px 8px", fontSize: 11 } : undefined}
          onClick={(e) => { e.stopPropagation(); send(dest, ids); }}
        >
          {label}
        </button>
      ))}
    </>
  );
}
