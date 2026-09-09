import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
  info: string;
}

/**
 * Stops one component's render error from blanking the whole app.
 *
 * Without this, any exception below the root unmounts everything and leaves a
 * white window with no way back — while the vault is still unlocked and a
 * generation may still be running in the engine.
 */
export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { error: null, info: "" };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep it in the console too: the panel shows a summary, not a stack.
    console.error("Render error:", error, info.componentStack);
    this.setState({ info: info.componentStack ?? "" });
  }

  render() {
    const { error, info } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="setup-wrap">
        <div className="setup-card">
          <h1>Something broke on screen</h1>
          <p className="lede">
            A part of the interface failed to render. Your vault is untouched and
            anything already generated is safe.
          </p>
          <div className="panel">
            <div className="notice bad" style={{ marginBottom: 12 }}>
              <strong>{error.name}: {error.message}</strong>
              This is a bug in the app, not something you did.
            </div>
            {info && (
              <div className="setup-log" style={{ maxHeight: 160 }}>
                {info.trim().split("\n").slice(0, 8).join("\n")}
              </div>
            )}
            <button
              className="btn primary full"
              style={{ marginTop: 12 }}
              onClick={() => this.setState({ error: null, info: "" })}
            >
              Try again
            </button>
            <button
              className="btn full"
              style={{ marginTop: 8 }}
              onClick={() => window.location.reload()}
            >
              Reload the interface
            </button>
            <div style={{ marginTop: 10, fontSize: 10.5, color: "var(--text-faint)" }}>
              Reloading does not unlock or re-lock the vault by itself, but you
              will be asked for your passphrase again.
            </div>
          </div>
        </div>
      </div>
    );
  }
}
