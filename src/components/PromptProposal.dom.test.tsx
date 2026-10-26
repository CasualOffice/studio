/**
 * @vitest-environment jsdom
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import PromptProposal from "./PromptProposal";
import type { AssistResult, AssistVariant } from "../lib/types";

/**
 * The first component test in this project, and the reason there is a harness.
 *
 * `PromptProposal` is where every honesty fix in the enhancer either reaches
 * the user or does not, and all of it was previously unverifiable: the engine
 * building a correct reply proves nothing if the branch that renders it drops
 * a field. That is exactly what had happened -- `note` was built, typed,
 * styled, sent, and rendered by two of the three branches.
 */
const base: AssistResult = {
  prompt: "a tabby cat curled on a worn oak chair",
  original: "a masterpiece 8k cat on a chair",
  changed: true,
  saw_image: false,
  outcome: "proposal",
};

const noop = () => {};

// Each test mounts its own card; without this they pile up in one document.
afterEach(cleanup);

function show(over: Partial<AssistResult>) {
  render(
    <PromptProposal
      result={{ ...base, ...over }}
      onAccept={noop}
      onDismiss={noop}
    />,
  );
}

describe("PromptProposal tells the user what it knows", () => {
  it("shows the note on a successful rewrite", () => {
    // The blind-mode warning. It is how a rewrite made without looking at the
    // picture says so, and it is the common edit path now that an edit is
    // routed as an edit whether or not a reader is installed.
    show({ note: "Rewritten from your words only." });
    expect(screen.getByText(/Rewritten from your words only/)).toBeTruthy();
  });

  it("shows the note when it kept your words instead", () => {
    show({
      outcome: "rewrite_rejected",
      changed: false,
      note: "The rewrite drifted away from what you asked for.",
    });
    expect(screen.getByText(/drifted away/)).toBeTruthy();
  });

  it("shows the note when the request named nothing to draw", () => {
    show({
      outcome: "unclear",
      unclear: true,
      note: "Name something visible instead of quality labels.",
    });
    expect(screen.getByText(/Name something visible/)).toBeTruthy();
  });

  it("separates what was stripped from what the rewrite lost", () => {
    // Two different facts under two labels. They used to collide: `dropped`
    // was diffed against the prompt as typed, so a word the cleaner had
    // already removed appeared in both lists as the same red chip, and the
    // user was told one word was both deliberately removed and accidentally
    // lost.
    show({ removed: ["masterpiece", "8k"], dropped: ["chair"] });
    expect(screen.getByText("removed")).toBeTruthy();
    expect(screen.getByText("no longer says")).toBeTruthy();
    expect(screen.getAllByText("masterpiece")).toHaveLength(1);
  });

  it("does not claim to have read a picture it never saw", () => {
    show({ saw_image: false, description: "subject: a grey cat" });
    expect(screen.queryByText(/read from your picture/)).toBeNull();
  });

  it("says what it read when it did look", () => {
    show({ saw_image: true, description: "subject: a grey cat" });
    expect(screen.getByText(/read from your picture/)).toBeTruthy();
  });

  it("shows the rewrite it refused, so the choice is the user's", () => {
    // The checks are literal by design, so the text they refuse is often the
    // text that was wanted: asked for a cat "setting" on a chair the writer
    // says "sitting", which no string comparison can prove was meant.
    show({
      outcome: "rewrite_rejected",
      changed: false,
      note: "The rewrite lost something you asked for.",
      attempt: "a tabby cat sitting on a worn oak chair",
    });
    expect(screen.getByText(/sitting on a worn oak chair/)).toBeTruthy();
  });
});

describe("PromptProposal offers versions to compare", () => {
  const variants: AssistVariant[] = [
    { kind: "faithful", label: "Faithful", prompt: "Two red cars without a sunroof.",
      additions: [], missing: [], reworded: [] },
    { kind: "enriched", label: "Enriched",
      prompt: "Two red cars without a sunroof, parked on wet asphalt under a grey sky.",
      additions: ["parked on wet asphalt", "under a grey sky"], missing: [], reworded: [] },
    { kind: "stylistic", label: "Stylistic",
      prompt: "Three red cars on wet asphalt. Pencil sketch.",
      additions: ["Pencil sketch"], missing: ["2 red cars", "no sunroof"], reworded: [] },
  ];
  const reply: AssistResult = {
    prompt: variants[1].prompt, original: "2 red cars with no sunroof", changed: true,
    outcome: "proposal", variants, target_label: "FLUX.2 Klein 4B",
  };

  function mount(onAccept = vi.fn()) {
    render(<PromptProposal result={reply} onAccept={onAccept} onDismiss={noop} />);
    return onAccept;
  }

  it("opens on the version the engine put first, among tabs for all three", () => {
    mount();
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Faithful", "Enriched", "Stylistic ⚠"]);
    expect(tabs[1].getAttribute("aria-selected")).toBe("true");
    expect(screen.getByText(/Suggested for FLUX.2 Klein 4B/)).toBeTruthy();
  });

  it("highlights what a version added to your words", () => {
    mount();
    const marked = [...screen.getByTestId("proposal-text").querySelectorAll("ins")]
      .map((n) => n.textContent);
    expect(marked).toContain("parked");
    expect(marked).toContain("asphalt");
    expect(marked).not.toContain("red");
  });

  it("flags what a version lost, from the checks and from your words", () => {
    mount();
    fireEvent.click(screen.getByRole("tab", { name: /Stylistic/ }));
    expect(screen.getByText("no longer says")).toBeTruthy();
    expect(screen.getByText("no sunroof")).toBeTruthy();
    expect(screen.getByText("not in this version")).toBeTruthy();
    expect(screen.getByText("sunroof")).toBeTruthy();
  });

  it("rebuilds the prompt when an addition is switched off, and uses that", () => {
    const onAccept = mount();
    const chip = screen.getByRole("button", { name: "under a grey sky" });
    expect(chip.getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(chip);
    expect(chip.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByTestId("proposal-text").textContent)
      .toBe("Two red cars without a sunroof, parked on wet asphalt.");
    fireEvent.click(screen.getByRole("button", { name: "Use this" }));
    expect(onAccept).toHaveBeenCalledWith(
      "Two red cars without a sunroof, parked on wet asphalt.");
  });

  it("keeps each version's switches its own", () => {
    mount();
    fireEvent.click(screen.getByRole("button", { name: "under a grey sky" }));
    fireEvent.click(screen.getByRole("tab", { name: "Faithful" }));
    expect(screen.getByTestId("proposal-text").textContent)
      .toBe("Two red cars without a sunroof.");
    fireEvent.click(screen.getByRole("tab", { name: "Enriched" }));
    expect(screen.getByRole("button", { name: "under a grey sky" })
      .getAttribute("aria-pressed")).toBe("false");
  });

  it("says the picture model will load again", () => {
    render(<PromptProposal result={{ ...reply, released: "FLUX.2 Klein 4B" }}
                           onAccept={noop} onDismiss={noop} />);
    expect(screen.getByText(/unloaded to make room for the writer/)).toBeTruthy();
  });

  it("says which region it read 'it' as", () => {
    render(<PromptProposal
      result={{ ...reply, saw_image: true, regions: ["man in a jacket (centre)"],
                target_region: "man in a jacket (centre)" }}
      onAccept={noop} onDismiss={noop} />);
    expect(screen.getByText(/“it” read as man in a jacket/)).toBeTruthy();
  });
});
