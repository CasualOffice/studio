//! Turn engine failures into something a person can act on.
//!
//! The worker surfaces whatever Python raised, which is usually accurate and
//! rarely useful: a memory ceiling reads as `[metal::malloc] Attempting to
//! allocate 14400000000 bytes...`. That tells you nothing about what to do,
//! and the ceiling exists precisely so people hit it instead of watching the
//! machine swap.
//!
//! Anything unrecognised is passed through untouched. Inventing a friendly
//! message for an error nobody has seen would hide the only real information.

/// A failure, restated with a suggested next step where one exists.
pub fn humanize(raw: &str) -> String {
    let lower = raw.to_lowercase();

    // Order matters: the memory ceiling is deliberately narrow, so check it
    // before the broader out-of-memory patterns.
    if lower.contains("metal::malloc") || lower.contains("maximum allowed buffer size") {
        return "That run needed more memory than the app is allowed to use. \
                Try a smaller size or fewer steps, turn on reduced-memory mode, \
                or raise the ceiling in Activity."
            .into();
    }
    if lower.contains("out of memory") || lower.contains("insufficient memory") {
        return "Ran out of memory. Close other applications, or try a smaller \
                size and fewer steps."
            .into();
    }
    if lower.contains("downloadrequirederror") || lower.contains("will not download") {
        return "That model's files are not on this Mac. Install it from the \
                Models tab."
            .into();
    }
    if lower.contains("could not infer a supported backend") || lower.contains("pass family") {
        return "This model is not one the engine recognises. If it is a variant \
                of a supported family, remove it and add it again so its family \
                can be detected; otherwise it cannot run here."
            .into();
    }
    if lower.contains("does not support") || lower.contains("is not a parameter") {
        return format!(
            "This model does not support that combination of settings. {}",
            first_sentence(raw)
        );
    }
    if lower.contains("no such file") || lower.contains("filenotfounderror") {
        return "A file this run needed is missing. If the model is on an \
                external drive, check it is still connected."
            .into();
    }
    if lower.contains("cancelled") {
        return "Cancelled.".into();
    }
    if lower.contains("engine process exited") || lower.contains("could not reach the engine") {
        return "The engine stopped unexpectedly. It restarts on the next run; \
                if this keeps happening, the Activity tab has the details."
            .into();
    }
    if lower.contains("cas client error")
        || lower.contains("file reconstruction error")
        || lower.contains("error decoding response body")
    {
        return "The download was interrupted partway through. This is usually \
                temporary: start it again and it will resume."
            .into();
    }
    if lower.contains("connection") || lower.contains("timed out") || lower.contains("timeout") {
        return "The network request failed. Check your connection and try again.".into();
    }
    if lower.contains("gated") || lower.contains("401") || lower.contains("403") {
        return "That repository is gated. Accept its licence on Hugging Face, \
                then try again."
            .into();
    }
    if lower.contains("no space left") || lower.contains("enospc") {
        return "The disk is full. Remove a model from the Models tab to free space.".into();
    }

    raw.trim().to_string()
}

/// First sentence of a message, for appending to an explanation without
/// dragging a whole traceback along.
fn first_sentence(raw: &str) -> String {
    let trimmed = raw.trim();
    let end = trimmed
        .find(". ")
        .map(|i| i + 1)
        .unwrap_or_else(|| trimmed.len().min(160));
    trimmed[..end].trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_ceiling_becomes_actionable() {
        let raw = "[metal::malloc] Attempting to allocate 14400000000 bytes which is \
                   greater than the maximum allowed buffer size";
        let out = humanize(raw);
        assert!(out.contains("more memory"), "{out}");
        assert!(out.contains("smaller size"), "should suggest a fix: {out}");
        assert!(!out.contains("metal::malloc"), "internals leaked: {out}");
    }

    #[test]
    fn missing_model_points_at_the_models_tab() {
        let out = humanize("DownloadRequiredError: MLX-Gen will not download model files");
        assert!(out.contains("Models tab"), "{out}");
    }

    #[test]
    fn unsupported_settings_keep_the_specifics() {
        // Here the original text is the useful part, so it is kept.
        let out = humanize("FLUX.2 does not support edit-reference image-to-image generation.");
        assert!(out.contains("does not support"), "{out}");
    }

    #[test]
    fn cancellation_is_not_an_error_report() {
        assert_eq!(humanize("Cancelled"), "Cancelled.");
        assert_eq!(humanize("cancelled"), "Cancelled.");
    }

    #[test]
    fn full_disk_says_how_to_fix_it() {
        assert!(humanize("OSError: [Errno 28] No space left on device").contains("disk is full"));
    }

    #[test]
    fn an_interrupted_transfer_says_to_retry() {
        // Hugging Face's chunked backend fails this way on a truncated
        // response, and the raw text names none of that.
        let out = humanize(
            "RuntimeError: Task error: File reconstruction error: CAS Client Error: \
             Format error: I/O error: error decoding response body",
        );
        assert!(out.contains("interrupted"), "{out}");
        assert!(!out.contains("CAS"), "internals leaked: {out}");
    }

    #[test]
    fn an_unplaceable_model_explains_itself() {
        // The engine names the option but not the value, which is no help.
        let out = humanize(
            "TaskInferenceError: could not infer a supported backend for model \
             'someone/unusual-repo', pass family=",
        );
        assert!(out.contains("not one the engine recognises"), "{out}");
        assert!(!out.contains("pass family"), "internals leaked: {out}");
    }

    #[test]
    fn unknown_errors_pass_through_untouched() {
        // Inventing a friendly message here would hide the only real signal.
        let odd = "TypeError: unhashable type: 'list'";
        assert_eq!(humanize(odd), odd);
    }

    #[test]
    fn a_dead_engine_says_it_recovers() {
        let out = humanize("the engine process exited unexpectedly");
        assert!(out.contains("restarts"), "{out}");
    }
}
