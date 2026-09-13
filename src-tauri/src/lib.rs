mod catalog;
mod commands;
mod diagnose;
mod engine;
mod error;
mod hostinfo;
mod models;
pub mod paths;
mod setup;
mod timings;
mod vault;

/// Re-exported so `examples/vault_vectors.rs` can generate cross-language test
/// vectors against the real implementation rather than a copy of it.
/// Exposed for `examples/storage_move.rs`, which checks that relocating the
/// model store preserves symlinks rather than duplicating every weight file.
pub mod storage_move {
    pub use crate::models::{copy_tree, dir_size_of};
}

/// Exposed for `examples/vault_lifecycle.rs`, which exercises creation,
/// deduplication, repair and export against a throwaway directory.
pub mod vault_lifecycle {
    pub use crate::vault::{Vault, VaultItem};

    /// Seal bytes with the vault's own key, to plant an orphan for repair to find.
    pub fn seal_for_test(vault: &Vault, bytes: &[u8]) -> Option<Vec<u8>> {
        vault.seal_for_test(bytes)
    }
}

/// Exposed for `examples/keychain_probe.rs`, which exercises the Touch ID
/// path without needing the whole app running.
#[cfg(target_os = "macos")]
pub mod keychain_probe {
    pub use crate::vault::keychain::{
        biometry_usable, delete_kek, kek_present, load_kek, provision_kek,
    };
}

/// Exposed for `examples/setup_probe.rs`, which exercises the first-run
/// bootstrap without touching a real installation.
pub mod setup_probe {
    pub use crate::setup::{bootstrap, bootstrap_runtime_only};
    use std::path::PathBuf;

    pub fn paths_at(root: PathBuf) -> crate::paths::AppPaths {
        crate::paths::AppPaths::at(root)
    }
}

pub mod vault_test_support {
    pub use crate::vault::crypto::{derive_file_key, open, seal, seal_with_file_key};
}

use commands::AppState;
use paths::AppPaths;
use std::sync::Arc;
use tauri::http;
use tauri::Manager;
use vault::Vault;

/// Serve decrypted vault content to the WebView.
///
/// This is what makes "no plaintext on disk" compatible with actually showing
/// the user their images: bytes are decrypted into memory and handed straight
/// to the renderer. Requests are refused while the vault is locked.
fn vault_protocol(
    app: &tauri::AppHandle,
    request: http::Request<Vec<u8>>,
) -> http::Response<Vec<u8>> {
    let deny = |code: u16, msg: &str| {
        http::Response::builder()
            .status(code)
            .header("Content-Type", "text/plain")
            .header("Cache-Control", "no-store")
            .body(msg.as_bytes().to_vec())
            .expect("static response is always valid")
    };

    let state = app.state::<AppState>();
    if !state.vault.is_unlocked() {
        return deny(403, "vault is locked");
    }

    // The id is the final path segment; the host part varies by platform.
    let id = request
        .uri()
        .path()
        .rsplit('/')
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .to_string();
    if id.is_empty() {
        return deny(400, "missing item id");
    }

    let item = match state.vault.get_item(&id) {
        Ok(i) => i,
        Err(_) => return deny(404, "no such item"),
    };
    let bytes = match state.vault.get(&id) {
        Ok(b) => b,
        Err(_) => return deny(404, "could not read item"),
    };

    http::Response::builder()
        .status(200)
        .header("Content-Type", item.mime)
        .header("Content-Length", bytes.len().to_string())
        // Decrypted bytes must not be cached anywhere they could outlive a lock.
        .header("Cache-Control", "no-store, no-cache, must-revalidate")
        // Readable back off a canvas.
        //
        // `vault://` is a different origin from the page, so a canvas that has
        // drawn one of these images is tainted and every pixel read off it
        // throws SecurityError -- "The operation is insecure." That is what
        // Adjust hit on save: it rotates and crops through a canvas and then
        // calls `toBlob`, so straightening a picture failed at the last step,
        // after the work, with a message that names nothing the user did.
        //
        // Nothing is exposed by allowing it. A custom scheme is not reachable
        // from a web page; the only thing that can ask is this app's own
        // webview, which is already displaying the pixels it wants to read.
        .header("Access-Control-Allow-Origin", "*")
        .body(bytes)
        .unwrap_or_else(|_| deny(500, "could not build response"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let paths = AppPaths::resolve().expect("could not resolve application directories");
    // Boot must not depend on every directory being creatable. It used to
    // `.expect()` here, so a models_root on a detached external drive -- a path
    // under /Volumes, which belongs to root and answers EACCES -- killed the
    // app before `tauri::Builder` ran: no window, no dialog, and no way to
    // reach the Storage panel that had set the path in the first place. Create
    // what this machine allows, report the rest, and let the window open. The
    // model location has already fallen back to the internal one for this
    // session, so the Storage panel shows where the models are actually going
    // and can point them back at the drive once it is attached again.
    if let Err(error) = paths.ensure_dirs() {
        eprintln!("could not create every application directory: {error}");
    }
    let vault = Arc::new(Vault::new(paths.vault()));

    tauri::Builder::default()
        // The standard macOS menu. It carries Cmd-C, Cmd-V, Cmd-A and the
        // window controls; without a menu those shortcuts do not exist, so
        // selected text cannot be copied out of the app at all.
        .menu(tauri::menu::Menu::default)
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .register_uri_scheme_protocol("vault", |ctx, request| {
            vault_protocol(ctx.app_handle(), request)
        })
        .manage(AppState::new(paths, vault))
        .invoke_handler(tauri::generate_handler![
            commands::host_info,
            commands::setup_state,
            commands::run_setup,
            commands::vault_status,
            commands::vault_create,
            commands::vault_unlock_passphrase,
            commands::vault_unlock_biometry,
            commands::vault_lock,
            commands::vault_enable_biometry,
            commands::vault_disable_biometry,
            commands::vault_change_passphrase,
            commands::vault_list,
            commands::sweep_transient,
            commands::board_state_get,
            commands::board_state_set,
            commands::board_state_list,
            commands::board_state_forget,
            commands::vault_repair,
            commands::vault_delete,
            commands::vault_export,
            commands::vault_import,
            commands::vault_import_bytes,
            commands::list_models,
            commands::list_loras,
            commands::resolve_lora,
            commands::add_lora,
            commands::remove_lora,
            commands::storage_info,
            commands::find_orphans,
            commands::sweep_orphans,
            commands::set_models_location,
            commands::download_model,
            commands::delete_model,
            commands::resolve_model,
            commands::add_custom_model,
            commands::remove_custom_model,
            commands::shot_list,
            commands::story_cast,
            commands::compose_board,
            commands::bind_comic,
            commands::enrich_panels,
            commands::hf_token_status,
            commands::set_hf_token,
            commands::unload_model,
            commands::engine_ping,
            commands::cancel_job,
            commands::generate,
            commands::edit_image,
            commands::upscale,
            commands::generate_video,
            commands::assist_prompt,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<AppState>();
                // Lock first: the data key should be gone before anything else
                // gets a chance to fail.
                state.vault.lock();
                let engine = state.engine.blocking_lock().take();
                if let Some(engine) = engine {
                    tauri::async_runtime::block_on(engine.shutdown());
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Model Studio");
}
