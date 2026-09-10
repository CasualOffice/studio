//! Verify the model-store relocation preserves a Hugging Face cache exactly.
//!
//!   cargo run --example storage_move
//!
//! This matters more than most checks: the real command deletes the originals
//! once the copy lands. The cache stores real bytes in `blobs/` and symlinks
//! them into `snapshots/`, so following those links would duplicate every
//! weight file and silently double the space used.

use melp_model_studio_lib::storage_move as sm;
use std::fs;
use std::path::Path;

fn main() {
    let root = std::env::temp_dir().join(format!("store-probe-{}", std::process::id()));
    let from = root.join("from");
    let to = root.join("to");
    let _ = fs::remove_dir_all(&root);
    let mut failures = 0;

    macro_rules! check {
        ($label:expr, $cond:expr) => {{
            let ok = $cond;
            println!("  {} {}", if ok { "OK  " } else { "FAIL" }, $label);
            if !ok {
                failures += 1;
            }
        }};
    }

    // A miniature of the real layout: blobs hold content, snapshots link to it.
    let repo = from.join("hub/models--acme--demo");
    fs::create_dir_all(repo.join("blobs")).unwrap();
    fs::create_dir_all(repo.join("snapshots/abc123")).unwrap();
    fs::create_dir_all(repo.join("refs")).unwrap();
    fs::write(repo.join("blobs/deadbeef"), vec![7u8; 512 * 1024]).unwrap();
    fs::write(repo.join("blobs/cafe"), b"{\"config\": true}").unwrap();
    fs::write(repo.join("refs/main"), b"abc123").unwrap();
    std::os::unix::fs::symlink(
        "../../blobs/deadbeef",
        repo.join("snapshots/abc123/model.safetensors"),
    )
    .unwrap();
    std::os::unix::fs::symlink(
        "../../blobs/cafe",
        repo.join("snapshots/abc123/config.json"),
    )
    .unwrap();

    let before = sm::dir_size_of(&from);
    println!("Source is {} bytes of real content\n", before);

    check!("copy succeeds", sm::copy_tree(&from, &to).is_ok());

    let dest_repo = to.join("hub/models--acme--demo");
    let link = dest_repo.join("snapshots/abc123/model.safetensors");

    check!(
        "snapshot entry is still a symlink, not a copy",
        fs::symlink_metadata(&link)
            .map(|m| m.is_symlink())
            .unwrap_or(false)
    );
    check!(
        "symlink still points at the blob",
        fs::read_link(&link)
            .map(|t| t == Path::new("../../blobs/deadbeef"))
            .unwrap_or(false)
    );
    check!(
        "symlink resolves to the real content",
        fs::read(&link)
            .map(|b| b.len() == 512 * 1024)
            .unwrap_or(false)
    );
    check!(
        "blob content is byte-identical",
        fs::read(dest_repo.join("blobs/deadbeef")).unwrap() == vec![7u8; 512 * 1024]
    );
    check!(
        "plain files come across",
        fs::read(dest_repo.join("refs/main")).unwrap() == b"abc123"
    );
    check!(
        "nested directories are recreated",
        dest_repo.join("snapshots/abc123").is_dir()
    );

    // The whole point: following links would report roughly double.
    let after = sm::dir_size_of(&to);
    check!(
        &format!("size is preserved, not doubled ({before} -> {after})"),
        after == before
    );

    let _ = fs::remove_dir_all(&root);
    println!();
    if failures == 0 {
        println!("STORAGE MOVE OK");
    } else {
        println!("{failures} check(s) failed");
        std::process::exit(1);
    }
}
