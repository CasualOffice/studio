//! Exercise the vault paths that the interface can reach but no test covers:
//! creation, dedup, repair after an orphaned blob, and export.
//!
//!   cargo run --example vault_lifecycle
//!
//! Runs entirely in a temporary directory; the real vault is never touched.

use melp_model_studio_lib::vault_lifecycle as v;

fn main() {
    let dir = std::env::temp_dir().join(format!("vault-probe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let vault = v::Vault::new(dir.clone());
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

    println!("Vault lifecycle in {}\n", dir.display());

    check!(
        "create with a passphrase",
        vault.create("correct horse battery", false).is_ok()
    );
    check!("is unlocked after creation", vault.is_unlocked());

    let item = |name: &str| v::VaultItem {
        id: uuid_like(),
        content_hash: None,
        kind: "import".into(),
        name: name.into(),
        mime: "image/png".into(),
        bytes: 0,
        model: String::new(),
        prompt: String::new(),
        seed: 0,
        width: None,
        height: None,
        steps: None,
        guidance: None,
        inputs: vec![],
        created_at: "now".into(),
        duration_ms: 0,
        project: None,
        project_name: None,
        project_index: None,
    };

    let first = vault.put(b"identical bytes", item("a.png")).unwrap();
    let second = vault.put(b"identical bytes", item("b.png")).unwrap();
    check!("same content is deduplicated", first == second);
    check!("only one item is listed", vault.list().unwrap().len() == 1);

    let other = vault.put(b"different bytes", item("c.png")).unwrap();
    check!("different content is stored separately", other != first);

    check!(
        "content round-trips",
        vault.get(&first).unwrap() == b"identical bytes"
    );

    // Simulate a crash between writing a blob and saving the index.
    let orphan = dir.join("blobs").join("orphaned-blob");
    let sealed = v::seal_for_test(&vault, b"\x89PNG\r\n\x1a\nrescued").unwrap();
    std::fs::write(&orphan, sealed).unwrap();

    let report = vault.repair().unwrap();
    check!("orphaned blob is recovered", report.recovered == 1);
    check!("nothing was wrongly dropped", report.dropped == 0);
    check!("recovered item is listed", vault.list().unwrap().len() == 3);
    check!(
        "recovered content is identified as PNG",
        vault
            .list()
            .unwrap()
            .iter()
            .any(|i| i.kind == "recovered" && i.mime == "image/png")
    );

    // A file this vault's key cannot open must be left alone, not deleted.
    let foreign = dir.join("blobs").join("not-ours");
    std::fs::write(&foreign, b"MSV1 but not really").unwrap();
    let report = vault.repair().unwrap();
    check!(
        "foreign file is reported, not destroyed",
        report.unreadable == 1 && foreign.exists()
    );

    let out = dir.join("exported.png");
    check!(
        "export writes plaintext where asked",
        vault.export(&first, &out, false).is_ok()
    );
    check!(
        "exported bytes match",
        std::fs::read(&out).unwrap() == b"identical bytes"
    );

    vault.lock();
    check!("locked vault refuses reads", vault.get(&first).is_err());
    check!(
        "wrong passphrase is rejected",
        vault.unlock_with_passphrase("wrong").is_err()
    );
    check!(
        "correct passphrase reopens it",
        vault
            .unlock_with_passphrase("correct horse battery")
            .is_ok()
    );
    check!(
        "items survived the lock cycle",
        vault.list().unwrap().len() == 3
    );

    check!(
        "passphrase can be changed",
        vault
            .change_passphrase("correct horse battery", "a longer new one")
            .is_ok()
    );
    vault.lock();
    check!(
        "new passphrase works",
        vault.unlock_with_passphrase("a longer new one").is_ok()
    );
    check!("old passphrase no longer works", {
        vault.lock();
        let bad = vault
            .unlock_with_passphrase("correct horse battery")
            .is_err();
        let _ = vault.unlock_with_passphrase("a longer new one");
        bad
    });

    let _ = std::fs::remove_dir_all(&dir);
    println!();
    if failures == 0 {
        println!("VAULT LIFECYCLE OK");
    } else {
        println!("{failures} check(s) failed");
        std::process::exit(1);
    }
}

fn uuid_like() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("item-{n}")
}
