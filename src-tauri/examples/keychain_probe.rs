//! Exercise the Keychain-backed Touch ID path.
//!
//!   cargo run --example keychain_probe            # no prompt: store/presence/delete
//!   cargo run --example keychain_probe -- --read   # ALSO reads, which prompts for Touch ID
//!
//! Reading is what triggers the biometric prompt, so it is opt-in: an
//! unattended run would leave a modal dialog on screen.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("macOS only");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    use melp_model_studio_lib::keychain_probe as kc;

    let read = std::env::args().any(|a| a == "--read");
    let mut failures = 0;

    macro_rules! check {
        ($label:expr, $cond:expr) => {{
            let ok = $cond;
            println!("  {} {}", if ok { "OK  " } else { "FAIL" }, $label);
            if !ok {
                failures += 1;
            }
            ok
        }};
    }

    // An unsigned build cannot use the data-protection Keychain at all. That is
    // a property of the build, not a bug, so report it as such.
    if !kc::biometry_usable() {
        println!("Biometric unlock is NOT available in this build.");
        println!();
        println!("  The data-protection Keychain, which is the only one supporting");
        println!("  biometric access control, requires a signature with an Apple");
        println!("  Developer team identifier. Storing returns errSecMissingEntitlement");
        println!("  (-34018) without one, and adding a keychain-access-groups");
        println!("  entitlement to an ad-hoc signature gets the process SIGKILLed.");
        println!();
        println!("  The app detects this and hides the Touch ID option.");
        println!("  Passphrase unlock is unaffected.");
        println!();
        println!("EXPECTED FOR AN UNSIGNED BUILD");
        return;
    }

    // Never clobber a real vault key: refuse to run if one is already stored.
    if kc::kek_present() {
        println!("A Keychain key already exists for this vault.");
        println!("Refusing to overwrite it. Remove Touch ID unlock in the app first.");
        return;
    }

    println!("Keychain round-trip (no prompt expected):");
    let stored = match kc::provision_kek() {
        Ok(k) => {
            check!("store a key behind a userPresence ACL", true);
            Some(k)
        }
        Err(e) => {
            println!("  FAIL store: {e}");
            failures += 1;
            None
        }
    };

    check!("presence check sees it without prompting", kc::kek_present());

    if read {
        println!("\nReading the key — expect a Touch ID prompt now:");
        match kc::load_kek("Model Studio keychain probe") {
            Ok(loaded) => {
                let matches = stored.map(|s| s == loaded).unwrap_or(false);
                check!("read back after authentication", true);
                check!("round-tripped key is identical", matches);
            }
            Err(e) => println!("  read returned: {e}"),
        }
    } else {
        println!("\n(skipping the read; pass --read to test the Touch ID prompt)");
    }

    let _ = kc::delete_kek();
    check!("delete removes it", !kc::kek_present());

    println!();
    if failures == 0 {
        println!("KEYCHAIN OK");
    } else {
        println!("{failures} check(s) failed");
        std::process::exit(1);
    }
}
