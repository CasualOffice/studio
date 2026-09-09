//! Vault state, manifest and encrypted item index.
//!
//! On disk the vault is opaque:
//!
//! ```text
//! vault/
//!   vault.json     wrapped data keys and KDF parameters -- no secrets
//!   index.enc      encrypted item metadata (names, prompts, seeds)
//!   blobs/<uuid>   encrypted content, random names, no extensions
//! ```
//!
//! Filenames carry no information, so even the *shape* of the library --
//! what is an image, what a document, what it was called -- is inside the
//! encrypted index rather than readable from a directory listing.

use super::crypto::{self, CryptoError};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::Instant;
use zeroize::Zeroize;

const B64: base64::engine::general_purpose::GeneralPurpose = base64::engine::general_purpose::STANDARD;

#[derive(Debug, thiserror::Error)]
pub enum VaultError {
    #[error("the vault is locked")]
    Locked,
    #[error("no vault has been created yet")]
    Absent,
    #[error("a vault already exists")]
    AlreadyExists,
    #[error("that passphrase is not correct")]
    BadPassphrase,
    #[error("no item with id {0}")]
    NoSuchItem(String),
    #[error("this vault has no Touch ID key; unlock with your passphrase")]
    NoBiometricKey,
    #[error("passphrase must be at least {0} characters")]
    WeakPassphrase(usize),
    #[error(transparent)]
    Crypto(#[from] CryptoError),
    #[error("vault io: {0}")]
    Io(#[from] std::io::Error),
    #[error("vault data is corrupt: {0}")]
    Corrupt(String),
    #[error("key derivation failed: {0}")]
    Kdf(String),
}

pub const MIN_PASSPHRASE: usize = 10;

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KdfParams {
    pub algo: String,
    /// Memory cost in KiB.
    pub m_cost: u32,
    pub t_cost: u32,
    pub p_cost: u32,
    pub salt: String,
}

impl Default for KdfParams {
    fn default() -> Self {
        Self {
            algo: "argon2id".into(),
            // 64 MiB / 3 passes: costly enough to make offline guessing
            // expensive, cheap enough that unlocking stays under a second.
            m_cost: 65536,
            t_cost: 3,
            p_cost: 4,
            salt: B64.encode(crypto::random_bytes::<16>()),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Wrap {
    pub nonce: String,
    pub ct: String,
}

impl Wrap {
    fn seal(kek: &[u8; 32], dek: &[u8; 32]) -> Self {
        let (nonce, ct) = crypto::wrap_dek(kek, dek);
        Self { nonce: B64.encode(nonce), ct: B64.encode(ct) }
    }

    fn open(&self, kek: &[u8; 32]) -> Result<[u8; 32], VaultError> {
        let nonce = B64.decode(&self.nonce).map_err(|e| VaultError::Corrupt(e.to_string()))?;
        let ct = B64.decode(&self.ct).map_err(|e| VaultError::Corrupt(e.to_string()))?;
        crypto::unwrap_dek(kek, &nonce, &ct).map_err(|_| VaultError::BadPassphrase)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Manifest {
    pub version: u32,
    pub kdf: KdfParams,
    pub passphrase_wrap: Wrap,
    pub keychain_wrap: Option<Wrap>,
    pub created_at: String,
}

fn derive_kek(passphrase: &str, kdf: &KdfParams) -> Result<[u8; 32], VaultError> {
    use argon2::{Algorithm, Argon2, Params, Version};
    let salt = B64.decode(&kdf.salt).map_err(|e| VaultError::Corrupt(e.to_string()))?;
    let params = Params::new(kdf.m_cost, kdf.t_cost, kdf.p_cost, Some(32))
        .map_err(|e| VaultError::Kdf(e.to_string()))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out = [0u8; 32];
    argon
        .hash_password_into(passphrase.as_bytes(), &salt, &mut out)
        .map_err(|e| VaultError::Kdf(e.to_string()))?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct VaultItem {
    pub id: String,
    /// "generate" | "edit" | "upscale" | "import" | "doc"
    pub kind: String,
    pub name: String,
    pub mime: String,
    pub bytes: u64,
    pub model: String,
    pub prompt: String,
    pub seed: i64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    /// Ids of other vault items used as inputs, never filesystem paths.
    pub inputs: Vec<String>,
    pub created_at: String,
    pub duration_ms: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Index {
    items: Vec<VaultItem>,
}

// ---------------------------------------------------------------------------
// Vault
// ---------------------------------------------------------------------------

struct Unlocked {
    dek: [u8; 32],
    index: Index,
    since: Instant,
    /// True when unlocked via Keychain, so the UI can say how.
    via_biometry: bool,
}

impl Drop for Unlocked {
    fn drop(&mut self) {
        self.dek.zeroize();
    }
}

pub struct Vault {
    root: PathBuf,
    inner: RwLock<Option<Unlocked>>,
}

#[derive(Serialize, Clone, Debug)]
pub struct VaultStatus {
    pub exists: bool,
    pub unlocked: bool,
    pub biometry_available: bool,
    pub biometry_enrolled: bool,
    pub via_biometry: bool,
    pub item_count: usize,
    pub unlocked_seconds: u64,
    pub root: String,
}

impl Vault {
    pub fn new(root: PathBuf) -> Self {
        Self { root, inner: RwLock::new(None) }
    }

    pub fn manifest_path(&self) -> PathBuf { self.root.join("vault.json") }
    fn index_path(&self) -> PathBuf { self.root.join("index.enc") }
    fn blobs(&self) -> PathBuf { self.root.join("blobs") }
    fn blob_path(&self, id: &str) -> PathBuf { self.blobs().join(id) }

    pub fn exists(&self) -> bool { self.manifest_path().exists() }

    pub fn is_unlocked(&self) -> bool {
        self.inner.read().map(|g| g.is_some()).unwrap_or(false)
    }

    fn read_manifest(&self) -> Result<Manifest, VaultError> {
        if !self.exists() {
            return Err(VaultError::Absent);
        }
        let raw = std::fs::read(self.manifest_path())?;
        serde_json::from_slice(&raw).map_err(|e| VaultError::Corrupt(e.to_string()))
    }

    fn write_manifest(&self, m: &Manifest) -> Result<(), VaultError> {
        std::fs::create_dir_all(&self.root)?;
        let tmp = self.manifest_path().with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(m).map_err(|e| VaultError::Corrupt(e.to_string()))?)?;
        std::fs::rename(&tmp, self.manifest_path())?;
        Ok(())
    }

    pub fn status(&self) -> VaultStatus {
        let guard = self.inner.read().ok();
        let unlocked = guard.as_ref().and_then(|g| g.as_ref());
        VaultStatus {
            exists: self.exists(),
            unlocked: unlocked.is_some(),
            biometry_available: biometry_available(),
            // Enrolled means both halves are present: a wrap in the manifest
            // *and* the key still in the Keychain. If the user cleared the
            // Keychain, Touch ID unlock would fail, so do not offer it.
            biometry_enrolled: self
                .read_manifest()
                .map(|m| m.keychain_wrap.is_some())
                .unwrap_or(false)
                && biometric_key_present(),
            via_biometry: unlocked.map(|u| u.via_biometry).unwrap_or(false),
            item_count: unlocked.map(|u| u.index.items.len()).unwrap_or(0),
            unlocked_seconds: unlocked.map(|u| u.since.elapsed().as_secs()).unwrap_or(0),
            root: self.root.to_string_lossy().to_string(),
        }
    }

    // ---- lifecycle ------------------------------------------------------

    pub fn create(&self, passphrase: &str, enable_biometry: bool) -> Result<(), VaultError> {
        if self.exists() {
            return Err(VaultError::AlreadyExists);
        }
        if passphrase.chars().count() < MIN_PASSPHRASE {
            return Err(VaultError::WeakPassphrase(MIN_PASSPHRASE));
        }

        let kdf = KdfParams::default();
        let mut pass_kek = derive_kek(passphrase, &kdf)?;
        let dek = crypto::random_bytes::<32>();

        let passphrase_wrap = Wrap::seal(&pass_kek, &dek);
        pass_kek.zeroize();

        // The passphrase wrap is written first and unconditionally, so a
        // failure to provision Touch ID can never produce an unopenable vault.
        let keychain_wrap = if enable_biometry {
            match provision_biometric_kek() {
                Ok(mut kek) => {
                    let w = Wrap::seal(&kek, &dek);
                    kek.zeroize();
                    Some(w)
                }
                Err(_) => None,
            }
        } else {
            None
        };

        let manifest = Manifest {
            version: 1,
            kdf,
            passphrase_wrap,
            keychain_wrap,
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        std::fs::create_dir_all(self.blobs())?;
        self.write_manifest(&manifest)?;

        let index = Index::default();
        self.write_index(&dek, &index)?;
        *self.inner.write().unwrap() = Some(Unlocked {
            dek,
            index,
            since: Instant::now(),
            via_biometry: false,
        });
        Ok(())
    }

    pub fn unlock_with_passphrase(&self, passphrase: &str) -> Result<(), VaultError> {
        let manifest = self.read_manifest()?;
        let mut kek = derive_kek(passphrase, &manifest.kdf)?;
        let dek = manifest.passphrase_wrap.open(&kek);
        kek.zeroize();
        let dek = dek?;
        self.finish_unlock(dek, false)
    }

    pub fn unlock_with_biometry(&self) -> Result<(), VaultError> {
        let manifest = self.read_manifest()?;
        let wrap = manifest.keychain_wrap.ok_or(VaultError::NoBiometricKey)?;
        let mut kek = load_biometric_kek()?;
        let dek = wrap.open(&kek);
        kek.zeroize();
        // A Keychain key that no longer unwraps means the vault was re-keyed;
        // the passphrase remains the source of truth.
        let dek = dek.map_err(|_| VaultError::NoBiometricKey)?;
        self.finish_unlock(dek, true)
    }

    fn finish_unlock(&self, dek: [u8; 32], via_biometry: bool) -> Result<(), VaultError> {
        let index = self.read_index(&dek)?;
        *self.inner.write().unwrap() = Some(Unlocked {
            dek,
            index,
            since: Instant::now(),
            via_biometry,
        });
        Ok(())
    }

    pub fn lock(&self) {
        // Dropping `Unlocked` zeroizes the data key.
        *self.inner.write().unwrap() = None;
    }

    /// Add or replace the Touch ID unlock path. Requires the vault to be open,
    /// which means the caller has already proved they hold the passphrase.
    pub fn enable_biometry(&self) -> Result<(), VaultError> {
        let guard = self.inner.read().unwrap();
        let u = guard.as_ref().ok_or(VaultError::Locked)?;
        let mut kek = provision_biometric_kek()?;
        let wrap = Wrap::seal(&kek, &u.dek);
        kek.zeroize();
        drop(guard);

        let mut manifest = self.read_manifest()?;
        manifest.keychain_wrap = Some(wrap);
        self.write_manifest(&manifest)
    }

    pub fn disable_biometry(&self) -> Result<(), VaultError> {
        let mut manifest = self.read_manifest()?;
        manifest.keychain_wrap = None;
        self.write_manifest(&manifest)?;
        let _ = delete_biometric_kek();
        Ok(())
    }

    pub fn change_passphrase(&self, current: &str, next: &str) -> Result<(), VaultError> {
        if next.chars().count() < MIN_PASSPHRASE {
            return Err(VaultError::WeakPassphrase(MIN_PASSPHRASE));
        }
        let mut manifest = self.read_manifest()?;
        let mut old_kek = derive_kek(current, &manifest.kdf)?;
        let dek = manifest.passphrase_wrap.open(&old_kek);
        old_kek.zeroize();
        let mut dek = dek?;

        // A fresh salt on every change, so the new wrap shares nothing with the old.
        manifest.kdf = KdfParams::default();
        let mut new_kek = derive_kek(next, &manifest.kdf)?;
        manifest.passphrase_wrap = Wrap::seal(&new_kek, &dek);
        new_kek.zeroize();
        dek.zeroize();

        self.write_manifest(&manifest)
    }

    // ---- index ----------------------------------------------------------

    fn read_index(&self, dek: &[u8; 32]) -> Result<Index, VaultError> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(Index::default());
        }
        let sealed = std::fs::read(&path)?;
        let plain = crypto::open(dek, &sealed)?;
        serde_json::from_slice(&plain).map_err(|e| VaultError::Corrupt(e.to_string()))
    }

    fn write_index(&self, dek: &[u8; 32], index: &Index) -> Result<(), VaultError> {
        let mut plain = serde_json::to_vec(index).map_err(|e| VaultError::Corrupt(e.to_string()))?;
        let sealed = crypto::seal(dek, &plain);
        plain.zeroize();
        let tmp = self.index_path().with_extension("enc.tmp");
        std::fs::write(&tmp, &sealed)?;
        std::fs::rename(&tmp, self.index_path())?;
        Ok(())
    }

    fn with_unlocked<T>(&self, f: impl FnOnce(&Unlocked) -> Result<T, VaultError>) -> Result<T, VaultError> {
        let guard = self.inner.read().unwrap();
        let u = guard.as_ref().ok_or(VaultError::Locked)?;
        f(u)
    }

    pub fn list(&self) -> Result<Vec<VaultItem>, VaultError> {
        self.with_unlocked(|u| {
            let mut items = u.index.items.clone();
            items.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            Ok(items)
        })
    }

    pub fn get_item(&self, id: &str) -> Result<VaultItem, VaultError> {
        self.with_unlocked(|u| {
            u.index
                .items
                .iter()
                .find(|i| i.id == id)
                .cloned()
                .ok_or_else(|| VaultError::NoSuchItem(id.to_string()))
        })
    }

    /// Reserve an id and hand out the file key for it. The engine seals its own
    /// output with this key; the vault's data key never leaves this process.
    pub fn reserve_slot(&self) -> Result<(String, [u8; 16], [u8; 32], PathBuf), VaultError> {
        self.with_unlocked(|u| {
            let id = uuid::Uuid::new_v4().to_string();
            let file_id = crypto::random_bytes::<16>();
            let key = crypto::derive_file_key(&u.dek, &file_id);
            std::fs::create_dir_all(self.blobs())?;
            Ok((id.clone(), file_id, key, self.blob_path(&id)))
        })
    }

    /// Derive the key for one existing blob, so the engine can open exactly
    /// that input and nothing else. The file id lives in the blob's header.
    pub fn input_key(&self, id: &str) -> Result<([u8; 16], [u8; 32], PathBuf), VaultError> {
        self.with_unlocked(|u| {
            let path = self.blob_path(id);
            if !path.exists() {
                return Err(VaultError::NoSuchItem(id.to_string()));
            }
            let mut header = [0u8; crypto::HEADER_LEN];
            {
                use std::io::Read;
                let mut f = std::fs::File::open(&path)?;
                f.read_exact(&mut header).map_err(|_| {
                    VaultError::Corrupt(format!("blob {id} is shorter than a header"))
                })?;
            }
            let file_id = crypto::file_id_of(&header)?;
            let key = crypto::derive_file_key(&u.dek, &file_id);
            Ok((file_id, key, path))
        })
    }

    /// Drop reserved slots that never received content, so a failed or
    /// cancelled run leaves nothing behind.
    pub fn discard_slots(&self, ids: &[String]) {
        for id in ids {
            let _ = std::fs::remove_file(self.blob_path(id));
        }
    }

    /// Record an already-sealed blob that the engine wrote into place.
    pub fn commit_slot(&self, item: VaultItem) -> Result<(), VaultError> {
        let mut guard = self.inner.write().unwrap();
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        if !self.blob_path(&item.id).exists() {
            return Err(VaultError::NoSuchItem(item.id));
        }
        u.index.items.push(item);
        let index = u.index.clone();
        let dek = u.dek;
        drop(guard);
        self.write_index(&dek, &index)
    }

    /// Seal and store plaintext bytes the host already holds.
    pub fn put(&self, plaintext: &[u8], item: VaultItem) -> Result<String, VaultError> {
        let mut guard = self.inner.write().unwrap();
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        let sealed = crypto::seal(&u.dek, plaintext);
        std::fs::create_dir_all(self.blobs())?;
        std::fs::write(self.blob_path(&item.id), &sealed)?;

        let id = item.id.clone();
        u.index.items.push(item);
        let index = u.index.clone();
        let dek = u.dek;
        drop(guard);
        self.write_index(&dek, &index)?;
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Result<Vec<u8>, VaultError> {
        self.with_unlocked(|u| {
            let path = self.blob_path(id);
            if !path.exists() {
                return Err(VaultError::NoSuchItem(id.to_string()));
            }
            let sealed = std::fs::read(&path)?;
            Ok(crypto::open(&u.dek, &sealed)?)
        })
    }

    pub fn delete(&self, id: &str) -> Result<(), VaultError> {
        let mut guard = self.inner.write().unwrap();
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        u.index.items.retain(|i| i.id != id);
        let index = u.index.clone();
        let dek = u.dek;
        drop(guard);
        let _ = std::fs::remove_file(self.blob_path(id));
        self.write_index(&dek, &index)
    }

    /// The one sanctioned way plaintext leaves the vault.
    pub fn export(&self, id: &str, dest: &Path) -> Result<u64, VaultError> {
        let plain = self.get(id)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(dest, &plain)?;
        Ok(plain.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// Keychain shims, so non-macOS builds still compile
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn biometry_available() -> bool { super::keychain::biometry_usable() }

#[cfg(target_os = "macos")]
fn biometric_key_present() -> bool { super::keychain::kek_present() }
#[cfg(not(target_os = "macos"))]
fn biometric_key_present() -> bool { false }
#[cfg(not(target_os = "macos"))]
fn biometry_available() -> bool { false }

#[cfg(target_os = "macos")]
fn provision_biometric_kek() -> Result<[u8; 32], VaultError> {
    super::keychain::provision_kek().map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn provision_biometric_kek() -> Result<[u8; 32], VaultError> { Err(VaultError::NoBiometricKey) }

#[cfg(target_os = "macos")]
fn load_biometric_kek() -> Result<[u8; 32], VaultError> {
    super::keychain::load_kek("Unlock your Model Studio vault")
        .map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn load_biometric_kek() -> Result<[u8; 32], VaultError> { Err(VaultError::NoBiometricKey) }

#[cfg(target_os = "macos")]
fn delete_biometric_kek() -> Result<(), VaultError> {
    super::keychain::delete_kek().map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn delete_biometric_kek() -> Result<(), VaultError> { Ok(()) }
