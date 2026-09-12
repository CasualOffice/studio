//! Vault state, manifest and encrypted item index.
//!
//! On disk the vault is opaque:
//!
//! ```text
//! vault/
//!   vault.json     wrapped data keys and KDF parameters -- no secrets
//!   vault.json.bak second copy of the same, so losing one is survivable
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

const B64: base64::engine::general_purpose::GeneralPurpose =
    base64::engine::general_purpose::STANDARD;

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
        Self {
            nonce: B64.encode(nonce),
            ct: B64.encode(ct),
        }
    }

    fn open(&self, kek: &[u8; 32]) -> Result<[u8; 32], VaultError> {
        let nonce = B64
            .decode(&self.nonce)
            .map_err(|e| VaultError::Corrupt(e.to_string()))?;
        let ct = B64
            .decode(&self.ct)
            .map_err(|e| VaultError::Corrupt(e.to_string()))?;
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

/// Content digest used for deduplication.
pub fn content_hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn derive_kek(passphrase: &str, kdf: &KdfParams) -> Result<[u8; 32], VaultError> {
    use argon2::{Algorithm, Argon2, Params, Version};
    if kdf.algo != "argon2id"
        || !(8 * 1024..=1024 * 1024).contains(&kdf.m_cost)
        || !(1..=10).contains(&kdf.t_cost)
        || !(1..=16).contains(&kdf.p_cost)
    {
        return Err(VaultError::Corrupt(
            "unsafe or unsupported KDF parameters".into(),
        ));
    }
    let salt = B64
        .decode(&kdf.salt)
        .map_err(|e| VaultError::Corrupt(e.to_string()))?;
    if !(16..=64).contains(&salt.len()) {
        return Err(VaultError::Corrupt("KDF salt has an invalid length".into()));
    }
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
    /// SHA-256 of the plaintext. Lets an import recognise a file already in the
    /// vault instead of storing a second copy, and lets a rebuilt index tell
    /// whether two blobs hold the same picture. Optional so indexes written
    /// before hashing existed still load.
    #[serde(default)]
    pub content_hash: Option<String>,
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
    /// The project this item belongs to, if any.
    ///
    /// A picture board is one thing a person made, not seventeen loose
    /// pictures that happen to share a timestamp. Items that carry the same
    /// project id are shown as a single entry in the library and are acted on
    /// together. Optional and defaulted so every index written before projects
    /// existed still loads, and so a one-off generation carries nothing.
    #[serde(default)]
    pub project: Option<String>,
    /// Human-readable name for the project. Carried on every member so the
    /// library can title the group without a second lookup.
    #[serde(default)]
    pub project_name: Option<String>,
    /// Position within the project: panel number, page number. Sorts the
    /// members back into the order they were meant to be read in.
    #[serde(default)]
    pub project_index: Option<u32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Index {
    items: Vec<VaultItem>,
    /// Small private application documents, such as the in-progress Board.
    /// They live inside the encrypted index so drafts never touch localStorage.
    #[serde(default)]
    state: std::collections::BTreeMap<String, String>,
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

/// Read the vault lock, recovering if a previous holder panicked.
///
/// `unwrap()` on a poisoned lock turns one unrelated panic into a permanently
/// unusable vault: every later operation panics too, and the only escape is
/// quitting the app. Poisoning here means some other thread failed partway
/// through, not that the guarded data is unreadable, so recovering is both
/// safe and the difference between one lost operation and all of them.
macro_rules! read_guard {
    ($lock:expr) => {
        $lock
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
}

macro_rules! write_guard {
    ($lock:expr) => {
        $lock
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    };
}

#[derive(Serialize, Clone, Debug)]
pub struct RepairReport {
    /// Blobs that had no index entry and were identified and re-listed.
    pub recovered: usize,
    /// Index entries whose blob is gone.
    pub dropped: usize,
    /// Files in the blob directory this vault's key cannot open.
    pub unreadable: usize,
}

/// Identify content from its leading bytes, since a recovered blob has no
/// filename or stored metadata to go on.
fn sniff_mime(bytes: &[u8]) -> &'static str {
    match bytes {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [b'G', b'I', b'F', b'8', ..] => "image/gif",
        [b'%', b'P', b'D', b'F', ..] => "application/pdf",
        _ if bytes.len() > 12 && &bytes[4..8] == b"ftyp" => match &bytes[8..12] {
            b"heic" | b"heix" | b"mif1" => "image/heic",
            b"avif" => "image/avif",
            _ => "video/mp4",
        },
        _ if bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" => {
            "image/webp"
        }
        _ => "application/octet-stream",
    }
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
        Self {
            root,
            inner: RwLock::new(None),
        }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("vault.json")
    }
    /// Second copy of the manifest. Not a previous version: both copies are
    /// kept current, see `write_manifest`.
    pub fn manifest_backup_path(&self) -> PathBuf {
        self.root.join("vault.json.bak")
    }
    fn index_path(&self) -> PathBuf {
        self.root.join("index.enc")
    }
    fn blobs(&self) -> PathBuf {
        self.root.join("blobs")
    }
    fn blob_path(&self, id: &str) -> PathBuf {
        self.blobs().join(id)
    }

    fn validate_id(id: &str) -> Result<(), VaultError> {
        uuid::Uuid::parse_str(id)
            .map(|_| ())
            .map_err(|_| VaultError::NoSuchItem(id.to_string()))
    }

    /// Whether there is a vault here at all.
    ///
    /// The backup counts. This used to look only at `vault.json`, so a vault
    /// whose live manifest had been lost read as "no vault yet" -- and `create`
    /// would then write a brand-new data key over the one surviving copy of the
    /// old one, turning a recoverable accident into every blob being unopenable
    /// forever.
    pub fn exists(&self) -> bool {
        self.manifest_path().exists() || self.manifest_backup_path().exists()
    }

    pub fn is_unlocked(&self) -> bool {
        self.inner.read().map(|g| g.is_some()).unwrap_or(false)
    }

    fn read_manifest_file(path: &Path) -> Result<Manifest, VaultError> {
        if !path.exists() {
            return Err(VaultError::Absent);
        }
        let raw = std::fs::read(path)?;
        serde_json::from_slice(&raw).map_err(|e| VaultError::Corrupt(e.to_string()))
    }

    /// Read the manifest, falling back to the backup copy.
    ///
    /// There is no other record anywhere of the wrapped data key, `repair`
    /// cannot reconstruct it and nothing re-creates it, so an unreadable
    /// `vault.json` used to mean every blob in the vault was lost. Reading the
    /// second copy instead turns that into an inconvenience.
    fn read_manifest(&self) -> Result<Manifest, VaultError> {
        let primary = match Self::read_manifest_file(&self.manifest_path()) {
            Ok(m) => return Ok(m),
            Err(e) => e,
        };
        let backup = self.manifest_backup_path();
        if !backup.exists() {
            return Err(primary);
        }
        let m = Self::read_manifest_file(&backup)?;
        let reason = match primary {
            VaultError::Absent => "vault.json is missing".to_string(),
            other => format!("vault.json is unreadable: {other}"),
        };
        eprintln!("vault: {reason}; recovered the backup manifest");
        Ok(m)
    }

    /// Write the manifest to both the backup path and the live path.
    ///
    /// `vault.json` was written as a single temp-file-and-rename with no second
    /// copy and no `fsync`, which meant a crash, a full disk or a power loss
    /// during `change_passphrase`, `enable_biometry` or `disable_biometry` could
    /// leave a populated vault with no readable wrapped key and nothing to fall
    /// back on.
    ///
    /// The backup is written first and the live copy second, so at every moment
    /// in between at least one of the two is a complete manifest. Both end up
    /// holding the *current* wraps rather than one lagging a version behind: a
    /// deliberately stale backup would mean a passphrase the user had just
    /// replaced still unwrapped the data key, which is the opposite of what
    /// changing a passphrase is for.
    fn write_manifest(&self, m: &Manifest) -> Result<(), VaultError> {
        std::fs::create_dir_all(&self.root)?;
        let bytes = serde_json::to_vec_pretty(m).map_err(|e| VaultError::Corrupt(e.to_string()))?;
        self.write_manifest_file(&self.manifest_backup_path(), &bytes)?;
        self.write_manifest_file(&self.manifest_path(), &bytes)
    }

    /// One durable copy: write a temporary file, flush it, rename it into
    /// place, then flush the directory entry as well.
    ///
    /// Without the `sync_all` the rename could reach the disk while the
    /// contents had not, which is precisely how a truncated or empty
    /// `vault.json` appears after a power loss.
    fn write_manifest_file(&self, path: &Path, bytes: &[u8]) -> Result<(), VaultError> {
        use std::io::Write;
        let name = path
            .file_name()
            .ok_or_else(|| VaultError::Corrupt("manifest path has no file name".into()))?
            .to_string_lossy()
            .to_string();
        let tmp = path.with_file_name(format!("{name}.tmp"));
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        // Best-effort on purpose: not every filesystem lets a directory handle
        // be flushed, and a refusal here must not fail a write whose data is
        // already safely on disk.
        let _ = std::fs::File::open(&self.root).and_then(|dir| dir.sync_all());
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
        *write_guard!(self.inner) = Some(Unlocked {
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
        let index = self.read_index_with_fallback(&dek)?;
        *write_guard!(self.inner) = Some(Unlocked {
            dek,
            index,
            since: Instant::now(),
            via_biometry,
        });
        Ok(())
    }

    pub fn lock(&self) {
        // Dropping `Unlocked` zeroizes the data key.
        *write_guard!(self.inner) = None;
    }

    /// Add or replace the Touch ID unlock path. Requires the vault to be open,
    /// which means the caller has already proved they hold the passphrase.
    pub fn enable_biometry(&self) -> Result<(), VaultError> {
        let guard = read_guard!(self.inner);
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

    /// Write the index, keeping one previous copy.
    ///
    /// The index is the only record of what every blob *is*: lose it and the
    /// content is still decryptable but anonymous. Rotating a `.bak` costs one
    /// rename and turns a corrupt write from total metadata loss into the loss
    /// of whatever changed since the last save.
    fn write_index(&self, dek: &[u8; 32], index: &Index) -> Result<(), VaultError> {
        let mut plain =
            serde_json::to_vec(index).map_err(|e| VaultError::Corrupt(e.to_string()))?;
        let sealed = crypto::seal(dek, &plain);
        plain.zeroize();

        let path = self.index_path();
        let tmp = path.with_extension("enc.tmp");
        std::fs::write(&tmp, &sealed)?;

        if path.exists() {
            let backup = path.with_extension("enc.bak");
            // A failed backup must not block the write; the new index is still
            // strictly better than no write at all.
            let _ = std::fs::rename(&path, &backup);
        }
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn read_index_with_fallback(&self, dek: &[u8; 32]) -> Result<Index, VaultError> {
        match self.read_index(dek) {
            Ok(i) => Ok(i),
            Err(primary) => {
                let backup = self.index_path().with_extension("enc.bak");
                if !backup.exists() {
                    return Err(primary);
                }
                // The live index is unreadable. The previous one loses only the
                // most recent change, which beats losing every name and prompt.
                let sealed = std::fs::read(&backup)?;
                let plain = crypto::open(dek, &sealed)?;
                let index: Index = serde_json::from_slice(&plain)
                    .map_err(|e| VaultError::Corrupt(e.to_string()))?;
                eprintln!("vault: index was unreadable ({primary}); recovered the backup");
                Ok(index)
            }
        }
    }

    fn with_unlocked<T>(
        &self,
        f: impl FnOnce(&Unlocked) -> Result<T, VaultError>,
    ) -> Result<T, VaultError> {
        let guard = read_guard!(self.inner);
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
        Self::validate_id(id)?;
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
        Self::validate_id(id)?;
        self.with_unlocked(|u| {
            if !u.index.items.iter().any(|item| item.id == id) {
                return Err(VaultError::NoSuchItem(id.to_string()));
            }
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
    pub fn commit_slot(&self, mut item: VaultItem) -> Result<(), VaultError> {
        Self::validate_id(&item.id)?;
        let mut guard = write_guard!(self.inner);
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        if !self.blob_path(&item.id).exists() {
            return Err(VaultError::NoSuchItem(item.id));
        }
        // Hash generated output too: an exported copy re-imported later should
        // be recognised rather than stored twice.
        if item.content_hash.is_none() {
            if let Ok(sealed) = std::fs::read(self.blob_path(&item.id)) {
                if let Ok(plain) = crypto::open(&u.dek, &sealed) {
                    item.content_hash = Some(content_hash(&plain));
                }
            }
        }
        u.index.items.push(item);
        let index = u.index.clone();
        let dek = u.dek;
        self.write_index(&dek, &index)
    }

    /// Seal and store plaintext bytes the host already holds.
    ///
    /// Returns the existing id when the same content is already stored, so
    /// importing a file twice does not fill the vault with copies.
    ///
    /// The lock is held across the index write. Cloning the index and
    /// releasing first let two concurrent writers interleave, so the later
    /// write clobbered the earlier one's entry and orphaned its blob.
    pub fn put(&self, plaintext: &[u8], mut item: VaultItem) -> Result<String, VaultError> {
        Self::validate_id(&item.id)?;
        let mut guard = write_guard!(self.inner);
        let u = guard.as_mut().ok_or(VaultError::Locked)?;

        let hash = content_hash(plaintext);
        if item.kind != "mask" {
            if let Some(existing) = u
                .index
                .items
                .iter()
                .find(|i| i.content_hash.as_deref() == Some(hash.as_str()))
            {
                // Same bytes already stored; hand back what is already there.
                return Ok(existing.id.clone());
            }
        }
        item.content_hash = Some(hash);

        let sealed = crypto::seal(&u.dek, plaintext);
        std::fs::create_dir_all(self.blobs())?;
        std::fs::write(self.blob_path(&item.id), &sealed)?;

        let id = item.id.clone();
        u.index.items.push(item);
        let index = u.index.clone();
        let dek = u.dek;
        self.write_index(&dek, &index)?;
        Ok(id)
    }

    pub fn get(&self, id: &str) -> Result<Vec<u8>, VaultError> {
        Self::validate_id(id)?;
        self.with_unlocked(|u| {
            if !u.index.items.iter().any(|item| item.id == id) {
                return Err(VaultError::NoSuchItem(id.to_string()));
            }
            let path = self.blob_path(id);
            if !path.exists() {
                return Err(VaultError::NoSuchItem(id.to_string()));
            }
            let sealed = std::fs::read(&path)?;
            Ok(crypto::open(&u.dek, &sealed)?)
        })
    }

    pub fn delete(&self, id: &str) -> Result<(), VaultError> {
        Self::validate_id(id)?;
        let mut guard = write_guard!(self.inner);
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        if !u.index.items.iter().any(|item| item.id == id) {
            return Err(VaultError::NoSuchItem(id.to_string()));
        }
        u.index.items.retain(|i| i.id != id);
        let index = u.index.clone();
        let dek = u.dek;
        let _ = std::fs::remove_file(self.blob_path(id));
        self.write_index(&dek, &index)
    }

    /// Reconcile the index against what is actually on disk.
    ///
    /// Two failure modes this repairs. Blobs with no index entry -- orphaned by
    /// a crash between writing content and saving the index -- are decrypted,
    /// identified and re-listed rather than silently wasting space forever.
    /// Index entries with no blob are dropped, since they can never be opened.
    ///
    /// This is possible at all because every blob carries its own file id in
    /// its header, so a blob is self-describing given the data key.
    pub fn repair(&self) -> Result<RepairReport, VaultError> {
        let mut guard = write_guard!(self.inner);
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        let dek = u.dek;

        let known: std::collections::HashSet<String> =
            u.index.items.iter().map(|i| i.id.clone()).collect();

        let mut recovered = Vec::new();
        let mut unreadable = 0usize;
        if let Ok(entries) = std::fs::read_dir(self.blobs()) {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if known.contains(&name) || name.ends_with(".tmp") {
                    continue;
                }
                // Not a blob name at all, so not damaged vault data. `blobs/`
                // is a directory Finder can reach, and counting a `.DS_Store`
                // here raised "1 file could not be read" on the Security
                // screen after every repair -- a data-loss alarm for a file
                // this vault never wrote. Left on disk, as everything
                // unrecognised here is, but not reported as ours.
                if Self::validate_id(&name).is_err() {
                    continue;
                }
                let Ok(sealed) = std::fs::read(e.path()) else {
                    continue;
                };
                match crypto::open(&dek, &sealed) {
                    Ok(plain) => {
                        let mime = sniff_mime(&plain);
                        recovered.push(VaultItem {
                            id: name.clone(),
                            content_hash: Some(content_hash(&plain)),
                            kind: "recovered".into(),
                            name: format!("recovered-{}", &name[..8.min(name.len())]),
                            mime: mime.into(),
                            bytes: plain.len() as u64,
                            model: String::new(),
                            prompt: String::new(),
                            seed: 0,
                            width: None,
                            height: None,
                            steps: None,
                            guidance: None,
                            inputs: vec![],
                            created_at: chrono::Local::now().to_rfc3339(),
                            duration_ms: 0,
                            // Recovery reads blobs off disk with no index, so
                            // there is nothing left that says which project a
                            // file belonged to. Left unset rather than guessed.
                            project: None,
                            project_name: None,
                            project_index: None,
                        });
                    }
                    // Not ours, or corrupt. Left alone rather than deleted:
                    // destroying data during a repair is the wrong default.
                    Err(_) => unreadable += 1,
                }
            }
        }

        let before = u.index.items.len();
        u.index.items.retain(|i| self.blob_path(&i.id).exists());
        let dropped = before - u.index.items.len();

        let recovered_count = recovered.len();
        u.index.items.extend(recovered);
        let index = u.index.clone();
        drop(guard);

        if recovered_count > 0 || dropped > 0 {
            self.write_index(&dek, &index)?;
        }
        Ok(RepairReport {
            recovered: recovered_count,
            dropped,
            unreadable,
        })
    }

    /// Seal bytes with this vault's data key.
    ///
    /// Only used to plant an orphaned blob for the repair probe: a realistic
    /// test needs content the vault can actually open.
    pub fn seal_for_test(&self, bytes: &[u8]) -> Option<Vec<u8>> {
        let guard = read_guard!(self.inner);
        let u = guard.as_ref()?;
        Some(crypto::seal(&u.dek, bytes))
    }

    /// The one sanctioned way plaintext leaves the vault.
    pub fn export(&self, id: &str, dest: &Path, overwrite: bool) -> Result<u64, VaultError> {
        let plain = self.get(id)?;
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .create_new(!overwrite)
            .truncate(overwrite)
            .open(dest)?;
        file.write_all(&plain)?;
        file.sync_all()?;
        Ok(plain.len() as u64)
    }

    pub fn get_state(&self, key: &str) -> Result<Option<String>, VaultError> {
        self.with_unlocked(|u| Ok(u.index.state.get(key).cloned()))
    }

    pub fn set_state(&self, key: &str, value: String) -> Result<(), VaultError> {
        if key.is_empty()
            || key.len() > 64
            || !key.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return Err(VaultError::Corrupt("invalid application-state key".into()));
        }
        let mut guard = write_guard!(self.inner);
        let u = guard.as_mut().ok_or(VaultError::Locked)?;
        let mut index = u.index.clone();
        index.state.insert(key.to_string(), value);
        self.write_index(&u.dek, &index)?;
        u.index = index;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Keychain shims, so non-macOS builds still compile
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn biometry_available() -> bool {
    super::keychain::biometry_usable()
}

#[cfg(target_os = "macos")]
fn biometric_key_present() -> bool {
    super::keychain::kek_present()
}
#[cfg(not(target_os = "macos"))]
fn biometric_key_present() -> bool {
    false
}
#[cfg(not(target_os = "macos"))]
fn biometry_available() -> bool {
    false
}

#[cfg(target_os = "macos")]
fn provision_biometric_kek() -> Result<[u8; 32], VaultError> {
    super::keychain::provision_kek().map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn provision_biometric_kek() -> Result<[u8; 32], VaultError> {
    Err(VaultError::NoBiometricKey)
}

#[cfg(target_os = "macos")]
fn load_biometric_kek() -> Result<[u8; 32], VaultError> {
    super::keychain::load_kek("Unlock your Model Studio vault")
        .map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn load_biometric_kek() -> Result<[u8; 32], VaultError> {
    Err(VaultError::NoBiometricKey)
}

#[cfg(target_os = "macos")]
fn delete_biometric_kek() -> Result<(), VaultError> {
    super::keychain::delete_kek().map_err(|_| VaultError::NoBiometricKey)
}
#[cfg(not(target_os = "macos"))]
fn delete_biometric_kek() -> Result<(), VaultError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("modelstudio-{name}-{}", uuid::Uuid::new_v4()))
    }

    #[test]
    fn rejects_blob_ids_that_are_paths() {
        for id in ["../vault.json", "/tmp/file", "not-a-uuid"] {
            assert!(Vault::validate_id(id).is_err(), "accepted {id}");
        }
        assert!(Vault::validate_id(&uuid::Uuid::new_v4().to_string()).is_ok());
    }

    #[test]
    fn encrypted_state_survives_lock_and_unlock() {
        let root = scratch("state");
        let vault = Vault::new(root.clone());
        vault.create("a long test passphrase", false).unwrap();
        vault
            .set_state("board-draft", r#"{"story":"private"}"#.into())
            .unwrap();
        vault.lock();
        assert!(matches!(
            vault.get_state("board-draft"),
            Err(VaultError::Locked)
        ));
        vault
            .unlock_with_passphrase("a long test passphrase")
            .unwrap();
        assert_eq!(
            vault.get_state("board-draft").unwrap().as_deref(),
            Some(r#"{"story":"private"}"#)
        );
        let raw_index = std::fs::read(root.join("index.enc")).unwrap();
        assert!(!raw_index.windows(7).any(|bytes| bytes == b"private"));
        let _ = std::fs::remove_dir_all(root);
    }

    fn an_item(name: &str) -> VaultItem {
        VaultItem {
            id: uuid::Uuid::new_v4().to_string(),
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
        }
    }

    /// Losing or corrupting `vault.json` used to destroy the only copy of the
    /// wrapped data key, and with it every blob in the vault.
    #[test]
    fn vault_opens_from_the_manifest_backup() {
        let root = scratch("manifest-backup");
        let vault = Vault::new(root.clone());
        vault.create("a long test passphrase", false).unwrap();
        let id = vault.put(b"the only copy", an_item("a.png")).unwrap();
        assert!(
            vault.manifest_backup_path().exists(),
            "no backup was written"
        );

        std::fs::remove_file(vault.manifest_path()).unwrap();
        assert!(
            vault.exists(),
            "a vault with only a backup manifest is still a vault"
        );
        vault.lock();
        vault
            .unlock_with_passphrase("a long test passphrase")
            .unwrap();
        assert_eq!(vault.get(&id).unwrap(), b"the only copy");

        // A manifest that is present but unparseable takes the same path.
        std::fs::write(vault.manifest_path(), b"{ truncated").unwrap();
        vault.lock();
        vault
            .unlock_with_passphrase("a long test passphrase")
            .unwrap();
        assert_eq!(vault.get(&id).unwrap(), b"the only copy");
        let _ = std::fs::remove_dir_all(root);
    }

    /// The backup must never lag a version behind: a superseded passphrase
    /// unwrapping the data key out of `vault.json.bak` would make changing a
    /// passphrase meaningless.
    #[test]
    fn manifest_backup_holds_the_current_wrap() {
        let root = scratch("manifest-current");
        let vault = Vault::new(root.clone());
        vault.create("a long test passphrase", false).unwrap();
        let id = vault.put(b"still readable", an_item("a.png")).unwrap();
        vault
            .change_passphrase("a long test passphrase", "a longer new one")
            .unwrap();

        std::fs::remove_file(vault.manifest_path()).unwrap();
        vault.lock();
        assert!(
            vault
                .unlock_with_passphrase("a long test passphrase")
                .is_err(),
            "the replaced passphrase still opens the backup"
        );
        vault.unlock_with_passphrase("a longer new one").unwrap();
        assert_eq!(vault.get(&id).unwrap(), b"still readable");
        let _ = std::fs::remove_dir_all(root);
    }
}
