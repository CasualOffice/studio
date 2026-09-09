//! Content encryption for the vault.
//!
//! Format (all integers little-endian):
//!
//! ```text
//! header (28 bytes)
//!   magic       "MSV1"   4
//!   version     u8 = 1   1
//!   alg         u8 = 1   1   ChaCha20-Poly1305 (IETF, 96-bit nonce)
//!   flags       u16 = 0  2
//!   file_id     [u8;16] 16   random per file
//!   chunk_size  u32      4
//! body
//!   repeated sealed chunks, each `chunk_size` plaintext bytes except the last
//! ```
//!
//! Each file gets its own subkey, so a plain counter nonce is safe: a nonce is
//! never reused under the same key. The AAD binds every chunk to the header,
//! its index, and whether it is final, which makes truncation, reordering and
//! cross-file splicing detectable rather than silent.

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use rand::RngCore;
use zeroize::Zeroize;

pub const MAGIC: &[u8; 4] = b"MSV1";
pub const VERSION: u8 = 1;
pub const ALG_CHACHA20POLY1305: u8 = 1;
pub const HEADER_LEN: usize = 28;
pub const CHUNK_SIZE: u32 = 256 * 1024;
pub const TAG_LEN: usize = 16;

/// Domain separators. Changing one invalidates old data by design.
const HKDF_INFO_FILE: &[u8] = b"melp-vault-file-v1";
pub const AAD_DEK_WRAP: &[u8] = b"melp-vault-dek-v1";

#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("not a vault file")]
    BadMagic,
    #[error("unsupported vault format version {0}")]
    BadVersion(u8),
    #[error("unsupported cipher {0}")]
    BadAlg(u8),
    #[error("file is truncated or corrupt")]
    Truncated,
    #[error("decryption failed: wrong key, or the data has been altered")]
    Decrypt,
    #[error("chunk size {0} is out of range")]
    BadChunkSize(u32),
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut b = [0u8; N];
    rand::thread_rng().fill_bytes(&mut b);
    b
}

/// Per-file subkey. `file_id` is public and unique, so it works as the salt.
///
/// Public because the engine subprocess is handed a *file* key rather than the
/// vault's data key: it can seal the one artifact it just produced and nothing
/// else, and the master key never crosses the process boundary.
pub fn derive_file_key(dek: &[u8; 32], file_id: &[u8; 16]) -> [u8; 32] {
    file_key(dek, file_id)
}

fn file_key(dek: &[u8; 32], file_id: &[u8; 16]) -> [u8; 32] {
    let hk = hkdf::Hkdf::<sha2::Sha256>::new(Some(file_id), dek);
    let mut out = [0u8; 32];
    hk.expand(HKDF_INFO_FILE, &mut out)
        .expect("32 bytes is a valid HKDF output length");
    out
}

fn nonce_for(counter: u64) -> Nonce {
    // 12-byte IETF nonce: 4 zero bytes then the counter.
    let mut n = [0u8; 12];
    n[4..].copy_from_slice(&counter.to_le_bytes());
    *Nonce::from_slice(&n)
}

fn aad_for(header: &[u8], counter: u64, final_chunk: bool) -> Vec<u8> {
    let mut aad = Vec::with_capacity(HEADER_LEN + 9);
    aad.extend_from_slice(header);
    aad.extend_from_slice(&counter.to_le_bytes());
    aad.push(final_chunk as u8);
    aad
}

/// Read the file id out of a sealed blob's header, so the host can re-derive
/// that blob's key without decrypting it.
pub fn file_id_of(sealed_prefix: &[u8]) -> Result<[u8; 16], CryptoError> {
    let (id, _) = parse_header(sealed_prefix)?;
    Ok(*id)
}

pub fn build_header(file_id: &[u8; 16], chunk_size: u32) -> [u8; HEADER_LEN] {
    let mut h = [0u8; HEADER_LEN];
    h[0..4].copy_from_slice(MAGIC);
    h[4] = VERSION;
    h[5] = ALG_CHACHA20POLY1305;
    // h[6..8] flags, left zero
    h[8..24].copy_from_slice(file_id);
    h[24..28].copy_from_slice(&chunk_size.to_le_bytes());
    h
}

fn parse_header(buf: &[u8]) -> Result<(&[u8; 16], u32), CryptoError> {
    if buf.len() < HEADER_LEN {
        return Err(CryptoError::Truncated);
    }
    if &buf[0..4] != MAGIC {
        return Err(CryptoError::BadMagic);
    }
    if buf[4] != VERSION {
        return Err(CryptoError::BadVersion(buf[4]));
    }
    if buf[5] != ALG_CHACHA20POLY1305 {
        return Err(CryptoError::BadAlg(buf[5]));
    }
    let file_id: &[u8; 16] = buf[8..24].try_into().unwrap();
    let chunk_size = u32::from_le_bytes(buf[24..28].try_into().unwrap());
    if chunk_size == 0 || chunk_size > 16 * 1024 * 1024 {
        return Err(CryptoError::BadChunkSize(chunk_size));
    }
    Ok((file_id, chunk_size))
}

/// Encrypt a whole buffer. Vault items are images and documents, which comfortably
/// fit in memory; streaming would buy nothing here and cost clarity.
pub fn seal(dek: &[u8; 32], plaintext: &[u8]) -> Vec<u8> {
    let file_id = random_bytes::<16>();
    let key = file_key(dek, &file_id);
    seal_with_file_key(&key, &file_id, plaintext)
}

/// Seal using a pre-derived file key. Mirrors `engine/vaultcrypto.py`; the two
/// implementations must stay byte-compatible.
pub fn seal_with_file_key(file_key: &[u8; 32], file_id: &[u8; 16], plaintext: &[u8]) -> Vec<u8> {
    let header = build_header(file_id, CHUNK_SIZE);
    let mut key = *file_key;
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));

    let chunk = CHUNK_SIZE as usize;
    let n_chunks = plaintext.len().div_ceil(chunk).max(1);
    let mut out = Vec::with_capacity(header.len() + plaintext.len() + n_chunks * TAG_LEN);
    out.extend_from_slice(&header);

    for i in 0..n_chunks {
        let start = i * chunk;
        let end = ((i + 1) * chunk).min(plaintext.len());
        let is_final = i == n_chunks - 1;
        let sealed = cipher
            .encrypt(
                &nonce_for(i as u64),
                Payload {
                    msg: &plaintext[start..end],
                    aad: &aad_for(&header, i as u64, is_final),
                },
            )
            .expect("ChaCha20-Poly1305 encryption cannot fail on valid input");
        out.extend_from_slice(&sealed);
    }

    key.zeroize();
    out
}

pub fn open(dek: &[u8; 32], data: &[u8]) -> Result<Vec<u8>, CryptoError> {
    let (file_id, chunk_size) = parse_header(data)?;
    let mut key = file_key(dek, file_id);
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key));
    let header = &data[..HEADER_LEN];

    let sealed_chunk = chunk_size as usize + TAG_LEN;
    let body = &data[HEADER_LEN..];
    if body.len() < TAG_LEN {
        key.zeroize();
        return Err(CryptoError::Truncated);
    }

    let n_chunks = body.len().div_ceil(sealed_chunk).max(1);
    let mut out = Vec::with_capacity(body.len());
    for i in 0..n_chunks {
        let start = i * sealed_chunk;
        let end = ((i + 1) * sealed_chunk).min(body.len());
        let is_final = i == n_chunks - 1;
        match cipher.decrypt(
            &nonce_for(i as u64),
            Payload {
                msg: &body[start..end],
                aad: &aad_for(header, i as u64, is_final),
            },
        ) {
            Ok(mut pt) => {
                out.extend_from_slice(&pt);
                pt.zeroize();
            }
            Err(_) => {
                key.zeroize();
                out.zeroize();
                return Err(CryptoError::Decrypt);
            }
        }
    }
    key.zeroize();
    Ok(out)
}

/// Wrap or unwrap the data key with a key-encryption key.
pub fn wrap_dek(kek: &[u8; 32], dek: &[u8; 32]) -> ([u8; 12], Vec<u8>) {
    let nonce_bytes = random_bytes::<12>();
    let cipher = ChaCha20Poly1305::new(Key::from_slice(kek));
    let ct = cipher
        .encrypt(
            Nonce::from_slice(&nonce_bytes),
            Payload { msg: dek, aad: AAD_DEK_WRAP },
        )
        .expect("wrapping a 32-byte key cannot fail");
    (nonce_bytes, ct)
}

pub fn unwrap_dek(kek: &[u8; 32], nonce: &[u8], ct: &[u8]) -> Result<[u8; 32], CryptoError> {
    if nonce.len() != 12 {
        return Err(CryptoError::Decrypt);
    }
    let cipher = ChaCha20Poly1305::new(Key::from_slice(kek));
    let mut pt = cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload { msg: ct, aad: AAD_DEK_WRAP },
        )
        .map_err(|_| CryptoError::Decrypt)?;
    if pt.len() != 32 {
        pt.zeroize();
        return Err(CryptoError::Decrypt);
    }
    let mut dek = [0u8; 32];
    dek.copy_from_slice(&pt);
    pt.zeroize();
    Ok(dek)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_various_sizes() {
        let dek = random_bytes::<32>();
        for len in [0usize, 1, 1024, CHUNK_SIZE as usize, CHUNK_SIZE as usize + 1,
                    CHUNK_SIZE as usize * 3 + 7] {
            let pt: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let ct = seal(&dek, &pt);
            assert_eq!(open(&dek, &ct).unwrap(), pt, "failed at len {len}");
        }
    }

    #[test]
    fn wrong_key_is_rejected() {
        let dek = random_bytes::<32>();
        let other = random_bytes::<32>();
        let ct = seal(&dek, b"secret");
        assert!(matches!(open(&other, &ct), Err(CryptoError::Decrypt)));
    }

    #[test]
    fn tampering_is_detected() {
        let dek = random_bytes::<32>();
        let mut ct = seal(&dek, b"secret payload");
        let last = ct.len() - 1;
        ct[last] ^= 0x01;
        assert!(open(&dek, &ct).is_err());
    }

    #[test]
    fn truncation_is_detected() {
        let dek = random_bytes::<32>();
        let pt = vec![7u8; CHUNK_SIZE as usize * 2];
        let ct = seal(&dek, &pt);
        // Drop the final chunk: the previous chunk is not marked final, so the
        // AAD no longer matches and decryption must fail.
        let cut = HEADER_LEN + (CHUNK_SIZE as usize + TAG_LEN);
        assert!(open(&dek, &ct[..cut]).is_err());
    }

    #[test]
    fn engine_sealed_file_opens_with_the_dek() {
        // What the Python worker does: seal with a handed-over file key, then
        // the host opens it with the vault key it never shared.
        let dek = random_bytes::<32>();
        let file_id = random_bytes::<16>();
        let fk = derive_file_key(&dek, &file_id);
        let ct = seal_with_file_key(&fk, &file_id, b"png bytes here");
        assert_eq!(open(&dek, &ct).unwrap(), b"png bytes here");
    }

    #[test]
    fn dek_wrap_roundtrip() {
        let kek = random_bytes::<32>();
        let dek = random_bytes::<32>();
        let (nonce, ct) = wrap_dek(&kek, &dek);
        assert_eq!(unwrap_dek(&kek, &nonce, &ct).unwrap(), dek);
        let bad = random_bytes::<32>();
        assert!(unwrap_dek(&bad, &nonce, &ct).is_err());
    }
}
