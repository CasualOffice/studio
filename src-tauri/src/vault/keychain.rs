//! macOS Keychain storage for the biometric key-encryption key.
//!
//! The KEK is a generic-password item guarded by a `SecAccessControl` with
//! `kSecAccessControlUserPresence`, so reading it makes the system present
//! Touch ID (falling back to the login password). Accessibility is
//! `WhenUnlockedThisDeviceOnly`, which keeps the item off iCloud Keychain and
//! out of backups -- it never leaves this Mac.
//!
//! `security-framework`'s safe `ItemAddOptions` cannot attach an access
//! control, so the add/copy/delete calls go through `SecItem*` directly.

#![cfg(target_os = "macos")]

use core_foundation::base::{CFType, TCFType};
use core_foundation::boolean::CFBoolean;
use core_foundation::data::CFData;
use core_foundation::dictionary::CFDictionary;
use core_foundation::string::{CFString, CFStringRef};
use core_foundation_sys::base::CFRelease;
use security_framework::access_control::{ProtectionMode, SecAccessControl};
use security_framework_sys::access_control::kSecAccessControlUserPresence;
use security_framework_sys::base::{errSecItemNotFound, errSecSuccess};
use security_framework_sys::item::{
    kSecAttrAccessControl, kSecAttrAccount, kSecAttrService, kSecClass, kSecClassGenericPassword,
    kSecMatchLimit, kSecReturnData, kSecValueData,
};
use security_framework_sys::keychain_item::{SecItemAdd, SecItemCopyMatching, SecItemDelete};
use zeroize::Zeroize;

// Exported by Security.framework but absent from security-framework-sys.
#[link(name = "Security", kind = "framework")]
extern "C" {
    static kSecMatchLimitOne: CFStringRef;
    static kSecUseOperationPrompt: CFStringRef;
}

/// `errSecUserCanceled`, also missing from the sys crate. A cancelled Touch ID
/// prompt must be distinguishable from a real failure so the UI can offer the
/// passphrase instead of showing an error.
const ERR_SEC_USER_CANCELED: i32 = -128;

const SERVICE: &str = "com.melp.modelstudio.vault";
const ACCOUNT: &str = "vault-kek";
const PROBE_ACCOUNT: &str = "capability-probe";

#[derive(Debug, thiserror::Error)]
pub enum KeychainError {
    #[error("no key is stored in the Keychain for this vault")]
    NotFound,
    #[error("authentication was cancelled")]
    Cancelled,
    #[error("could not build the access control policy: {0}")]
    AccessControl(String),
    #[error("Keychain call failed with status {0}")]
    Status(i32),
    #[error("the stored key has the wrong length")]
    BadLength,
}

fn cfstr(r: CFStringRef) -> CFString {
    unsafe { CFString::wrap_under_get_rule(r) }
}

/// `userPresence` = Touch ID with a password fallback, which is what a desktop
/// app wants: a Mac without a sensor, or a wet finger, must still be usable.
fn access_control() -> Result<SecAccessControl, KeychainError> {
    SecAccessControl::create_with_protection(
        Some(ProtectionMode::AccessibleWhenUnlockedThisDeviceOnly),
        kSecAccessControlUserPresence,
    )
    .map_err(|e| KeychainError::AccessControl(e.to_string()))
}

fn base_query() -> Vec<(CFString, CFType)> {
    unsafe {
        vec![
            (
                cfstr(kSecClass),
                cfstr(kSecClassGenericPassword).as_CFType(),
            ),
            (cfstr(kSecAttrService), CFString::new(SERVICE).as_CFType()),
            (cfstr(kSecAttrAccount), CFString::new(ACCOUNT).as_CFType()),
        ]
    }
}

fn status_to_err(status: i32) -> KeychainError {
    if status == errSecItemNotFound {
        KeychainError::NotFound
    } else if status == ERR_SEC_USER_CANCELED {
        KeychainError::Cancelled
    } else {
        KeychainError::Status(status)
    }
}

/// Store the KEK, replacing any existing one.
pub fn store_kek(kek: &[u8; 32]) -> Result<(), KeychainError> {
    // SecItemAdd refuses duplicates; removing first makes this idempotent.
    let _ = delete_kek();

    let acl = access_control()?;
    let mut pairs = base_query();
    unsafe {
        pairs.push((cfstr(kSecValueData), CFData::from_buffer(kek).as_CFType()));
        pairs.push((cfstr(kSecAttrAccessControl), acl.as_CFType()));
    }
    let dict = CFDictionary::from_CFType_pairs(&pairs);

    let status = unsafe { SecItemAdd(dict.as_concrete_TypeRef(), std::ptr::null_mut()) };
    if status != errSecSuccess {
        return Err(status_to_err(status));
    }
    Ok(())
}

/// Read the KEK. This is the call that prompts for Touch ID.
pub fn load_kek(prompt: &str) -> Result<[u8; 32], KeychainError> {
    let mut pairs = base_query();
    unsafe {
        pairs.push((cfstr(kSecReturnData), CFBoolean::true_value().as_CFType()));
        pairs.push((cfstr(kSecMatchLimit), cfstr(kSecMatchLimitOne).as_CFType()));
        pairs.push((
            cfstr(kSecUseOperationPrompt),
            CFString::new(prompt).as_CFType(),
        ));
    }
    let dict = CFDictionary::from_CFType_pairs(&pairs);

    let mut result: core_foundation::base::CFTypeRef = std::ptr::null();
    let status = unsafe { SecItemCopyMatching(dict.as_concrete_TypeRef(), &mut result) };
    if status != errSecSuccess {
        return Err(status_to_err(status));
    }
    if result.is_null() {
        return Err(KeychainError::NotFound);
    }

    let data = unsafe { CFData::wrap_under_create_rule(result as _) };
    let bytes = data.bytes();
    if bytes.len() != 32 {
        return Err(KeychainError::BadLength);
    }
    let mut kek = [0u8; 32];
    kek.copy_from_slice(bytes);
    Ok(kek)
}

pub fn delete_kek() -> Result<(), KeychainError> {
    let dict = CFDictionary::from_CFType_pairs(&base_query());
    let status = unsafe { SecItemDelete(dict.as_concrete_TypeRef()) };
    if status != errSecSuccess && status != errSecItemNotFound {
        return Err(status_to_err(status));
    }
    Ok(())
}

/// Whether an item exists, without prompting for authentication.
/// Asking only for attributes (not data) does not trigger the ACL.
pub fn kek_present() -> bool {
    let mut pairs = base_query();
    unsafe {
        pairs.push((cfstr(kSecMatchLimit), cfstr(kSecMatchLimitOne).as_CFType()));
    }
    let dict = CFDictionary::from_CFType_pairs(&pairs);
    let mut result: core_foundation::base::CFTypeRef = std::ptr::null();
    let status = unsafe { SecItemCopyMatching(dict.as_concrete_TypeRef(), &mut result) };
    if !result.is_null() {
        unsafe { CFRelease(result as _) };
    }
    status == errSecSuccess
}

/// `errSecMissingEntitlement`. The data-protection Keychain -- the only one
/// that supports biometric access control -- refuses binaries without a
/// team-identified signature.
const ERR_SEC_MISSING_ENTITLEMENT: i32 = -34018;

/// Whether this build can actually use the Keychain for biometric unlock.
///
/// A locally built, unsigned or ad-hoc-signed app cannot: storing an item with
/// a `SecAccessControl` returns `errSecMissingEntitlement`, and adding a
/// `keychain-access-groups` entitlement without a real team identifier gets the
/// process killed outright. Rather than offer a Touch ID switch that always
/// fails, probe once with a throwaway item and report honestly.
///
/// The probe never prompts: writing and deleting do not evaluate the ACL.
pub fn biometry_usable() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();

    *CACHED.get_or_init(|| {
        let Ok(acl) = access_control() else {
            return false;
        };
        let probe = crate::vault::crypto::random_bytes::<32>();

        let mut pairs = unsafe {
            vec![
                (
                    cfstr(kSecClass),
                    cfstr(kSecClassGenericPassword).as_CFType(),
                ),
                (cfstr(kSecAttrService), CFString::new(SERVICE).as_CFType()),
                (
                    cfstr(kSecAttrAccount),
                    CFString::new(PROBE_ACCOUNT).as_CFType(),
                ),
                (
                    cfstr(kSecValueData),
                    CFData::from_buffer(&probe).as_CFType(),
                ),
                (cfstr(kSecAttrAccessControl), acl.as_CFType()),
            ]
        };
        let dict = CFDictionary::from_CFType_pairs(&pairs);
        let status = unsafe { SecItemAdd(dict.as_concrete_TypeRef(), std::ptr::null_mut()) };

        // Clean up whether or not it worked.
        pairs.truncate(3);
        let del = CFDictionary::from_CFType_pairs(&pairs);
        unsafe { SecItemDelete(del.as_concrete_TypeRef()) };

        if status == ERR_SEC_MISSING_ENTITLEMENT {
            log_unavailable();
        }
        status == errSecSuccess
    })
}

fn log_unavailable() {
    eprintln!(
        "keychain: biometric unlock unavailable (errSecMissingEntitlement). \
         The data-protection Keychain requires a signature with a team identifier; \
         this build is unsigned or ad-hoc signed. Passphrase unlock is unaffected."
    );
}

/// Generate, store and return a fresh KEK.
pub fn provision_kek() -> Result<[u8; 32], KeychainError> {
    let kek = super::crypto::random_bytes::<32>();
    match store_kek(&kek) {
        Ok(()) => Ok(kek),
        Err(e) => {
            let mut k = kek;
            k.zeroize();
            Err(e)
        }
    }
}
