use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    Msg(String),
    #[error("setup incomplete: {0}")]
    SetupIncomplete(String),
    #[error("engine not running")]
    EngineDown,
    #[error("engine error: {0}")]
    Engine(String),
    #[error("the vault is locked")]
    VaultLocked,
    #[error("{0}")]
    Vault(#[from] crate::vault::VaultError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl AppError {
    pub fn msg(s: impl Into<String>) -> Self {
        AppError::Msg(s.into())
    }
}

/// Tauri commands must return something serializable; keep a stable shape so
/// the frontend can branch on `kind` rather than string-matching messages.
impl Serialize for AppError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let kind = match self {
            AppError::SetupIncomplete(_) => "setup_incomplete",
            AppError::EngineDown => "engine_down",
            AppError::Engine(_) => "engine",
            AppError::VaultLocked => "vault_locked",
            AppError::Vault(crate::vault::VaultError::Locked) => "vault_locked",
            AppError::Vault(crate::vault::VaultError::Absent) => "vault_absent",
            AppError::Vault(crate::vault::VaultError::BadPassphrase) => "bad_passphrase",
            AppError::Vault(crate::vault::VaultError::NoBiometricKey) => "no_biometric_key",
            AppError::Vault(_) => "vault",
            AppError::Io(_) => "io",
            AppError::Http(_) => "http",
            AppError::Json(_) => "json",
            AppError::Msg(_) => "error",
        };
        let mut st = s.serialize_struct("AppError", 2)?;
        st.serialize_field("kind", kind)?;
        st.serialize_field("message", &self.to_string())?;
        st.end()
    }
}

pub type Result<T> = std::result::Result<T, AppError>;
