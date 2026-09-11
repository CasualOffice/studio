# Security

Model Studio is built to keep what it produces on the machine that produced
it. This describes what that means in practice, and what it does not cover.

## What is protected

Everything the app generates or imports is sealed before it reaches disk,
with ChaCha20-Poly1305 and a per-file subkey derived by HKDF. The index of
what exists is encrypted too, so filenames and prompts are not readable from
the filesystem. The key that wraps the rest is derived from your passphrase
with Argon2id, or held in the macOS Keychain behind a biometric access
control when the build is signed with a team identifier.

Plaintext exists only in memory, and only while a model needs it. Source
images are decrypted into a private staging directory for as long as a job
runs and removed afterwards, including on failure.

Nothing is sent anywhere. There is no account, no API key, no telemetry, and
no cloud inference. The only network traffic is model downloads, which go to
Hugging Face and are initiated by you.

## What is not protected

- **An unlocked app on an unattended machine.** The vault auto-locks after
  fifteen minutes of inactivity; before that, anything open is readable.
- **Your passphrase.** There is no recovery. A forgotten passphrase means the
  vault cannot be opened, by us or by anyone.
- **A compromised machine.** Malware running as your user can read the keys
  from memory while the vault is unlocked. This is a privacy tool, not a
  defence against local code execution.
- **The models themselves.** Weights are downloaded from Hugging Face and are
  whatever their publishers made them. Nothing here inspects them.

## Reporting something

Open a GitHub issue for anything that is not itself sensitive. For a problem
that would put users at risk if described publicly, use GitHub's private
vulnerability reporting on this repository instead.

Please include the macOS version, the Mac's chip and memory, and the steps
that reproduce it. A crash log from `~/Library/Logs/DiagnosticReports` helps
if there is one.
