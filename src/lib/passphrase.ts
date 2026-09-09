export const MIN_PASSPHRASE = 10;

/**
 * A deliberately rough strength hint. It rewards length and variety, which is
 * what actually matters against offline guessing once Argon2id has done its
 * part -- it is not a substitute for a real estimator.
 */
export function strength(pass: string): { score: number; label: string } {
  if (!pass) return { score: 0, label: "" };

  const classes =
    Number(/[a-z]/.test(pass)) + Number(/[A-Z]/.test(pass)) +
    Number(/[0-9]/.test(pass)) + Number(/[^A-Za-z0-9]/.test(pass));
  const words = pass.trim().split(/\s+/).filter(Boolean).length;

  let score = 0;
  if (pass.length >= MIN_PASSPHRASE) score += 1;
  if (pass.length >= 16 || words >= 4) score += 1;
  if (pass.length >= 24 || words >= 5) score += 1;
  if (classes >= 3 || words >= 4) score += 1;

  const labels = [
    "Too short to protect much.",
    "Weak — add length rather than symbols.",
    "Reasonable. Longer is better than more exotic.",
    "Strong.",
    "Very strong.",
  ];
  return { score, label: labels[Math.min(score, 4)] };
}
