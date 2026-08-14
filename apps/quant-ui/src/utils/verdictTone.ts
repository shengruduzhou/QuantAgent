/**
 * Severity for backend verdict/status strings.
 *
 * Substring matching on these strings is negation-blind, and every instance of
 * it found so far painted a refusal as success:
 *
 *   "DO_NOT_ENABLE".includes("ENABLE")            === true
 *   "U0_BAR_NOT_READY_COVERAGE".includes("ready") === true   (lowercased)
 *
 * Both are real values the API returns. The first made the 做T workstation show
 * a green check on a hard research refusal while the *less* severe PAPER_ONLY
 * got the warning icon; the second would render every U0 NOT_READY state green.
 *
 * So: detect negation first, and never let an affirmative substring win once a
 * negation is present. Unknown input resolves to "unknown", never to success —
 * the backend deliberately models three states and the UI must not collapse
 * them at the last mile.
 */

export type Tone = "success" | "warning" | "danger" | "unknown";

/** Tokens that invert whatever affirmative word follows them. */
const NEGATIONS = [
  "do_not",
  "do not",
  "not_",
  "not ",
  "no_",
  "non_",
  "never",
  "disabled",
  "disable",
  "blocked",
  "block",
  "forbid",
  "denied",
  "deny",
  "reject",
  "refus",
  "unavailable",
  "missing",
  "insufficient",
];

const DANGER = ["error", "fail", "fatal", "critical", "violation", "breach"];
const WARNING = ["warn", "partial", "degraded", "stale", "paper_only", "paper only"];
const SUCCESS = ["ready", "normal", "success", "passed", "pass", "enable", "ok", "healthy"];
const UNKNOWN = ["unknown", "unverified", "not_evaluated", "unevaluated", "pending", "n/a"];

function contains(haystack: string, needles: string[]): boolean {
  return needles.some((n) => haystack.includes(n));
}

/**
 * Classify a backend verdict/status string.
 *
 * Order matters: explicit danger, then explicit unknown, then negation, and only
 * an un-negated affirmative earns "success".
 */
export function verdictTone(raw: string | null | undefined): Tone {
  if (raw === null || raw === undefined) return "unknown";
  const s = String(raw).trim().toLowerCase();
  if (s.length === 0) return "unknown";

  if (contains(s, DANGER)) return "danger";
  if (contains(s, UNKNOWN)) return "unknown";

  const negated = contains(s, NEGATIONS);

  // A negated affirmative ("DO_NOT_ENABLE", "NOT_READY") is a refusal, not a win.
  if (negated) {
    return contains(s, SUCCESS) || contains(s, WARNING) ? "danger" : "warning";
  }

  if (contains(s, WARNING)) return "warning";
  if (contains(s, SUCCESS)) return "success";
  return "unknown";
}

/** True only for a verdict that affirmatively permits the action. */
export function isAffirmative(raw: string | null | undefined): boolean {
  return verdictTone(raw) === "success";
}
