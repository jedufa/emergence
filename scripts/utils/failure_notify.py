"""scripts.utils.failure_notify

Failure notification utility for long-running Emergence jobs (nightly, farms, etc).

Goal:
- When something fails (nox session, registry module, CLI run), send an email
  containing:
    - the captured console output
    - a zip of the most relevant failed run folder (best-effort)

This is intentionally boring and dependency-free: stdlib only.

Configuration (environment variables):
  EMERGENCE_NOTIFY_EMAIL_TO        Required. Comma-separated recipients.
  EMERGENCE_NOTIFY_EMAIL_FROM      Optional. Defaults to EMERGENCE_NOTIFY_EMAIL_TO.
  EMERGENCE_NOTIFY_SMTP_HOST       Optional. If unset, we try local sendmail.
  EMERGENCE_NOTIFY_SMTP_PORT       Optional. Defaults to 587 (TLS) or 25 (no TLS).
  EMERGENCE_NOTIFY_SMTP_USER       Optional. If set, we will login.
  EMERGENCE_NOTIFY_SMTP_PASS       Optional. Password for login.
  EMERGENCE_NOTIFY_SMTP_TLS        Optional. "1" (default) enables STARTTLS.

Optional knobs:
  EMERGENCE_NOTIFY_SUBJECT_PREFIX  Optional. Prefix for the subject line.
  EMERGENCE_NOTIFY_MAX_ZIP_MB      Optional. Default 40. If the zip would exceed
                                   this limit, we include only the log.

Notes:
- If EMERGENCE_NOTIFY_EMAIL_TO is not set, this tool becomes a no-op (returns 0).
- We do NOT crash the main process if email delivery fails; we print a warning
  to stderr and continue.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import smtplib
import sys
import tempfile
import zipfile
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse common boolean environment variable values."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _max_zip_bytes() -> int:
    """Return max zip size in bytes."""
    mb = os.getenv("EMERGENCE_NOTIFY_MAX_ZIP_MB", "40").strip()
    try:
        return int(float(mb) * 1024 * 1024)
    except Exception:
        return 40 * 1024 * 1024


def _read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    """Read a text file with an upper bound (prevents sending 5GB logs)."""
    try:
        data = path.read_bytes()
    except Exception:
        return f"<could not read log: {path}>"
    if len(data) > max_bytes:
        data = data[-max_bytes:]
        prefix = "<log truncated: showing last bytes>\n\n"
    else:
        prefix = ""
    return prefix + data.decode("utf-8", errors="replace")


def _guess_run_dir_from_log(log_text: str) -> Optional[Path]:
    """Best-effort guess of the most relevant run folder from console output.

    Heuristic:
    - Find the LAST occurrence of a 'Metrics written to:' line.
    - Convert the metrics.csv path to the run directory path:
        .../runs/<run_id>/metrics.csv  ->  .../runs/<run_id>

    Returns:
        Path to run directory, or None if not found / not valid.
    """
    m = None
    for m in re.finditer(r"Metrics written to:\s*(.+?/runs/[^\s]+?/metrics\.csv)", log_text):
        pass
    if not m:
        return None

    metrics_path = Path(m.group(1)).expanduser()
    run_dir = metrics_path.parent
    if run_dir.exists() and run_dir.is_dir():
        return run_dir
    return None


def _zip_dir(src_dir: Path, zip_path: Path) -> Tuple[bool, int]:
    """Zip a directory, returning (ok, size_bytes)."""
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in src_dir.rglob("*"):
                if p.is_dir():
                    continue
                # Preserve relative structure inside the run directory.
                zf.write(p, arcname=str(p.relative_to(src_dir)))
        return True, zip_path.stat().st_size
    except Exception:
        return False, 0


def _send_email(
    *,
    subject: str,
    body: str,
    to_addrs: List[str],
    from_addr: str,
    attachments: List[Tuple[str, bytes, str]],
) -> None:
    """Send an email either via SMTP (preferred) or local sendmail fallback."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.set_content(body)

    for filename, data, mime in attachments:
        maintype, subtype = mime.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

    host = os.getenv("EMERGENCE_NOTIFY_SMTP_HOST", "").strip()
    if host:
        port_s = os.getenv("EMERGENCE_NOTIFY_SMTP_PORT", "").strip()
        use_tls = _env_bool("EMERGENCE_NOTIFY_SMTP_TLS", True)
        port = int(port_s) if port_s else (587 if use_tls else 25)

        user = os.getenv("EMERGENCE_NOTIFY_SMTP_USER", "").strip()
        pw = os.getenv("EMERGENCE_NOTIFY_SMTP_PASS", "").strip()

        with smtplib.SMTP(host, port, timeout=20) as s:
            s.ehlo()
            if use_tls:
                s.starttls()
                s.ehlo()
            if user:
                s.login(user, pw)
            s.send_message(msg)
        return

    # Fallback: local sendmail
    sendmail = shutil.which("sendmail")
    if not sendmail:
        raise RuntimeError(
            "No SMTP host configured (EMERGENCE_NOTIFY_SMTP_HOST unset) and 'sendmail' not found."
        )
    import subprocess  # stdlib

    p = subprocess.Popen([sendmail, "-t", "-i"], stdin=subprocess.PIPE)
    assert p.stdin is not None
    p.stdin.write(msg.as_bytes())
    p.stdin.close()
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"sendmail failed with exit code {rc}")


def notify_failure(
    *,
    title: str,
    log_path: Path,
    context: str,
    extra_attach_paths: Optional[List[Path]] = None,
) -> int:
    """Send an email about a failure. Returns 0 on success, 0 if disabled, 1 on notify error."""
    to_raw = os.getenv("EMERGENCE_NOTIFY_EMAIL_TO", "").strip()
    if not to_raw:
        return 0  # disabled by config

    to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]
    from_addr = os.getenv("EMERGENCE_NOTIFY_EMAIL_FROM", "").strip() or to_addrs[0]
    prefix = os.getenv("EMERGENCE_NOTIFY_SUBJECT_PREFIX", "[EMERGENCE]").strip()

    log_text = _read_text(log_path)
    run_dir = _guess_run_dir_from_log(log_text)

    body_lines = [
        f"Context: {context}",
        f"Log: {log_path}",
    ]
    if run_dir:
        body_lines.append(f"Guessed failed run dir: {run_dir}")
    else:
        body_lines.append("Guessed failed run dir: <not found in log>")
    body_lines.append("")
    body_lines.append("--- console (tail) ---")
    body_lines.append(log_text[-12000:])  # keep email body sane

    attachments: List[Tuple[str, bytes, str]] = []
    attachments.append(("console.log", log_text.encode("utf-8", errors="replace"), "text/plain"))

    # Zip the run dir (best effort, size limited).
    max_zip = _max_zip_bytes()
    tmpdir = Path(tempfile.mkdtemp(prefix="emergence_fail_notify_"))
    try:
        if run_dir:
            zip_path = tmpdir / "failed_run.zip"
            ok, size = _zip_dir(run_dir, zip_path)
            if ok and size <= max_zip:
                attachments.append((zip_path.name, zip_path.read_bytes(), "application/zip"))
            elif ok:
                body_lines.insert(2, f"NOTE: run zip omitted (size {size/1024/1024:.1f}MB > limit {max_zip/1024/1024:.1f}MB)")
        # Any extra attachments explicitly requested.
        if extra_attach_paths:
            for p in extra_attach_paths:
                try:
                    if p.exists() and p.is_file():
                        attachments.append((p.name, p.read_bytes(), "application/octet-stream"))
                except Exception:
                    continue

        subject = f"{prefix} FAIL: {title}"
        _send_email(subject=subject, body="\n".join(body_lines), to_addrs=to_addrs, from_addr=from_addr, attachments=attachments)
        return 0
    except Exception as e:
        print(f"[failure_notify] WARNING: failed to send email: {e}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI args."""
    ap = argparse.ArgumentParser(description="Send Emergence failure notification email.")
    ap.add_argument("--title", required=True, help="Short title (e.g. nightly / module name)")
    ap.add_argument("--context", required=True, help="Context string (host, tag, etc)")
    ap.add_argument("--log", required=True, help="Path to captured console log")
    ap.add_argument("--attach", action="append", default=[], help="Extra file to attach (repeatable)")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    log_path = Path(args.log).expanduser().resolve()
    extra = [Path(p).expanduser().resolve() for p in args.attach]
    return int(notify_failure(title=args.title, log_path=log_path, context=args.context, extra_attach_paths=extra))


if __name__ == "__main__":
    raise SystemExit(main())
