# MEMORY.md - Long-Term Memory

## Email Sending (Gmail via App Password)
- Method: Python smtplib or Himalaya CLI
- App Password stored in `~/.secrets/mail/gmail.pass` (chmod 600)
- Python example: use smtplib with STARTTLS on port 587
- Himalaya config: `~/.config/himalaya/config.toml` with `auth.cmd = "cat ~/.secrets/mail/gmail.pass"`
- Verified: Test emails sent successfully (see memory/2026-03-27.md)
