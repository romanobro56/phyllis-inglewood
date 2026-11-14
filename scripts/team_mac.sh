#!/bin/bash
# Owner only (Roman). Gives each teammate their own standard macOS account on this Mac, reachable by SSH
# with their own key, so everyone runs Claude Code under their own claude.ai login. No shared passwords,
# API keys or private keys. Teammates read docs/SHARED_MAC.md.
#
#   sudo bash scripts/team_mac.sh lockdown                       # once: make your home folder private
#   sudo bash scripts/team_mac.sh add <user> "ssh-ed25519 AAAA... name@laptop"
#   sudo bash scripts/team_mac.sh status
#   sudo bash scripts/team_mac.sh remove <user>                  # after the hackathon (deletes their home)
#
# Before the first `add`: System Settings > General > Sharing > Remote Login: on,
# "Allow access for: Only these users". `add` puts each teammate on that list.
#
# `add` is safe to re-run: it creates the account if needed, installs (or adds) the login key, makes a GPU
# key inside their account, adds that key to the GPU pod (if your .remote.json is set up) and clones the repo
# to ~/flyhack. The account gets a long random password that is never shown: SSH keys are the only way in.
set -euo pipefail

REPO_URL="https://github.com/romanobro56/phyllis-inglewood.git"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OWNER="${SUDO_USER:-}"

die() { echo "error: $*" >&2; exit 1; }
cd /   # commands run as a teammate can't read your (private) home, so never start them from inside it
[ "$(id -u)" -eq 0 ] || die "run with sudo: sudo bash scripts/team_mac.sh $*"
[ -n "$OWNER" ] && [ "$OWNER" != root ] || die "run it from your own account with sudo, not as root"

user_exists() { dscl . -read "/Users/$1" >/dev/null 2>&1; }

check_name() {
  [[ "$1" =~ ^[a-z][a-z0-9_]{1,15}$ ]] || die "username must be lowercase letters/digits, e.g. mitchell"
  [ "$1" != "$OWNER" ] || die "that's your own account"
}

cmd_lockdown() {
  # New standard users are in the 'staff' group, and macOS home folders are group-readable by staff,
  # so without this a teammate could browse your home folder (dotfiles, ~/.claude history, projects).
  chmod 700 "/Users/$OWNER"
  echo "/Users/$OWNER is now private ($(stat -f %Sp "/Users/$OWNER"))."
}

cmd_add() {
  local u="${1:-}" key="${2:-}"
  check_name "$u"
  [ -n "$key" ] || die "usage: sudo bash scripts/team_mac.sh add <user> \"ssh-ed25519 AAAA... name@laptop\""
  # Repair keys copied out of email/chat: drop wrap markers (│) and rejoin a base64 body that got
  # split across lines, e.g. "ssh-ed25519 AAAA...Jfg6\n │ ce7I... ali-flyhack".
  [[ "$key" == *"ssh-ed25519 "* ]] && key="ssh-ed25519 ${key#*ssh-ed25519 }"   # drop a pasted "cat ..." line
  key="${key//│/ }"
  local -a t; read -ra t <<< "$(echo $key)"
  if [ "${t[0]:-}" = "ssh-ed25519" ]; then
    local body="${t[1]:-}" i=2
    while [ ${#body} -lt 68 ] && [ $i -lt ${#t[@]} ] && [[ "${t[$i]}" =~ ^[A-Za-z0-9+/=]+$ ]]; do
      body="$body${t[$i]}"; i=$((i + 1))
    done
    key="ssh-ed25519 $body ${t[*]:$i}"; key="${key% }"
  fi
  [[ "$key" != *$'\n'* ]] || die "paste exactly one key line"
  local re='^(ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp[0-9]+|sk-ssh-ed25519@openssh\.com) '
  [[ "$key" =~ $re ]] || die "that doesn't look like a public key (it should start with ssh-ed25519). Never accept a private key."
  local fp; fp="$(echo "$key" | ssh-keygen -l -f - 2>/dev/null)" || die "ssh-keygen can't read that public key"
  echo "key for $u: $fp"
  echo "  (the teammate can confirm it matches: ssh-keygen -lf ~/.ssh/id_ed25519.pub)"

  if user_exists "$u"; then
    echo "account $u already exists"
  else
    local pw; pw="$(openssl rand -base64 30)"
    sysadminctl -addUser "$u" -fullName "$u (flyhack)" -password "$pw" -home "/Users/$u" -shell /bin/zsh
    unset pw
    createhomedir -c -u "$u" >/dev/null
    echo "created standard account $u"
  fi
  dseditgroup -o checkmember -m "$u" admin >/dev/null 2>&1 && die "$u is an admin; teammates must be standard users"
  local home="/Users/$u"
  chmod 700 "$home"

  # Login key (from their laptop).
  install -d -m 700 -o "$u" -g staff "$home/.ssh"
  touch "$home/.ssh/authorized_keys"
  grep -qxF "$key" "$home/.ssh/authorized_keys" || echo "$key" >> "$home/.ssh/authorized_keys"
  chown "$u:staff" "$home/.ssh/authorized_keys"; chmod 600 "$home/.ssh/authorized_keys"

  # Allow SSH login (Remote Login's "Only these users" list).
  if dscl . -read /Groups/com.apple.access_ssh >/dev/null 2>&1; then
    dseditgroup -o edit -a "$u" -t user com.apple.access_ssh
  else
    echo "note: Remote Login is set to 'All users' (or off). Set it to 'Only these users' and re-run add."
  fi

  # GPU key, made and kept inside their account (the private half never leaves it).
  if [ ! -f "$home/.ssh/id_ed25519" ]; then
    sudo -u "$u" -H ssh-keygen -q -t ed25519 -N "" -C "$u-flyhack-mac" -f "$home/.ssh/id_ed25519"
  fi
  local gpub; gpub="$(cat "$home/.ssh/id_ed25519.pub")"
  if [ -f "$ROOT/.remote.json" ]; then
    local py="$ROOT/.venv/bin/python"; [ -x "$py" ] || py=python3
    sudo -u "$OWNER" -H "$py" "$ROOT/scripts/remote.py" add-key "$gpub" \
      || echo "couldn't reach the pod; later run: python scripts/remote.py add-key \"$gpub\""
  else
    echo "no .remote.json here; give the pod this key: python scripts/remote.py add-key \"$gpub\""
  fi

  # Their own clone (public repo, no credentials needed to clone).
  if [ ! -d "$home/flyhack/.git" ]; then
    sudo -u "$u" -H git clone -q "$REPO_URL" "$home/flyhack" || die "git clone failed for $u"
    sudo -u "$u" -H git -C "$home/flyhack" checkout -q integrate-flies 2>/dev/null || true   # until main has it
    echo "cloned repo to $home/flyhack"
  fi

  echo "done: $u can now run  ssh $u@$(scutil --get LocalHostName).local"
}

cmd_status() {
  echo "Remote Login (port 22): $(nc -z -G 2 localhost 22 >/dev/null 2>&1 && echo on || echo OFF)"
  echo "your home: $(stat -f %Sp "/Users/$OWNER")  (should be drwx------; run lockdown if not)"
  for h in /Users/*; do
    local u; u="$(basename "$h")"
    [ -f "$h/.ssh/authorized_keys" ] && [ "$u" != "$OWNER" ] || continue
    local ssh_ok="no"; dseditgroup -o checkmember -m "$u" com.apple.access_ssh >/dev/null 2>&1 && ssh_ok="yes"
    echo "  $u: keys=$(grep -c . "$h/.ssh/authorized_keys") ssh-allowed=$ssh_ok repo=$([ -d "$h/flyhack/.git" ] && echo yes || echo no)"
  done
}

cmd_remove() {
  local u="${1:-}"; check_name "$u"
  user_exists "$u" || die "no account $u"
  dseditgroup -o edit -d "$u" -t user com.apple.access_ssh 2>/dev/null || true
  pkill -u "$u" 2>/dev/null || true
  sysadminctl -deleteUser "$u"
  echo "deleted $u and /Users/$u. Remove its GPU key on the pod too (~/.ssh/authorized_keys, comment $u-flyhack-mac)."
}

case "${1:-}" in
  lockdown) cmd_lockdown ;;
  add) shift; cmd_add "$@" ;;
  status) cmd_status ;;
  remove) shift; cmd_remove "$@" ;;
  *) sed -n 2,16p "$0"; exit 1 ;;
esac
