# Working on Roman's Mac (shared hackathon machine)

Each teammate has their **own standard macOS account** on Roman's Mac. You SSH in with your own key and run
Claude Code there under **your own claude.ai login**. Nobody shares a password, an API key, a Claude login or a
private key. Your agent should read this file and `AGENTS.md`. All the `AGENTS.md` rules still apply.

**Not in this repo (it's public):** your username and the Mac's address. Roman sends you both privately.
Below they're written `<user>` and `<mac-address>`.

## 1. On your laptop (once)

1. **SSH key.** If you made one for the GPU (`docs/COMPUTE.md`), reuse it. Otherwise:
   `ssh-keygen -t ed25519 -C "<name>-flyhack"`. Send Roman the one line from `~/.ssh/id_ed25519.pub`
   and say it's for the Mac. Never send the file without `.pub`.
2. **Reaching the Mac.** Roman sends two addresses: a `.local` name for when you're on the same Wi-Fi, and a
   Tailscale `100.x.x.x` address for everywhere else. Venue Wi-Fi often blocks device-to-device traffic, so
   if the `.local` name times out, install Tailscale, accept the share invite Roman sends you, and use the
   `100.x` address.
3. **SSH shortcut.** Add this to `~/.ssh/config`:
   ```
   Host flymac
     HostName <mac-address>
     User <user>
     IdentityFile ~/.ssh/id_ed25519
     ForwardAgent yes
     ServerAliveInterval 30
   ```
   Then `ssh flymac` should log you in without a password. `ForwardAgent` lets you push to GitHub from the
   Mac with your laptop's key. Run `ssh-add ~/.ssh/<your github key>` on the laptop first.

## 2. On the Mac, in your account (once, about 10 minutes)

Roman's setup already made `~/flyhack` (a clone of this repo) and a GPU key at `~/.ssh/id_ed25519`, and
added that key to the GPU pod.

```bash
ssh flymac
curl -fsSL https://claude.ai/install.sh | bash          # Claude Code, installed only for you
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
claude                                                  # then type /login and use YOUR claude.ai account
```

`/login` prints a link. Open it in your laptop's browser, sign in, and paste the code back. Your login is
saved inside your account only.

```bash
cd ~/flyhack
/opt/homebrew/bin/python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py && python -m core.graph --size all && python -m core.check --all
python scripts/remote.py setup <gpu-ip> <gpu-port> --name <user>     # GPU address is in AGENTS.md
```

To push to GitHub, connect with `ssh flymac` (agent forwarding is on), then run
`git remote set-url origin git@github.com:romanobro56/phyllis-inglewood.git`. You need collaborator access
on the repo. Ask Roman if `git push` is refused.

## 3. Three ways to work

- **Terminal:** `ssh flymac`, then `tmux new -A -s claude`, then `claude`. tmux keeps your session running
  if Wi-Fi drops. Reconnect with the same command.
- **Claude desktop app:** in the Code tab, add an SSH connection to `flymac` (or `<user>@<mac-address>`).
  Claude Code then runs on the Mac in your account.
- **Remote Control (phone or claude.ai/code):** in a tmux session on the Mac, run `claude remote-control`
  and open the session from your own Claude app. It only ever shows up in your account. On a Team or
  Enterprise plan, an org Owner has to turn Remote Control on first.

## 4. Rules for agents on the shared Mac

1. **Stay in your own account.** Work in `~/flyhack`. Never use `sudo`. Never read or change other users'
   home folders, system settings, or other people's processes.
2. **Edit only `flies/<your_fly>/`**, exactly as `AGENTS.md` says. Commit on your own branch.
3. **Your own port.** Several people share this machine, so never use the default port 8000:
   ```bash
   PORT=$((8000 + $(id -u) % 1000))
   python -m server.serve --only <fly> --port $PORT        # stays bound to 127.0.0.1
   ```
   To view it from your laptop, run `ssh -L $PORT:localhost:$PORT flymac` and open
   `http://localhost:$PORT/web/`. Use the actual number in place of `$PORT`. Run `echo $PORT` on the Mac
   to see it.
4. **The CPU is shared, and it's a laptop.** Train on the GPU (`scripts/remote.py submit`). Local training
   is only for quick smoke tests (`--size tiny --minutes 3`), one at a time. Don't leave long CPU jobs or
   servers running when you're done.
5. **No secrets in the repo or in chat.** Never print, copy or commit private keys, `.credentials.json`, or
   tokens.

## 5. Privacy, and when you're done

Roman is the Mac's administrator, so technically he can access anything in your account, including your
saved Claude login. When the hackathon ends, type `/logout` in `claude` (or delete `~/.claude`). Roman
deletes the accounts afterwards.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied (publickey)` | Roman hasn't added your key yet, or SSH is offering a different key. Check `IdentityFile`. |
| `Operation timed out` / `Connection refused` | The Mac is asleep, Remote Login is off, or you're on a different network. Try Tailscale. |
| `claude: command not found` | Run the PATH line in step 2 again. |
| `could not connect` from `remote.py setup` | Your Mac-account GPU key isn't on the pod. Ask Roman to run `remote.py add-key` with `~/.ssh/id_ed25519.pub`. |
| `Address already in use` | Someone else has that port. Use your own `$PORT` (rule 3). |
