# QuantoniumOS Mobile App

> React Native / Expo mobile client for QuantoniumOS concepts.

**Parent repo:** [quantoniumos](https://github.com/mandcony/quantoniumos)

---

## Quick Start

```bash
# Install dependencies
npm install

# Start Expo dev server
npx expo start
```

Use the Expo Go app on iOS/Android to scan the QR code.

---

## Neural Brain Integration

1. **Build Neural Brain Packages**
   ```bash
   python -m quantonium_os_src.ai.neural_brain --domains physics mathematics medicine
   ```
2. **Start the Python API server**
   ```bash
   python -m quantonium_os_src.ai.neural_brain_server --port 8765
   ```
3. **Expose the server to Expo**
   ```bash
   export EXPO_PUBLIC_NEURAL_BRAIN_URL="http://192.168.1.23:8765"
   export EXPO_PUBLIC_NEURAL_BRAIN_TOP_K=3
   npx expo start
   ```
   Replace the host IP with the machine running the Python server.
4. Launch the *AI Chat* screen — use the domain chips to route questions to the corresponding vertex-compressed neural brain matrices.

The client will show errors in-chat if the server is unreachable or no neural brain package exists for the requested domain.

---

## Quick Automation (Windows PowerShell)

```powershell
tools/mobile/run_neural_brain_expo.ps1 `
  -ServerHost 0.0.0.0 `
  -Port 8765 `
  -ClientHost 192.168.0.42 `
  -Domains "physics,mathematics,medicine"
```

The script will (optionally) rebuild neural brain packages, start the Python server, export the Expo environment variables, and then run `npx expo start`. Change `ClientHost` to the LAN IP your device should target (use `127.0.0.1` for iOS Simulator, `10.0.2.2` for Android emulator). Pass `-SkipBuild` if you only need to restart the server.

---

## Project Structure

```
quantonium-mobile/
├── App.tsx          # Root component
├── index.ts         # Entry point
├── src/             # App source (screens, components, hooks)
├── assets/          # Images, fonts
├── app.json         # Expo config
├── package.json     # Dependencies
└── tsconfig.json    # TypeScript config
```

---

## See Also

- [REPO_ORGANIZATION.md](../REPO_ORGANIZATION.md) — repo structure map
- [quantonium_os_src/](../quantonium_os_src/) — desktop OS apps
- [ui/](../ui/) — shared UI assets
