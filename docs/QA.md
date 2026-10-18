# Verification and deployment — 2026-10-18

## Automated checks

- API: 19 local tests passed with both PyTorch and ONNX selected, covering uploads, authentication, decision policy, disease catalog, field comparison, weather handling and private feedback regression. One upstream Starlette/httpx deprecation warning remains.
- ONNX conversion: all 260 held-out decisions match PyTorch; maximum absolute score difference 0.0000131. This is conversion parity, not field accuracy.
- Production smoke: ready endpoint, catalog, actual ONNX inference and input rejection passed without importing PyTorch. Local Windows process used approximately 95 MB RSS after inference; host use may differ.
- Web: TypeScript, Worker type checking, formatting and Vite build passed. Three Worker weather tests passed (source values, coordinate validation/no credentials, provider failure without fabricated forecasts).
- Public browser: 12 workflow groups passed through the real Cloudflare/Render deployment in an isolated headless session; no JavaScript runtime exceptions.
- All five pages fit desktop 1440px and mobile 390px/320px. The five-icon mobile navigation was checked after fixing a 1px overflow.
- Disease catalog loads four disease cards plus healthy-pattern information; Rust is explicitly unsupported and leaf-only RedRot is withheld. A simulated catalog HTTP 404 recovers using the retry button without resetting leaf screening.
- Actual leaf inference includes sourced general cause/care information. Field mode preserves leaf state, finds the synthetic color patch, resets old references and does not force markers onto a uniform image.
- Real Open-Meteo weather returns three days through the Cloudflare Worker. Simulated provider failure shows an error with no replacement forecast or disease probability.
- Invalid/oversized files, result history and JSON export were checked. Native file chooser clicks for leaf and field inputs were smoke-tested separately; the regression suite also assigns actual files to those inputs.

## Hosted application

- Web: https://sugarcane-care.khampheeraphop-thon.workers.dev
- Disease catalog: https://sugarcane-care.khampheeraphop-thon.workers.dev/diseases
- API: https://sugarcane-care-api.onrender.com

Cloudflare watches `main` in Sugarcane_web. Build checks run before deployment. Render Free watches `main` in Sugarcane_api and deploys after GitHub CI passes. A push-driven Cloudflare deployment and a Render Auto-Deploy were observed successfully. Code pushes do not train a new model. Render can sleep when idle, delaying the first scan by 50 seconds or longer.

The original `Not Found` screenshot was traced to an older local API process. Restarting the workspace API loaded the new route. A separate Worker redirect-mode incompatibility was corrected; the final public regression went through the actual proxy. Cloud weather is served directly at the edge, separately from the model host.

## Scope of evidence

Field test images are synthetic. The leaf fixture is a licensed held-out dataset sample. These tests verify behavior and deployment, not agricultural performance or accuracy in Thai farms. The model remains experimental and has no trained Rust class.

Private feedback, original downloaded photographs, PyTorch checkpoints, browser screenshots, secrets, local histories and environments are excluded from public Git. The compact ONNX inference weights/reference features are public with explicit user approval and retained source attribution. Application secrets are stored in Render and Cloudflare provider secret storage. Supabase is a future extension and has not been deployed.
