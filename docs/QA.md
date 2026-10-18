# Local verification — 2026-10-18

- API: 18 tests passed, covering invalid uploads, actual image formats, authentication, shared image checks, pure decision policy, field references, weather response handling and private local model regression. One upstream Starlette/httpx deprecation warning remains.
- Clean API source snapshot without local models/data: 14 tests passed, 4 local-artifact tests skipped as intended.
- Frontend: TypeScript, optional Worker type check and Vite production build passed; formatting checks passed.
- Browser: 10 workflow groups passed in an isolated headless Playwright CLI session using the actual local API; no JavaScript runtime exceptions.
- Desktop 1440px and mobile 390px/320px checked; all four routes fit without horizontal overflow.
- Native leaf and field file input clicks each opened a browser file chooser and accepted the supplied fixture through that chooser.
- Leaf image/result persists through field mode; changing the field image clears references and prior markers. Uniform-color input gets no forced markers.
- Weather success used a real three-day response for Bangkok test coordinates. Simulated provider failure showed an error with no invented forecast.
- Uploaded invalid and oversized images show readable errors. Selected leaf preview has no filename overlay.
- Current policy withholds the previously reported private misclassification. It does not claim to diagnose the user's out-of-scope condition.

Field fixtures are synthetic color patches. The leaf fixture is a licensed held-out dataset sample. Neither validates agricultural performance. Browser screenshots, private feedback, downloaded datasets, model weights, local histories and environments are excluded from the public repositories.

Automated GitHub workflow definitions are included. Their remote run status is separate from these local results. Optional Cloudflare/Supabase adapters have not been deployed.
