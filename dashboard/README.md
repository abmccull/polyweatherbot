# Station Sniper Dashboard

## Local run

```bash
npm install
API_BASE_URL=http://localhost:8000 \
API_BEARER_TOKEN=your-token \
npm run dev
```

## Vercel env vars

- `API_BASE_URL`: base URL of the server-side FastAPI (`https://your-server`).
- `API_BEARER_TOKEN`: same static bearer token configured on the API host.
