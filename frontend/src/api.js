const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

async function request(path, options) {
  const response = await fetch(`${BASE}${path}`, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = body?.detail;
    let message = "La requête a échoué.";
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = "La saisie ne respecte pas le format attendu.";
    }
    throw new Error(message);
  }
  return body;
}

export function fetchMetrics() {
  return request("/metrics-summary");
}

export function predict(payload) {
  return request("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export { BASE };
