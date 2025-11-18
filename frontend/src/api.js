export async function searchApi(path) {
  // First try environment variable, otherwise fallback to localhost
  const base =
    process.env.REACT_APP_API_BASE_URL || "http://localhost:5000";

  const res = await fetch(base + path);

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "API error");
  }

  return res.json();
}
