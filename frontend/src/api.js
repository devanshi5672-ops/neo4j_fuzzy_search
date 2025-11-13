export async function searchApi(path) {
  // backend assumed to run on http://localhost:5000
  const base = "http://localhost:5000";
  const res = await fetch(base + path);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(txt || "API error");
  }
  return res.json();
}
