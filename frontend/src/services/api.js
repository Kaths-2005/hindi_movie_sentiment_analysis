const BASE = "/api";

async function handle(res) {
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "अज्ञात त्रुटि हुई।");
  return data;
}

export const api = {
  getDashboard: () => fetch(`${BASE}/dashboard`, { credentials: "include" }).then(handle),
  getMovies: () => fetch(`${BASE}/movies`, { credentials: "include" }).then(handle),
  getMovie: (name) => fetch(`${BASE}/movie/${encodeURIComponent(name)}`, { credentials: "include" }).then(handle),
  predict: (review) => fetch(`${BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ review }) }).then(handle),
  register: (username, password, confirm_password) => fetch(`${BASE}/auth/register`, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ username, password, confirm_password }) }).then(handle),
  login: (username, password) => fetch(`${BASE}/auth/login`, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ username, password }) }).then(handle),
  logout: () => fetch(`${BASE}/auth/logout`, { method: "POST", credentials: "include" }).then(handle),
  me: () => fetch(`${BASE}/auth/me`, { credentials: "include" }).then(handle),
  createReview: (movie_name, review) => fetch(`${BASE}/reviews`, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "include", body: JSON.stringify({ movie_name, review }) }).then(handle),
  myReviews: () => fetch(`${BASE}/reviews/my`, { credentials: "include" }).then(handle),
};
