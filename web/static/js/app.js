const mailForm = document.getElementById("mail-form");
const mailList = document.getElementById("mail-list");
const listTitle = document.getElementById("list-title");
const statusEl = document.getElementById("status");
const primaryCount = document.getElementById("primary-count");
const spamCount = document.getElementById("spam-count");
const seedBtn = document.getElementById("seed-btn");
const menuButtons = Array.from(document.querySelectorAll(".menu-btn"));
const cardTemplate = document.getElementById("mail-card-template");

let cache = { primary: [], spam: [] };
let activeTab = "primary";

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#bf2f21" : "#6b7280";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function renderList() {
  const mails = cache[activeTab] || [];
  listTitle.textContent = activeTab === "primary" ? "Primary Inbox" : "Spam Folder";
  mailList.innerHTML = "";

  if (mails.length === 0) {
    mailList.innerHTML = `<p class="status">No mails in ${activeTab}.</p>`;
    return;
  }

  mails.forEach((mail) => {
    const node = cardTemplate.content.cloneNode(true);
    node.querySelector(".sender").textContent = mail.sender;
    node.querySelector(".time").textContent = mail.received_at;
    node.querySelector(".subject").textContent = mail.subject;
    node.querySelector(".snippet").textContent = mail.body.slice(0, 170);

    const badge = node.querySelector(".badge");
    badge.textContent = mail.bucket === "spam" ? "Spam" : "Primary";
    badge.classList.add(mail.bucket);

    const score = node.querySelector(".score");
    score.textContent = `spam score: ${(mail.spam_probability * 100).toFixed(1)}%`;

    mailList.appendChild(node);
  });
}

async function refresh() {
  const data = await api("/api/emails");
  cache = data;
  primaryCount.textContent = String(cache.primary.length);
  spamCount.textContent = String(cache.spam.length);
  renderList();
}

menuButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    activeTab = btn.dataset.tab;
    menuButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    renderList();
  });
});

mailForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Analyzing mail...");

  const formData = new FormData(mailForm);
  const payload = {
    sender: String(formData.get("sender") || "").trim(),
    subject: String(formData.get("subject") || "").trim(),
    body: String(formData.get("body") || "").trim(),
  };

  try {
    await api("/api/emails", { method: "POST", body: JSON.stringify(payload) });
    mailForm.reset();
    await refresh();
    setStatus("Mail routed successfully.");
  } catch (error) {
    setStatus(error.message, true);
  }
});

seedBtn.addEventListener("click", async () => {
  setStatus("Loading demo mails...");
  try {
    const res = await api("/api/seed", { method: "POST", body: "{}" });
    await refresh();
    setStatus(`${res.added} demo mails added.`);
  } catch (error) {
    setStatus(error.message, true);
  }
});

refresh().catch((error) => setStatus(error.message, true));
