document.addEventListener("DOMContentLoaded", () => {
    const seedBtn = document.getElementById("seed-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const customEmailText = document.getElementById("custom-email-text");
    const fileInput = document.getElementById("email-file-input");
    const emailList = document.getElementById("email-list");
    const emailCounter = document.getElementById("email-counter");
    const emptyState = document.getElementById("empty-detail-state");
    const activeState = document.getElementById("active-detail-state");

    let emailsData = [];

    async function fetchEmails() {
        const response = await fetch("/api/emails");
        emailsData = await response.json();
        renderEmailList();
    }

    function getPriorityClass(priority) {
        if (!priority) return "badge-p3";
        const normalized = String(priority).toLowerCase();
        if (normalized.includes("p1") || normalized.includes("urgent")) return "badge-p1";
        if (normalized.includes("p2") || normalized.includes("high")) return "badge-p2";
        return "badge-p3";
    }

    function renderEmailList() {
        emailList.innerHTML = "";
        emailCounter.textContent = `${emailsData.length} emails`;

        if (emailsData.length === 0) {
            emailList.innerHTML = '<p class="placeholder-text">Click "Seed Demo Data" to load emails.</p>';
            return;
        }

        emailsData.forEach((email, index) => {
            const card = document.createElement("div");
            card.className = "email-card";
            card.innerHTML = `
                <div class="card-top">
                    <span class="badge ${email.is_spam ? "badge-spam" : "badge-ham"}">
                        ${email.is_spam ? "SPAM" : "INBOX"}
                    </span>
                    <span class="badge ${getPriorityClass(email.priority)}">
                        ${email.priority || "P3 - Normal"}
                    </span>
                    <span class="badge badge-topic">${email.topic || "General"}</span>
                </div>
                <div class="card-subject">${email.subject}</div>
                <div class="card-sender">${email.sender}</div>
            `;
            card.addEventListener("click", () => selectEmail(index));
            emailList.appendChild(card);
        });
    }

    function selectEmail(index) {
        const email = emailsData[index];
        if (!email) return;

        emptyState.classList.add("hidden");
        activeState.classList.remove("hidden");

        document.getElementById("detail-subject").textContent = email.subject;
        document.getElementById("detail-sender").textContent = `From: ${email.sender}`;
        document.getElementById("detail-body").textContent = email.body;
        document.getElementById("detail-tone").textContent = email.overall_tone || "Neutral";

        const badgesContainer = document.getElementById("detail-badges");
        badgesContainer.innerHTML = `
            <span class="badge ${email.is_spam ? "badge-spam" : "badge-ham"}">${email.is_spam ? "SPAM" : "HAM"}</span>
            <span class="badge ${getPriorityClass(email.priority)}">${email.priority || "P3 - Normal"}</span>
            <span class="badge badge-topic">${email.topic || "General"}</span>
        `;

        const entityContainer = document.getElementById("detail-entities");
        entityContainer.innerHTML = "";
        if (email.action_items && email.action_items.length > 0) {
            email.action_items.forEach((item) => {
                const chip = document.createElement("span");
                chip.className = "chip chip-entity";
                const label = typeof item === "string" ? item : `${item.text || item.name} (${item.type || "entity"})`;
                chip.textContent = label;
                entityContainer.appendChild(chip);
            });
        } else {
            entityContainer.innerHTML = '<span class="text-muted">None detected</span>';
        }

        const phraseContainer = document.getElementById("detail-phrases");
        phraseContainer.innerHTML = "";
        if (email.key_phrases && email.key_phrases.length > 0) {
            email.key_phrases.forEach((phrase) => {
                const chip = document.createElement("span");
                chip.className = "chip chip-phrase";
                chip.textContent = phrase;
                phraseContainer.appendChild(chip);
            });
        } else {
            phraseContainer.innerHTML = '<span class="text-muted">None detected</span>';
        }
    }

    analyzeBtn.addEventListener("click", analyzeCustomEmail);

    async function analyzeCustomEmail() {
        const text = customEmailText.value.trim();
        if (!text) {
            customEmailText.focus();
            return;
        }

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing...";

        try {
            const response = await fetch("/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });
            const result = await response.json();
            const classification = result.classification || {};
            const customEmail = {
                id: Date.now(),
                subject: "Custom Email",
                sender: "user@local",
                body: text,
                is_spam: Boolean(result.is_spam),
                topic: classification.topic || "General",
                priority: classification.priority || "P3 - Normal",
                overall_tone: classification.overall_tone || "Neutral",
                action_items: result.action_items || [],
                key_phrases: result.key_phrases || []
            };

            emailsData.unshift(customEmail);
            renderEmailList();
            selectEmail(0);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = "Analyze email";
        }
    }

    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;

        const text = await file.text();
        customEmailText.value = text;
    });

    seedBtn.addEventListener("click", async () => {
        seedBtn.disabled = true;
        seedBtn.textContent = "Seeding & Analyzing...";

        try {
            await fetch("/api/seed", { method: "POST" });
            await fetchEmails();
            if (emailsData.length > 0) {
                selectEmail(0);
            }
        } finally {
            seedBtn.disabled = false;
            seedBtn.textContent = "Seed Demo Data";
        }
    });

    fetchEmails();
});
