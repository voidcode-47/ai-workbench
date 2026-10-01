/* ============================================================
   AI Workbench 前端逻辑 v4 (2026-09-16)
   - 用户登录 / 角色权限 / toast 通知 / 流式 Agent
   ============================================================ */
const API = "/api/v1";
const TOKEN_KEY = "aw_token";
const USER_KEY = "aw_user";

// ---------- 通用 ----------
const $ = (id) => document.getElementById(id);
let currentKb = null;
let currentUser = null; // {username, role}
const agentHistory = [];
const ragHistory = [];

function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Toast 通知（替代 alert，非阻塞） ----------
let toastTimer = null;
function toast(message, type = "info") {
    const el = $("toast");
    el.textContent = message;
    el.className = "toast" + (type === "error" ? " error" : type === "success" ? " success" : "");
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

// ---------- API（带 Bearer token） ----------
async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    // 发送 JSON body 时自动补 Content-Type，避免 FastAPI 422
    if (options.body && typeof options.body === "string" && !headers["Content-Type"]) {
        headers["Content-Type"] = "application/json";
    }
    if (currentToken()) headers["Authorization"] = "Bearer " + currentToken();
    const resp = await fetch(API + path, { ...options, headers });
    const data = await resp.json().catch(() => ({}));
    if (resp.status === 401) {
        // INVALID_CREDENTIALS = 用户名/密码类错误（登录、旧密码错），不视为会话过期
        if (data.error?.code !== "INVALID_CREDENTIALS") {
            logout(false);
            throw new Error("登录已过期，请重新登录");
        }
    }
    if (!resp.ok) {
        const msg = data.error?.message || `HTTP ${resp.status}`;
        throw new Error(msg);
    }
    return data;
}

function currentToken() { return localStorage.getItem(TOKEN_KEY) || ""; }

// ---------- 登录 / 退出 ----------
async function login(e) {
    e.preventDefault();
    const username = $("loginUser").value.trim();
    const password = $("loginPass").value;
    if (!username || !password) {
        $("loginError").textContent = "请输入用户名和密码";
        return;
    }
    $("loginError").textContent = "";
    $("btnLogin").disabled = true;
    try {
        const resp = await fetch(`${API}/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username, password }),
        });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error?.message || "登录失败");
        localStorage.setItem(TOKEN_KEY, data.token);
        currentUser = { username: data.username, role: data.role };
        localStorage.setItem(USER_KEY, JSON.stringify(currentUser));
        enterApp();
        toast("欢迎回来，" + data.username, "success");
    } catch (err) {
        $("loginError").textContent = err.message;
    } finally {
        $("btnLogin").disabled = false;
    }
}

function logout(notify = true) {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    currentUser = null;
    currentKb = null;
    $("pwdModal").hidden = true;   // 关闭可能挂着的弹窗
    $("appView").hidden = true;
    $("loginView").hidden = false;
    $("loginPass").value = "";
    if (notify) toast("已退出登录");
}

function enterApp() {
    $("loginView").hidden = true;
    $("appView").hidden = false;
    $("currentUser").textContent = currentUser.username;
    const roleEl = $("currentRole");
    roleEl.textContent = currentUser.role === "admin" ? "管理员" : "普通员工";
    roleEl.className = "role-badge" + (currentUser.role === "admin" ? " admin" : "");
    // 用户管理仅管理员可见
    $("navUsers").hidden = currentUser.role !== "admin";
    switchTab("rag");
    loadKbs().catch((e) => toast(e.message, "error"));
    loadTools();
    loadTemplates();
    if (currentUser.role === "admin") loadUsers().catch(() => {});
}

// ---------- 导航 ----------
function switchTab(tab) {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    const panel = $("panel-" + tab);
    if (panel) panel.classList.add("active");
    const titles = { rag: "知识库", agent: "智能助手", multimodal: "图片理解", workflow: "工作流", users: "用户管理" };
    $("pageTitle").textContent = titles[tab] || "AI Workbench";
    if (tab === "users" && currentUser?.role === "admin") loadUsers().catch(() => {});
}

document.querySelectorAll(".nav-item").forEach((b) =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));

// ============================================================
// 聊天气泡工具
// ============================================================
function addBubble(container, content, role = "ai", extraHtml = "") {
    const div = document.createElement("div");
    div.className = "bubble " + role;
    div.textContent = content;
    if (extraHtml) div.innerHTML += extraHtml;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function addSystem(container, text) {
    const div = document.createElement("div");
    div.className = "bubble system";
    div.textContent = text;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function renderCitations(citations) {
    if (!citations?.length) return "";
    const items = citations.map((c) =>
        `<div class="cite">[${c.index}] ${escapeHtml(c.source || "未知来源")}${c.page ? ` · 第 ${c.page} 页` : ""}（score=${c.score}）<br>${escapeHtml(c.content.slice(0, 180))}…</div>`
    ).join("");
    return `<div class="citations"><details><summary>引用来源（${citations.length}）</summary>${items}</details></div>`;
}

// ============================================================
// RAG：知识库 + 问答（带权限）
// ============================================================
function currentKbRole() {
    const sel = $("kbSelect");
    const opt = sel.selectedOptions[0];
    return opt ? opt.dataset.role || "" : "";
}

function applyKbPerms() {
    const role = currentKbRole();
    const writable = role === "owner" || role === "admin";
    $("btnUpload").disabled = !writable;
    $("btnReindex").disabled = !writable;
    $("btnDeleteKb").disabled = !writable;
}

async function loadKbs(selectId) {
    const kbs = await api("/rag/knowledge-bases");
    const sel = $("kbSelect");
    sel.innerHTML = "";
    kbs.forEach((kb) => {
        const opt = document.createElement("option");
        const name = kb.name || kb.kb_id;
        const roleText = kb.role === "owner" ? "（可编辑）" : kb.role === "readonly" ? "（只读）" : "";
        const ownerText = kb.owner && kb.owner !== currentUser.username ? ` · ${kb.owner}` : "";
        opt.value = kb.kb_id;
        opt.textContent = `${name} [${kb.kb_id}]${ownerText}${roleText} · ${kb.documents} 文档`;
        opt.dataset.role = kb.role;
        opt.dataset.owner = kb.owner || "";
        sel.appendChild(opt);
    });
    if (selectId) sel.value = selectId;
    currentKb = sel.value || null;
    applyKbPerms();
    await loadDocs();
}

async function loadDocs() {
    const list = $("docList");
    list.innerHTML = "";
    if (!currentKb) return;
    try {
        const docs = await api(`/rag/knowledge-bases/${currentKb}/documents`);
        docs.forEach((doc) => {
            const row = document.createElement("div");
            row.className = "doc-item";
            const name = document.createElement("span");
            name.className = "doc-name";
            name.textContent = `${doc.filename}（${doc.chunks} 块）`;
            const del = document.createElement("button");
            del.className = "doc-del";
            del.textContent = "✕";
            del.title = "删除该文档";
            del.disabled = currentKbRole() !== "owner" && currentKbRole() !== "admin";
            del.onclick = async () => {
                if (!confirm(`确定删除文档「${doc.filename}」？`)) return;
                try {
                    await api(`/rag/knowledge-bases/${currentKb}/documents/${doc.doc_id}`, { method: "DELETE" });
                    toast("文档已删除", "success");
                    loadDocs();
                } catch (e) { toast(e.message, "error"); }
            };
            row.appendChild(name);
            row.appendChild(del);
            list.appendChild(row);
        });
    } catch (e) { /* 知识库无权限时静默 */ }
}

$("kbSelect").addEventListener("change", (e) => {
    currentKb = e.target.value || null;
    applyKbPerms();
    loadDocs();
});

$("btnRefreshKb").onclick = () => loadKbs().catch((e) => toast(e.message, "error"));

$("btnCreateKb").onclick = async () => {
    const kb_id = $("kbId").value.trim();
    const name = $("kbName").value.trim() || null;
    if (!kb_id) return toast("请输入知识库 ID", "error");
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(kb_id)) {
        return toast("知识库 ID 只能包含字母、数字、中划线 - 和下划线 _（1-64 字符）", "error");
    }
    try {
        await api("/rag/knowledge-bases", { method: "POST", body: JSON.stringify({ kb_id, name }) });
        $("kbId").value = "";
        $("kbName").value = "";
        toast("知识库创建成功", "success");
        await loadKbs(kb_id);
    } catch (e) { toast(e.message, "error"); }
};

$("btnUpload").onclick = async () => {
    if (!currentKb) return toast("请先选择或创建知识库", "error");
    const files = $("kbFiles").files;
    if (!files.length) return toast("请选择文件", "error");
    const fd = new FormData();
    [...files].forEach((f) => fd.append("files", f));
    const btn = $("btnUpload");
    btn.disabled = true; btn.textContent = "解析与向量化中…";
    try {
        await fetch(`${API}/rag/knowledge-bases/${currentKb}/documents`, {
            method: "POST",
            headers: { Authorization: "Bearer " + currentToken() },
            body: fd,
        }).then(async (r) => {
            if (!r.ok) throw new Error((await r.json()).error?.message || "上传失败");
        });
        $("kbFiles").value = "";
        toast("上传并解析完成", "success");
        await loadKbs(currentKb);
    } catch (e) { toast(e.message, "error"); }
    finally { btn.disabled = false; btn.textContent = "上传文档"; }
};

$("btnReindex").onclick = async () => {
    if (!currentKb) return toast("请先选择知识库", "error");
    try {
        const r = await api(`/rag/knowledge-bases/${currentKb}/reindex`, { method: "POST" });
        toast(`重建完成，共 ${r.reindexed.length} 个文档`, "success");
        loadKbs(currentKb);
    } catch (e) { toast(e.message, "error"); }
};

$("btnDeleteKb").onclick = async () => {
    if (!currentKb) return toast("请先选择知识库", "error");
    if (!confirm(`确定删除知识库「${currentKb}」？该操作不可恢复！`)) return;
    try {
        await api(`/rag/knowledge-bases/${currentKb}`, { method: "DELETE" });
        toast("知识库已删除", "success");
        await loadKbs();
    } catch (e) { toast(e.message, "error"); }
};

async function askRag() {
    const question = $("ragQuestion").value.trim();
    if (!question) return;
    if (!currentKb) return toast("请先选择知识库", "error");
    addBubble($("ragChat"), question, "user");
    $("ragQuestion").value = "";
    const waiting = addSystem($("ragChat"), "检索与生成中…");
    try {
        const data = await api("/rag/query", {
            method: "POST",
            body: JSON.stringify({ kb_id: currentKb, question, history: ragHistory }),
        });
        waiting.remove();
        addBubble($("ragChat"), data.answer, "ai", renderCitations(data.citations));
        ragHistory.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
    } catch (e) { if (waiting) waiting.textContent = "出错：" + e.message; }
}
$("btnAsk").onclick = askRag;
$("ragQuestion").addEventListener("keydown", (e) => e.key === "Enter" && askRag());

// ============================================================
// Agent：流式对话
// ============================================================
async function loadTools() {
    try {
        const tools = await api("/agent/tools");
        $("toolListHint").textContent = "可用工具：" + tools.map((t) => t.name).join(" / ");
    } catch (e) { /* 忽略 */ }
}

function appendStep(chat, s) {
    const step = document.createElement("div");
    step.className = "step";
    if (s.type === "tool_call") {
        const calls = (s.tool_calls || []).map((c) => `${c.name}(${JSON.stringify(c.args)})`).join("; ");
        step.innerHTML = `调用工具：${escapeHtml(calls)}`;
    } else if (s.type === "observation") {
        step.innerHTML = `观察结果（${escapeHtml(s.tool || "")}）：${escapeHtml((s.content || "").slice(0, 300))}`;
    }
    if (step.innerHTML) chat.appendChild(step);
}

async function sendAgent() {
    const message = $("agentMessage").value.trim();
    if (!message) return;
    const chat = $("agentChat");
    addBubble(chat, message, "user");
    $("agentMessage").value = "";
    const waiting = addSystem(chat, "助手思考中…");
    let answerBubble = null;
    try {
        const resp = await fetch(`${API}/agent/chat/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: "Bearer " + currentToken(),
            },
            body: JSON.stringify({ message, history: agentHistory }),
        });
        if (resp.status === 401) { logout(false); throw new Error("登录已过期"); }
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.error?.message || `HTTP ${resp.status}`);
        }
        waiting.remove();
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const events = buf.split("\n\n");
            buf = events.pop();
            for (const raw of events) {
                let type = "", data = "";
                for (const line of raw.split("\n")) {
                    if (line.startsWith("event: ")) type = line.slice(7);
                    else if (line.startsWith("data: ")) data += line.slice(6);
                }
                if (!data) continue;
                const payload = JSON.parse(data);
                if (type === "step") {
                    appendStep(chat, payload.step);
                } else if (type === "answer") {
                    if (!answerBubble) answerBubble = addBubble(chat, "", "ai");
                    answerBubble.textContent += payload.content;
                    chat.scrollTop = chat.scrollHeight;
                } else if (type === "error") {
                    addBubble(chat, "出错：" + payload.message, "ai");
                }
            }
        }
        if (answerBubble && answerBubble.textContent) {
            agentHistory.push({ role: "user", content: message }, { role: "assistant", content: answerBubble.textContent });
        }
    } catch (e) {
        if (waiting) waiting.remove();
        if (!answerBubble) addBubble(chat, "出错：" + e.message, "ai");
    }
    chat.scrollTop = chat.scrollHeight;
}
$("btnAgentSend").onclick = sendAgent;
$("agentMessage").addEventListener("keydown", (e) => e.key === "Enter" && sendAgent());

// ============================================================
// 多模态
// ============================================================
$("imageFile").addEventListener("change", () => {
    const file = $("imageFile").files[0];
    const img = $("imagePreview");
    if (file) {
        img.src = URL.createObjectURL(file);
        img.style.display = "block";
    }
});

$("mmTask").addEventListener("change", (e) => {
    $("mmQuestion").style.display = e.target.value === "question" ? "block" : "none";
});

$("btnAnalyze").onclick = async () => {
    const file = $("imageFile").files[0];
    if (!file) return toast("请先选择图片", "error");
    const task = $("mmTask").value;
    const question = $("mmQuestion").value.trim() || "";
    if (task === "question" && !question) return toast("请填写问题", "error");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("task", task);
    if (question) fd.append("question", question);
    const btn = $("btnAnalyze");
    btn.disabled = true;
    $("mmResult").textContent = "分析中…";
    try {
        const resp = await fetch(`${API}/multimodal/analyze`, {
            method: "POST",
            headers: { Authorization: "Bearer " + currentToken() },
            body: fd,
        });
        if (resp.status === 401) { logout(false); throw new Error("登录已过期"); }
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) throw new Error(data.error?.message || "分析失败");
        $("mmResult").textContent = data.result;
        toast("分析完成", "success");
    } catch (e) {
        $("mmResult").textContent = "出错：" + e.message;
        toast(e.message, "error");
    } finally { btn.disabled = false; }
};

// ============================================================
// 工作流
// ============================================================
async function loadTemplates() {
    try {
        const tpls = await api("/workflow/templates");
        const sel = $("tplSelect");
        sel.innerHTML = "";
        tpls.forEach((t) => {
            const opt = document.createElement("option");
            opt.value = t.key;
            opt.textContent = t.name;
            sel.appendChild(opt);
        });
    } catch (e) { /* 忽略 */ }
}

$("btnLoadTpl").onclick = async () => {
    const key = $("tplSelect").value;
    if (!key) return toast("请选择模板", "error");
    try {
        const data = await api(`/workflow/templates/${key}`);
        $("wfDefinition").value = JSON.stringify(data.definition, null, 2);
        toast("模板已载入", "success");
    } catch (e) { toast(e.message, "error"); }
};

$("btnRunWf").onclick = async () => {
    let definition, inputs;
    try {
        definition = JSON.parse($("wfDefinition").value || "{}");
        inputs = JSON.parse($("wfInputs").value || "{}");
    } catch (e) { return toast("JSON 格式错误：" + e.message, "error"); }
    const btn = $("btnRunWf");
    btn.disabled = true;
    try {
        const data = await api("/workflow/run", {
            method: "POST",
            body: JSON.stringify({ definition, inputs }),
        });
        $("wfOutputs").textContent = JSON.stringify(data.outputs, null, 2);
        const logs = $("wfLogs");
        logs.innerHTML = "";
        (data.logs || []).forEach((log) => {
            const div = document.createElement("div");
            div.className = "log-line" + (log.status === "error" ? " error" : log.status === "ok" ? " ok" : "");
            div.textContent = `[${log.node}] ${log.status}${log.elapsed_ms != null ? ` (${log.elapsed_ms}ms)` : ""}${log.error ? " — " + log.error : ""}`;
            logs.appendChild(div);
        });
        toast(data.status === "ok" ? "工作流执行完成" : "工作流执行异常", data.status === "ok" ? "success" : "error");
    } catch (e) { toast(e.message, "error"); }
    finally { btn.disabled = false; }
};

// ============================================================
// 用户管理（管理员）
// ============================================================
async function loadUsers() {
    const users = await api("/users");
    const tbody = $("usersTable");
    tbody.innerHTML = "";
    users.forEach((u) => {
        const tr = document.createElement("tr");
        const td1 = document.createElement("td"); td1.textContent = u.username;
        const td2 = document.createElement("td"); td2.textContent = u.role === "admin" ? "管理员" : "普通员工";
        const td3 = document.createElement("td"); td3.textContent = u.created_at || "";
        const td4 = document.createElement("td");
        // 管理员账号与当前登录账号不可删除
        if (u.role !== "admin" && u.username !== currentUser.username) {
            const btn = document.createElement("button");
            btn.className = "btn danger small";
            btn.textContent = "删除";
            btn.onclick = async () => {
                if (!confirm(`确定删除用户「${u.username}」？其名下的知识库将转移给管理员。`)) return;
                try {
                    await api("/users/" + encodeURIComponent(u.username), { method: "DELETE" });
                    toast("用户已删除", "success");
                    await loadUsers();
                } catch (e) { toast(e.message, "error"); }
            };
            td4.appendChild(btn);
        } else {
            td4.textContent = "—";
        }
        tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
        tbody.appendChild(tr);
    });
}

$("btnAddUser").onclick = async () => {
    const username = $("newUserName").value.trim();
    const password = $("newUserPass").value;
    const role = $("newUserRole").value;
    if (!username) return toast("请输入用户名", "error");
    if (password.length < 6) return toast("密码至少 6 位", "error");
    try {
        await api("/users", { method: "POST", body: JSON.stringify({ username, password, role }) });
        $("newUserName").value = "";
        $("newUserPass").value = "";
        toast("用户创建成功", "success");
        await loadUsers();
    } catch (e) { toast(e.message, "error"); }
};

// ============================================================
// 修改密码
// ============================================================
$("btnChangePwd").onclick = () => {
    $("pwdError").textContent = "";
    $("chgOld").value = ""; $("chgNew").value = ""; $("chgConfirm").value = "";
    $("pwdModal").hidden = false;
};
$("btnChgCancel").onclick = () => { $("pwdModal").hidden = true; };
$("btnChgOk").onclick = async () => {
    const oldPwd = $("chgOld").value;
    const newPwd = $("chgNew").value;
    const confirmPwd = $("chgConfirm").value;
    $("pwdError").textContent = "";
    if (!oldPwd || !newPwd) return $("pwdError").textContent = "请填写完整";
    if (newPwd.length < 6) return $("pwdError").textContent = "新密码至少 6 位";
    if (newPwd !== confirmPwd) return $("pwdError").textContent = "两次输入的新密码不一致";
    try {
        await api("/auth/change-password", {
            method: "POST",
            body: JSON.stringify({ old_password: oldPwd, new_password: newPwd }),
        });
        $("pwdModal").hidden = true;
        toast("密码修改成功，请用新密码重新登录", "success");
        logout(true);
    } catch (e) { $("pwdError").textContent = e.message; }
};

// ============================================================
// 启动
// ============================================================
$("loginForm").addEventListener("submit", login);
$("btnLogout").onclick = () => logout(true);

(function boot() {
    const saved = localStorage.getItem(USER_KEY);
    if (saved && currentToken()) {
        try {
            currentUser = JSON.parse(saved);
            // 校验 token 有效性
            api("/auth/me").then((me) => {
                currentUser = { username: me.username, role: me.role };
                localStorage.setItem(USER_KEY, JSON.stringify(currentUser));
                enterApp();
            }).catch(() => logout(false));
        } catch (e) { logout(false); }
    } else {
        logout(false);
    }
})();
