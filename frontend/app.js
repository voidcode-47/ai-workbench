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
    // 清空会话数据：否则同一浏览器里下一位登录者会看到上一位的对话，
    // 且这些历史会被当作上下文重新发给模型
    agentHistory.length = 0;
    ragHistory.length = 0;
    $("agentChat").replaceChildren();
    $("ragChat").replaceChildren();
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
    loadModelConfig();
    if (currentUser.role === "admin") loadUsers().catch(() => {});
}

// ---------- 导航 ----------
function switchTab(tab) {
    document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    const panel = $("panel-" + tab);
    if (panel) panel.classList.add("active");
    const titles = { rag: "知识库", agent: "智能助手", multimodal: "图片理解", workflow: "工作流", models: "模型接入", users: "用户管理" };
    $("pageTitle").textContent = titles[tab] || "AI Workbench";
    if (tab === "users" && currentUser?.role === "admin") loadUsers().catch(() => {});
    if (tab === "models") loadModelConfig();
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
    // 历史属于具体知识库：切换后必须清掉，否则上一库的问答会被当成上下文发给模型
    ragHistory.length = 0;
    $("ragChat").replaceChildren();
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
        // 复用 api()：自动带 token、401 时统一登出、非 JSON 错误体也不会抛 SyntaxError
        await api(`/rag/knowledge-bases/${currentKb}/documents`, {
            method: "POST",
            body: fd,
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
        // 空回答不能入历史：后端要求 content 至少 1 个字符，
        // 一旦存进去，本会话后续每次提问都会因校验失败而 422
        if (data.answer) {
            ragHistory.push({ role: "user", content: question }, { role: "assistant", content: data.answer });
        }
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
                } else if (type === "streaming_fallback") {
                    // 网关不支持 SSE，已降级为整段输出（功能不受影响，仅无打字机效果）
                    if (payload.reason === "sse_unsupported") {
                        addSystem(chat, "（该模型网关不支持逐字流式，已整段返回）");
                    }
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
    // 释放上一个预览的对象 URL：不回收会让每次选图都留下一个常驻内存的 blob
    if (img.dataset.objectUrl) {
        URL.revokeObjectURL(img.dataset.objectUrl);
        delete img.dataset.objectUrl;
    }
    if (file) {
        const url = URL.createObjectURL(file);
        img.dataset.objectUrl = url;
        img.src = url;
        img.style.display = "block";
    } else {
        img.removeAttribute("src");
        img.style.display = "none";
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
        // 后端状态取值：success / skipped / failed（见 workflow/engine.py）
        const LOG_TEXT = { success: "完成", skipped: "跳过（分支未命中）", failed: "失败" };
        (data.logs || []).forEach((log) => {
            const div = document.createElement("div");
            const cls = log.status === "failed" ? " error" : log.status === "success" ? " ok" : " muted";
            div.className = "log-line" + cls;
            div.textContent = `[${log.node}] ${LOG_TEXT[log.status] || log.status}`
                + (log.elapsed_ms != null ? ` (${log.elapsed_ms}ms)` : "")
                + (log.error ? " — " + log.error : "");
            logs.appendChild(div);
        });
        const ok = data.status === "success" || data.status === "ok";
        toast(ok ? "工作流执行完成" : "工作流执行异常", ok ? "success" : "error");
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
// 模型接入
// ============================================================
let modelConfig = null;   // {catalog, effective, saved}

function isAdmin() { return currentUser?.role === "admin"; }

function providerMeta(key) {
    return (modelConfig?.catalog || []).find((p) => p.key === key) || null;
}

function updateModelChip() {
    const text = $("modelChipText");
    const dot = $("modelChipDot");
    if (!modelConfig) {
        text.textContent = "模型未配置";
        dot.className = "model-chip-dot warn";
        return;
    }
    const e = modelConfig.effective;
    text.textContent = `${e.provider_label || "沿用 .env"} · ${e.llm_model || "未指定模型"}`;
    const meta = providerMeta(e.provider);
    const usable = e.has_api_key || meta?.local || !e.provider;
    dot.className = "model-chip-dot" + (usable ? " ok" : " warn");
    $("modelChip").title = `提供商：${e.provider_label || "沿用 .env"}｜模型：${e.llm_model || "未指定"}｜端点：${e.base_url}`;
}

async function loadModelConfig() {
    try {
        modelConfig = await api("/models/config");
        renderModelPanel();
        updateModelChip();
    } catch (e) {
        $("modelChipText").textContent = "模型信息加载失败";
        $("modelChipDot").className = "model-chip-dot warn";
        $("mdCurrent").textContent = "加载失败：" + e.message;
    }
}

function renderModelPanel() {
    const catalog = modelConfig.catalog || [];
    const sel = $("mdProvider");
    // 重渲染时保留用户当前选中的提供商，否则回落到实际生效的那家
    const keep = sel.value || modelConfig.effective.provider || catalog[0]?.key || "";
    sel.innerHTML = "";
    catalog.forEach((p) => {
        const opt = document.createElement("option");
        opt.value = p.key;
        opt.textContent = p.label + (p.local ? " · 本地" : "");
        sel.appendChild(opt);
    });
    sel.value = keep;
    fillModelForm(keep);

    const e = modelConfig.effective;
    $("mdCurrent").innerHTML = [
        ["提供商", e.provider_label || "未配置（沿用 .env）"],
        ["Base URL", e.base_url || "—"],
        ["对话模型", e.llm_model || "—"],
        ["Embedding 模型", e.embedding_model || "—"],
        ["Embedding 端点", e.embedding_base_url || "与 Base URL 相同"],
        ["视觉模型", e.vision_model || "—"],
        ["API Key", e.has_api_key ? "已配置 " + e.api_key_hint : "未配置"],
    ].map(([k, v]) => `<div class="kv-row"><span>${escapeHtml(k)}</span><code>${escapeHtml(v)}</code></div>`).join("");

    // 接入参数会影响所有用户，仅管理员可修改
    const editable = isAdmin();
    ["mdProvider", "mdBaseUrl", "mdApiKey", "mdModelSelect", "mdModel", "mdEmbedding",
     "mdEmbeddingUrl", "mdVision", "btnFetchModels", "btnTestModel", "btnSaveModel", "btnResetModel"]
        .forEach((id) => { const el = $(id); if (el) el.disabled = !editable; });
    $("modelAdminHint").textContent = editable ? "" : "仅管理员可修改，以下为只读展示";
}

function fillModelForm(provider) {
    const meta = providerMeta(provider);
    const saved = modelConfig.saved?.[provider] || {};
    $("mdBaseUrl").value = saved.base_url || meta?.default_base_url || modelConfig.effective.base_url || "";
    $("mdApiKey").value = "";
    $("mdEmbeddingUrl").value = saved.embedding_base_url || "";
    fillModelOptions(meta?.models || []);

    const chosen = saved.llm_model || meta?.default_model || "";
    $("mdModel").value = chosen;
    if (chosen) $("mdModelSelect").value = chosen;  // 不在列表内时 select 保持首项，以手填框为准
    $("mdEmbedding").value = saved.embedding_model || meta?.default_embedding_model || "";
    $("mdVision").value = saved.vision_model || meta?.default_vision_model || "";

    $("mdProviderHint").textContent = meta
        ? `${meta.label}｜${meta.supports_vision ? "支持图片理解" : "无视觉能力（图片理解会失败）"}`
        : "";
    const keyState = saved.has_api_key ? "已保存 Key"
        : meta?.has_env_key ? "使用 .env 中的 Key"
        : meta?.local ? "无需 Key" : "未配置 Key";
    $("mdKeyHint").textContent = keyState + (meta?.local ? "（本地/自建服务可留空）" : "（留空保存不会清除已存 Key）");
}

function fillModelOptions(models) {
    const ms = $("mdModelSelect");
    ms.innerHTML = "";
    if (!models.length) {
        const opt = document.createElement("option");
        opt.value = ""; opt.textContent = "（暂无列表，可点「拉取列表」或直接填写）";
        ms.appendChild(opt);
        return;
    }
    models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m; opt.textContent = m;
        ms.appendChild(opt);
    });
    const cur = $("mdModel").value.trim();
    if (cur && models.includes(cur)) ms.value = cur;
}

function currentProfile() {
    const typed = $("mdModel").value.trim();
    return {
        base_url: $("mdBaseUrl").value.trim(),
        api_key: $("mdApiKey").value,   // 留空 = 不修改（由后端处理）
        llm_model: typed || $("mdModelSelect").value || "",
        embedding_model: $("mdEmbedding").value.trim(),
        embedding_base_url: $("mdEmbeddingUrl").value.trim(),
        vision_model: $("mdVision").value.trim(),
    };
}

$("mdProvider").addEventListener("change", () => {
    fillModelForm($("mdProvider").value);
    $("mdStatus").textContent = "尚未测试";
});

// 显式从下拉列表选模型 = 以列表为准，同步到手填框
$("mdModelSelect").addEventListener("change", () => {
    if ($("mdModelSelect").value) $("mdModel").value = $("mdModelSelect").value;
});

$("btnFetchModels").onclick = async () => {
    const btn = $("btnFetchModels");
    btn.disabled = true;
    $("mdStatus").textContent = "正在拉取模型列表…";
    try {
        const r = await api("/models/list", {
            method: "POST",
            body: JSON.stringify({ provider: $("mdProvider").value, profile: currentProfile() }),
        });
        fillModelOptions(r.models || []);
        const note = r.source === "remote"
            ? `已从服务端获取 ${r.models.length} 个模型`
            : "未能连接服务端，已回退到建议列表";
        $("mdStatus").textContent = (r.error ? "⚠️ " : "✅ ") + note + (r.error ? `（${r.error}）` : "");
    } catch (e) {
        $("mdStatus").textContent = "❌ " + e.message;
    } finally { btn.disabled = false; }
};

$("btnTestModel").onclick = async () => {
    const btn = $("btnTestModel");
    btn.disabled = true;
    $("mdStatus").textContent = "测试中…";
    try {
        const r = await api("/models/test", {
            method: "POST",
            body: JSON.stringify({ provider: $("mdProvider").value, profile: currentProfile() }),
        });
        if (r.models?.length) fillModelOptions(r.models);
        $("mdStatus").textContent = (r.ok ? "✅ " : "❌ ") + r.detail;
        toast(r.ok ? "连接正常" : "连接失败", r.ok ? "success" : "error");
    } catch (e) {
        $("mdStatus").textContent = "❌ " + e.message;
        toast(e.message, "error");
    } finally { btn.disabled = false; }
};

$("btnSaveModel").onclick = async () => {
    const profile = currentProfile();
    if (!profile.llm_model) return toast("请填写对话模型 ID", "error");
    const btn = $("btnSaveModel");
    btn.disabled = true;
    try {
        modelConfig = await api("/models/config", {
            method: "POST",
            body: JSON.stringify({ provider: $("mdProvider").value, profile }),
        });
        renderModelPanel();
        updateModelChip();
        $("mdStatus").textContent = "✅ 已保存并生效（无需重启）";
        toast("模型配置已生效", "success");
    } catch (e) {
        $("mdStatus").textContent = "❌ 保存失败：" + e.message;
        toast(e.message, "error");
    } finally { btn.disabled = false; }
};

$("btnResetModel").onclick = async () => {
    if (!confirm("恢复为 .env 中的模型配置？界面保存的接入参数（含 API Key）将被清除。")) return;
    const btn = $("btnResetModel");
    btn.disabled = true;
    try {
        modelConfig = await api("/models/config", { method: "DELETE" });
        renderModelPanel();
        updateModelChip();
        $("mdStatus").textContent = "已恢复 .env 默认配置";
        toast("已恢复默认配置", "success");
    } catch (e) {
        $("mdStatus").textContent = "❌ " + e.message;
        toast(e.message, "error");
    } finally { btn.disabled = false; }
};

$("modelChip").onclick = () => switchTab("models");

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
