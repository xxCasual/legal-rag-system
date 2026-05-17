const state = {
  lastJson: null,
};

const $ = (id) => document.getElementById(id);

const els = {
  lastAction: $("lastAction"),
  healthDot: $("healthDot"),
  healthText: $("healthText"),
  healthBtn: $("healthBtn"),
  chatInput: $("chatInput"),
  sendChatBtn: $("sendChatBtn"),
  clearChatBtn: $("clearChatBtn"),
  chatResult: $("chatResult"),
  companyFile: $("companyFile"),
  uploadCompanyDocBtn: $("uploadCompanyDocBtn"),
  refreshCompanyDocsBtn: $("refreshCompanyDocsBtn"),
  companyDocs: $("companyDocs"),
  lawFile: $("lawFile"),
  uploadLawDocBtn: $("uploadLawDocBtn"),
  refreshLawDocsBtn: $("refreshLawDocsBtn"),
  rebuildLawIndexBtn: $("rebuildLawIndexBtn"),
  lawDocs: $("lawDocs"),
  contractInput: $("contractInput"),
  sampleContractBtn: $("sampleContractBtn"),
  reviewContractBtn: $("reviewContractBtn"),
  contractResult: $("contractResult"),
  refreshReviewsBtn: $("refreshReviewsBtn"),
  pendingReviews: $("pendingReviews"),
};

function setLastAction(message) {
  els.lastAction.textContent = message;
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label || "处理中";
    button.disabled = true;
    return;
  }
  button.textContent = button.dataset.label || button.textContent;
  button.disabled = false;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = payload.detail || payload.error || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  state.lastJson = payload;
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function jsonBlock(payload) {
  return `<details><summary>原始 JSON</summary><pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre></details>`;
}

function pill(label, value, tone = "") {
  if (value === null || value === undefined || value === "") return "";
  return `<span class="pill ${tone}">${escapeHtml(label)}: ${escapeHtml(value)}</span>`;
}

function renderError(target, error) {
  target.className = "result";
  target.innerHTML = `<div class="pill danger">请求失败</div><p class="answer">${escapeHtml(error.message)}</p>`;
  setLastAction(`请求失败：${error.message}`);
}

async function checkHealth() {
  setBusy(els.healthBtn, true);
  try {
    const payload = await requestJson("/api/health");
    els.healthDot.className = "dot ok";
    els.healthText.textContent = payload.status;
    setLastAction("服务健康检查完成");
  } catch (error) {
    els.healthDot.className = "dot bad";
    els.healthText.textContent = "异常";
    setLastAction(`服务健康检查失败：${error.message}`);
  } finally {
    setBusy(els.healthBtn, false);
  }
}

function renderChatResult(payload) {
  const citations = payload.citations || [];
  const riskTone = payload.risk_level === "high" ? "danger" : payload.risk_level === "medium" ? "warn" : "";
  els.chatResult.className = "result";
  els.chatResult.innerHTML = `
    <div class="answer">${escapeHtml(payload.answer || "无回答")}</div>
    <div class="meta">
      ${pill("intent", payload.intent)}
      ${pill("route", payload.route)}
      ${pill("tools", (payload.tools_used || []).join(", "))}
      ${pill("type", payload.result_type)}
      ${pill("latency", `${payload.latency ?? 0}s`)}
      ${pill("risk", payload.risk_level, riskTone)}
      ${pill("review", payload.review_status, payload.review_status === "pending_review" ? "warn" : "")}
      ${pill("review_id", payload.review_id)}
    </div>
    ${renderCitations(citations)}
    ${renderContractFindings(payload.contract_review)}
    ${jsonBlock(payload)}
  `;
}

function renderCitations(citations) {
  if (!citations.length) return "";
  return `
    <details>
      <summary>引用依据 ${citations.length}</summary>
      ${citations.map((item) => `<div class="citation">${escapeHtml(item)}</div>`).join("")}
    </details>
  `;
}

function renderContractFindings(contractReview) {
  if (!contractReview || !Array.isArray(contractReview.findings) || !contractReview.findings.length) {
    return "";
  }
  return `
    <details open>
      <summary>风险明细 ${contractReview.findings.length}</summary>
      ${contractReview.findings.map(renderFinding).join("")}
    </details>
  `;
}

function renderFinding(finding) {
  const tone = finding.risk_level === "high" ? "danger" : finding.risk_level === "medium" ? "warn" : "";
  return `
    <div class="finding">
      <div class="item-title">${escapeHtml(finding.clause_name)}</div>
      <div class="meta">
        ${pill("status", finding.status)}
        ${pill("risk", finding.risk_level, tone)}
      </div>
      <div class="item-subtitle">${escapeHtml(finding.analysis)}</div>
      <div class="item-subtitle">${escapeHtml(finding.suggestion)}</div>
    </div>
  `;
}

async function sendChat() {
  const query = els.chatInput.value.trim();
  if (!query) {
    setLastAction("请输入问题");
    return;
  }
  setBusy(els.sendChatBtn, true, "发送中");
  els.chatResult.className = "result empty";
  els.chatResult.textContent = "请求中";
  try {
    const payload = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    renderChatResult(payload);
    setLastAction("聊天请求完成");
    if (payload.review_status === "pending_review") {
      await loadPendingReviews();
    }
  } catch (error) {
    renderError(els.chatResult, error);
  } finally {
    setBusy(els.sendChatBtn, false);
  }
}

async function uploadFile(input, url, button, afterUpload) {
  const file = input.files[0];
  if (!file) {
    setLastAction("请选择文件");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  setBusy(button, true, "上传中");
  try {
    const payload = await requestJson(url, {
      method: "POST",
      body: formData,
    });
    input.value = "";
    setLastAction(`已上传：${payload.file_name}`);
    await afterUpload(payload);
  } catch (error) {
    setLastAction(`上传失败：${error.message}`);
  } finally {
    setBusy(button, false);
  }
}

async function loadCompanyDocs() {
  setBusy(els.refreshCompanyDocsBtn, true);
  try {
    const payload = await requestJson("/api/documents");
    renderCompanyDocs(payload.documents || []);
    setLastAction("企业制度列表已刷新");
  } catch (error) {
    renderListError(els.companyDocs, error);
  } finally {
    setBusy(els.refreshCompanyDocsBtn, false);
  }
}

function renderCompanyDocs(documents) {
  if (!documents.length) {
    els.companyDocs.className = "list empty";
    els.companyDocs.textContent = "暂无文档";
    return;
  }
  els.companyDocs.className = "list";
  els.companyDocs.innerHTML = documents.map((doc) => `
    <div class="item">
      <div class="item-title">${escapeHtml(doc.file_name)}</div>
      <div class="item-subtitle">${escapeHtml(doc.source_type)} · ${doc.chunk_count} chunks · ${escapeHtml(doc.created_at)}</div>
    </div>
  `).join("");
}

async function loadLawDocs() {
  setBusy(els.refreshLawDocsBtn, true);
  try {
    const payload = await requestJson("/api/law-documents");
    renderLawDocs(payload.documents || []);
    setLastAction("法律条文列表已刷新");
  } catch (error) {
    renderListError(els.lawDocs, error);
  } finally {
    setBusy(els.refreshLawDocsBtn, false);
  }
}

function renderLawDocs(documents) {
  if (!documents.length) {
    els.lawDocs.className = "list empty";
    els.lawDocs.textContent = "暂无法律条文";
    return;
  }
  const needsRebuild = documents.some((doc) => doc.rebuild_required);
  els.lawDocs.className = "list";
  els.lawDocs.innerHTML = `
    ${needsRebuild ? '<div class="pill warn">需要重建索引</div>' : '<div class="pill">索引无待处理变更</div>'}
    ${documents.map((doc) => `
      <div class="item">
        <div class="item-title">${escapeHtml(doc.file_name)}</div>
        <div class="item-subtitle">${doc.size_bytes} bytes · ${escapeHtml(doc.updated_at)}</div>
      </div>
    `).join("")}
  `;
}

function renderListError(target, error) {
  target.className = "list";
  target.innerHTML = `<span class="pill danger">加载失败</span><div class="item-subtitle">${escapeHtml(error.message)}</div>`;
  setLastAction(`加载失败：${error.message}`);
}

async function rebuildLawIndex() {
  setBusy(els.rebuildLawIndexBtn, true, "重建中");
  try {
    const payload = await requestJson("/api/law-documents/rebuild-index", {
      method: "POST",
    });
    setLastAction(payload.message || "法律索引已重建");
    await loadLawDocs();
    els.lawDocs.insertAdjacentHTML("afterbegin", `
      <div class="meta">
        ${pill("documents", payload.indexed_document_count)}
        ${pill("nodes", payload.indexed_node_count)}
      </div>
    `);
  } catch (error) {
    setLastAction(`重建失败：${error.message}`);
  } finally {
    setBusy(els.rebuildLawIndexBtn, false);
  }
}

async function reviewContract() {
  const contractText = els.contractInput.value.trim();
  if (!contractText) {
    setLastAction("请输入合同文本");
    return;
  }
  setBusy(els.reviewContractBtn, true, "审查中");
  els.contractResult.className = "result empty";
  els.contractResult.textContent = "请求中";
  try {
    const payload = await requestJson("/api/review/contract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ contract_text: contractText }),
    });
    els.contractResult.className = "result";
    const tone = payload.risk_level === "high" ? "danger" : payload.risk_level === "medium" ? "warn" : "";
    els.contractResult.innerHTML = `
      <div class="meta">
        ${pill("risk", payload.risk_level, tone)}
        ${pill("review", payload.review_status, payload.review_status === "pending_review" ? "warn" : "")}
        ${pill("review_id", payload.review_id)}
        ${pill("latency", `${payload.latency ?? 0}s`)}
      </div>
      ${renderContractFindings(payload)}
      ${renderSuggestions(payload.suggestions || [])}
      ${renderCitations(payload.evidence || [])}
      ${jsonBlock(payload)}
    `;
    setLastAction("合同审查完成");
    if (payload.review_status === "pending_review") {
      await loadPendingReviews();
    }
  } catch (error) {
    renderError(els.contractResult, error);
  } finally {
    setBusy(els.reviewContractBtn, false);
  }
}

function renderSuggestions(suggestions) {
  if (!suggestions.length) return "";
  return `
    <details open>
      <summary>建议 ${suggestions.length}</summary>
      ${suggestions.map((item) => `<div class="item-subtitle">${escapeHtml(item)}</div>`).join("")}
    </details>
  `;
}

async function loadPendingReviews() {
  setBusy(els.refreshReviewsBtn, true);
  try {
    const payload = await requestJson("/api/reviews/pending");
    renderPendingReviews(payload.reviews || []);
    setLastAction("审批队列已刷新");
  } catch (error) {
    renderListError(els.pendingReviews, error);
  } finally {
    setBusy(els.refreshReviewsBtn, false);
  }
}

function renderPendingReviews(reviews) {
  if (!reviews.length) {
    els.pendingReviews.className = "list empty";
    els.pendingReviews.textContent = "暂无待审批记录";
    return;
  }
  els.pendingReviews.className = "list";
  els.pendingReviews.innerHTML = reviews.map((review) => `
    <div class="item" data-review-id="${escapeHtml(review.review_id)}">
      <div class="item-title">${escapeHtml(review.review_id)}</div>
      <div class="meta">
        ${pill("source", review.source_type)}
        ${pill("status", review.status, "warn")}
      </div>
      <div class="item-subtitle">${escapeHtml(review.payload?.summary || "")}</div>
      <details>
        <summary>payload</summary>
        <pre>${escapeHtml(JSON.stringify(review.payload || {}, null, 2))}</pre>
      </details>
      <div class="review-actions">
        <button type="button" data-decision="approve">通过</button>
        <button type="button" class="danger" data-decision="reject">拒绝</button>
      </div>
    </div>
  `).join("");
}

async function decideReview(reviewId, decision, button) {
  setBusy(button, true);
  try {
    const payload = await requestJson(`/api/reviews/${encodeURIComponent(reviewId)}/${decision}`, {
      method: "POST",
    });
    setLastAction(`审批结果：${payload.status}`);
    await loadPendingReviews();
    els.chatResult.className = "result";
    els.chatResult.innerHTML = jsonBlock(payload);
  } catch (error) {
    setLastAction(`审批失败：${error.message}`);
  } finally {
    setBusy(button, false);
  }
}

function bindEvents() {
  els.healthBtn.addEventListener("click", checkHealth);
  els.sendChatBtn.addEventListener("click", sendChat);
  els.clearChatBtn.addEventListener("click", () => {
    els.chatInput.value = "";
    els.chatResult.className = "result empty";
    els.chatResult.textContent = "暂无结果";
  });
  document.querySelectorAll("[data-query]").forEach((button) => {
    button.addEventListener("click", () => {
      els.chatInput.value = button.dataset.query || "";
      sendChat();
    });
  });
  els.uploadCompanyDocBtn.addEventListener("click", () => uploadFile(
    els.companyFile,
    "/api/documents/upload",
    els.uploadCompanyDocBtn,
    loadCompanyDocs,
  ));
  els.refreshCompanyDocsBtn.addEventListener("click", loadCompanyDocs);
  els.uploadLawDocBtn.addEventListener("click", () => uploadFile(
    els.lawFile,
    "/api/law-documents/upload",
    els.uploadLawDocBtn,
    loadLawDocs,
  ));
  els.refreshLawDocsBtn.addEventListener("click", loadLawDocs);
  els.rebuildLawIndexBtn.addEventListener("click", rebuildLawIndex);
  els.sampleContractBtn.addEventListener("click", () => {
    els.contractInput.value = "合同期限为三年。试用期一年。工资另行约定。员工自愿放弃社保。甲方可随时解除合同且不支付经济补偿。";
  });
  els.reviewContractBtn.addEventListener("click", reviewContract);
  els.refreshReviewsBtn.addEventListener("click", loadPendingReviews);
  els.pendingReviews.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-decision]");
    if (!button) return;
    const item = button.closest("[data-review-id]");
    decideReview(item.dataset.reviewId, button.dataset.decision, button);
  });
}

bindEvents();
checkHealth();
loadCompanyDocs();
loadLawDocs();
loadPendingReviews();
