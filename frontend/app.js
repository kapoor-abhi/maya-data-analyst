'use strict';

/* ══════════════════════════════════════════════════════════════
   Maya AI — Frontend (v3.2)

   Key fixes vs original:
   1. launchPipeline() no longer awaits the upload fetch.
      Upload returns {status:"running"} immediately.
      SSE stream drives all progress updates.
   2. SSE is opened BEFORE the upload POST fires.
   3. SSE handles: log | progress | done | paused | error events.
   4. Resume also uses non-blocking pattern (POST → SSE).
   5. Activity feed renders WHILE pipeline runs, not after.
   6. Progress bar driven by SSE progress events.
   ══════════════════════════════════════════════════════════════ */

const state = {
  userId: localStorage.getItem('maya_uid') || genId(),
  threadId: null,
  files: [],
  working: {},
  isPaused: false,
  isRunning: false,
  logCount: 0,
  taskPlan: null,
  taskResults: [],
  currentTaskIndex: 0,
  currentTaskTitle: '',
  currentTaskType: '',
  activeAgent: '',
  pipelineStatus: 'idle',
  datasetIssues: [],
  cleaningStrategy: [],
  interruptMsg: '',
  planDraft: '',
  mlReport: null,
  charts: [],
  stats: null,
  sse: null,
  voicePlayed: false,
};
localStorage.setItem('maya_uid', state.userId);
const API = resolveApiBase();

function genId() { return 'u' + Math.random().toString(36).slice(2, 10); }

function resolveApiBase() {
  const stored = localStorage.getItem('maya_api_base')?.trim();
  if (stored) return stored.replace(/\/+$/, '');
  const { protocol, hostname, port, origin } = window.location;
  if (protocol === 'file:') return 'http://localhost:8000';
  if (['localhost', '127.0.0.1', '0.0.0.0'].includes(hostname) && port !== '8000')
    return `${protocol}//${hostname}:8000`;
  return origin === 'null' ? '' : origin;
}

function explainFetchError(err) {
  if (!(err instanceof TypeError)) return err.message || 'Request failed.';
  return `Cannot reach Maya API at ${API || window.location.origin}. ` +
    'Make sure the FastAPI server is running on port 8000.';
}

function applyServerState(data = {}) {
  if (data.working_files) state.working = data.working_files;
  if (Array.isArray(data.task_plan)) state.taskPlan = data.task_plan;
  else if (data.task_plan) {
    try { state.taskPlan = JSON.parse(data.task_plan); } catch { }
  }
  if (Array.isArray(data.task_results)) state.taskResults = data.task_results;
  state.currentTaskIndex = data.current_task_index ?? state.currentTaskIndex ?? 0;
  state.currentTaskTitle = data.current_task_title || state.currentTaskTitle || '';
  state.currentTaskType = data.current_task_type || state.currentTaskType || '';
  state.activeAgent = data.active_agent || state.activeAgent || '';
  state.pipelineStatus = data.pipeline_status || state.pipelineStatus || 'idle';
  if (Array.isArray(data.dataset_issues)) state.datasetIssues = data.dataset_issues;
  if (Array.isArray(data.cleaning_strategy)) state.cleaningStrategy = data.cleaning_strategy;
  if (data.ml_report) state.mlReport = data.ml_report;
  if (Array.isArray(data.charts_generated)) state.charts = data.charts_generated;
  if (typeof data.interrupt_msg === 'string') state.interruptMsg = data.interrupt_msg;
}

/* ── Bubbles ────────────────────────────────────────────────── */
function spawnBubbles(container) {
  const colors = ['rgba(99,102,241,.12)', 'rgba(167,139,250,.1)',
    'rgba(192,132,252,.08)', 'rgba(129,140,248,.1)', 'rgba(99,102,241,.06)'];
  function addBubble() {
    const b = document.createElement('div');
    b.className = 'bubble';
    const size = 6 + Math.random() * 40;
    b.style.cssText = `width:${size}px;height:${size}px;left:${Math.random() * 100}%;` +
      `bottom:-${size}px;background:${colors[Math.floor(Math.random() * colors.length)]};` +
      `backdrop-filter:blur(1px);animation-duration:${8 + Math.random() * 12}s;` +
      `animation-delay:${Math.random() * 2}s;`;
    container.appendChild(b);
    b.addEventListener('animationend', () => b.remove());
  }
  for (let i = 0; i < 15; i++) setTimeout(() => addBubble(), i * 400);
  setInterval(addBubble, 1200);
}

/* ── Voice ──────────────────────────────────────────────────── */
function speakGreeting() {
  if (state.voicePlayed || !('speechSynthesis' in window)) return;
  state.voicePlayed = true;
  const msg = new SpeechSynthesisUtterance("Hi! I am Maya, your analytical intelligence assistant.");
  msg.rate = 0.95; msg.pitch = 1.15; msg.volume = 0.8;
  function setVoice() {
    const voices = speechSynthesis.getVoices();
    msg.voice = voices.find(v => /female|samantha|victoria|karen|zira|fiona/i.test(v.name))
      || voices.find(v => /en.*us|en.*gb/i.test(v.lang) && !/male/i.test(v.name))
      || voices[0];
    speechSynthesis.speak(msg);
  }
  if (speechSynthesis.getVoices().length) setVoice();
  else speechSynthesis.onvoiceschanged = setVoice;
}

/* ── Init ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  setupDrop(); setupInput(); tryRestore();
  const bc = document.querySelector('.bubble-canvas');
  if (bc) spawnBubbles(bc);
  document.querySelector('.hero-orb')?.addEventListener('click', speakGreeting);
  setTimeout(speakGreeting, 1500);
  document.getElementById('modal-provider')?.addEventListener('change', e => {
    document.getElementById('provider-badge').textContent =
      e.target.value.charAt(0).toUpperCase() + e.target.value.slice(1);
  });
});

/* ── Session restore ────────────────────────────────────────── */
async function tryRestore() {
  const saved = localStorage.getItem('maya_thread');
  if (!saved) return;
  try {
    const r = await fetch(`${API}/state/${saved}`);
    if (!r.ok) return;
    const d = await r.json();
    if (d.status === 'not_found') return;
    state.threadId = saved;
    applyServerState(d);
    (d.agent_log || []).forEach(renderEntry);
    state.logCount = (d.agent_log || []).length;
    showChat();
    if (d.status === 'paused') {
      state.isPaused = true;
      state.interruptMsg = d.interrupt_msg || '';
      renderInterrupt(d.interrupt_msg);
      toast('Session restored — awaiting your input.', 'info');
    } else {
      toast('Previous session restored.', 'ok');
    }
    updateStats(d);
    refreshList();
    renderPlan();
  } catch { }
}

/* ── File handling ──────────────────────────────────────────── */
function setupDrop() {
  const dz = document.getElementById('drop-zone');
  const fi = document.getElementById('file-input');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('over');
    stageFiles([...e.dataTransfer.files]);
  });
  fi.addEventListener('change', () => { if (fi.files.length) stageFiles([...fi.files]); fi.value = ''; });
}

function stageFiles(files) {
  const ok = ['.csv', '.xlsx', '.xls', '.parquet', '.json'];
  files.forEach(f => {
    const ext = '.' + f.name.split('.').pop().toLowerCase();
    if (ok.includes(ext) && state.files.length < 20 && !state.files.find(x => x.name === f.name))
      state.files.push(f);
  });
  refreshList();
  if (state.files.length) openModal();
}

function refreshList() {
  const el = document.getElementById('file-list');
  el.innerHTML = '';
  state.files.forEach(f => {
    el.innerHTML += `<div class="file-item"><span class="file-dot staged"></span><span class="file-name">${f.name}</span></div>`;
  });
  Object.keys(state.working).forEach(name => {
    el.innerHTML += `<div class="file-item"><span class="file-dot loaded"></span><span class="file-name">${name}</span></div>`;
  });
  document.getElementById('stat-files').textContent =
    state.files.length + Object.keys(state.working).length;
}

/* ── Modal ──────────────────────────────────────────────────── */
function openModal() {
  document.getElementById('modal-files').innerHTML =
    state.files.map(f => `<div>${f.name} — ${(f.size / 1048576).toFixed(1)} MB</div>`).join('');
  document.getElementById('upload-modal').classList.remove('hidden');
}
function closeModal() { document.getElementById('upload-modal').classList.add('hidden'); }

/* ── Pipeline launch — NON-BLOCKING ────────────────────────── */
async function launchPipeline() {
  const prompt = document.getElementById('modal-prompt').value.trim() || 'Load and analyse this dataset.';
  if (!state.files.length) { toast('No files selected.', 'error'); return; }
  closeModal();
  setRunning(true);
  showChat();
  switchTab('plan');

  state.taskPlan = null;
  state.taskResults = [];
  state.currentTaskIndex = 0;
  state.currentTaskTitle = '';
  state.currentTaskType = '';
  state.activeAgent = 'planner';
  state.pipelineStatus = 'planning';
  state.datasetIssues = [];
  state.cleaningStrategy = [];
  state.interruptMsg = '';
  renderPlan();

  state.threadId = 't' + Date.now() + Math.random().toString(36).slice(2, 6);
  localStorage.setItem('maya_thread', state.threadId);
  setProgress(0, 'starting');

  appendMsg('user', prompt);
  appendMsg('maya', '<span class="spin">⟳</span> Starting pipeline…');

  // ★ Open SSE BEFORE firing upload — so we don't miss any early log entries
  startSSE();

  const fd = new FormData();
  fd.append('thread_id', state.threadId);
  fd.append('user_id', state.userId);
  fd.append('user_input', prompt);
  state.files.forEach(f => fd.append('files', f));

  try {
    const r = await fetch(`${API}/upload`, { method: 'POST', body: fd });
    const data = await r.json();

    if (data.status === 'error') {
      removeSpinner();
      appendMsg('maya', '❌ ' + (data.message || 'Upload failed.'));
      toast(data.message || 'Upload failed.', 'error');
      setRunning(false);
      return;
    }

    // status === "running" → pipeline is running in background, SSE will notify us
    state.files = [];
    refreshList();
    // spinner + running state are cleared by the SSE "done"/"paused"/"error" event handler

  } catch (e) {
    const msg = explainFetchError(e);
    removeSpinner();
    appendMsg('maya', '❌ ' + msg);
    toast(msg, 'error');
    setRunning(false);
  }
}

/* ── SSE ────────────────────────────────────────────────────── */
function startSSE() {
  if (state.sse) { state.sse.close(); state.sse = null; }
  const url = `${API}/stream/${state.threadId}`;
  state.sse = new EventSource(url);

  state.sse.onmessage = e => {
    try {
      const m = JSON.parse(e.data);

      if (m.type === 'log' && m.entry) {
        renderEntry(m.entry);
      }

      if (m.type === 'state') {
        applyServerState(m);
        renderPlan();
        if (state.mlReport) renderML();
        refreshList();
      }

      if (m.type === 'progress') {
        setProgress(m.pct, m.stage);
      }

      if (m.type === 'done') {
        state.sse.close(); state.sse = null;
        handlePipelineResult(m.result || {});
      }

      if (m.type === 'paused') {
        state.sse.close(); state.sse = null;
        removeSpinner();
        state.isPaused = true;
        state.interruptMsg = m.interrupt_msg || '';
        applyServerState(m);
        renderPlan();
        if (state.mlReport) renderML();
        renderInterrupt(m.interrupt_msg);
        setRunning(false);
        switchTab('plan');
        refreshList();
        loadStats();
      }

      if (m.type === 'error') {
        state.sse.close(); state.sse = null;
        removeSpinner();
        appendMsg('maya', '❌ Pipeline error: ' + (m.message || 'Unknown error'));
        toast(m.message || 'Pipeline error', 'error');
        setProgress(0, 'error');
        setRunning(false);
      }
    } catch { }
  };

  state.sse.onerror = () => {
    state.sse?.close();
    state.sse = null;
  };
}

function handlePipelineResult(result) {
  removeSpinner();
  state.isPaused = false;
  state.interruptMsg = '';
  applyServerState(result);

  renderPlan();
  if (state.mlReport) renderML();
  if (result.insights && result.insights.length) {
    appendMsg('maya', '💡 **Key Insights**\n\n' + result.insights.slice(0, 5).map(i => '- ' + i).join('\n'));
  }

  // Show the last message or a generic done message
  const finalMsg = result.response || 'Pipeline complete. You can now query your data.';
  appendMsg('maya', finalMsg);

  if (state.charts.length) renderCharts();
  setProgress(100, 'complete');
  setRunning(false);
  refreshList();
  loadStats();
}

/* ── Chat / Resume ──────────────────────────────────────────── */
async function sendMessage() {
  const ta = document.getElementById('msg-input');
  const text = ta.value.trim();
  if (!text || state.isRunning) return;
  ta.value = ''; ta.style.height = 'auto';
  setRunning(true);
  appendMsg('user', text);

  if (state.isPaused) {
    // Resume the paused pipeline
    appendMsg('maya', '<span class="spin">⟳</span> Resuming…');
    startSSE();  // re-open SSE to watch resume progress

    const fd = new FormData();
    fd.append('thread_id', state.threadId);
    fd.append('user_id', state.userId);
    fd.append('feedback', text);

    try {
      const r = await fetch(`${API}/resume`, { method: 'POST', body: fd });
      const data = await r.json();
      if (data.status === 'error') {
        removeSpinner();
        appendMsg('maya', '❌ ' + (data.message || 'Resume failed.'));
        setRunning(false);
      } else {
        state.isPaused = false;
        state.interruptMsg = '';
      }
    } catch (e) {
      removeSpinner();
      appendMsg('maya', explainFetchError(e));
      setRunning(false);
    }

  } else if (state.threadId && Object.keys(state.working).length) {
    // Regular chat query
    appendMsg('maya', '<span class="spin">⟳</span> Thinking…');

    const fd = new FormData();
    fd.append('message', text);
    fd.append('thread_id', state.threadId);
    fd.append('user_id', state.userId);

    try {
      const r = await fetch(`${API}/chat`, { method: 'POST', body: fd });
      const data = await r.json();
      removeSpinner();

      if (data.status === 'paused') {
        state.isPaused = true;
        state.interruptMsg = data.interrupt_msg || '';
        applyServerState(data);
        renderPlan();
        switchTab('plan');
        renderInterrupt(data.interrupt_msg);
      } else if (data.status === 'error') {
        appendMsg('maya', '❌ ' + (data.message || 'Error.'));
      } else {
        applyServerState(data);
        renderPlan();
        if (data.response) appendMsg('maya', data.response);
        if (data.plot_path) renderChartMsg(data.plot_path);
        if (data.ml_report) { state.mlReport = data.ml_report; renderML(); }
        if (data.agent_log) data.agent_log.slice(state.logCount).forEach(renderEntry);
      }
    } catch (e) {
      removeSpinner();
      appendMsg('maya', explainFetchError(e));
    }
    setRunning(false);

  } else {
    appendMsg('maya', 'Please upload data files first.');
    setRunning(false);
  }
}

async function sendPlanFeedback(feedback) {
  const text = (feedback || '').trim();
  if (!text || !state.threadId || state.isRunning) return;

  setRunning(true);
  startSSE();

  const fd = new FormData();
  fd.append('thread_id', state.threadId);
  fd.append('user_id', state.userId);
  fd.append('feedback', text);

  try {
    const r = await fetch(`${API}/resume`, { method: 'POST', body: fd });
    const data = await r.json();
    if (data.status === 'error') {
      removeSpinner();
      toast(data.message || 'Plan update failed.', 'error');
      setRunning(false);
      return;
    }
    state.isPaused = false;
    state.interruptMsg = '';
    state.planDraft = '';
    const editor = document.getElementById('plan-feedback');
    if (editor) editor.value = '';
  } catch (e) {
    toast(explainFetchError(e), 'error');
    setRunning(false);
  }
}

function approvePlan() {
  sendPlanFeedback('approve');
}

function submitPlanEdits() {
  const editor = document.getElementById('plan-feedback');
  if (!editor) return;
  const text = editor.value.trim();
  if (!text) {
    toast('Describe the plan changes you want Maya to make.', 'info');
    return;
  }
  state.planDraft = text;
  sendPlanFeedback(text);
}

/* ── UI helpers ─────────────────────────────────────────────── */
function showChat() {
  document.getElementById('greeting-view').classList.add('hidden');
  const cv = document.getElementById('chat-view');
  cv.classList.remove('hidden');
  cv.style.display = 'flex';
}

function appendMsg(role, html) {
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = `msg ${role}`;
  d.innerHTML = `<div class="msg-by">${role === 'maya' ? 'Maya' : 'You'}</div>` +
    `<div class="msg-bub">${mdparse(html)}</div>`;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
  return d;
}

function renderInterrupt(msg) {
  if (!msg) return;
  state.interruptMsg = msg;
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'intr-card';
  d.innerHTML = `<div class="intr-lbl">Awaiting your input</div>` +
    `<div class="intr-body">${mdparse(msg)}</div>`;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
  // If plan JSON is embedded, parse it
  const m = msg.match(/```json\n([\s\S]+?)\n```/);
  if (m) { try { state.taskPlan = JSON.parse(m[1]); renderPlan(); } catch { } }
}

function removeSpinner() {
  document.querySelectorAll('#messages .spin').forEach(s => s.closest('.msg')?.remove());
}

/* ── Activity feed ──────────────────────────────────────────── */
function renderEntry(entry) {
  const feed = document.getElementById('activity-feed');
  const d = document.createElement('div');
  d.className = 'ae';
  const marks = { running: '…', success: '✓', error: '×', waiting: '·' };
  const icon = marks[entry.status] || '·';

  // Build a human-readable description of what the agent is doing
  const agentLabels = {
    Ingestion: '📥 Loading data',
    Merge: '🔀 Merging datasets',
    Cleaning: '🧹 Cleaning data',
    FeatureEngineer: '⚙️ Engineering features',
    ML: '🤖 Training model',
    Chat: '🔍 Analysing',
    Maya: '📋 Planning',
  };
  const agentLabel = agentLabels[entry.agent] || entry.agent;

  d.innerHTML =
    `<div class="ae-top">` +
    `<span class="ae-ag">${agentLabel}</span>` +
    `<span class="ae-act">${entry.action}</span>` +
    `<span class="ae-mark ${entry.status}">${icon}</span>` +
    `</div>` +
    (entry.detail ? `<div class="ae-det">${entry.detail}</div>` : '');

  feed.appendChild(d);
  feed.scrollTop = feed.scrollHeight;
  state.logCount++;
  document.getElementById('feed-count').textContent = state.logCount;
}

/* ── Progress bar ───────────────────────────────────────────── */
function setProgress(pct, stageName) {
  const fill = document.getElementById('progress-fill');
  const label = document.getElementById('progress-pct');
  const stageEl = document.getElementById('progress-stage');
  const cur = parseInt(fill.style.width) || 0;
  const next = Math.max(cur, pct);
  fill.style.width = next + '%';
  if (label) label.textContent = next > 0 ? next + '%' : '—';
  if (stageEl) stageEl.textContent = stageName || 'running';
}

/* ── Plan / ML / Data / Charts ──────────────────────────────── */
function renderPlan() {
  const c = document.getElementById('plan-container');
  const plan = Array.isArray(state.taskPlan) ? state.taskPlan : [];
  const resultsById = new Map((state.taskResults || []).map(r => [r.task_id, r]));
  const stage = state.activeAgent || state.pipelineStatus || 'idle';
  const currentStep = state.currentTaskTitle || (plan[state.currentTaskIndex] || {}).title || 'Awaiting plan review';
  const issueItems = (state.datasetIssues || []).slice(0, 8);
  const cleaningItems = (state.cleaningStrategy || []).slice(0, 8);
  const editorVisible = state.isPaused && /execution plan|approve|change task|skip task/i.test(state.interruptMsg || '');

  let h = `<div class="plan-shell">` +
    `<div class="plan-overview">` +
    `<div class="plan-card"><div class="plan-kicker">Stage</div><div class="plan-value">${stage || 'idle'}</div></div>` +
    `<div class="plan-card"><div class="plan-kicker">Tasks</div><div class="plan-value">${plan.length || 0}</div></div>` +
    `<div class="plan-card"><div class="plan-kicker">Current Step</div><div class="plan-value plan-value-sm">${currentStep}</div></div>` +
    `</div>`;

  if (issueItems.length) {
    h += `<div class="plan-block"><div class="plan-block-title">Dataset Problems Maya Found</div>`;
    issueItems.forEach(issue => {
      h += `<div class="plan-note">` +
        `<div class="plan-note-title">${issue.file || 'dataset'} · ${issue.title || 'Issue'}</div>` +
        `<div class="plan-note-body">${issue.detail || ''}</div>` +
        `</div>`;
    });
    h += `</div>`;
  }

  if (cleaningItems.length) {
    h += `<div class="plan-block"><div class="plan-block-title">Planned Fix Strategy</div>`;
    cleaningItems.forEach(item => {
      h += `<div class="plan-bullet">${item}</div>`;
    });
    h += `</div>`;
  }

  if (editorVisible) {
    h += `<div class="plan-block plan-edit">` +
      `<div class="plan-block-title">Edit The Plan Before Execution</div>` +
      `<div class="plan-note-body">Change any task, add steps, remove steps, or approve the current plan.</div>` +
      `<textarea id="plan-feedback" class="plan-textarea" placeholder="Example: change task 2 to clean duplicate order_id values before merging, then add an export step at the end.">${state.planDraft || ''}</textarea>` +
      `<div class="plan-actions">` +
      `<button class="btn btn-primary" onclick="approvePlan()">Approve Plan</button>` +
      `<button class="btn btn-ghost" onclick="submitPlanEdits()">Send Changes</button>` +
      `</div></div>`;
  }

  if (!plan.length) {
    h += `<div class="plan-empty">Plan is being prepared. Maya will show the tasks here before execution continues.</div>`;
  } else {
    h += `<div class="plan-block"><div class="plan-block-title">Execution Timeline</div>`;
    plan.forEach((task, i) => {
      const result = resultsById.get(task.task_id);
      const status = result?.status || (i < state.currentTaskIndex ? 'success' : i === state.currentTaskIndex ? (state.isPaused ? 'waiting' : 'running') : 'pending');
      const badgeLabel = status === 'success' ? 'Done' : status === 'error' ? 'Failed' : status === 'waiting' ? 'Waiting' : status === 'running' ? 'Running' : 'Pending';
      h += `<div class="plan-task ${status}">` +
        `<div class="plan-task-num">${String(i + 1).padStart(2, '0')}</div>` +
        `<div class="plan-task-body">` +
        `<div class="plan-task-top"><div class="plan-task-title">${task.title}</div><div class="plan-badge ${status}">${badgeLabel}</div></div>` +
        `<div class="plan-task-meta">${task.task_type || 'task'} · ${task.estimated_duration || 'n/a'} · ${task.estimated_complexity || 'n/a'}</div>` +
        `<div class="plan-task-desc">${task.description || ''}</div>` +
        `${result?.error ? `<div class="plan-task-error">${result.error}</div>` : ''}` +
        `</div></div>`;
    });
    h += `</div>`;
  }

  h += `</div>`;
  c.innerHTML = h;
  document.querySelector('[data-tab="plan"]').style.color = 'var(--accent)';
}

function renderML() {
  if (!state.mlReport) return;
  let report;
  try { report = typeof state.mlReport === 'string' ? JSON.parse(state.mlReport) : state.mlReport; }
  catch { return; }
  const c = document.getElementById('ml-container');
  const m = report.metrics || {};
  let h = `<div style="margin-bottom:24px;">` +
    `<div style="font-size:22px;font-weight:600;color:var(--ink);margin-bottom:4px;">${report.best_model || '—'}</div>` +
    `<div style="font-family:var(--mono);font-size:11px;color:var(--ink3);">${report.task_type || ''} · target: ${report.target_col || '—'}</div>` +
    `</div><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>`;
  Object.entries(m).forEach(([k, v]) => {
    h += `<tr><td>${k}</td><td>${typeof v === 'number' ? v.toFixed(4) : v}</td></tr>`;
  });
  h += '</tbody></table>';
  if (report.feature_importance) {
    const top = Object.entries(report.feature_importance).slice(0, 8);
    if (top.length) {
      h += `<div style="margin-top:24px;font-weight:500;margin-bottom:8px;">Top Features</div>`;
      const maxV = top[0][1];
      top.forEach(([feat, score]) => {
        const pct = Math.round((score / maxV) * 100);
        h += `<div style="margin:4px 0;font-size:12px;">` +
          `<div style="display:flex;justify-content:space-between;margin-bottom:2px;">` +
          `<span>${feat}</span><span style="color:var(--accent)">${score.toFixed(3)}</span></div>` +
          `<div style="background:var(--glass-border);border-radius:2px;height:3px;">` +
          `<div style="width:${pct}%;height:3px;background:var(--accent);border-radius:2px;"></div></div></div>`;
      });
    }
  }
  c.innerHTML = h;
  document.querySelector('[data-tab="ml"]').style.color = 'var(--accent)';
}

function renderCharts() {
  if (state.charts.length) renderChartMsg(state.charts[state.charts.length - 1]);
}

function renderChartMsg(path) {
  const fname = path.split('/').pop();
  const url = path.startsWith('storage/')
    ? `${API}/chart/${state.userId}/${fname}`
    : `${API}/${path}`;
  const msgs = document.getElementById('messages');
  const d = document.createElement('div');
  d.className = 'msg maya';
  d.innerHTML = `<div class="msg-by">Maya</div>` +
    `<div class="msg-bub" style="padding:0;overflow:hidden;border-radius:var(--radius-sm);">` +
    `<img src="${url}" style="width:100%;display:block;max-height:400px;object-fit:contain;" ` +
    `onerror="this.style.display='none'" /></div>`;
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}

async function loadStats() {
  try {
    const r = await fetch(`${API}/statistics?user_id=${state.userId}`);
    if (!r.ok) return;
    state.stats = await r.json();
    updateStats(state.stats);
    if (document.querySelector('[data-tab="data"]')?.classList.contains('active'))
      renderData(state.stats);
  } catch { }
}

function updateStats(d) {
  document.getElementById('stat-rows').textContent = fmtN(d.total_rows || 0);
  document.getElementById('stat-cols').textContent = d.total_columns || '—';
  document.getElementById('stat-mb').textContent = d.storage_mb?.toFixed(1) || '—';
}

function renderData(stats) {
  const c = document.getElementById('data-table-container');
  if (!stats?.columns) return;
  const cols = Object.entries(stats.columns);
  let h = `<div style="font-family:var(--mono);font-size:11px;color:var(--ink3);margin-bottom:16px;">` +
    `${fmtN(stats.total_rows)} observations · ${stats.total_columns} variables</div>` +
    `<table><thead><tr><th>Variable</th><th>Type</th><th>Missing</th><th>Unique</th><th>Mean</th></tr></thead><tbody>`;
  cols.slice(0, 80).forEach(([col, info]) => {
    h += `<tr><td>${col}</td><td style="color:var(--accent)">${info.dtype}</td>` +
      `<td>${info.missing_pct?.toFixed(1) ?? '0.0'}%</td><td>${fmtN(info.unique)}</td>` +
      `<td>${info.mean != null ? info.mean.toFixed(3) : '—'}</td></tr>`;
  });
  h += '</tbody></table>';
  c.innerHTML = h;
}

/* ── Tabs ────────────────────────────────────────────────────── */
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  const map = { chat: 'chat-view', data: 'data-view', plan: 'plan-view', ml: 'ml-view' };
  Object.values(map).forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.add('hidden'); el.style.display = ''; }
  });
  document.getElementById('greeting-view')?.classList.add('hidden');
  const t = document.getElementById(map[tab]);
  if (t) { t.classList.remove('hidden'); if (tab === 'chat') t.style.display = 'flex'; }
  if (tab === 'data' && state.stats) renderData(state.stats);
  if (tab === 'ml' && state.mlReport) renderML();
  if (tab === 'plan') renderPlan();
}

/* ── Input setup ────────────────────────────────────────────── */
function setupInput() {
  const ta = document.getElementById('msg-input');
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 100) + 'px';
  });
}

function useHint(el) {
  document.getElementById('msg-input').value = el.textContent;
  showChat(); switchTab('chat');
  document.getElementById('msg-input').focus();
}

function setRunning(v) {
  state.isRunning = v;
  document.getElementById('status-dot').className = 'status-dot' + (v ? ' busy' : '');
  document.getElementById('status-label').textContent = v ? 'processing' : state.isPaused ? 'waiting' : 'ready';
  const btn = document.getElementById('send-btn');
  if (btn) btn.disabled = v;
}

/* ── Markdown ────────────────────────────────────────────────── */
function mdparse(s) {
  if (!s) return '';
  return s
    .replace(/```(\w*)\n?([\s\S]*?)```/g, (_, l, c) => `<pre><code>${esc(c.trim())}</code></pre>`)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^- (.+)$/gm, '<div style="padding-left:12px;margin:2px 0">— $1</div>')
    .replace(/\n/g, '<br/>');
}

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function toast(msg, type = 'info') {
  const d = document.createElement('div');
  d.className = `toast ${type}`;
  d.textContent = msg;
  document.getElementById('toasts').appendChild(d);
  setTimeout(() => d.remove(), 3200);
}

function fmtN(n) {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return String(n);
}
