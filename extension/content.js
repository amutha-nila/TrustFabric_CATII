(() => {
  'use strict';
  if (window.__trustfabricLoaded) return;
  window.__trustfabricLoaded = true;
  const API_DEFAULT = 'http://127.0.0.1:8000';
  let settings = { apiBase: API_DEFAULT, extensionApiBase: 'http://127.0.0.1:8765', apiKey: '', autoPrivacyWarnings: true };
  let verificationInProgress = false;
  let capturedAnswer = '';
  let capturedDiagnostics = null;
  const $ = (s, root = document) => root.querySelector(s);

  chrome.storage.sync.get(['apiBase', 'extensionApiBase', 'autoPrivacyWarnings'], saved => {
    const syncError = chrome.runtime.lastError;
    chrome.storage.local.get(['apiKey'], local => {
      const localError = chrome.runtime.lastError;
      settings = {
        ...settings,
        ...(syncError ? {} : saved),
        apiKey: localError ? '' : (local.apiKey || '')
      };
      mountDock(); observeUploads();
      if (syncError || localError) {
        showResult('Settings unavailable', 'Could not load all extension settings. Open TrustFabric settings and check Chrome storage permissions.', 'error');
      }
    });
  });

  function h(tag, attrs = {}, text = '') {
    const el = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => k === 'class' ? el.className = v : k.startsWith('on') ? null : el.setAttribute(k, v));
    if (text) el.textContent = text;
    return el;
  }
  function mountDock() {
    if ($('#tf-dock')) return;
    const dock = h('section', { id: 'tf-dock', 'aria-label': 'TrustFabric AI trust tools' });
    const header = h('div', { class: 'tf-header' });
    header.append(h('span', { class: 'tf-brand' }, '◈ TrustFabric'), h('span', { class: 'tf-subtitle' }, 'AI trust & privacy'));
    const row = h('div', { class: 'tf-actions' });
    const privacy = h('button', { class: 'tf-btn' }, 'Check privacy');
    privacy.addEventListener('click', () => checkPrivacy());
    const media = h('button', { class: 'tf-btn' }, 'Check image/video');
    media.addEventListener('click', () => $('#tf-media-input').click());
    const claimInput = h('textarea', {
      id: 'tf-claim-input',
      'aria-label': 'Optional manual claim',
      placeholder: 'Optional: enter a claim if automatic answer detection is unavailable.',
      rows: '3'
    });
    const verify = h('button', { class: 'tf-btn tf-dock-verify' }, 'Verify answer');
    verify.addEventListener('click', () => {
      const extractor = window.TrustFabricAnswerExtraction;
      const diagnostics = getExtractionDiagnostics(extractor);
      if (!extractor || typeof extractor.extractLatestCompletedAssistantAnswer !== 'function') {
        showExtractionDiagnostics('Answer extractor is not initialized.', diagnostics);
        return;
      }
      let answer;
      try {
        answer = extractor.extractLatestCompletedAssistantAnswer(document, location.hostname);
      } catch (error) {
        const detail = { ...diagnostics, rejections: [`Extractor threw: ${error.message || String(error)}`] };
        showExtractionDiagnostics('Answer extraction failed before verification.', detail);
        return;
      }
      const detail = { ...diagnostics, ...(answer.diagnostics || {}), contentScriptRunning: true, extractorAvailable: true };
      if (!answer.text) {
        showExtractionDiagnostics(answer.reason, detail);
        return;
      }
      verifyAnswer(answer.text, verify, detail);
    });
    const file = h('input', { id: 'tf-media-input', type: 'file', accept: 'image/*,video/mp4,video/webm,video/quicktime', hidden: 'hidden' });
    file.addEventListener('change', async () => { if (file.files?.[0]) await checkMedia(file.files[0]); file.value = ''; });
    row.append(privacy, media, file);
    const verifyForm = h('div', { class: 'tf-verify-form' });
    verifyForm.append(verify);
    const manual = h('details', { class: 'tf-manual-entry' });
    const manualVerify = h('button', { class: 'tf-btn' }, 'Verify entered claim');
    manualVerify.addEventListener('click', () => {
      const text = claimInput.value.trim();
      if (!text) {
        showResult('Claim needed', 'Enter a claim here, or use Verify answer to check the latest completed assistant response.', 'warning');
        claimInput.focus();
        return;
      }
      verifyAnswer(text, manualVerify);
    });
    manual.append(h('summary', {}, 'Optional: enter a claim manually'), claimInput, manualVerify);
    const results = h('div', { id: 'tf-results', role: 'status', 'aria-live': 'polite' }, 'Verify an AI answer or claim, check a platform policy, or analyze selected media.');
    dock.append(header, row, verifyForm, manual, results); document.body.append(dock);
  }
  function observeUploads() {
    if (!settings.autoPrivacyWarnings) return;
    document.addEventListener('change', async ev => {
      const input = ev.target;
      if (!(input instanceof HTMLInputElement) || input.type !== 'file' || !input.files?.length) return;
      const file = input.files[0];
      const findings = localFileWarnings(file);
      if (findings.length) showResult('Pre-upload privacy warning', findings.map(x => `• ${x}`).join('\n') + '\n\nThis is a local heuristic warning, not proof of misuse. TrustFabric does not block or control the provider upload.', 'warning');
    }, true);
  }
  function localFileWarnings(file) {
    const out = [];
    if (file.size > 20 * 1024 * 1024) out.push('Large file (' + (file.size / 1024 / 1024).toFixed(1) + ' MB): check whether this upload is necessary.');
    if (/identity|passport|aadhaar|license|medical|bank|statement/i.test(file.name)) out.push('Filename may indicate identity, medical, or financial information. Check the file before uploading.');
    if (file.type.startsWith('image/')) out.push('Personal images may contain faces and location metadata. Consider removing metadata or using a less sensitive copy.');
    if (file.type.startsWith('video/')) out.push('Videos may reveal faces, voices, locations, and bystanders. Review the content before uploading.');
    return out;
  }
  async function request(path, options = {}, timeoutMs = 90000, apiBase = settings.apiBase) {
    const headers = { ...(options.headers || {}) };
    if (settings.apiKey) headers['X-TrustFabric-Key'] = settings.apiKey;
    const res = await fetch(apiBase.replace(/\/$/, '') + path, { ...options, headers, signal: AbortSignal.timeout(timeoutMs) });
    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data?.detail;
      const message = typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map(item => item.msg || 'Invalid request').join('; ')
          : `Backend error (${res.status})`;
      throw new Error(message);
    }
    if (!data) throw new Error('The backend returned an invalid JSON response.');
    return data;
  }
  function getExtractionDiagnostics(extractor) {
    if (extractor && typeof extractor.getDiagnostics === 'function') {
      return extractor.getDiagnostics(document, true, window.__trustfabricLoaded === true);
    }
    const count = selector => {
      try { return document.querySelectorAll(selector).length; }
      catch { return 0; }
    };
    return {
      contentScriptRunning: window.__trustfabricLoaded === true,
      extractorAvailable: Boolean(extractor && typeof extractor.extractLatestCompletedAssistantAnswer === 'function'),
      assistantCount: count('[data-message-author-role="assistant"]'),
      userCount: count('[data-message-author-role="user"]'),
      otherSelectorCounts: {
        '[data-testid^="conversation-turn-"]': count('main [data-testid^="conversation-turn-"]'),
        'main .text-message': count('main .text-message'),
        'main [class*="markdown"]': count('main [class*="markdown"]'),
        'assistant response actions': count('main [data-testid*="feedback-turn-action-button"], main button[aria-label*="Copy response" i]')
      },
      candidateCount: 0,
      fallbackCandidateCount: 0,
      fallbackUsed: false,
      selectedSource: '',
      rejections: ['Answer extractor API is missing or incomplete.']
    };
  }
  function showExtractionDiagnostics(reason, diagnostics) {
    const lines = [
      `Content script running: ${diagnostics.contentScriptRunning ? 'yes' : 'no'}`,
      `Answer extractor available: ${diagnostics.extractorAvailable ? 'yes' : 'no'}`,
      `Assistant-role elements: ${diagnostics.assistantCount ?? 'unavailable'}`,
      `User-role elements: ${diagnostics.userCount ?? 'unavailable'}`,
      ...Object.entries(diagnostics.otherSelectorCounts || {}).map(([selector, count]) => `${selector}: ${count}`),
      `Candidates considered: ${diagnostics.candidateCount ?? 0}`,
      `Fallback candidates: ${diagnostics.fallbackCandidateCount ?? 0}`,
      `Fallback used: ${diagnostics.fallbackUsed ? 'yes' : 'no'}`,
      ...(diagnostics.selectedSource ? [`Selected source: ${diagnostics.selectedSource}`] : []),
      ...(diagnostics.rejections || []).map(entry => `Candidate rejected: ${entry}`)
    ];
    showResult('Answer extraction diagnostics', `${reason}\n\n${lines.join('\n')}`, 'warning');
  }
  async function verifyAnswer(text, target, diagnostics = null) {
    if (verificationInProgress) return;
    verificationInProgress = true;
    capturedAnswer = text;
    capturedDiagnostics = diagnostics;
    const button = target;
    if (button) { button.disabled = true; button.textContent = 'Checking…'; }
    showCapturedPreview('Verifying captured answer', 'Extracting factual claims and checking available evidence…', 'working');
    try {
      const data = await request('/api/v1/verify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) }, 120000);
      showVerification(data);
    } catch (e) {
      const reason = e.name === 'TimeoutError'
        ? 'Verification timed out after two minutes. A first-time NLI model download may still be in progress; check the backend logs and retry.'
        : e instanceof TypeError
          ? `Could not connect to the verification API at ${settings.apiBase}. Check that the backend is running and that the URL is correct.`
          : e.message;
      showCapturedPreview('Could not verify', `${reason}\nBackend: ${settings.apiBase}\nCheck that the TrustFabric backend is running and settings are correct.`, 'error');
    }
    finally {
      verificationInProgress = false;
      if (button) {
        button.disabled = false;
        button.textContent = button.classList.contains('tf-dock-verify') ? 'Verify answer' : 'Verify entered claim';
      }
    }
  }
  function humanVerdict(status) {
    return ({
      supported: 'Supported',
      contradicted: 'Contradicted',
      mixed_or_conflicting_evidence: 'Mixed evidence',
      insufficient_evidence: 'Unable to verify'
    })[status] || 'Unable to verify';
  }
  function verdictIcon(status) {
    return ({ supported: '✓', contradicted: '⚠', mixed_or_conflicting_evidence: '⚠' })[status] || '?';
  }
  function retrievalLabel(status) {
    return ({
      available: 'Search completed; evidence was retrieved.',
      no_results: 'Search completed successfully but returned no results.',
      unavailable: 'Search provider unavailable; no evidence was retrieved.',
      partial: 'Some searches failed; available results are shown.',
      not_needed: 'No factual claims required a search.'
    })[status] || 'Search status was not reported.';
  }
  function claimSearchLabel(status) {
    return ({
      available: 'Evidence retrieved.',
      no_results: 'Search completed successfully but returned no results.',
      unavailable: 'Search failed before evidence could be retrieved.',
      irrelevant: 'Search succeeded, but returned results were not relevant enough.',
      inconclusive: 'Relevant results were found, but evidence was insufficient or conflicting.',
      not_needed: 'No evidence search was needed.'
    })[status] || 'Claim search status was not reported.';
  }
  function showVerification(data) {
    const root = $('#tf-results');
    if (!root) return;
    root.replaceChildren();
    const claims = data.claims || [];
    const counts = claims.reduce((totals, claim) => {
      const verdict = humanVerdict(claim.status);
      totals[verdict] = (totals[verdict] || 0) + 1;
      return totals;
    }, {});
    const summary = Object.entries(counts).map(([verdict, count]) => `${count} ${verdict.toLowerCase()}`).join(', ');
    root.append(h(
      'div',
      { class: 'tf-result-title tf-' + (data.overall_status || 'insufficient_evidence') },
      `Overall: ${humanVerdict(data.overall_status)}`
    ));
    root.append(h('p', { class: 'tf-result-body' }, summary || 'No factual claims were identified.'));
    root.append(h('p', { class: 'tf-evidence-status' }, retrievalLabel(data.evidence_retrieval_status)));
    appendCapturedPreview(root);
    for (const claim of claims) {
      const section = h('details', { class: 'tf-claim' });
      section.append(h(
        'summary',
        { class: 'tf-' + claim.status },
        `${verdictIcon(claim.status)} ${humanVerdict(claim.status)} — ${claim.claim}`
      ));
      section.append(
        h('p', {}, claim.explanation || 'No further explanation was provided.'),
        h('p', { class: 'tf-evidence-status' }, claimSearchLabel(claim.evidence_status))
      );
      for (const item of claim.evidence || []) {
        let url;
        try { url = new URL(item.url); } catch { continue; }
        if (!['http:', 'https:'].includes(url.protocol)) continue;
        const source = h('div', { class: 'tf-source' });
        const link = h('a', { href: url.href, target: '_blank', rel: 'noopener noreferrer' }, item.title || 'Open source');
        source.append(link, h('div', { class: 'tf-source-url' }, url.href));
        if (item.snippet) source.append(h('p', {}, item.snippet));
        section.append(source);
      }
      root.append(section);
    }
    root.append(h('p', { class: 'tf-result-body' }, data.message || 'Automated evidence triage only; inspect sources independently.'));
    const close = h('button', { class: 'tf-close', 'aria-label': 'Clear result' }, 'Clear');
    close.addEventListener('click', () => {
      root.replaceChildren(h('div', {}, 'Ready. Results are not saved by this extension.'));
    });
    root.append(close);
  }
  async function checkMedia(file) {
    if (file.size > 25 * 1024 * 1024) { showResult('File too large', 'Maximum supported upload size is 25 MB. For privacy, use a smaller copy if possible.', 'warning'); return; }
    const form = new FormData(); form.append('file', file, file.name);
    showResult('Analysing media', 'Processing selected media. It will be sent to your configured TrustFabric backend for analysis.', 'working');
    try {
      const data = await request('/api/v1/media/analyze', { method: 'POST', body: form }, 90000, settings.extensionApiBase);
      showResult('Media analysis', `${data.summary}\n\nType: ${data.media_type}\nResult: ${data.label}\nScore: ${data.score == null ? 'Unavailable' : (data.score * 100).toFixed(1) + '%'}\nFrames analysed: ${data.frames_analyzed ?? 'N/A'}\n\n${data.limitations}`, data.label === 'likely_ai_generated' || data.label === 'potentially_manipulated' ? 'warning' : 'info');
    } catch (e) {
      const nextStep = e.message.includes('(404)')
        ? '\nMedia analysis is provided by the separate extension_backend on port 8765. Start it and set the backend URL to http://127.0.0.1:8765.'
        : '';
      showResult('Media analysis unavailable', `${e.message}${nextStep}`, 'error');
    }
  }
  async function checkPrivacy() {
    showResult('Checking platform privacy', 'Reviewing TrustFabric’s maintained public policy summary and documented risk indicators…', 'working');
    try {
      const host = location.hostname;
      const data = await request('/api/v1/privacy/platform?host=' + encodeURIComponent(host), {}, 15000, settings.extensionApiBase);
      const lines = [data.summary, '', ...(data.findings || []).map(f => `${f.level.toUpperCase()}: ${f.topic}\n${f.detail}\nSource: ${f.source}`), '', 'This report summarises public documentation; it cannot inspect internal provider systems or prove misuse.'];
      showResult('Platform privacy report', lines.join('\n'), data.risk_level || 'info');
    } catch (e) {
      const nextStep = e.message.includes('(404)')
        ? '\nPlatform privacy reports are provided by the separate extension_backend on port 8765. Start it and set the backend URL to http://127.0.0.1:8765.'
        : '';
      showResult('Privacy report unavailable', `${e.message}${nextStep}`, 'error');
    }
  }
  function showResult(title, body, kind = 'info') {
    const root = $('#tf-results'); if (!root) return;
    root.replaceChildren();
    const t = h('div', { class: 'tf-result-title tf-' + kind }, title);
    const p = h('pre', { class: 'tf-result-body' }, body);
    const close = h('button', { class: 'tf-close', 'aria-label': 'Clear result' }, 'Clear'); close.addEventListener('click', () => { root.replaceChildren(h('div', {}, 'Ready. Results are not saved by this extension.')); });
    root.append(t, p, close);
  }
  function appendCapturedPreview(root) {
    if (!capturedAnswer) return;
    const preview = h('details', { class: 'tf-answer-preview', open: 'open' });
    preview.append(h('summary', {}, 'Captured answer preview'));
    preview.append(h('pre', { class: 'tf-result-body' }, capturedAnswer));
    if (capturedDiagnostics) {
      const details = h('details', { class: 'tf-answer-diagnostics' });
      details.append(h('summary', {}, 'Local extraction diagnostics'));
      details.append(h('pre', { class: 'tf-result-body' }, extractionDiagnosticText(capturedDiagnostics)));
      preview.append(details);
    }
    root.append(preview);
  }
  function extractionDiagnosticText(diagnostics) {
    return [
      `Content script running: ${diagnostics.contentScriptRunning ? 'yes' : 'no'}`,
      `Answer extractor available: ${diagnostics.extractorAvailable ? 'yes' : 'no'}`,
      `Assistant-role elements: ${diagnostics.assistantCount ?? 'unavailable'}`,
      `User-role elements: ${diagnostics.userCount ?? 'unavailable'}`,
      ...Object.entries(diagnostics.otherSelectorCounts || {}).map(([selector, count]) => `${selector}: ${count}`),
      `Candidates considered: ${diagnostics.candidateCount ?? 0}`,
      `Fallback candidates: ${diagnostics.fallbackCandidateCount ?? 0}`,
      `Fallback used: ${diagnostics.fallbackUsed ? 'yes' : 'no'}`,
      ...(diagnostics.selectedSource ? [`Selected source: ${diagnostics.selectedSource}`] : []),
      ...(diagnostics.rejections || []).map(entry => `Candidate rejected: ${entry}`)
    ].join('\n');
  }
  function showCapturedPreview(title, body, kind) {
    const root = $('#tf-results'); if (!root) return;
    root.replaceChildren(
      h('div', { class: 'tf-result-title tf-' + kind }, title),
      h('p', { class: 'tf-result-body' }, body)
    );
    appendCapturedPreview(root);
  }
})();
