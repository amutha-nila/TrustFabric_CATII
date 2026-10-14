'use strict';

const DEFAULTS = {
  apiBase: 'http://127.0.0.1:8000',
  extensionApiBase: 'http://127.0.0.1:8765',
  autoPrivacyWarnings: true
};
const apiBaseInput = document.querySelector('#apiBase');
const extensionApiBaseInput = document.querySelector('#extensionApiBase');
const apiKeyInput = document.querySelector('#apiKey');
const autoPrivacyWarningsInput = document.querySelector('#autoPrivacyWarnings');
const status = document.querySelector('#status');

chrome.storage.sync.get(['apiBase', 'extensionApiBase', 'autoPrivacyWarnings', 'apiKey'], synced => {
  const syncError = chrome.runtime.lastError;
  chrome.storage.local.get(['apiKey'], local => {
    const localError = chrome.runtime.lastError;
    if (syncError || localError) {
      status.textContent = `Could not load settings: ${(syncError || localError).message}`;
      return;
    }
    apiBaseInput.value = synced.apiBase || DEFAULTS.apiBase;
    extensionApiBaseInput.value = synced.extensionApiBase || DEFAULTS.extensionApiBase;
    autoPrivacyWarningsInput.checked = synced.autoPrivacyWarnings ?? DEFAULTS.autoPrivacyWarnings;
    apiKeyInput.value = local.apiKey || synced.apiKey || '';
    if (!local.apiKey && synced.apiKey) {
      chrome.storage.local.set({ apiKey: synced.apiKey }, () => {
        if (chrome.runtime.lastError) {
          status.textContent = `Could not migrate the API key to local storage: ${chrome.runtime.lastError.message}`;
          return;
        }
        chrome.storage.sync.remove('apiKey', () => {
          if (chrome.runtime.lastError) {
            status.textContent = `The API key was saved locally, but could not be removed from sync storage: ${chrome.runtime.lastError.message}`;
          }
        });
      });
    }
  });
});

document.querySelector('#save').addEventListener('click', () => {
  const apiBase = apiBaseInput.value.trim().replace(/\/$/, '');
  const extensionApiBase = extensionApiBaseInput.value.trim().replace(/\/$/, '');
  for (const [label, value] of [
    ['claim-verification API', apiBase],
    ['privacy/media backend', extensionApiBase]
  ]) {
    try {
      const url = new URL(value);
      if (!['http:', 'https:'].includes(url.protocol)) throw new Error();
    } catch {
      status.textContent = `Enter a valid http:// or https:// URL for the ${label}.`;
      return;
    }
  }

  let pending = 2;
  const errors = [];
  const onSaved = () => {
    if (chrome.runtime.lastError) errors.push(chrome.runtime.lastError.message);
    pending -= 1;
    if (pending === 0) {
      status.textContent = errors.length
        ? `Could not save all settings: ${errors.join('; ')}`
        : 'Settings saved. Refresh supported AI tabs to apply changes.';
    }
  };

  chrome.storage.sync.set({
    apiBase,
    extensionApiBase,
    autoPrivacyWarnings: autoPrivacyWarningsInput.checked
  }, onSaved);
  chrome.storage.local.set({
    apiKey: apiKeyInput.value.trim()
  }, onSaved);
});
