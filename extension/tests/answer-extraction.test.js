'use strict';

const assert = require('node:assert/strict');
const {
  extractLatestCompletedAssistantAnswer,
  getDiagnostics
} = require('../answer-extraction.js');

class FakeNode {
  constructor(tagName, text = '', position = 0, attributes = {}) {
    this.tagName = tagName.toUpperCase();
    this.directText = text;
    this.position = position;
    this.attributes = attributes;
    this.children = [];
    this.parentElement = null;
    this.isConnected = true;
    this.hidden = false;
    this.disabled = false;
  }

  append(child) {
    child.parentElement = this;
    this.children.push(child);
    return child;
  }

  get innerText() { return this.textContent; }
  get textContent() { return this.directText + this.children.map(child => child.textContent).join(''); }
  getAttribute(name) { return this.attributes[name] ?? null; }

  matches(selector) {
    return selector.split(',').some(part => this.matchesOne(part.trim()));
  }

  matchesOne(selector) {
    if (selector.includes(' ')) {
      const parts = selector.split(/\s+/);
      const last = parts.pop();
      if (!this.matchesOne(last)) return false;
      let ancestor = this.parentElement;
      while (ancestor) {
        if (ancestor.matchesOne(parts.join(' '))) return true;
        ancestor = ancestor.parentElement;
      }
      return false;
    }
    if (selector === '#tf-dock') return this.attributes.id === 'tf-dock';
    const parsed = selector.match(/^([a-z][\w-]*)?(?:\.([\w-]+))?(?:\[([^\]]+)\])?$/i);
    if (!parsed || (!parsed[1] && !parsed[2] && !parsed[3])) return false;
    const [, tag, className, expression] = parsed;
    if (tag && this.tagName !== tag.toUpperCase()) return false;
    if (className && !(this.attributes.class || '').split(/\s+/).includes(className)) return false;
    if (!expression) return true;
    const attribute = expression.match(/^([\w-]+)(~=|\*=|\^=|=)"([^"]*)"(?:\s+i)?$/i);
    if (!attribute) return false;
    const [, name, operator, expected] = attribute;
    const actual = this.attributes[name] || '';
    if (operator === '=') return actual === expected;
    if (operator === '~=') return actual.split(/\s+/).includes(expected);
    if (operator === '*=') return actual.toLowerCase().includes(expected.toLowerCase());
    if (operator === '^=') return actual.startsWith(expected);
    return false;
  }

  closest(selector) {
    let current = this;
    while (current) {
      if (current.matches(selector)) return current;
      current = current.parentElement;
    }
    return null;
  }

  contains(other) {
    return this.children.some(child => child === other || child.contains(other));
  }

  querySelectorAll(selector) {
    return this.children.flatMap(child => [
      ...(child.matches(selector) ? [child] : []),
      ...child.querySelectorAll(selector)
    ]);
  }

  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }

  cloneNode(deep) {
    const clone = new FakeNode(this.tagName, this.directText, this.position, { ...this.attributes });
    if (deep) this.children.forEach(child => clone.append(child.cloneNode(true)));
    return clone;
  }

  remove() {
    if (this.parentElement) {
      this.parentElement.children = this.parentElement.children.filter(child => child !== this);
      this.parentElement = null;
    }
  }

  compareDocumentPosition(other) {
    return this.position < other.position ? 4 : this.position > other.position ? 2 : 0;
  }
}

function fakeDocument(roots = []) {
  const documentRoots = roots.some(root => root.tagName === 'MAIN')
    ? roots
    : (() => {
      const main = new FakeNode('main');
      roots.forEach(root => main.append(root));
      return roots.length ? [main] : [];
    })();
  return {
    querySelector(selector) {
      return documentRoots.find(root => root.matches(selector)) ||
        documentRoots.flatMap(root => root.querySelectorAll(selector))[0] ||
        null;
    },
    querySelectorAll(selector) {
      return documentRoots.flatMap(root => [
        ...(root.matches(selector) ? [root] : []),
        ...root.querySelectorAll(selector)
      ]);
    }
  };
}

function createTurn(position, { role, text, markdown = false, userBubble = false, turnId = null }) {
  const turn = new FakeNode('section', '', position, turnId ? { 'data-testid': turnId } : {});
  const roleAttrs = role ? { 'data-message-author-role': role } : {};
  const message = turn.append(new FakeNode('div', '', position + 1, {
    class: 'min-h-8 text-message relative',
    ...roleAttrs
  }));
  if (userBubble) {
    message.append(new FakeNode('div', text, position + 2, { class: 'user-message-bubble-color' }));
  } else if (markdown) {
    message.append(new FakeNode('div', text, position + 2, { class: 'markdown prose' }));
  } else {
    message.append(new FakeNode('div', text, position + 2));
  }
  if (role === 'assistant' || (!role && markdown)) {
    turn.append(new FakeNode('button', '', position + 3, {
      'aria-label': 'Copy response',
      'data-testid': 'copy-turn-action-button'
    }));
    turn.append(new FakeNode('button', '', position + 4, {
      'aria-label': 'Rate response',
      'data-testid': 'feedback-turn-action-button'
    }));
  } else if (role === 'user' || userBubble) {
    turn.append(new FakeNode('button', '', position + 3, { 'aria-label': 'Copy message' }));
  }
  return { turn, message };
}

function createActionLinkedTurn(position, text, { user = false } = {}) {
  const turn = new FakeNode('section', '', position);
  const frame = turn.append(new FakeNode('div', '', position + 1, { class: 'turn-layout' }));
  const content = frame.append(new FakeNode('div', '', position + 2, { class: 'message-content' }));
  const message = content.append(new FakeNode('div', '', position + 3, { class: 'message-block' }));
  message.append(new FakeNode('div', text, position + 4, {
    class: user ? 'user-message-bubble-color' : 'markdown prose'
  }));
  const actionRow = frame.append(new FakeNode('div', '', position + 5, { class: 'action-row' }));
  const toolbar = actionRow.append(new FakeNode('div', '', position + 6, { role: 'group' }));
  toolbar.append(new FakeNode('button', '', position + 7, {
    'aria-label': user ? 'Copy message' : 'Copy response',
    'data-testid': 'copy-turn-action-button'
  }));
  if (!user) {
    toolbar.append(new FakeNode('button', '', position + 8, {
      'aria-label': 'Rate response',
      'data-testid': 'feedback-turn-action-button'
    }));
  }
  return { turn, frame, message };
}

const answer = 'The Eiffel Tower is located in Paris, France, in Europe.';
const firstUser = createTurn(1, {
  role: 'user',
  text: 'Where is the Eiffel Tower located?',
  userBubble: true,
  turnId: 'conversation-turn-1'
});
const firstAssistant = createTurn(5, {
  role: 'assistant',
  text: 'An earlier assistant response',
  markdown: true,
  turnId: 'conversation-turn-2'
});
const latestUser = createTurn(9, {
  role: 'user',
  text: 'And what is its height?',
  userBubble: true,
  turnId: 'conversation-turn-3'
});
const latestAssistant = createTurn(13, {
  role: 'assistant',
  text: answer,
  markdown: true,
  turnId: 'conversation-turn-4'
});
const main = new FakeNode('main', '', 0);
[firstUser, firstAssistant, latestUser, latestAssistant].forEach(pair => main.append(pair.turn));
const extracted = extractLatestCompletedAssistantAnswer(fakeDocument([main]), 'chatgpt.com');
assert.equal(extracted.text, answer, 'chooses latest assistant markdown content rather than latest user prompt');
assert.equal(extracted.diagnostics.assistantCount, 2);
assert.equal(extracted.diagnostics.userCount, 2);
assert.equal(extracted.diagnostics.candidateCount, 2);

const noRoleNoTestId = new FakeNode('main', '', 20);
const rolelessAssistant = createTurn(21, {
  text: answer,
  markdown: true
});
noRoleNoTestId.append(rolelessAssistant.turn);
const fallbackResult = extractLatestCompletedAssistantAnswer(fakeDocument([noRoleNoTestId]), 'chatgpt.com');
assert.equal(fallbackResult.text, answer, 'uses actual text-message/markdown/response-action structure without role or test IDs');
assert.equal(fallbackResult.diagnostics.assistantCount, 0);
assert.equal(fallbackResult.diagnostics.fallbackUsed, true);
assert.equal(fallbackResult.diagnostics.fallbackCandidateCount, 1);

const linkedConversation = new FakeNode('main', '', 32);
linkedConversation.append(createActionLinkedTurn(33, 'User question one', { user: true }).turn);
linkedConversation.append(createActionLinkedTurn(43, 'First assistant response').turn);
linkedConversation.append(createActionLinkedTurn(53, 'User question two', { user: true }).turn);
linkedConversation.append(createActionLinkedTurn(63, answer).turn);
const linkedResult = extractLatestCompletedAssistantAnswer(fakeDocument([linkedConversation]), 'chatgpt.com');
assert.equal(linkedResult.text, answer, 'associates nested response action buttons with their own sibling text block');
assert.match(linkedResult.diagnostics.selectedSource, /adjacent response content/);

const rolelessPrevious = createTurn(23, { text: 'Previous answer', markdown: true });
const rolelessEmptyLatest = createTurn(24, { text: '', markdown: true });
const rolelessEmptyMain = new FakeNode('main', '', 22);
rolelessEmptyMain.append(rolelessPrevious.turn);
rolelessEmptyMain.append(rolelessEmptyLatest.turn);
assert.equal(
  extractLatestCompletedAssistantAnswer(fakeDocument([rolelessEmptyMain]), 'chatgpt.com').text,
  '',
  'does not fall back to a previous response when the latest identifiable response is empty'
);

const rolelessUserTurn = new FakeNode('section', '', 25);
const rolelessUser = rolelessUserTurn.append(new FakeNode('div', '', 26, { class: 'text-message' }));
rolelessUser.append(new FakeNode('div', answer, 27, {
  class: 'markdown user-message-bubble-color'
}));
rolelessUserTurn.append(new FakeNode('button', '', 28, {
  'aria-label': 'Copy message',
  'data-testid': 'copy-turn-action-button'
}));
const rolelessUserMain = new FakeNode('main', '', 24);
rolelessUserMain.append(rolelessUserTurn);
assert.equal(
  extractLatestCompletedAssistantAnswer(fakeDocument([rolelessUserMain]), 'chatgpt.com').text,
  '',
  'does not mistake a formatted user prompt and its copy control for an assistant response'
);

const emptyAssistant = createTurn(30, {
  role: 'assistant',
  text: '',
  markdown: true,
  turnId: 'conversation-turn-5'
});
const emptyResult = extractLatestCompletedAssistantAnswer(fakeDocument([emptyAssistant.turn]), 'chatgpt.com');
assert.equal(emptyResult.text, '', 'does not produce an answer from an empty response');
assert.match(emptyResult.diagnostics.rejections.join(' '), /no extractable text/i);
assert.equal(extractLatestCompletedAssistantAnswer(fakeDocument([]), 'chatgpt.com').text, '');

const streaming = createTurn(40, {
  role: 'assistant',
  text: answer,
  markdown: true,
  turnId: 'conversation-turn-6'
});
streaming.message.attributes['data-is-streaming'] = 'true';
assert.match(
  extractLatestCompletedAssistantAnswer(fakeDocument([streaming.turn]), 'chatgpt.com').reason,
  /still generating/
);

const unrelatedStreaming = new FakeNode('div', '', 50, { 'aria-busy': 'true' });
const completed = createTurn(51, {
  role: 'assistant',
  text: answer,
  markdown: true,
  turnId: 'conversation-turn-7'
});
const unrelatedPage = new FakeNode('main', '', 50);
unrelatedPage.append(unrelatedStreaming);
unrelatedPage.append(completed.turn);
assert.equal(
  extractLatestCompletedAssistantAnswer(fakeDocument([unrelatedPage]), 'chatgpt.com').text,
  answer,
  'does not reject a completed response due to unrelated busy content elsewhere'
);

const stopButton = completed.turn.append(new FakeNode('button', '', 60, { 'data-testid': 'stop-button' }));
assert.match(
  extractLatestCompletedAssistantAnswer(fakeDocument([completed.turn]), 'chatgpt.com').reason,
  /still generating/
);

const stopElsewhere = new FakeNode('main', '', 61);
stopElsewhere.append(createTurn(62, {
  role: 'assistant',
  text: answer,
  markdown: true
}).turn);
stopElsewhere.append(new FakeNode('button', '', 70, {
  'data-testid': 'stop-button'
}));
assert.match(
  extractLatestCompletedAssistantAnswer(fakeDocument([stopElsewhere]), 'chatgpt.com').reason,
  /still generating/,
  'a visible ChatGPT stop-generation control signals an active response even without role attributes'
);

const dock = new FakeNode('section', '', 70, { id: 'tf-dock' });
const dockAssistant = dock.append(new FakeNode('div', answer, 71, {
  'data-message-author-role': 'assistant'
}));
assert.equal(dockAssistant.textContent, answer);
assert.equal(extractLatestCompletedAssistantAnswer(fakeDocument([dock]), 'chatgpt.com').text, '');

const sidebar = new FakeNode('nav', '', 80);
sidebar.append(createTurn(81, {
  role: 'assistant',
  text: answer,
  markdown: true,
  turnId: 'conversation-turn-8'
}).turn);
assert.equal(extractLatestCompletedAssistantAnswer(fakeDocument([sidebar]), 'chatgpt.com').text, '');

const missingExtractorDiagnostics = getDiagnostics(fakeDocument([main]), false, true);
assert.equal(missingExtractorDiagnostics.contentScriptRunning, true);
assert.equal(missingExtractorDiagnostics.extractorAvailable, false);
assert.equal(missingExtractorDiagnostics.assistantCount, 2);

const gemini = new FakeNode('model-response', 'Gemini final response', 90);
assert.equal(
  extractLatestCompletedAssistantAnswer(fakeDocument([gemini]), 'gemini.google.com').text,
  'Gemini final response'
);

console.log('Automatic answer extraction tests passed.');
