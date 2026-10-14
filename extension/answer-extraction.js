(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TrustFabricAnswerExtraction = api;
}(globalThis, function () {
  'use strict';

  const ASSISTANT_SELECTOR = '[data-message-author-role="assistant"]';
  const USER_SELECTOR = '[data-message-author-role="user"]';
  const TURN_SELECTOR = '[data-testid^="conversation-turn-"]';
  const TEXT_MESSAGE_SELECTOR = '.text-message';
  const MARKDOWN_SELECTOR = '[class~="markdown"], [class*="markdown"]';
  const GEMINI_SELECTORS = [
    'model-response',
    '[data-test-id="model-response"]',
    '.model-response-text',
    'message-content'
  ];
  const ASSISTANT_ACTION_SELECTOR = [
    'button[aria-label*="Copy response" i]',
    'button[aria-label*="Good response" i]',
    'button[aria-label*="Bad response" i]',
    'button[aria-label*="Rate response" i]',
    'button[aria-label*="Read aloud response" i]',
    '[data-testid*="feedback-turn-action-button"]',
    '[data-testid*="good-response-turn-action-button"]',
    '[data-testid*="bad-response-turn-action-button"]',
    '[data-testid*="read-aloud-turn-action-button"]'
  ].join(',');
  const STOP_GENERATION_SELECTORS = [
    'button[data-testid="stop-button"]',
    'button[aria-label*="Stop generating" i]',
    'button[title*="Stop generating" i]',
    'button[aria-label*="Stop streaming" i]',
    'button[title*="Stop streaming" i]'
  ];
  const MESSAGE_STREAMING_SELECTOR = '[data-is-streaming="true"], [aria-busy="true"]';
  const EXCLUDED_CONTENT_SELECTOR = '#tf-dock, nav, [role="navigation"], button, [role="button"]';
  const EXCLUDED_ANCESTOR_SELECTOR = '#tf-dock, nav, [role="navigation"]';
  const EMPTY_RESULT_REASON = 'No completed assistant answer was found on this page. Wait for an answer or use the optional manual entry.';

  function extractLatestCompletedAssistantAnswer(doc, hostname) {
    const isGemini = hostname === 'gemini.google.com';
    const diagnostics = collectCounts(doc);
    const candidates = isGemini
      ? collectGeminiCandidates(doc)
      : collectChatgptCandidates(doc, diagnostics);
    candidates.sort((left, right) => {
      const position = left.node.compareDocumentPosition(right.node);
      if (position & 4) return -1;
      if (position & 2) return 1;
      return 0;
    });
    diagnostics.candidateCount = candidates.length;

    if (!candidates.length) {
      if (!diagnostics.rejections.length) {
        diagnostics.rejections.push(diagnostics.assistantCount
          ? 'Assistant-role elements were excluded or contained no usable message candidate.'
          : 'No assistant-role element or unambiguous ChatGPT message/turn structure was identified.');
      }
      return result('', EMPTY_RESULT_REASON, diagnostics);
    }

    const latest = candidates.at(-1);
    const conversation = doc.querySelector('main') || doc.querySelector('[role="main"]');
    if (hasStreamingSignal(latest.node) ||
        (latest.scope !== latest.node && hasStreamingSignal(latest.scope)) ||
        hasActiveStopControl(latest.scope || latest.node) ||
        (!isGemini && conversation && hasActiveStopControl(conversation))) {
      diagnostics.rejections.push('An active stop-generation control or a streaming/busy marker indicates the latest response is incomplete.');
      return result('', 'The assistant is still generating a response. Wait for it to finish, then retry.', diagnostics);
    }
    const text = getMessageText(latest.node);
    if (!text) {
      diagnostics.rejections.push('The latest assistant candidate contains no extractable text.');
      return result('', EMPTY_RESULT_REASON, diagnostics);
    }
    diagnostics.selectedSource = latest.selector;
    return result(text, '', diagnostics);
  }

  function collectChatgptCandidates(doc, diagnostics) {
    const conversation = doc.querySelector('main') || doc.querySelector('[role="main"]');
    if (!conversation) {
      diagnostics.rejections.push('No main conversation container was found; refusing page-wide text fallback.');
      return [];
    }

    const matches = [...conversation.querySelectorAll(ASSISTANT_SELECTOR)]
      .filter(node => node.isConnected !== false && !node.closest(EXCLUDED_ANCESTOR_SELECTOR))
      .map(node => ({ node, scope: findTurnScope(node) || node, selector: ASSISTANT_SELECTOR }));

    const turns = [...conversation.querySelectorAll(TURN_SELECTOR)]
      .filter(turn => turn.isConnected !== false && !turn.closest(EXCLUDED_ANCESTOR_SELECTOR));
    for (const turn of turns) {
      const messages = [...turn.querySelectorAll(TEXT_MESSAGE_SELECTOR)];
      for (const message of messages) {
        if (isUserMessage(message)) continue;
        if (message.matches(ASSISTANT_SELECTOR) || hasAssistantResponseStructure(message, turn)) {
          matches.push({
            node: message,
            scope: turn,
            selector: `${TURN_SELECTOR} > ${TEXT_MESSAGE_SELECTOR} assistant structure`
          });
        }
      }

      if (!messages.length && !turn.matches(ASSISTANT_SELECTOR) &&
          !isUserMessage(turn) && hasAssistantResponseAction(turn)) {
        const blocks = [...turn.querySelectorAll(MARKDOWN_SELECTOR)]
          .filter(block => !block.closest(EXCLUDED_CONTENT_SELECTOR) && !isUserMessage(block));
        if (blocks.length === 1) {
          matches.push({
            node: blocks[0],
            scope: turn,
            selector: `${TURN_SELECTOR} markdown block with response-action fallback`
          });
        }
      }
    }

    if (diagnostics.assistantCount === 0) {
      const fallback = [...conversation.querySelectorAll(TEXT_MESSAGE_SELECTOR)]
        .filter(message =>
          message.isConnected !== false &&
          !message.closest(EXCLUDED_ANCESTOR_SELECTOR) &&
          !isUserMessage(message) &&
          hasAssistantResponseStructure(message, findTurnScope(message))
        )
        .map(node => ({
          node,
          scope: findTurnScope(node) || node,
          selector: `${TEXT_MESSAGE_SELECTOR} markdown/response-action fallback`
        }));
      diagnostics.fallbackCandidateCount = fallback.length;
      diagnostics.fallbackUsed = fallback.length > 0;
      matches.push(...fallback);
    }

    for (const action of conversation.querySelectorAll(ASSISTANT_ACTION_SELECTOR)) {
      if (action.closest(EXCLUDED_ANCESTOR_SELECTOR)) continue;
      if (!isAssistantResponseAction(action)) continue;
      const associated = findResponseText(action, conversation);
      if (associated) {
        matches.push(associated);
      } else {
        diagnostics.rejections.push('A response-specific control was found, but no single safe adjacent answer block could be associated with it.');
      }
    }

    return uniqueMatches(matches);
  }

  function collectGeminiCandidates(doc) {
    return uniqueMatches(GEMINI_SELECTORS.flatMap(selector =>
      [...doc.querySelectorAll(selector)]
        .filter(node => node.isConnected !== false && !node.closest(EXCLUDED_ANCESTOR_SELECTOR))
        .map(node => ({ node, scope: node, selector }))
    ));
  }

  function hasAssistantResponseStructure(message, turn) {
    if (isUserMessage(message)) return false;
    return hasAssistantResponseAction(turn || findTurnScope(message));
  }

  function findTurnScope(message) {
    let current = message.parentElement;
    while (current && current.tagName !== 'MAIN') {
      if (current.matches(TURN_SELECTOR)) return current;
      const messages = current.querySelectorAll(TEXT_MESSAGE_SELECTOR);
      if (hasAssistantResponseAction(current) && messages.length <= 1) return current;
      current = current.parentElement;
    }
    return null;
  }

  function hasAssistantResponseAction(scope) {
    if (!scope) return false;
    return [...scope.querySelectorAll(ASSISTANT_ACTION_SELECTOR)].some(isAssistantResponseAction);
  }

  function isAssistantResponseAction(action) {
    const label = `${action.getAttribute('aria-label') || ''} ${action.getAttribute('title') || ''}`.toLowerCase();
    const testId = action.getAttribute('data-testid') || '';
    return /copy response|good response|bad response|rate response|read aloud response/.test(label) ||
      /feedback-turn-action-button|good-response-turn-action-button|bad-response-turn-action-button|read-aloud-turn-action-button/.test(testId);
  }

  function findResponseText(action, conversation) {
    let branch = action;
    let ancestor = action.parentElement;
    let depth = 0;
    while (ancestor && ancestor !== conversation) {
      if (depth++ >= 10) break;
      if (ancestor.matches(EXCLUDED_ANCESTOR_SELECTOR)) return null;
      if (branch.parentElement !== ancestor) {
        while (branch.parentElement && branch.parentElement !== ancestor) {
          branch = branch.parentElement;
        }
      }
      const siblingBranches = [...ancestor.children].filter(child =>
        child !== branch &&
        !child.matches(EXCLUDED_CONTENT_SELECTOR) &&
        !isUserMessage(child)
      );
      const content = siblingBranches
        .map(node => {
          if (node.querySelector(ASSISTANT_ACTION_SELECTOR) ||
              [...node.querySelectorAll('[data-testid="copy-turn-action-button"]')].some(button =>
                !isAssistantResponseAction(button)
              )) return null;
          const messages = [
            ...(node.matches(TEXT_MESSAGE_SELECTOR) ? [node] : []),
            ...node.querySelectorAll(TEXT_MESSAGE_SELECTOR)
          ].filter(message => !isUserMessage(message));
            if (messages.length > 1) return null;
            const responseNode = messages.length === 1 ? messages[0] : node;
            if (isUserMessage(responseNode)) return null;
            return { node: responseNode, text: getMessageText(responseNode) };
          })
          .filter(item => item && item.text);
      if (content.length === 1) {
        return {
          node: content[0].node,
          scope: ancestor,
          selector: `${ASSISTANT_ACTION_SELECTOR} -> adjacent response content`
        };
      }
      if (content.length > 1) return null;
      branch = ancestor;
      ancestor = ancestor.parentElement;
    }
    return null;
  }

  function isUserMessage(node) {
    return node.matches(USER_SELECTOR) ||
      Boolean(node.closest(USER_SELECTOR)) ||
      node.matches('[class*="user-message-bubble-color"]') ||
      Boolean(node.querySelector('[class*="user-message-bubble-color"]'));
  }

  function collectCounts(doc) {
    const conversation = doc.querySelector('main') || doc.querySelector('[role="main"]') || doc;
    const selectors = {
      '[data-testid^="conversation-turn-"]': TURN_SELECTOR,
      'main .text-message': TEXT_MESSAGE_SELECTOR,
      'main [class*="markdown"]': 'main [class*="markdown"]',
      'assistant response actions': ASSISTANT_ACTION_SELECTOR,
      'copy response controls': 'button[aria-label*="Copy response" i]',
      'good/bad response controls': 'button[aria-label*="Good response" i], button[aria-label*="Bad response" i]'
    };
    return {
      assistantCount: safeCount(conversation, ASSISTANT_SELECTOR),
      userCount: safeCount(conversation, USER_SELECTOR),
      otherSelectorCounts: Object.fromEntries(
        Object.entries(selectors).map(([label, selector]) => [label, safeCount(conversation, selector)])
      ),
      textMessageCount: safeCount(conversation, TEXT_MESSAGE_SELECTOR),
      markdownCount: safeCount(conversation, MARKDOWN_SELECTOR),
      candidateCount: 0,
      fallbackCandidateCount: 0,
      fallbackUsed: false,
      selectedSource: '',
      rejections: []
    };
  }

  function safeCount(doc, selector) {
    try {
      return doc.querySelectorAll(selector).length;
    } catch {
      return 0;
    }
  }

  function uniqueMatches(matches) {
    const seen = new Set();
    return matches.filter(({ node }) => {
      if (seen.has(node)) return false;
      seen.add(node);
      return true;
    });
  }

  function hasActiveStopControl(scope) {
    return STOP_GENERATION_SELECTORS.some(selector =>
      [...scope.querySelectorAll(selector)].some(node =>
        node.isConnected !== false &&
        !node.disabled &&
        !node.closest(EXCLUDED_ANCESTOR_SELECTOR) &&
        isVisible(node)
      )
    );
  }

  function isVisible(node) {
    let current = node;
    while (current) {
      if (current.hidden || current.getAttribute('aria-hidden') === 'true') return false;
      const view = current.ownerDocument?.defaultView;
      if (view?.getComputedStyle) {
        const style = view.getComputedStyle(current);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
      }
      current = current.parentElement;
    }
    return true;
  }

  function hasStreamingSignal(node) {
    return node.matches(MESSAGE_STREAMING_SELECTOR) ||
      Boolean(node.querySelector(MESSAGE_STREAMING_SELECTOR));
  }

  function getMessageText(node) {
    const content = node.cloneNode ? node.cloneNode(true) : node;
    if (content !== node) {
      content.querySelectorAll(EXCLUDED_CONTENT_SELECTOR).forEach(element => element.remove());
    }
    const text = typeof content.innerText === 'string' ? content.innerText : content.textContent || '';
    return text.trim();
  }

  function result(text, reason, diagnostics) {
    return { text, reason, diagnostics };
  }

  function getDiagnostics(doc, extractorAvailable, contentScriptRunning) {
    const counts = collectCounts(doc);
    return {
      extractorAvailable,
      contentScriptRunning,
      assistantCount: counts.assistantCount,
      userCount: counts.userCount,
      otherSelectorCounts: counts.otherSelectorCounts
    };
  }

  return { extractLatestCompletedAssistantAnswer, getDiagnostics };
}));
