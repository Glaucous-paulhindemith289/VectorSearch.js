/**
 * A class to handle visual representation of tokens in the DOM.
 * Coded by Jason Mayes 2026.
 */
export class VisualizeTokens {
  /**
   * Renders the given tokens into the container element.
   * @param {Array<number>} tokens Array of token IDs.
   * @param {HTMLElement} containerEl The DOM element to render into.
   * @param {number} seqLength The total sequence length to show padding for.
   */
  render(tokens, containerEl, seqLength) {
    containerEl.innerHTML = '';
    
    // Show actual tokens.
    tokens.forEach(id => {
      const CHIP = document.createElement('div');
      CHIP.className = 'token-chip';
      CHIP.innerText = id;
      containerEl.appendChild(CHIP);
    });
    
    // Show a few padding tokens to illustrate the seq length.
    const paddingCount = Math.min(20, seqLength - tokens.length);
    for (let i = 0; i < paddingCount; i++) {
      const CHIP2 = document.createElement('div');
      CHIP2.className = 'token-chip padded';
      CHIP2.innerText = '0';
      containerEl.appendChild(CHIP2);
    }
    
    if (tokens.length < seqLength) {
      const MORE = document.createElement('div');
      MORE.className = 'mini-subtitle';
      MORE.innerText = `... and ${seqLength - tokens.length - paddingCount} more padding tokens`;
      containerEl.appendChild(MORE);
    }
  }
}
