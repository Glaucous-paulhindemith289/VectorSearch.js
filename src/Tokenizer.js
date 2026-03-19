import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.0';

/**
 * A class to handle loading and using the Transformers.js AutoTokenizer.
 * Coded by Jason Mayes 2026.
 */
export class Tokenizer {
  constructor() {
    this.tokenizer = undefined;
  }

  /**
   * Loads the tokenizer from a pretrained model ID.
   * @param {string} tokenizerId The ID of the pretrained tokenizer.
   * @return {Promise<void>}
   */
  async load(tokenizerId) {
    this.tokenizer = await AutoTokenizer.from_pretrained(tokenizerId);
  }

  /**
   * Encodes text into token IDs.
   * @param {string} text The text to encode.
   * @return {Promise<Array<number>>} Array of token IDs.
   */
  async encode(text) {
    if (!this.tokenizer) {
      throw new Error('Tokenizer not loaded. Call load() first.');
    }
    return await this.tokenizer.encode(text);
  }
}
