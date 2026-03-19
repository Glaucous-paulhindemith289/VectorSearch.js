import * as LiteRT from 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/+esm';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js';
import { VectorSearch } from '/src/VectorSearch.js';


// DOM references.
const DB_NAME_INPUT = document.getElementById('db-name-input');
const DB_SELECT = document.getElementById('db-select');
const STATUS_EL = document.getElementById('status');
const QUERY_EMBEDDING_TEXT = document.getElementById('query-embedding-text');
const QUERY_TOKENS_OUTPUT = document.getElementById('query-tokens-output');
const QUERY_EMBEDDING_VIZ = document.getElementById('query-embedding-viz');
const BEST_MATCH_EMBEDDING_VIZ = document.getElementById('best-match-embedding-viz');
const BEST_MATCH_EMBEDDING_TEXT = document.getElementById('best-match-embedding-text');
const INPUT_TEXT = document.getElementById('input-text');
const TARGET_TEXT = document.getElementById('target-text');
const STORE_BTN = document.getElementById('store-btn');
const PREDICT_BTN = document.getElementById('predict-btn');
const THRESHOLD_INPUT = document.getElementById('threshold-input');
const THRESHOLD_VALUE = document.getElementById('threshold-value');
const RESULTS_TEXT = document.getElementById('results-text');
const SIMILARITY_CONTAINER = document.getElementById('similarity-container');
const SIMILARITY_SCORE_EL = document.getElementById('similarity-score');
const SIMILARITY_LABEL_EL = document.getElementById('similarity-label');


// Configuration.
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
const TOKENIZER_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const SEQ_LENGTH = 1024;


// Instantiate VectorSearch Master Class.
const VECTOR_SEARCH = new VectorSearch(MODEL_URL, TOKENIZER_ID, SEQ_LENGTH);


let textBatch = [];
let tensorBatch = [];


async function predictBtnClickHandler() {
  const QUERY_TEXT_VALUE = TARGET_TEXT.value;
  const THRESHOLD = parseFloat(THRESHOLD_INPUT.value) || 0.5;
  const SELECTED_DB = DB_SELECT.value;

  if (QUERY_TEXT_VALUE && SELECTED_DB) {
    VECTOR_SEARCH.setDb(SELECTED_DB);
    PREDICT_BTN.disabled = true;
    STATUS_EL.innerText = `Searching VectorDB (${SELECTED_DB})...`;
    const t0 = performance.now();
    await predict(QUERY_TEXT_VALUE, THRESHOLD);
    const t1 = performance.now();
    console.log(`Total search time (query embedding + vector search) took ${t1 - t0} milliseconds.`);
    STATUS_EL.innerText = 'Search complete';
    PREDICT_BTN.disabled = false;
  }
}


async function storeBtnClickHandler() {
  const text = INPUT_TEXT.value.trim();
  const dbName = DB_NAME_INPUT.value.trim();
  if (!text || !dbName) return;

  VECTOR_SEARCH.setDb(dbName);

  STORE_BTN.disabled = true;
  STATUS_EL.innerText = `Storing paragraphs in VectorDB (${dbName})...`;

  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 0);
  
  for (let i = 0; i < paragraphs.length; i++) {
    const p = paragraphs[i];
    STATUS_EL.innerText = `Embedding paragraph ${i + 1} of ${paragraphs.length}...`;
    
    const { embedding } = await VECTOR_SEARCH.getEmbedding(p);
    tensorBatch.push(embedding);
    textBatch.push(p);
    
    if (tensorBatch.length >= 2 || i === paragraphs.length - 1) {
      const stackedTensors = tf.stack(tensorBatch);
      const t0 = performance.now();
      const allVectors = await stackedTensors.array();
      const t1 = performance.now();
      console.log(`Batch GPU -> CPU memory fetch took ${t1 - t0} milliseconds.`);
      const storagePayload = allVectors.map((vector, index) => ({
        embedding: vector[0],
        text: textBatch[index]
      }));

      await VECTOR_SEARCH.storeBatch(storagePayload);

      tensorBatch.forEach(t => t.dispose());
      stackedTensors.dispose();
      tensorBatch.splice(0);
      textBatch.splice(0);
      
      console.log('DB Batch Write');
    }
  }

  STATUS_EL.innerText = `Stored ${paragraphs.length} paragraphs.`;
  STORE_BTN.disabled = false;
  INPUT_TEXT.value = '';
  
  await VECTOR_SEARCH.deleteGPUVectorCache();
  await updateDbList();
} 


async function load() {
  try {
    await updateDbList();
    
    STATUS_EL.innerText = 'Loading WebGPU...';
    await tf.setBackend('webgpu');
    
    STATUS_EL.innerText = 'Initializing LiteRT...';
    await LiteRT.loadLiteRt('wasm/');
    const TF_BACKEND = tf.backend();
    LiteRT.setWebGpuDevice(TF_BACKEND.device);

    STATUS_EL.innerText = 'Loading Gemma & Tokenizer...';
    await VECTOR_SEARCH.load();

    STATUS_EL.innerText = 'Ready to store and search';
    STORE_BTN.disabled = false;
    PREDICT_BTN.disabled = false;

    STORE_BTN.addEventListener('click', storeBtnClickHandler);
    PREDICT_BTN.addEventListener('click', predictBtnClickHandler);
    THRESHOLD_INPUT.addEventListener('input', () => {
      THRESHOLD_VALUE.innerText = THRESHOLD_INPUT.value;
    });
  } catch (e) {
    console.error(e);
    STATUS_EL.innerText = 'Error: ' + e.message;
  }
}


async function predict(queryText, threshold) {
  const { embedding: EMBEDDING, tokens: TOKENS } = await VECTOR_SEARCH.getEmbedding(queryText);
  
  VECTOR_SEARCH.renderTokens(TOKENS, QUERY_TOKENS_OUTPUT);
  await VECTOR_SEARCH.renderEmbedding(EMBEDDING, QUERY_EMBEDDING_VIZ, QUERY_EMBEDDING_TEXT);
  
  const { results, bestScore, bestIndex } = await VECTOR_SEARCH.search(EMBEDDING, threshold, DB_SELECT.value);

  if (results.length > 0) {
    RESULTS_TEXT.value = results.map(m => `[Score: ${m.score.toFixed(4)}]\n${m.text}`).join('\n\n');
    updateSimilarityUI(bestScore);
    
    const bestMatchVector = results[bestIndex].vector;
    if (bestMatchVector) {
      const matchEmbedding = tf.tensor1d(bestMatchVector);
      await VECTOR_SEARCH.renderEmbedding(matchEmbedding, BEST_MATCH_EMBEDDING_VIZ, BEST_MATCH_EMBEDDING_TEXT);
      matchEmbedding.dispose();
    }
  } else {
    RESULTS_TEXT.value = "No matches found above threshold.";
    SIMILARITY_CONTAINER.classList.add('hidden');
    BEST_MATCH_EMBEDDING_VIZ.innerHTML = '';
    BEST_MATCH_EMBEDDING_TEXT.innerText = '';
  }
  
  EMBEDDING.dispose();
}


function updateSimilarityUI(score) {
  SIMILARITY_CONTAINER.classList.remove('hidden');
  SIMILARITY_SCORE_EL.innerText = score.toFixed(4);
  
  const HUE = Math.max(0, Math.min(120, score * 120));
  const BACKGROUND_COLOUR = `hsla(${HUE}, 70%, 20%, 0.4)`;
  const BORDER_COLOUR = `hsla(${HUE}, 70%, 50%, 0.6)`;
  
  SIMILARITY_CONTAINER.style.backgroundColor = BACKGROUND_COLOUR;
  SIMILARITY_CONTAINER.style.borderColor = BORDER_COLOUR;
  
  let label = 'Low Similarity';
  if (score > 0.8) {
    label = 'Very High Similarity';
  } else if (score > 0.6) {
    label = 'High Similarity';
  } else if (score > 0.4) {
    label = 'Moderate Similarity';
  }
  
  SIMILARITY_LABEL_EL.innerText = label;
}


async function updateDbList() {
  if (!window.indexedDB.databases) {
    console.warn('indexedDB.databases() is not supported in this browser.');
    return;
  }

  try {
    const dbs = await window.indexedDB.databases();
    const currentSelection = DB_SELECT.value;
    
    DB_SELECT.innerHTML = '';
    const currentInputName = DB_NAME_INPUT.value.trim();
    let names = dbs.map(db => db.name).filter(name => name !== undefined);
    
    if (currentInputName && !names.includes(currentInputName)) {
      names.push(currentInputName);
    }
    
    names.sort();
    
    names.forEach(name => {
      const option = document.createElement('option');
      option.value = name;
      option.text = name;
      if (name === currentSelection || (currentSelection === '' && name === currentInputName)) {
        option.selected = true;
      }
      DB_SELECT.appendChild(option);
    });
  } catch (e) {
    console.error('Error fetching databases:', e);
  }
}


load();
