import * as LiteRT from 'https://cdn.jsdelivr.net/npm/@litertjs/core@0.2.1/+esm';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/dist/tf.min.js';
import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgpu/dist/tf-backend-webgpu.js';
import { VectorStore } from '/src/VectorStore.js';
import { VectorSearch } from '/src/VectorSearch.js';
import { EmbeddingModel } from '/src/EmbeddingModel.js';
import { Tokenizer } from '/src/Tokenizer.js';
import { VisualizeTokens } from '/src/VisualizeTokens.js';
import { VisualizeEmbedding } from '/src/VisualizeEmbedding.js';


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


// Model configuration.
const MODEL_URL = 'model/embeddinggemma-300M_seq1024_mixed-precision.tflite';
const TOKENIZER_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const SEQ_LENGTH = 1024;


// Component instances.
const VECTOR_STORE = new VectorStore();
VECTOR_STORE.setDb(DB_NAME_INPUT.value);

const VECTOR_SEARCH = new VectorSearch();
const EMBEDDING_MODEL = new EmbeddingModel();
const TOKENIZER = new Tokenizer();
const VISUALIZE_TOKENS = new VisualizeTokens();
const VISUALIZE_EMBEDDING = new VisualizeEmbedding();


let allStoredData = undefined;
let lastDBName = '';
let textBatch = [];
let tensorBatch = [];


async function predictBtnClickHandler() {
  const QUERY_TEXT_VALUE = TARGET_TEXT.value;
  const THRESHOLD = parseFloat(THRESHOLD_INPUT.value) || 0.5;
  const SELECTED_DB = DB_SELECT.value;

  if (QUERY_TEXT_VALUE && SELECTED_DB) {
    VECTOR_STORE.setDb(SELECTED_DB);
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

  VECTOR_STORE.setDb(dbName);

  STORE_BTN.disabled = true;
  STATUS_EL.innerText = `Storing paragraphs in VectorDB (${dbName})...`;

  // Split by double newline - representing paragraph chunking.
  const paragraphs = text.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 0);
  
  for (let i = 0; i < paragraphs.length; i++) {
    const p = paragraphs[i];
    STATUS_EL.innerText = `Embedding paragraph ${i + 1} of ${paragraphs.length}...`;
    
    const tokens = await TOKENIZER.encode(p);
    const { embedding } = await EMBEDDING_MODEL.getEmbedding(tokens, SEQ_LENGTH);
    tensorBatch.push(embedding);
    textBatch.push(p);
    
    if (tensorBatch.length >= 2 || i === paragraphs.length - 1) {
      const stackedTensors = tf.stack(tensorBatch);
      const t0 = performance.now();
      // This next line actually makes it wait for the model to finish inference.
      // Otherwise GPU will be busy finishing inferences when next request comes.
      const allVectors = await stackedTensors.array();
      const t1 = performance.now();
      console.log(`Batch GPU -> CPU memory fetch took ${t1 - t0} milliseconds.`);
      const storagePayload = allVectors.map((vector, index) => ({
        embedding: vector[0],
        text: textBatch[index]
      }));

      await VECTOR_STORE.storeBatch(storagePayload);

      // Clean up the batch here
      tensorBatch.forEach(t => t.dispose());
      stackedTensors.dispose();
      tensorBatch.splice(0); // Clear the array
      textBatch.splice(0);
      
      console.log('DB Batch Write');
    }
  }

  STATUS_EL.innerText = `Stored ${paragraphs.length} paragraphs.`;
  STORE_BTN.disabled = false;
  INPUT_TEXT.value = '';
  
  // Refresh list and cache after storing in maybe a new DB.
  await VECTOR_SEARCH.deleteGPUVectorCache();
  await updateDbList();
} 


async function load() {
  try {
    // Refresh available DB list to GUI.
    await updateDbList();
    
    STATUS_EL.innerText = 'Loading WebGPU...';
    await tf.setBackend('webgpu');
    
    STATUS_EL.innerText = 'Initializing LiteRT...';
    await LiteRT.loadLiteRt('wasm/');
    const TF_BACKEND = tf.backend();
    LiteRT.setWebGpuDevice(TF_BACKEND.device);

    STATUS_EL.innerText = 'Loading Gemma Embedding Model...';
    await EMBEDDING_MODEL.load(MODEL_URL);

    STATUS_EL.innerText = 'Loading Transformers.js Tokenizer...';
    await TOKENIZER.load(TOKENIZER_ID);

    STATUS_EL.innerText = 'Ready to store and search';
    STORE_BTN.disabled = false;
    PREDICT_BTN.disabled = false;

    // Add key event handlers.
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
  // 1. Get embedding for the query.
  const tokens = await TOKENIZER.encode(queryText);
  const { embedding: EMBEDDING } = await EMBEDDING_MODEL.getEmbedding(tokens, SEQ_LENGTH);
  const QUERY_VECTOR = Array.from(await EMBEDDING.data());
  
  // 2. Update visualizations for the query.
  VISUALIZE_TOKENS.render(tokens, QUERY_TOKENS_OUTPUT, SEQ_LENGTH);
  await VISUALIZE_EMBEDDING.render(EMBEDDING, QUERY_EMBEDDING_VIZ, QUERY_EMBEDDING_TEXT);
  
  // 3. Fetch all vectors from VectorStore if not already done.
  let matrixData = undefined;
  if (lastDBName !== DB_SELECT.value) {
    await VECTOR_SEARCH.deleteGPUVectorCache();
    lastDBName = DB_SELECT.value;
    allStoredData = await VECTOR_STORE.getAllVectors();

    if (allStoredData.length === 0) {
      RESULTS_TEXT.value = "No data in database.";
      PREDICT_BTN.disabled = false;
      EMBEDDING.dispose();
      return;
    }

    matrixData = allStoredData.map(item => item.embedding);
  } else {
    // If same DB, we can still use the matrixData if we didn't cache it yet, 
    // but usually VECTOR_SEARCH handles caching internally.
    // If cached, it ignores matrixData.
    matrixData = allStoredData.map(item => item.embedding);
  }
  
  console.log('Searching ' + allStoredData.length + ' vectors');
  
  // 4. Prepare matrix and compute similarities across all in 1 pass.
  const t0 = performance.now();
  const MAX_MATCHES = 10;
  const {values, indices} = await VECTOR_SEARCH.cosineSimilarityTFJSGPUMatrix(matrixData, QUERY_VECTOR, MAX_MATCHES);
  
  // 5. Map scores back to original data and filter/sort.
  let topMatches = [];
  let bestIndex = 0;
  let bestScore = 0;
  
  for (let i = 0; i < values.length; i++) {
    if (values[i] >= threshold) {
      if (topMatches.length < MAX_MATCHES) {
        topMatches.push({
          id: allStoredData[indices[i]].id,
          score: values[i],
          vector: allStoredData[indices[i]].embedding
        });
        if (values[i] > bestScore) {
          bestIndex = i;
          bestScore = values[i];
        }
      }
    }
  }

  const t1 = performance.now();
  console.log(`Vector search took: ${t1 - t0} milliseconds.`);

  // 6. Fetch text for top matches and update Results UI.
  if (topMatches.length > 0) {
    const results = [];
    for (const match of topMatches) {
      const text = await VECTOR_STORE.getTextByID(match.id);
      results.push({ ...match, text });
    }

    RESULTS_TEXT.value = results.map(m => `[Score: ${m.score.toFixed(4)}]\n${m.text}`).join('\n\n');
    updateSimilarityUI(bestScore); // Show best match score in the existing UI
    
    // 7. Finalize by visualizing the actual best match embedding.
    const bestMatchVector = results[bestIndex].vector;
    if (bestMatchVector) {
      const matchEmbedding = tf.tensor1d(bestMatchVector);
      await VISUALIZE_EMBEDDING.render(matchEmbedding, BEST_MATCH_EMBEDDING_VIZ, BEST_MATCH_EMBEDDING_TEXT);
      matchEmbedding.dispose();
    }
  } else {
    RESULTS_TEXT.value = "No matches found above threshold.";
    SIMILARITY_CONTAINER.classList.add('hidden');
    BEST_MATCH_EMBEDDING_VIZ.innerHTML = '';
    BEST_MATCH_EMBEDDING_TEXT.innerText = '';
  }
  
  // 8. Cleanup.
  EMBEDDING.dispose();
}


function updateSimilarityUI(score) {
  SIMILARITY_CONTAINER.classList.remove('hidden');
  SIMILARITY_SCORE_EL.innerText = score.toFixed(4);
  
  // Dynamic coloring: Red (0) to Green (1)
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
    
    // Always include the one in the input box if not present.
    const currentInputName = DB_NAME_INPUT.value.trim();
    let names = dbs.map(db => db.name).filter(name => name !== undefined);
    
    if (currentInputName && !names.includes(currentInputName)) {
      names.push(currentInputName);
    }
    
    // Sort names.
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
