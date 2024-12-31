import { AutoTokenizer } from '../libs/transformers.min.js';

const ort = window.ort;
let session = null;

export async function initializeTokenizer() {
    const tokenizer = await AutoTokenizer.from_pretrained('all-MiniLM-L6-v2');
    const response = await fetch('https://huggingface.co/napronald/AI-Paper-Similarity-Search-App/resolve/main/model_quantized.onnx');
    const buffer = await response.arrayBuffer();
    session = await ort.InferenceSession.create(buffer);
    return tokenizer;
}

export async function inferModel(tokenizer, query) {
    const encoding = await tokenizer(query, { 
        add_special_tokens: true,
        padding: 'max_length', 
        truncation: true, 
        max_length: 128, 
        return_tensors: 'pt' 
    });

    const inputIds = Array.from(encoding.input_ids.data).map(id => Number(id));
    const attentionMask = Array.from(encoding.attention_mask.data).map(mask => Number(mask));

    const inputIdsArray = new BigInt64Array(inputIds.map(id => BigInt(id)));
    const attentionMaskArray = new BigInt64Array(attentionMask.map(mask => BigInt(mask)));
    const tensorType = 'int64';

    const inputTensor = new ort.Tensor(tensorType, inputIdsArray, [1, inputIds.length]);
    const attentionTensor = new ort.Tensor(tensorType, attentionMaskArray, [1, attentionMask.length]);

    const output = await session.run({
        input_ids: inputTensor, 
        attention_mask: attentionTensor
    });
    
    const lastHiddenState = output['last_hidden_state']; 

    const [batchSize, sequenceLength, hiddenSize] = lastHiddenState.dims;
    const reshapedHiddenState = [];
    for (let i = 0; i < batchSize; i++) {
        const sentence = [];
        for (let j = 0; j < sequenceLength; j++) {
            const token = [];
            for (let k = 0; k < hiddenSize; k++) {
                token.push(lastHiddenState.data[(i * sequenceLength + j) * hiddenSize + k]);
            }
            sentence.push(token);
        }
        reshapedHiddenState.push(sentence);
    }

    const meanPooled = [];
    for (let i = 0; i < reshapedHiddenState.length; i++) { 
        const tokens = reshapedHiddenState[i];
        const mask = attentionMask; 
        const hiddenDim = tokens[0].length;
        const sum = new Array(hiddenDim).fill(0);
        let validTokenCount = 0;

        for (let j = 0; j < tokens.length; j++) {
            if (mask[j] === 1) { 
                validTokenCount += 1;
                for (let k = 0; k < hiddenDim; k++) {
                    sum[k] += tokens[j][k];
                }
            }
        }

        if (validTokenCount === 0) {
            validTokenCount = 1;
        }

        const mean = sum.map(val => val / validTokenCount);
        meanPooled.push(mean);
    }

    const embedding = meanPooled[0]; 

    return embedding; 
}

export function normalize(vec) {
    const norm = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    return vec.map(val => val / norm);
}