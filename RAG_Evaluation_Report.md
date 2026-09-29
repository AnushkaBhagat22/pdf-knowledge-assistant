# RAG Evaluation Report

## Intelligent PDF Knowledge Assistant using Retrieval-Augmented Generation

---

## 1. Evaluation Overview

The PDF Knowledge Assistant was evaluated to verify the performance of its Retrieval-Augmented Generation (RAG) pipeline.

The evaluation focused on four main areas:

1. Single-document retrieval
2. Cross-document retrieval
3. Answer grounding
4. Handling of unsupported questions

The evaluation was performed using the following documents:

- `RNN_LSTM.pdf`
- `sample.pdf`

The retrieval system used **FAISS** with a **Top-k value of 5**.

---

## 2. Evaluation Dataset

A total of **10 questions** were prepared.

| Test Type | Number of Questions |
|---|---:|
| Single-document retrieval | 8 |
| Cross-document retrieval | 1 |
| Unsupported-question test | 1 |
| **Total** | **10** |

The questions were designed to test different parts of the RAG pipeline rather than simply checking whether the application could generate an answer.

---

## 3. Retrieval Evaluation

### 3.1 Single-Document Retrieval

The first eight questions tested whether the system could retrieve information from the correct document.

Examples included questions about:

- How RNNs use previous information
- Vanishing and exploding gradients
- RNN and LSTM architecture
- LSTM long-term dependencies
- LSTM feature extraction
- LSTM temporal state information
- LSTM training limitations
- LSTM accuracy information

### Result

**Single-document retrieval hit rate: 100%**

```text
8 / 8 questions successfully retrieved
the expected source document.