from FlagEmbedding import FlagModel
import numpy as np

def predict(model_id: str, query: list, doc: list) -> tuple[np.ndarray, np.ndarray, list[float]]:

    model = FlagModel(
        model_id,
        query_instruction_for_retrieval=None,
        pooling_method="cls",
        normalize_embeddings=True,
        use_fp16=True,
    )
    query_embeddings = model.encode_queries(query)
    doc_embeddings = model.encode_corpus(doc)
    scores = query_embeddings @ doc_embeddings.T

    return query_embeddings, doc_embeddings, scores