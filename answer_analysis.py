from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_answer(candidate_answer, expected_answer):

    emb1 = model.encode([candidate_answer])
    emb2 = model.encode([expected_answer])

    score = cosine_similarity(emb1, emb2)

    return score[0][0]


if __name__ == "__main__":
    
    candidate = "Machine learning allows computers to learn from data."
    expected = "Machine learning is a method where systems learn patterns from data."

    score = evaluate_answer(candidate, expected)

    print("Answer Score:", score)