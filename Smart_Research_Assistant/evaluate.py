

import sys
import types

fake_module = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:
    pass


fake_module.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = fake_module


import os
from dotenv import load_dotenv
from rag_pipeline import (
    load_documents,
    split_documents,
    get_embedding_model,
    build_vectorstore,
    retrieve,
    get_llm,
    generate_answer,
)
from langchain_openai import ChatOpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from dotenv import load_dotenv
load_dotenv()

TEST_CASES = [
    {
        "question": "What is the leave policy?",
        "ground_truth": "Employees are entitled to annual leave, sick leave, and maternity leave.",
    },
    {
        "question": "What are the applications of AI?",
        "ground_truth": "AI applications include healthcare, finance, education, and autonomous systems.",
    },
    {
        "question": "What are the ethical concerns with AI?",
        "ground_truth": "Ethical concerns include bias, privacy, and transparency.",
    },
]


def main():
    print("Loading documents and building vector store...")
    docs = load_documents()
    chunks = split_documents(docs)
    embedding_model = get_embedding_model()
    vectorstore = build_vectorstore(chunks, embedding_model)

    print("Loading Groq LLM for generation...")
    gen_llm = get_llm()

    questions, answers, contexts, ground_truths = [], [], [], []

    for case in TEST_CASES:
        query = case["question"]
        print(f"\nProcessing: {query}")

        retrieved_docs = retrieve(query, vectorstore, k=5)
        answer = generate_answer(query, retrieved_docs, gen_llm)
        print(f"Answer: {answer}")

        questions.append(query)
        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])
        ground_truths.append(case["ground_truth"])

    print("\nLoading OpenAI judge LLM for RAGAS...")
    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    print("\nSaved detailed results to ragas_results.csv")


if __name__ == "__main__":
    main()