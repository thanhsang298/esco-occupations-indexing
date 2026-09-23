"""One-off probe: Scrum Master query vs ICT project manager with enriched docs."""

import json

import httpx

INSTRUCTION = (
    "Given a job posting, retrieve the ESCO occupation that best represents "
    "its work and responsibilities."
)


def main() -> None:
    with open("/tmp/scrum_probe.json", encoding="utf-8") as handle:
        probe = json.loads(handle.read())
    query = probe["query"]
    base = probe["baseline_doc"]
    alt = probe["alt"]
    ess = [label for _, label in probe["essential"] if label]
    docs = {
        "D0_baseline": base,
        "D1_+alt-labels": base + "\nAlternative labels: " + "; ".join(alt),
        "D2_+alt+essential-skills": (
            base
            + "\nAlternative labels: "
            + "; ".join(alt)
            + "\nEssential skills and knowledge: "
            + "; ".join(ess)
        ),
    }
    client = httpx.Client(timeout=300.0)
    for name, doc in docs.items():
        response = client.post(
            "http://localhost:8989/v1/rerank",
            json={
                "model": "Qwen/Qwen3-Reranker-4B",
                "query": query,
                "documents": [doc],
                "top_n": 1,
                "instruction": INSTRUCTION,
            },
        ).json()
        score = response["results"][0]["relevance_score"]
        print(f"{name}: {score:.4f} (doc chars: {len(doc)})")


if __name__ == "__main__":
    main()
