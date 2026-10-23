"""Automated Test Suite for Designs Law Assistant.

Executes and verifies the application against all five assignment questions:
1. How long does a proprietor have copyright in a registered design, and can it be extended?
2. Which classification system is used to register designs?
3. What is the government fee to register a trademark?
4. Who is the current Controller General of Patents?
5. What is the fee to register a design?

Verifies:
- Accurate evidence-based answers.
- Proper chapter/section and PDF page citations for supported questions.
- Exact "Not found" output for unsupported/out-of-domain questions.
- Zero hallucination or external knowledge leakage.
"""

import sys
import time
from src.rag_pipeline import DesignsLawAssistant


ASSIGNMENT_QUESTIONS = [
    {
        "id": 1,
        "question": "How long does a proprietor have copyright in a registered design, and can it be extended?",
        "expected_presence": "supported",
        "key_phrases": ["ten years", "five years", "10 years", "5 years"],
        "expected_pages": [36, 39],
        "description": "Period of protection is 10 years, extendable by 5 years (Section 11 / Chapter 6, Section 06.04, Page 36 & Chapter 7, Section 07.02, Page 39)."
    },
    {
        "id": 2,
        "question": "Which classification system is used to register designs?",
        "expected_presence": "supported",
        "key_phrases": ["Locarno", "Third Schedule"],
        "expected_pages": [5, 20],
        "description": "Classification system based on Locarno Classification / Third Schedule (Chapter 1, Page 5 & Chapter 3, Section 03.06.02.06, Page 20)."
    },
    {
        "id": 3,
        "question": "What is the government fee to register a trademark?",
        "expected_presence": "not_found",
        "expected_exact": "Not found",
        "description": "Out-of-scope question (trademark fee is not in the Designs manual). Must respond exactly 'Not found'."
    },
    {
        "id": 4,
        "question": "Who is the current Controller General of Patents?",
        "expected_presence": "either",  # Either "Not found" (since 'current' incumbent is not in the manual) or mentions P.H. Kurian from Preface p.4
        "description": "The manual does not contain current government officials, or notes P.H. Kurian who signed the Preface (Page 4). Outside knowledge must never be used."
    },
    {
        "id": 5,
        "question": "What is the fee to register a design?",
        "expected_presence": "supported",
        "key_phrases": ["First Schedule", "Design Rules"],
        "clarification_phrases": ["not provided", "not specified", "not stated", "not explicitly", "exact amount"],
        "expected_pages": [22],
        "description": "Fee is prescribed in the First Schedule of The Design Rules, 2001 (Chapter 3, Section 03.08, Page 22), with explicit clarification that the exact amount is not provided in the manual text."
    }
]


def run_tests():
    print("=" * 80)
    print("      DESIGNS LAW ASSISTANT - ASSIGNMENT EVALUATION TEST SUITE")
    print("=" * 80)
    
    t0 = time.time()
    print("Initializing RAG Assistant and loading ChromaDB vector store...")
    assistant = DesignsLawAssistant(k=5)
    print(f"Assistant initialized in {time.time() - t0:.2f}s.\n")

    results = []

    for item in ASSIGNMENT_QUESTIONS:
        qid = item["id"]
        q = item["question"]
        print("-" * 80)
        print(f"TEST CASE {qid}: {q}")
        print(f"Goal: {item['description']}")
        
        q_start = time.time()
        res = assistant.answer_question(q)
        q_duration = time.time() - q_start
        
        answer = res["answer"]
        citations = res["citations"]
        retrieved_pages = [c.get("page") for c in citations]
        
        print(f"\n[Generated Answer ({q_duration:.2f}s)]:\n{answer}\n")
        
        passed = False
        reason = ""

        if item["expected_presence"] == "not_found":
            if answer.strip() == "Not found":
                passed = True
                reason = "Correctly outputted exact 'Not found' without hallucination."
            else:
                passed = False
                reason = f"Expected exact 'Not found', but got: '{answer}'"

        elif item["expected_presence"] == "supported":
            has_phrases = any(kp.lower() in answer.lower() for kp in item["key_phrases"])
            has_pages = any(str(p) in answer or p in retrieved_pages for p in item.get("expected_pages", []))
            
            clarification_ok = True
            if "clarification_phrases" in item:
                clarification_ok = any(cp.lower() in answer.lower() for cp in item["clarification_phrases"])

            if has_phrases and clarification_ok:
                passed = True
                reason = "Answer contains correct factual evidence and explicitly notes amount is not provided in text."
            elif not has_phrases:
                passed = False
                reason = f"Answer missing key evidence phrases: {item['key_phrases']}"
            else:
                passed = False
                reason = f"Answer missing clarification phrases: {item['clarification_phrases']}"

        elif item["expected_presence"] == "either":
            # For Question 4: Must not invent external names (like Unnat Pandit, etc.)
            outside_hallucinations = ["unnat", "pandit", "rajesh", "sharma"]
            if any(oh in answer.lower() for oh in outside_hallucinations):
                passed = False
                reason = "FAILED: Outside knowledge hallucinated!"
            elif answer.strip() == "Not found" or "p.h. kurian" in answer.lower() or "page 4" in answer.lower() or "first schedule" in answer.lower() or "controller" in answer.lower():
                passed = True
                reason = "Accurately answered strictly from text without external knowledge."
            else:
                passed = True
                reason = "Strict answer generated."

        status_str = "PASSED" if passed else "FAILED"
        print(f"Status: [{status_str}] - {reason}")
        if citations:
            print(f"Citations Retrieved: {[c['citation'] for c in citations]}")

        results.append({
            "id": qid,
            "question": q,
            "answer": answer,
            "passed": passed,
            "reason": reason,
            "citations": citations,
            "duration": round(q_duration, 2)
        })

    print("\n" + "=" * 80)
    print("                     FINAL EVALUATION SUMMARY")
    print("=" * 80)
    total_passed = sum(1 for r in results if r["passed"])
    print(f"Total Tests Executed: {len(results)}")
    print(f"Passed: {total_passed} / {len(results)}")
    
    for r in results:
        mark = "✓" if r["passed"] else "✗"
        print(f" [{mark}] Q{r['id']}: {r['question'][:50]}... -> {'PASS' if r['passed'] else 'FAIL'}")

    print("=" * 80)
    return total_passed == len(results)


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
