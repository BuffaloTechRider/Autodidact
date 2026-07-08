"""Inspect what GSA actually emits.

Runs the probe against a few representative questions and prints the raw
response, the top-5 logprobs at position 0, and the computed p_yes. Useful
for sanity-checking the assumption that the probe always generates a
single YES/NO token.
"""

from __future__ import annotations

import math
from pprint import pformat

from autodidact.llm_client import LLMClient, LLMConfig
from autodidact.signals.grounded_self_assessment import SelfAssessment


def _format_logprobs(d: dict[str, float]) -> str:
    if not d:
        return "(empty)"
    items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
    lines = []
    for tok, lp in items:
        prob = math.exp(lp)
        lines.append(f"    {tok!r:20s}  logprob={lp:8.4f}  prob={prob:.4f}")
    return "\n".join(lines)


def main():
    client = LLMClient(LLMConfig(provider="ollama", model="qwen3:8b"))
    gsa = SelfAssessment(client)

    questions = [
        "What is the capital of France?",
        "What is the airspeed velocity of an unladen swallow?",
        "Who is the current CEO of Anthropic?",
        "What is 2 + 2?",
        "Tell me about the BuffaloTechRider GitHub organization's autodidact repo.",
        "Explain quantum entanglement in one sentence.",
    ]

    for q in questions:
        print("=" * 78)
        print(f"Q: {q}")
        result = gsa.compute(q)
        print(f"  raw response  : {result.raw_response!r}")
        print(f"  raw repr      : {[ord(c) for c in result.raw_response[:8]]} (first 8 chars as codepoints)")
        print(f"  extraction    : {result.extraction_mode}")
        print(f"  yes_logprob   : {result.yes_logprob}")
        print(f"  no_logprob    : {result.no_logprob}")
        print(f"  p_yes         : {result.p_yes:.4f}")
        # Re-run the underlying probe to surface raw top-logprobs.
        prompt, *_ = gsa._build_prompt(q, None)
        from autodidact.llm_client import ChatMessage
        resp = client.chat_with_logprobs(
            [ChatMessage(role="user", content=prompt)],
            max_tokens=1,
            temperature=0.0,
            top_logprobs=5,
            think=False,
        )
        print(f"  raw content   : {resp.content!r}")
        if resp.top_logprobs_by_position:
            print(f"  top-5 at pos 0:")
            print(_format_logprobs(resp.top_logprobs_by_position[0]))
        else:
            print("  (no top_logprobs_by_position)")
    print("=" * 78)


if __name__ == "__main__":
    main()
