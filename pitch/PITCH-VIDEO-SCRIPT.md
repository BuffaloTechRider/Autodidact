# Autodidact — Pitch Video Script
## Vietnam AI Stars 2026

**Duration: 2:30–3:00**
**Format: Talking head + terminal demo clips + slide overlays**

---

## Production Notes

- Record in a quiet room, good lighting, plain background
- Cut between your face (storytelling) and terminal/slides (visuals)
- Speak naturally, slightly slower than conversation
- Add subtitles (CapCut auto-captions work well)
- Splice in 10-15 second clips from the actual demo recording

---

## Script

### Opening — The Hook (0:00–0:20)

*[On camera]*

"Your brain doesn't Google everything. When someone asks you a question, you think first. If you know the answer, you respond. If you're not sure, you search the internet or ask someone. And then you remember what you learned.

AI agents don't do this. Ask the same question tomorrow — same expensive cloud API call. They forget everything.

I'm [Your Name], and this is Autodidact — the local-first AI agent that actually learns and evolves."

---

### The Problem (0:20–0:45)

*[Cut to slide: flat line chart]*

"Today, AI agents call GPT-4 or Claude for every single query. Ten thousand queries a day? That's two to three thousand dollars a month. And that number never goes down — because the agent never remembers anything.

No human works this way. A good employee learns from every answer. They stop asking the same questions. AI agents should work the same way."

---

### The Solution + Demo (0:45–1:45)

*[Cut to slide: learning loop diagram]*

"Autodidact gives your AI agent a brain — a local model running on your machine — and teaches it how to learn.

When a query comes in, the brain thinks first. Checks its memory. If it knows the answer, it responds immediately — zero cost. If it's not confident, it asks a more powerful cloud model. But here's the key — it extracts the knowledge, skills and stores it locally. The search was temporary. The learning is permanent."

*[Cut to terminal — show actual demo clip]*

"Let me show you. This agent just started — empty brain, no knowledge. I ask it about fintech regulations in Vietnam."

*[Show Act 1: ESCALATE, Score 0.000, Cost $0.0015, learned 4 facts]*

"It escalated to the cloud. Cost a fraction of a cent. But it learned four facts and stored them.

Now I ask a related question — what compliance does a Vietnamese payment app need?"

*[Show Act 2: LOCAL, Score 0.718, Cost $0.0000, 3 seconds]*

"LOCAL. Zero cost. Three seconds. It recognized this relates to what it already learned.

And watch — after 14 queries across five different domains..."

*[Show metrics dashboard: 71.4% local, 15 knowledge entries, $0.03 cost avoided]*

"71% resolved from memory. The agent learned 15 facts. And it's only getting smarter."

---

### Why Now + Market (1:45–2:10)

*[Cut back to camera]*

"This is possible now because local models just got good enough. Llama 3, Mistral, Phi — they can actually reason. And cloud costs are at peak. The pain is real.

This matters for enterprises deploying AI at scale, startups controlling unit economics, small businesses in Vietnam who can't afford big API bills, and looking ahead — personal AI assistants that learn your domain on your own hardware."

---

### The Ask & Close (2:10–2:30)

*[On camera, direct to viewer]*

"We have a working prototype today. The full SDK ships by August. We're looking for mentorship on go-to-market in Southeast Asia, connections to early adopters, and investor introductions.

The future of AI agents isn't just smarter models. It's agents that remember what they've learned.

That's Autodidact. Thank you."

*[End card: AUTODIDACT logo, your name, contact info]*

---

## Recording Plan

1. **Record talking head segments first** (the non-demo parts) — ~5 takes, pick the best
2. **Record the demo** — run the full demo, screen capture the terminal
3. **Edit together** — splice demo clips into the talking head video at the marked points
4. **Add subtitles** — CapCut or DaVinci Resolve auto-captions
5. **Export** — 1080p MP4, upload to YouTube/Google Drive

## Quick Recording Commands

```bash
# Run the demo for recording (clean DB each time)
cd demo-prototype
LOCAL_MODEL="llama3.2" EMBEDDING_MODEL="hf.co/CompendiumLabs/bge-base-en-v1.5-gguf" npm run demo
```

## Timing Checklist
- [ ] Total video under 3 minutes
- [ ] Demo clips: 30-45 seconds total
- [ ] Each segment practiced 3+ times
- [ ] Subtitles added
- [ ] End card with contact info
