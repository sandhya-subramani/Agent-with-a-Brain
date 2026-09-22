# Agent with a Brain

A customer-support agent with a company brain, built with
[Strands Agents](https://strandsagents.com/) and [Cognee](https://docs.cognee.ai/).
Strands runs the agent loop. Cognee turns a folder of company documents into a knowledge
graph and serves as the agent's long-term memory.

The whole thing is about 160 lines of Python across two scripts. Swap the files in `data/`
for your own and it becomes an agent that knows your company.

## What it demonstrates

| Capability            | Where                    | What you see                                                        |
|-----------------------|--------------------------|---------------------------------------------------------------------|
| Memory injection      | `CogneeMemory.search`    | Relevant facts are recalled and placed in the prompt before every turn |
| Tools + memory        | `lookup_order`, `search_memory` | The agent combines a system of record with company knowledge  |
| Hooks                 | `AuditHook`              | Deterministic code that logs every tool call                        |
| Steering              | `RefundApproval`         | A Python rule that blocks a tool call before it runs                 |
| Memory write          | `CogneeMemory.add`       | The agent saves a new rule and knows it on the next run             |

## How it works

```
data/*.md, *.txt  ──►  ingest.py  ──►  Cognee knowledge graph (local, on disk)
                                              ▲            │
                                     remember │            │ recall
                                              │            ▼
user question  ──►  Strands Agent  ◄──  MemoryManager ◄──  CogneeMemory
                        │
                        ├── lookup_order, issue_refund   (tools)
                        ├── AuditHook                    (hook, after every tool call)
                        └── RefundApproval               (steering, before every tool call)
```

1. `ingest.py` reads every file in `data/` and passes it to `cognee.remember()`. Cognee
   extracts entities and relationships with an LLM, embeds them, and stores a graph in
   local databases inside this folder. It also renders the graph to `brain.html`.
2. `agent.py` wraps `cognee.recall()` and `cognee.remember()` in a small class that
   satisfies the Strands memory-store interface, then hands it to a `MemoryManager`.
3. Before every turn Strands calls `search`, and the results are injected into the prompt.
   The model also gets `search_memory` and `add_memory` tools for explicit lookups and
   writes.
4. A steering handler inspects each tool call before it executes. Refunds over $100 are
   redirected back to the model with an explanation, with no LLM involved in the decision.

## The integration

This is the entire bridge between the two libraries:

```python
class CogneeMemory:
    name = "company_brain"
    description = "Company policies, FAQ, product notes, and team chat."
    writable = True

    async def search(self, query, options=None):
        results = await cognee.recall(query)
        return [MemoryEntry(content=r.text) for r in results if getattr(r, "text", None)]

    async def add(self, content, metadata=None):
        await cognee.remember(content, self_improvement=False)


agent = Agent(
    tools=[lookup_order, issue_refund],
    memory_manager=MemoryManager(stores=[CogneeMemory()], add_tool_config=True),
)
```

## Requirements

- Python 3.10 to 3.14
- An AWS account with Amazon Bedrock access to a chat model and an embedding model.
  The defaults are `us.anthropic.claude-haiku-4-5-20251001-v1:0` and
  `amazon.titan-embed-text-v2:0` in `us-east-1`.
- AWS credentials configured as a named profile. No API keys are used. Both Strands
  (via boto3) and Cognee (via LiteLLM) read the standard AWS credential chain.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set AWS_PROFILE and AWS_PROFILE_NAME to your profile

python ingest.py          # builds the knowledge graph, about 90 seconds
```

`ingest.py` clears any previous graph first, so it is safe to rerun after editing `data/`.

## Run

```bash
python agent.py                          # four scripted questions
python agent.py "Can order 1003 be returned?"
python agent.py --chat                   # interactive, type exit to quit
open brain.html                          # the knowledge graph Cognee built
```

The scripted run walks through four questions:

1. **A return-window question.** No tool runs. Memory recall supplies the policy, including
   a holiday extension that overrides the product's normal window.
2. **A product complaint on a real order.** The agent looks up the order, then searches
   memory and finds a known issue and its fix.
3. **A refund request over the limit.** The model calls `issue_refund`, steering blocks
   it, and the agent explains that manager approval is needed. This rule is not in any data
   file. It lives only in code.
4. **A new rule from the manager.** The agent saves it with `add_memory`. Ask about it in a
   later run and the answer is there.

Every memory recall, tool call, hook, and steering decision prints a bracketed line so you
can follow what the agent is doing.

## Use your own data

1. Replace the files in `data/` with your own Markdown or text. Policies, FAQs, runbooks,
   exported chat logs all work.
2. Edit `ORDERS` and the two tools in `agent.py`, or point them at a real system.
3. Adjust the system prompt and the steering rule to match your domain.
4. Run `python ingest.py` again.

## Configuration

All settings live in `.env`. See `.env.example` for the full list. The important ones:

| Variable            | Purpose                                              |
|---------------------|------------------------------------------------------|
| `AWS_PROFILE`       | Profile Strands uses for Bedrock                     |
| `AWS_PROFILE_NAME`  | Same profile, read by Cognee                         |
| `AWS_REGION`        | Bedrock region                                       |
| `STRANDS_MODEL_ID`  | Chat model for the agent                             |
| `LLM_MODEL`         | Chat model Cognee uses to build the graph            |
| `EMBEDDING_MODEL`   | Embedding model Cognee uses for vector search        |

Cognee's databases are written to `.cognee_system/` and `.data_storage/` inside this folder,
set in `config.py`. Delete those two folders to start from nothing.

## Project layout

```
agent.py           the agent: memory store, tools, hook, steering, system prompt
ingest.py          data/ -> Cognee knowledge graph + brain.html
config.py          loads .env, pins Cognee storage paths, builds the Bedrock model
data/              sample company documents for a fictional retailer
requirements.txt   cognee[aws], strands-agents, python-dotenv
.env.example       configuration template
```

## Troubleshooting

- **Credential errors.** Confirm the profile works with
  `aws sts get-caller-identity --profile <name>` and that it has Bedrock model access.
- **Recall returns nothing.** The graph has not been built. Run `python ingest.py`.
- **Slow first turn.** Cognee initializes its local databases on first use. Later turns are
  faster.
- **Model access denied.** Enable the chat and embedding models in the Bedrock console for
  your region, or change the model IDs in `.env`.

## Learn more

- [Strands Agents SDK user guide](https://strandsagents.com/docs/user-guide/sdk/)
- [Strands memory overview](https://strandsagents.com/docs/user-guide/sdk/memory/overview/)
- [Strands hooks](https://strandsagents.com/docs/user-guide/sdk/agents/hooks/)
- [Strands steering](https://strandsagents.com/docs/user-guide/sdk/agents/interventions/steering/)
- [Cognee documentation](https://docs.cognee.ai/)
- [Cognee remember](https://docs.cognee.ai/core-concepts/main-operations/remember) and
  [recall](https://docs.cognee.ai/core-concepts/main-operations/recall)

## License

MIT-0. See [LICENSE](LICENSE).
