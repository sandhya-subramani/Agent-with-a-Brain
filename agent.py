"""XYZ Retail support agent

    python agent.py                # with pre-configured questions
    python agent.py "question"     # ask one at a time
    python agent.py --chat         # interactive

Read top to bottom:
  MEMORY    CogneeMemory: the company brain, plugged in as a Strands memory store
  TOOLS     lookup_order and issue_refund, mock functions like the Strands workshop
  HOOK      AuditHook prints every tool call
  STEERING  RefundApproval blocks refunds over $100 before the tool runs
"""

import asyncio
import sys

import config
import cognee
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, HookProvider, HookRegistry
from strands.memory import MemoryManager
from strands.memory.types import MemoryEntry, MemoryInjectionConfig
from strands.vended_plugins.steering import Guide, Proceed, SteeringHandler


# ── MEMORY ─────────────────────────────────────────────────────────────────────────────
# Strands' MemoryManager works with any object that has `search` and `add`.
# Cognee's recall and remember are exactly those two operations.

class CogneeMemory:
    name = "company_brain"
    description = "XYZ Retail policies, shipping FAQ, product notes, and support team Slack."
    max_search_results = 5
    writable = True
    extraction = None

    async def search(self, query, options=None):
        print(f"\n   [memory] cognee.recall({query!r})")
        results = await cognee.recall(query)
        return [MemoryEntry(content=r.text) for r in results if getattr(r, "text", None)]

    async def add(self, content, metadata=None):
        print(f"\n   [memory] cognee.remember({content!r})")
        await cognee.remember(content, self_improvement=False)


def show_injection(ctx) -> str:
    print(f"   [memory] {len(ctx.entries)} entr{'y' if len(ctx.entries) == 1 else 'ies'} injected into the prompt")
    return "<company_brain>\n" + "\n".join(e.content for e in ctx.entries) + "\n</company_brain>"


# ── TOOLS ──────────────────────────────────────────────────────────────────────────────
# Mock order system. Tools know nothing about policy; the brain supplies that.

ORDERS = {
    "1001": {"customer": "Maria Lopez", "item": "Aurora Blender", "price": 89, "delivered": "2025-12-13"},
    "1002": {"customer": "James Carter", "item": "Nordic Oak Standing Desk", "price": 349, "delivered": "2026-01-08"},
    "1003": {"customer": "Aisha Khan", "item": "Trailhead 2-Person Tent", "price": 229, "delivered": "2026-01-15"},
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its order number.

    Args:
        order_id: The order number, e.g. 1002.
    """
    order = ORDERS.get(order_id)
    return str(order) if order else f"No order {order_id}."


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Refund an order to the customer's original payment method.

    Args:
        order_id: The order number.
        amount: Refund amount in dollars.
        reason: One sentence on why.
    """
    return f"Refunded ${amount:.0f} on order {order_id}. Reason: {reason}"


# ── HOOK ───────────────────────────────────────────────────────────────────────────────
# Deterministic code in the agent loop. Runs after every tool call, always.

class AuditHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def after_tool(self, event: AfterToolCallEvent) -> None:
        status = event.result.get("status", "?") if event.result else "cancelled"
        print(f"   [hook] {event.tool_use['name']}({event.tool_use.get('input', {})}) -> {status}")


# ── STEERING ───────────────────────────────────────────────────────────────────────────
# Policy that runs BEFORE a tool executes and can redirect the model. No LLM involved.

class RefundApproval(SteeringHandler):
    def __init__(self):
        super().__init__(context_providers=[])

    async def steer_before_tool(self, *, agent, tool_use, **kwargs):
        if tool_use["name"] == "issue_refund" and float(tool_use["input"].get("amount", 0)) > 100:
            print("\n   [steering] BLOCKED issue_refund: over $100, needs manager approval")
            return Guide(
                reason="Company rule: refunds over $100 require approval from Sam Patel, the support "
                "manager, before they are issued. Do not issue this refund. Tell the user to get Sam's "
                "approval in #support-team, and offer the alternative the company brain suggests, if any."
            )
        return Proceed(reason="ok")


# ── THE AGENT ──────────────────────────────────────────────────────────────────────────

agent = Agent(
    model=config.bedrock_model(),
    tools=[lookup_order, issue_refund],
    hooks=[AuditHook()],
    plugins=[RefundApproval()],
    memory_manager=MemoryManager(
        stores=[CogneeMemory()],
        add_tool_config=True,                                    # agent may save new facts
        injection=MemoryInjectionConfig(format=show_injection),  # recall before every turn
    ),
    system_prompt=(
        "You are the support assistant at XYZ Retail, helping a support agent on the phone with a "
        "customer. Facts inside <company_brain> are company policy and are authoritative; use them "
        "and name the source file or channel. Look up orders with lookup_order before acting on them, "
        "then call search_memory about that product to check for known issues before you advise. "
        "When the user states a new company fact or rule, save it with add_memory. "
        "Be concise: a few lines, no preamble."
    ),
)

DEMO = [
    "A customer bought an Aurora Blender that was delivered December 13. Today is January 20. Can she still return it?",
    "Order 1002 arrived and the customer says the desk wobbles. What should I do?",
    "He is not interested in the kit. He wants a full refund on order 1002.",
    "New rule from Sam: starting today, the express shipping cutoff is 2pm Pacific. Remember that.",
]


async def ask(text: str) -> None:
    print(f"\n{'─' * 78}\n>>> {text}\n")
    await agent.invoke_async(text)


async def main(argv: list[str]) -> None:
    if argv == ["--chat"]:
        while (q := input("\nyou> ").strip()) not in {"", "exit", "quit"}:
            await ask(q)
    elif argv:
        await ask(" ".join(argv))
    else:
        for q in DEMO:
            await ask(q)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
