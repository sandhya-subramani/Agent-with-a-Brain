"""Step 1: build the company brain from the files in data/. Run once before the demo.

    python ingest.py     

Each file goes to cognee.remember(), which extracts entities and relationships with an
LLM and adds them to one knowledge graph. Writes brain.html when done.
"""

import asyncio
import time

import config
import cognee

DATA = config.ROOT / "data"


async def main() -> None:
    files = sorted(p for p in DATA.iterdir() if p.is_file() and not p.name.startswith("."))

    print("Clearing any previous graph...")
    await cognee.forget(everything=True)

    print(f"Building the XYZ Retail brain from {len(files)} files:\n")
    for path in files:
        t = time.time()
        await cognee.remember(f"Source file: {path.name}\n\n{path.read_text()}", self_improvement=False)
        print(f"  remembered {path.name:<22} {time.time() - t:5.1f}s")

    html = config.ROOT / "brain.html"
    await cognee.visualize_graph(str(html))
    print(f"\nDone. Open the graph:  open {html}")


if __name__ == "__main__":
    asyncio.run(main())
