# Testing your LLM

Here is a snippet of code to help you verify that your LLM-related environment variables are set up correctly.

```python
from gpt_researcher.config.config import Config
from gpt_researcher.utils.llm import create_chat_completion
import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    cfg = Config()

    try:
        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages = [{"role": "user", "content": "sup?"}],
            temperature=0.35,
            llm_provider=cfg.smart_llm_provider,
            stream=True,
            max_tokens=cfg.smart_token_limit,
            llm_kwargs=cfg.llm_kwargs
        )
    except Exception as e:
        print(f"Error in calling LLM: {e}")

# Run the async function
asyncio.run(main())
```

## Replaying a run without calling the model

Once the check above passes, the next thing that costs you money is re-running a whole
research task to see whether a prompt, model or retriever change made any difference.

GPT Researcher's LLM calls go through `langchain-openai`, which honours **both**
`OPENAI_BASE_URL` and `OPENAI_API_BASE` (measured on gpt-researcher 0.16.0 /
langchain-openai 1.6.2 — both redirect `ChatOpenAI` to the given origin). That means a
proxy recorder can capture the run without any change to your code:

```console
orca record generic-openai -- python your_research.py   # records; runs normally
orca replay last                                        # the same run, no model called
```

[OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay) is one such tool
(Apache-2.0, Node 20+). Replay serves the recorded turns back, so the second command
spends nothing, and `orca replay last --from N --model <other>` replays the run up to
step N from the recording and then continues on a different model — the model is the
only variable.

Three limits worth knowing before you rely on it:

- A matching replay is **not** a determinism result. It proves the recorded exchange
  reproduces, not that a fresh research run would behave the same way.
- Replay blocks **model-provider** egress only. It is not a sandbox, and the
  **retrievers still hit the real web** on replay.
- Embedding calls are not captured by the default adapter, so a run whose behaviour
  depends on vector search can replay cleanly at the LLM layer and still not reproduce.
