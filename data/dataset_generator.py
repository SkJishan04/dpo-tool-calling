"""Generate preference dataset for DPO training."""

import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel
from config.schema_definitions import TOOL_REGISTRY, ToolCallJson, DirectAnswerJson


class PreferencePair(BaseModel):
    """A single preference pair for DPO training."""
    prompt: str
    chosen: str  # Correct/preferred output
    rejected: str  # Incorrect/less preferred output
    metadata: Dict[str, Any] = {}


class DatasetGenerator:
    """Generate preference dataset for tool calling decisions."""

    def __init__(self, tools=None):
        """Initialize with tool registry."""
        self.tools = tools or TOOL_REGISTRY
        self.pairs: List[PreferencePair] = []

    def generate_tool_call_prompts(self, num_samples: int = 100) -> List[PreferencePair]:
        """Generate prompts that should trigger tool calls."""
        pairs = []
        
        tool_prompts = [
            ("What is the current weather in Paris?", "weather", {"location": "Paris"}),
            ("Search for information about climate change", "search", {"query": "climate change", "num_results": 10}),
            ("Calculate 2**100", "calculator", {"expression": "2**100"}),
            ("Get all users from the customers table", "database_query", {"table": "customers"}),
            ("What's the temperature in Tokyo right now?", "weather", {"location": "Tokyo", "units": "celsius"}),
            ("Search for Python machine learning tutorials", "search", {"query": "Python machine learning tutorials"}),
            ("Solve: (100 + 50) * 2 - 30", "calculator", {"expression": "(100 + 50) * 2 - 30"}),
            ("Find products with price > 100", "database_query", {"table": "products", "filters": {"price": "> 100"}}),
        ]

        for prompt, tool_name, params in tool_prompts * (num_samples // len(tool_prompts) + 1):
            # Chosen: Correct tool call
            chosen = ToolCallJson(
                should_call_tool=True,
                tool_name=tool_name,
                parameters=params,
                reasoning=f"This query requires real-time data from {tool_name}."
            ).model_dump_json()

            # Rejected: Hallucinated tool call (wrong params)
            bad_params = {list(params.keys())[0]: "INVALID_VALUE"}
            rejected = ToolCallJson(
                should_call_tool=True,
                tool_name=tool_name,
                parameters=bad_params,
                reasoning="Attempting with invalid parameters."
            ).model_dump_json()

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={"type": "tool_call", "tool": tool_name}
            ))

        return pairs[:num_samples]

    def generate_direct_answer_prompts(self, num_samples: int = 100) -> List[PreferencePair]:
        """Generate prompts that should NOT call tools."""
        pairs = []
        
        answer_prompts = [
            ("What is Python?", "Python is a high-level, interpreted programming language..."),
            ("Explain machine learning", "Machine learning is a subset of artificial intelligence..."),
            ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet in..."),
            ("What is photosynthesis?", "Photosynthesis is a process by which plants convert light energy..."),
            ("Define recursion", "Recursion is a programming technique where a function calls itself..."),
            ("What are the benefits of exercise?", "Regular exercise provides numerous health benefits including..."),
            ("Explain quantum computing", "Quantum computing uses quantum bits (qubits) to process information..."),
        ]

        for prompt, answer in answer_prompts * (num_samples // len(answer_prompts) + 1):
            # Chosen: Direct answer (no tool call)
            chosen = DirectAnswerJson(
                should_call_tool=False,
                answer=answer,
                reasoning="This question can be answered from training knowledge."
            ).model_dump_json()

            # Rejected: Unnecessary tool call
            rejected = ToolCallJson(
                should_call_tool=True,
                tool_name="search",
                parameters={"query": prompt},
                reasoning="Calling search unnecessarily."
            ).model_dump_json()

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                metadata={"type": "direct_answer"}
            ))

        return pairs[:num_samples]

    def generate_full_dataset(self, num_tool_calls: int = 500, num_direct_answers: int = 500) -> List[PreferencePair]:
        """Generate full preference dataset."""
        pairs = []
        pairs.extend(self.generate_tool_call_prompts(num_tool_calls))
        pairs.extend(self.generate_direct_answer_prompts(num_direct_answers))
        self.pairs = pairs
        return pairs

    def save_to_jsonl(self, filepath: str) -> None:
        """Save dataset to JSONL format."""
        with open(filepath, 'w') as f:
            for pair in self.pairs:
                f.write(pair.model_dump_json() + '\n')
        print(f"Saved {len(self.pairs)} pairs to {filepath}")

    def save_to_json(self, filepath: str) -> None:
        """Save dataset to JSON format."""
        data = [pair.model_dump() for pair in self.pairs]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.pairs)} pairs to {filepath}")