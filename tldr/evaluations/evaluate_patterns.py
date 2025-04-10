import weave
from weave import Evaluation, Scorer, MessagesPrompt, Dataset
import asyncio
import json
import typer
import functools
from pathlib import Path
from typing import List, Dict, Any

from tldr.agent import extract_code_patterns
from tldr.utils import get_openai_client

app = typer.Typer()


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Loads the dataset from a JSON file and converts repo_path to Path object."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # Convert repo_path string back to Path object
        for item in data:
            if "repo_path" in item:
                item["repo_path"] = Path(item["repo_path"])
        return data
    except FileNotFoundError:
        print(f"[red]Error: Dataset file not found at {file_path}[/red]")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        print(f"[red]Error: Could not decode JSON from {file_path}[/red]")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(code=1)


LLM_SCORER_PROMPT = MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator comparing lists of expected code patterns against actual patterns extracted by an AI model. "
                "Your goal is to provide a numerical score from 0.0 (no match) to 1.0 (perfect match) and brief reasoning. "
                "Consider:\\n- Are all expected patterns present in the actual output?\\n- Are there unexpected patterns in the actual output?\\n- How relevant are the actual patterns compared to the expected ones? (Exact matches are best)\\n\\n"
                "Respond ONLY with a valid JSON object containing 'score' (float) and 'reasoning' (string). "
                'Example format: {{{{ "score": 0.8, "reasoning": "Most expected patterns were found, but one was missing and one unexpected pattern was present." }}}}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Please evaluate the quality of the extracted patterns.\\n\\n"
                "Expected Patterns:\\n```json\\n{expected}\\n```\\n\\n"
                "Actual Extracted Patterns:\\n```json\\n{actual}\\n```\\n\\n"
                'Provide your evaluation ONLY in the specified JSON format: {{{{ "score": float, "reasoning": string }}}}.'
            ),
        },
    ]
)

PATTERN_MATCHER_PROMPT = MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator comparing two sets of code patterns. Your task is to find matches between the expected and actual patterns with some forgiveness for wording variations."
                "\n\nYou will analyze each pattern in the 'expected' list and find its best match (if any) in the 'actual' list."
                "\n\nFor each expected pattern, you should:"
                "\n1. Find its best match in the actual patterns list"
                "\n2. Determine a match score (0.0-1.0) where:"
                "\n   - 1.0: Perfect or near-perfect match (identical or only trivial differences)"
                "\n   - 0.7-0.9: Good match (same concept but different wording)"
                "\n   - 0.4-0.6: Partial match (overlapping concepts)"
                "\n   - 0.1-0.3: Poor match (slightly related)"
                "\n   - 0.0: No match found"
                "\n3. Provide brief reasoning for each match score"
                "\n\nAdditionally, identify any patterns in the 'actual' list that don't match any expected patterns."
                "\n\nRespond ONLY with a valid JSON object containing:"
                "\n- 'pattern_matches': Array of objects with 'expected', 'actual', 'score', and 'reasoning'"
                "\n- 'unmatched_actual': Array of strings (patterns from actual list with no corresponding expected pattern)"
                "\n- 'overall_score': Float representing the overall match quality (0.0-1.0)"
                "\n- 'overall_reasoning': String explaining the overall score"
            ),
        },
        {
            "role": "user",
            "content": (
                "Please directly compare these two sets of patterns with some forgiveness for slight variations in wording:\n\n"
                "Expected Patterns:\n```json\n{expected}\n```\n\n"
                "Actual Extracted Patterns:\n```json\n{actual}\n```\n\n"
                "Provide your evaluation ONLY in the specified JSON format."
            ),
        },
    ]
)


class LLMPatternScorer(Scorer):
    scorer_model: str = "gpt-4o"  # Default model can be set here

    @weave.op()
    def score(self, expected_patterns: List[str], model_output: Any) -> dict:
        """
        Uses an LLM to evaluate if the model_output patterns match the expected_patterns.
        Uses self.scorer_model.
        Returns a dictionary like {'score': float, 'reasoning': str}.
        """
        # scorer_model is now accessed via self.scorer_model

        # Handle non-list model_output
        if not isinstance(model_output, list):
            return {
                "score": 0.0,
                "reasoning": f"Invalid model output type: {type(model_output).__name__}. Expected list.",
                "error": True,
            }

        # Convert lists to JSON strings
        try:
            expected_json = json.dumps(expected_patterns, indent=2)
            actual_json = json.dumps(model_output, indent=2)
        except TypeError as e:
            return {
                "score": 0.0,
                "reasoning": f"Failed to serialize patterns to JSON: {e}",
                "error": True,
            }

        try:
            client = get_openai_client()

            messages = LLM_SCORER_PROMPT.format(
                expected=expected_json, actual=actual_json
            )

            response = client.chat.completions.create(
                model=self.scorer_model,  # Use model from class attribute
                messages=messages,
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            score_data = json.loads(content)

            if (
                not isinstance(score_data, dict)
                or "score" not in score_data
                or not isinstance(score_data["score"], (int, float))
                or "reasoning" not in score_data
                or not isinstance(score_data["reasoning"], str)
            ):
                raise ValueError("Scorer LLM response did not match expected format.")

            score_data["score"] = max(0.0, min(1.0, float(score_data["score"])))
            return score_data

        except json.JSONDecodeError as e:
            response_preview = "<unavailable>"
            try:
                response_preview = content[:100] + "..."
            except NameError:
                pass
            reason = f"Failed to parse scorer LLM JSON response: {e}. Response: {response_preview}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}
        except Exception as e:
            reason = f"Error calling scorer LLM: {e}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}


class PatternMatcher(Scorer):
    matcher_model: str = "gpt-4o"  # Default model

    @weave.op()
    def score(self, expected_patterns: List[str], model_output: Any) -> dict:
        """
        Directly compares expected and actual patterns with some forgiveness for variations.
        Returns detailed pattern matching information and overall score.
        """
        if not isinstance(model_output, list):
            return {
                "score": 0.0,
                "reasoning": f"Invalid model output type: {type(model_output).__name__}. Expected list.",
                "error": True,
            }

        try:
            expected_json = json.dumps(expected_patterns, indent=2)
            actual_json = json.dumps(model_output, indent=2)
        except TypeError as e:
            return {
                "score": 0.0,
                "reasoning": f"Failed to serialize patterns to JSON: {e}",
                "error": True,
            }

        try:
            client = get_openai_client()

            messages = PATTERN_MATCHER_PROMPT.format(
                expected=expected_json, actual=actual_json
            )

            response = client.chat.completions.create(
                model=self.matcher_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000,  # Increased for detailed matching
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            match_data = json.loads(content)

            # Validate response format
            required_fields = [
                "pattern_matches",
                "unmatched_actual",
                "overall_score",
                "overall_reasoning",
            ]
            if not all(field in match_data for field in required_fields):
                missing = [f for f in required_fields if f not in match_data]
                raise ValueError(f"Matcher response missing required fields: {missing}")

            # Ensure score is in valid range
            match_data["score"] = max(0.0, min(1.0, float(match_data["overall_score"])))
            match_data["reasoning"] = match_data["overall_reasoning"]

            return match_data

        except json.JSONDecodeError as e:
            response_preview = "<unavailable>"
            try:
                response_preview = content[:100] + "..."
            except NameError:
                pass
            reason = f"Failed to parse matcher LLM JSON response: {e}. Response: {response_preview}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}
        except Exception as e:
            reason = f"Error calling matcher LLM: {e}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}


@weave.op()
def extract_code_patterns_with_model(
    repo_path: Path, diff_content: str, eval_model: str
) -> List[str]:
    """Wrapper to call extract_code_patterns with a specific model."""
    # Note: repo_path is part of the signature to match the dataset format,
    # but it's not directly used by extract_code_patterns anymore.
    # If other preprocessing needed repo_path, it could be used here.
    return extract_code_patterns(diff_content=diff_content, model=eval_model)


# --- 4. Preprocessing Function ---
def preprocess_example(example: Dict[str, Any], eval_model: str) -> Dict[str, Any]:
    """Selects diff_content and adds eval_model to the example dictionary."""
    return {
        "repo_path": Path("."),
        "diff_content": example["diff_content"],
        "eval_model": eval_model,
    }


@app.command()
def main(
    dataset_path: Path = typer.Option(
        "tldr/evaluations/datasets/pattern_eval_dataset.json",
        "--dataset",
        "-d",
        help="Path to the evaluation dataset JSON file.",
        exists=True,
        readable=True,
        dir_okay=False,
    ),
    eval_model: str = typer.Option(
        "gpt-3.5-turbo",
        "--eval-model",
        help="Model name to use for pattern extraction.",
    ),
    scorer_model: str = typer.Option(
        "gpt-4o",
        "--scorer-model",
        help="Model name to use for the LLM scorer/matcher instance.",
    ),
    use_direct_matching: bool = typer.Option(
        False,
        "--direct-matching",
        help="Use direct pattern matching instead of general scoring.",
    ),
    project_name: str = typer.Option(
        "tldr",
        "--project",
        "-p",
        help="Weave project name to log results to.",
    ),
):
    """Runs the Weave evaluation for the extract_code_patterns op."""
    # Initialize Weave first
    weave.init(project_name)

    # Choose which prompt to publish based on evaluation mode
    try:
        if use_direct_matching:
            published_prompt = weave.publish(
                PATTERN_MATCHER_PROMPT, name="pattern-matcher-prompt"
            )
            print(f"Published pattern matcher prompt: {published_prompt.ref.uri()}")
        else:
            published_prompt = weave.publish(
                LLM_SCORER_PROMPT, name="llm-pattern-scorer-prompt"
            )
            print(f"Published scorer prompt: {published_prompt.ref.uri()}")
    except Exception as e:
        print(f"[yellow]Warning: Failed to publish prompt: {e}[/yellow]")

    print(f"Evaluating model: {eval_model}")
    print(f"Scorer model: {scorer_model}")
    print(f"Using dataset: {dataset_path}")
    print(
        f"Evaluation mode: {'Direct pattern matching' if use_direct_matching else 'General LLM scoring'}"
    )
    print(f"Logging to Weave project: {project_name}")

    raw_example_dataset = load_dataset(dataset_path)
    preprocessor = functools.partial(preprocess_example, eval_model=eval_model)

    # Create a named weave.Dataset
    dataset_name = "pattern_extractor_dataset"

    example_dataset = Dataset(name=dataset_name, rows=raw_example_dataset)

    # Publish the dataset
    weave.publish(example_dataset)
    print(f"Published dataset with name: {dataset_name}")

    # Create the appropriate scorer based on evaluation mode
    if use_direct_matching:
        scorer_instance = PatternMatcher(matcher_model=scorer_model)
    else:
        scorer_instance = LLMPatternScorer(scorer_model=scorer_model)

    evaluation = Evaluation(
        dataset=example_dataset,
        scorers=[scorer_instance],
        preprocess_model_input=preprocessor,
    )

    print("Running evaluation...")

    # Run the evaluation asynchronously
    async def run_eval():
        _results = await evaluation.evaluate(extract_code_patterns_with_model)
        print("Evaluation complete.")
        print(f"View results in Weave project: {project_name}")

    asyncio.run(run_eval())


if __name__ == "__main__":
    app()  # Run the typer app
