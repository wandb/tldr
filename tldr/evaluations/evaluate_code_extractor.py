import weave
from weave import Evaluation, Scorer, MessagesPrompt, Dataset
import asyncio
import json
import typer
import functools
from pathlib import Path
from typing import List, Dict, Any

from tldr.agent import extract_relevant_code
from tldr.utils import get_openai_client

app = typer.Typer()


def load_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Loads the dataset from a JSON file."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
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


# Precision Scorer Prompt - Evaluates how well the extraction avoided irrelevant code
PRECISION_SCORER_PROMPT = MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator analyzing code extraction precision. "
                "Your task is to evaluate how well the model extracted only the relevant parts "
                "of code based on a diff, avoiding the inclusion of irrelevant or unrelated code sections. "
                "High precision means the extraction is focused and efficient, containing just what's needed. "
                "You should consider:\n"
                "- Did the extraction avoid including unrelated functions or classes?\n"
                "- Did it exclude irrelevant code that wasn't affected by the diff?\n"
                "- Is the extraction concise while still capturing the important context?\n\n"
                "Respond ONLY with a valid JSON object containing 'score' (float 0.0-1.0) and 'reasoning' (string). "
                'Example format: {{{{ "score": 0.8, "reasoning": "The extraction is focused on the affected code but included some unnecessary imports." }}}}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the PRECISION of this code extraction.\\n\\n"
                "Original Diff:\\n```\\n{diff}\\n```\\n\\n"
                "Full Original File Content:\\n```\\n{file_content}\\n```\\n\\n"
                "Expected Extraction:\\n```\\n{expected}\\n```\\n\\n"
                "Actual Model Extraction:\\n```\\n{actual}\\n```\\n\\n"
                'Provide your evaluation ONLY in the specified JSON format focusing on PRECISION: {{{{ "score": float, "reasoning": string }}}}'
            ),
        },
    ]
)

# Recall Scorer Prompt - Evaluates how well the extraction included all necessary code
RECALL_SCORER_PROMPT = MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator analyzing code extraction recall. "
                "Your task is to evaluate how completely the model extracted all the necessary parts "
                "of code based on a diff, including all important functions, classes, and context needed. "
                "High recall means the extraction includes all affected code and necessary context. "
                "You should consider:\n"
                "- Did the extraction include all functions/classes affected by the diff?\n"
                "- Were important imports and dependencies included?\n"
                "- Is all necessary context present to understand the changes?\n\n"
                "Respond ONLY with a valid JSON object containing 'score' (float 0.0-1.0) and 'reasoning' (string). "
                'Example format: {{{{ "score": 0.8, "reasoning": "The extraction included all the modified functions but missed a relevant import." }}}}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the RECALL of this code extraction.\\n\\n"
                "Original Diff:\\n```\\n{diff}\\n```\\n\\n"
                "Full Original File Content:\\n```\\n{file_content}\\n```\\n\\n"
                "Expected Extraction:\\n```\\n{expected}\\n```\\n\\n"
                "Actual Model Extraction:\\n```\\n{actual}\\n```\\n\\n"
                'Provide your evaluation ONLY in the specified JSON format focusing on RECALL: {{{{ "score": float, "reasoning": string }}}}'
            ),
        },
    ]
)


class PrecisionScorer(Scorer):
    """Evaluates how well the extraction avoided irrelevant code"""

    scorer_model: str = "gpt-4o"  # Default model

    @weave.op()
    def score(
        self, file_content: str, diff: str, expected_extraction: str, model_output: Any
    ) -> dict:
        """Evaluate extraction precision - how well it avoided irrelevant code"""
        # Handle non-string model_output
        if not isinstance(model_output, str):
            return {
                "score": 0.0,
                "reasoning": f"Invalid model output type: {type(model_output).__name__}. Expected string.",
                "error": True,
            }

        try:
            client = get_openai_client()

            messages = PRECISION_SCORER_PROMPT.format(
                diff=diff,
                file_content=file_content,
                expected=expected_extraction,
                actual=model_output,
            )

            response = asyncio.run(
                client.chat.completions.create(
                    model=self.scorer_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
            )
            content = response.choices[0].message.content
            score_data = json.loads(content)

            # Validate response format
            required_fields = ["score", "reasoning"]
            if not all(field in score_data for field in required_fields):
                missing = [f for f in required_fields if f not in score_data]
                raise ValueError(
                    f"Precision scorer response missing required fields: {missing}"
                )

            # Ensure score is in valid range
            score_data["score"] = max(0.0, min(1.0, float(score_data["score"])))
            return score_data

        except json.JSONDecodeError as e:
            response_preview = "<unavailable>"
            try:
                response_preview = content[:100] + "..."
            except NameError:
                pass
            reason = f"Failed to parse precision scorer JSON response: {e}. Response: {response_preview}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}
        except Exception as e:
            reason = f"Error calling precision scorer: {e}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}


class RecallScorer(Scorer):
    """Evaluates how well the extraction included all necessary code"""

    scorer_model: str = "gpt-4o"  # Default model

    @weave.op()
    def score(
        self, file_content: str, diff: str, expected_extraction: str, model_output: Any
    ) -> dict:
        """Evaluate extraction recall - how well it included all necessary code"""
        # Handle non-string model_output
        if not isinstance(model_output, str):
            return {
                "score": 0.0,
                "reasoning": f"Invalid model output type: {type(model_output).__name__}. Expected string.",
                "error": True,
            }

        try:
            client = get_openai_client()

            messages = RECALL_SCORER_PROMPT.format(
                diff=diff,
                file_content=file_content,
                expected=expected_extraction,
                actual=model_output,
            )

            response = asyncio.run(
                client.chat.completions.create(
                    model=self.scorer_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
            )
            content = response.choices[0].message.content
            score_data = json.loads(content)

            # Validate response format
            required_fields = ["score", "reasoning"]
            if not all(field in score_data for field in required_fields):
                missing = [f for f in required_fields if f not in score_data]
                raise ValueError(
                    f"Recall scorer response missing required fields: {missing}"
                )

            # Ensure score is in valid range
            score_data["score"] = max(0.0, min(1.0, float(score_data["score"])))
            return score_data

        except json.JSONDecodeError as e:
            response_preview = "<unavailable>"
            try:
                response_preview = content[:100] + "..."
            except NameError:
                pass
            reason = f"Failed to parse recall scorer JSON response: {e}. Response: {response_preview}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}
        except Exception as e:
            reason = f"Error calling recall scorer: {e}"
            print(f"[red]{reason}[/red]")
            return {"score": 0.0, "reasoning": reason, "error": True}


def preprocess_example(example: Dict[str, Any], eval_model: str) -> Dict[str, Any]:
    """
    Preprocesses a single example by running the extractor on it.
    The file_content is now expected to already have the diff applied.
    """
    file_content = example["file_content"]
    diff = example["diff"]
    expected_extraction = example["expected_extracted_code"]

    return {
        "file_content": file_content,
        "diff": diff,
        "expected_extraction": expected_extraction,
    }


@app.command()
def main(
    dataset_path: Path = typer.Option(
        "tldr/evaluations/datasets/code_extractor_eval_dataset.json",
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
        help="Model name to use for code extraction.",
    ),
    scorer_model: str = typer.Option(
        "gpt-4o",
        "--scorer-model",
        help="Model name to use for the LLM scorer instance.",
    ),
    project_name: str = typer.Option(
        "tldr",
        "--project",
        "-p",
        help="Weave project name to log results to.",
    ),
):
    """
    Evaluate the code extraction model on a dataset of examples.
    """
    # Initialize Weave with the project name
    weave.init(project_name)

    print(f"Loading dataset from {dataset_path}")
    raw_dataset = load_dataset(dataset_path)
    print(f"Loaded {len(raw_dataset)} examples")

    # Create both precision and recall scorers
    print(f"Using {scorer_model} for evaluation")
    precision_scorer = PrecisionScorer(scorer_model=scorer_model)
    recall_scorer = RecallScorer(scorer_model=scorer_model)

    # Preprocess all examples in parallel
    try:
        preprocess_fn = functools.partial(preprocess_example, eval_model=eval_model)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_examples = list(executor.map(preprocess_fn, raw_dataset))
        print(f"Processed {len(processed_examples)} examples")
    except Exception as e:
        print(f"[red]Error during preprocessing: {e}[/red]")
        processed_examples = []
        for example in raw_dataset:
            try:
                processed_example = preprocess_example(example, eval_model)
                processed_examples.append(processed_example)
            except Exception as e_inner:
                print(f"[red]Error processing example: {e_inner}[/red]")
                # Add a placeholder to maintain order
                processed_examples.append(
                    {
                        "file_content": example["file_content"],
                        "diff": example["diff"],
                        "expected_extraction": example["expected_extracted_code"],
                        "model_extraction": f"ERROR: {e_inner}",
                    }
                )

    # Define the evaluation
    eval_name = f"code_extractor_eval_{eval_model.replace('-', '_')}"

    # Create a named weave.Dataset
    dataset_name = "code_extractor_dataset"
    processed_dataset = Dataset(name=dataset_name, rows=processed_examples)

    # Publish the dataset
    weave.publish(processed_dataset)
    print(f"Published dataset with name: {dataset_name}")

    # Create the evaluation object with both scorers
    evaluation = Evaluation(
        name=eval_name,
        dataset=processed_dataset,
        scorers=[precision_scorer, recall_scorer],
    )

    # Define the function to be evaluated
    @weave.op()
    async def extract_code_for_eval(file_content: str, diff: str):
        return await extract_relevant_code(file_content, diff, eval_model)

    # Run the evaluation
    async def run_eval():
        print("\n=== Running evaluation ===")

        # This will evaluate the model and log results automatically
        results = await evaluation.evaluate(extract_code_for_eval)

        print("\n=== Evaluation Results ===")

        # Process precision scores
        if "PrecisionScorer" in results:
            score_data = results["PrecisionScorer"]
            if "score" in score_data:
                precision = score_data["score"].get("mean", 0.0)
                print(
                    f"\nPrecision score: {precision:.2f} (how well it excluded irrelevant code)"
                )
        else:
            print("\nNo precision scores available")

        # Process recall scores
        if "RecallScorer" in results:
            score_data = results["RecallScorer"]
            if "score" in score_data:
                recall = score_data["score"].get("mean", 0.0)
                print(
                    f"Recall score: {recall:.2f} (how well it included all needed code)"
                )
        else:
            print("\nNo recall scores available")

        # Calculate and display F1 score (harmonic mean of precision and recall)
        try:
            if "PrecisionScorer" in results and "RecallScorer" in results:
                precision = results["PrecisionScorer"]["score"].get("mean", 0.0)
                recall = results["RecallScorer"]["score"].get("mean", 0.0)

                # Avoid division by zero
                if precision + recall > 0:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    print(
                        f"F1 score: {f1_score:.2f} (harmonic mean of precision and recall)"
                    )
        except Exception as e:
            print(f"[yellow]Warning: Error calculating F1 score: {e}[/yellow]")

        # Print model latency if available
        if "model_latency" in results and "mean" in results["model_latency"]:
            latency = results["model_latency"]["mean"]
            print(f"Average model latency: {latency:.4f} seconds")

        # Print the raw results for debugging (optional)
        print("\nDetailed results:")
        print(json.dumps(results, indent=2))

    asyncio.run(run_eval())


if __name__ == "__main__":
    app()
