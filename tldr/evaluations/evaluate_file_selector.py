import weave
from weave import Evaluation, Scorer, MessagesPrompt, Dataset
import asyncio
import json
import typer
import functools
from pathlib import Path
from typing import List, Dict, Any, Optional

from tldr.agent import select_relevant_files_and_lines
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


# Define the scorer prompt for file selection evaluation
FILE_SELECTOR_SCORER_PROMPT = MessagesPrompt(
    [
        {
            "role": "system",
            "content": (
                "You are an expert evaluator for file selection algorithms. Your task is to analyze how well a model selected relevant files and line ranges "
                "based on code diffs and grep results. You need to evaluate both the precision (did it select only relevant files/lines?) and "
                "recall (did it find all important dependencies?) of the selection."
                "\n\nYou should consider:"
                "\n- Did the model select files that show actual dependency relationships with the changed code?"
                "\n- Did the model identify appropriate line ranges that contain the relevant context?"
                "\n- Did the model miss any critical files or include irrelevant ones?"
                "\n- Are the line ranges too broad (including irrelevant code) or too narrow (missing context)?"
                "\n\nRespond ONLY with a valid JSON object containing:"
                "\n- 'score': A float from 0.0 to 1.0 indicating overall quality of selection"
                "\n- 'reasoning': String explaining your evaluation"
                "\n- 'file_matches': Array describing evaluation of each selected file"
                "\n- 'missed_files': If the expected selection contains files not found in the model output"
                '\n\nExample format: {{"score": 0.85, "reasoning": "The model selected most relevant files with appropriate line ranges, but missed one critical dependency."}}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the file selection quality.\\n\\n"
                "Diff Content:\\n```\\n{diff_content}\\n```\\n\\n"
                "Grep Results:\\n```\\n{grep_output}\\n```\\n\\n"
                "Expected Selection:\\n```json\\n{expected}\\n```\\n\\n"
                "Model Selection:\\n```json\\n{actual}\\n```\\n\\n"
                "Provide your evaluation ONLY in the specified JSON format with 'score' as a number between 0 and 1."
            ),
        },
    ]
)


class FileSelectorScorer(Scorer):
    """Evaluates how well the model selected relevant files and line ranges"""

    scorer_model: str = "gpt-4o"  # Default model

    @weave.op()
    def score(
        self,
        diff_content: str,
        grep_output: str,
        expected_selection: List[Dict[str, Any]],
        model_output: Any,
    ) -> dict:
        """Evaluate file selection quality"""
        # Handle non-list model_output
        if not isinstance(model_output, list):
            return {
                "score": 0.0,
                "reasoning": f"Invalid model output type: {type(model_output).__name__}. Expected list.",
                "error": True,
            }

        try:
            client = get_openai_client()

            # Convert expected and actual selections to JSON strings for comparison
            expected_json = json.dumps(expected_selection, indent=2)
            actual_json = json.dumps(model_output, indent=2)

            messages = FILE_SELECTOR_SCORER_PROMPT.format(
                diff_content=diff_content[:4000]
                + ("..." if len(diff_content) > 4000 else ""),
                grep_output=grep_output[:8000]
                + ("..." if len(grep_output) > 8000 else ""),
                expected=expected_json,
                actual=actual_json,
            )

            response = asyncio.run(
                client.chat.completions.create(
                    model=self.scorer_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
            )
            content = response.choices[0].message.content

            # Print the raw response for debugging
            print(f"Scorer raw response: {content[:200]}...")

            # Parse the JSON response with error handling
            try:
                score_data = json.loads(content)
                print(f"Parsed response keys: {list(score_data.keys())}")
                print(f"Score field type: {type(score_data.get('score'))}")
                print(f"Score field value: {score_data.get('score')}")
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return {
                    "score": 0.5,
                    "reasoning": f"Failed to parse JSON response: {e}",
                    "error": True,
                }

            # Create a fresh result dictionary with default values
            result = {
                "score": 0.5,  # Default middle score
                "reasoning": "Evaluation completed without detailed reasoning",
            }

            # Try to extract score from response
            if "score" in score_data:
                try:
                    # Handle both direct numbers and string representations
                    if isinstance(score_data["score"], (int, float)):
                        result["score"] = float(score_data["score"])
                    elif isinstance(score_data["score"], str):
                        # Remove any extra quotes or formatting
                        score_str = score_data["score"].strip("\"'")
                        result["score"] = float(score_str)
                    else:
                        print(f"Unknown score type: {type(score_data['score'])}")
                        if "overall_score" in score_data:
                            result["score"] = float(score_data["overall_score"])
                except (ValueError, TypeError) as e:
                    print(f"Error converting score to float: {e}")
                    # Keep default score
            elif "overall_score" in score_data:
                try:
                    result["score"] = float(score_data["overall_score"])
                except (ValueError, TypeError) as e:
                    print(f"Error converting overall_score to float: {e}")
                    # Keep default score

            # Try to extract reasoning
            if "reasoning" in score_data and score_data["reasoning"]:
                result["reasoning"] = score_data["reasoning"]
            elif "overall_reasoning" in score_data and score_data["overall_reasoning"]:
                result["reasoning"] = score_data["overall_reasoning"]

            # Ensure score is in valid range
            result["score"] = max(0.0, min(1.0, result["score"]))

            # Add any additional fields that might be useful
            if "file_matches" in score_data:
                result["file_matches"] = score_data["file_matches"]
            if "missed_files" in score_data:
                result["missed_files"] = score_data["missed_files"]

            return result

        except Exception as e:
            reason = f"Error calling scorer: {e}"
            print(f"Error: {reason}")
            print(f"Exception type: {type(e)}")
            import traceback

            traceback.print_exc()
            return {"score": 0.0, "reasoning": reason, "error": True}


@weave.op()
def select_files_with_model(
    diff_content: str, grep_output: str, eval_model: str
) -> List[Dict[str, Any]]:
    """
    Wrapper function that calls select_relevant_files_and_lines with provided model.
    """
    try:
        return asyncio.run(
            select_relevant_files_and_lines(
                combined_diff_content=diff_content,
                combined_grep_output=grep_output,
                model=eval_model,
            )
        )
    except Exception as e:
        print(f"Error in select_files_with_model: {e}")
        return []


def preprocess_example(example: Dict[str, Any], eval_model: str) -> Dict[str, Any]:
    """
    Preprocesses a single example for the evaluation.
    """
    return {
        "diff_content": example["diff_content"],
        "grep_output": example["grep_output"],
        "expected_selection": example["expected_selection"],
        "eval_model": eval_model,
    }


@app.command()
def main(
    dataset_path: Path = typer.Option(
        "tldr/evaluations/datasets/file_selector_eval_dataset.json",
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
        help="Model name to use for file selection.",
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
    Evaluate the file selection model on a dataset of examples.
    """
    # Initialize Weave with the project name
    weave.init(project_name)

    print(f"Loading dataset from {dataset_path}")
    raw_dataset = load_dataset(dataset_path)
    print(f"Loaded {len(raw_dataset)} examples")

    # Create the scorer
    print(f"Using {scorer_model} for evaluation")
    file_selector_scorer = FileSelectorScorer(scorer_model=scorer_model)

    # Preprocess examples
    try:
        preprocess_fn = functools.partial(preprocess_example, eval_model=eval_model)
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            processed_examples = list(executor.map(preprocess_fn, raw_dataset))
        print(f"Processed {len(processed_examples)} examples")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        processed_examples = []
        for example in raw_dataset:
            try:
                processed_example = preprocess_example(example, eval_model)
                processed_examples.append(processed_example)
            except Exception as e_inner:
                print(f"Error processing example: {e_inner}")
                processed_examples.append(
                    {
                        "diff_content": example["diff_content"],
                        "grep_output": example["grep_output"],
                        "expected_selection": example["expected_selection"],
                        "model_selection": [],
                    }
                )

    # Define the evaluation name
    eval_name = f"file_selector_eval_{eval_model.replace('-', '_')}"

    # Create a named weave.Dataset
    dataset_name = "file_selector_dataset"
    processed_dataset = Dataset(name=dataset_name, rows=processed_examples)

    # Publish the dataset
    weave.publish(processed_dataset)
    print(f"Published dataset with name: {dataset_name}")

    # Create the evaluation object
    evaluation = Evaluation(
        name=eval_name,
        dataset=processed_dataset,
        scorers=[file_selector_scorer],
    )

    # Define a wrapped function to be evaluated
    @weave.op()
    def select_files_for_eval(diff_content: str, grep_output: str):
        return select_files_with_model(diff_content, grep_output, eval_model)

    # Run the evaluation
    async def run_eval():
        print("\n=== Running evaluation ===")

        # This will evaluate the model and log results automatically
        results = await evaluation.evaluate(select_files_for_eval)

        print("\n=== Evaluation Results ===")

        # Process scores
        if "FileSelectorScorer" in results:
            score_data = results["FileSelectorScorer"]
            if "score" in score_data:
                score = score_data["score"].get("mean", 0.0)
                print(
                    f"\nFile selection score: {score:.2f} (overall quality of file and line selection)"
                )
        else:
            print("\nNo scores available")

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
