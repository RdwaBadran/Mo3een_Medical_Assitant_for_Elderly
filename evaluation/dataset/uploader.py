"""
evaluation/dataset/uploader.py
--------------------------------
Uploads test cases to LangSmith as datasets.

Creates four LangSmith datasets (one per tool):
  - mo3een-eval-symptoms
  - mo3een-eval-drug
  - mo3een-eval-lab
  - mo3een-eval-agent

Each dataset example contains:
  - inputs: query, language, tool
  - outputs: gold-standard expected values
  - metadata: metric IDs, weights, gate, scoring type

Idempotent: checks if datasets exist before creating them.
"""

from __future__ import annotations
import logging
from langsmith import Client

logger = logging.getLogger(__name__)

# Dataset name mapping
DATASET_NAMES = {
    "symptoms": "mo3een-eval-symptoms",
    "drug":     "mo3een-eval-drug",
    "lab":      "mo3een-eval-lab",
    "agent":    "mo3een-eval-agent",
}


def get_langsmith_client() -> Client:
    """Get a LangSmith client (reads LANGSMITH_API_KEY from env)."""
    return Client()


def dataset_exists(client: Client, dataset_name: str) -> bool:
    """Check if a dataset already exists in LangSmith."""
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        return len(datasets) > 0
    except Exception:
        return False


def upload_dataset(
    client: Client,
    tool_name: str,
    cases: list[dict],
    force: bool = False,
) -> str:
    """
    Upload test cases to LangSmith as a dataset.

    Args:
        client: LangSmith client
        tool_name: Tool key (symptoms, drug, lab, agent)
        cases: List of test case dicts
        force: If True, delete and recreate existing dataset

    Returns:
        Dataset name string
    """
    dataset_name = DATASET_NAMES.get(tool_name, f"mo3een-eval-{tool_name}")

    # Check if dataset already exists
    if dataset_exists(client, dataset_name):
        if not force:
            logger.info(f"[uploader] Dataset '{dataset_name}' already exists — skipping upload")
            return dataset_name
        else:
            logger.info(f"[uploader] Force mode — deleting existing dataset '{dataset_name}'")
            try:
                existing = list(client.list_datasets(dataset_name=dataset_name))
                for ds in existing:
                    client.delete_dataset(dataset_id=ds.id)
            except Exception as e:
                logger.warning(f"[uploader] Failed to delete existing dataset: {e}")

    # Create the dataset
    logger.info(f"[uploader] Creating dataset '{dataset_name}' with {len(cases)} examples")

    try:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"Mo3een evaluation dataset for {tool_name} tool. "
                        f"Contains {len(cases)} curated and synthetic test cases "
                        f"covering safety and effectiveness metrics.",
        )

        # Upload each case as an example
        for i, case in enumerate(cases):
            try:
                client.create_example(
                    inputs=case.get("inputs", {}),
                    outputs=case.get("outputs", {}),
                    metadata=case.get("metadata", {}),
                    dataset_id=dataset.id,
                )
            except Exception as e:
                logger.warning(f"[uploader] Failed to upload case {i} to {dataset_name}: {e}")

        logger.info(f"[uploader] Successfully uploaded {len(cases)} examples to '{dataset_name}'")
        return dataset_name

    except Exception as e:
        logger.error(f"[uploader] Failed to create dataset '{dataset_name}': {e}")
        raise


def upload_all_datasets(
    cases_by_tool: dict[str, list[dict]],
    force: bool = False,
) -> dict[str, str]:
    """
    Upload all test cases to LangSmith.

    Args:
        cases_by_tool: Dict mapping tool_name -> list of cases
        force: If True, recreate existing datasets

    Returns:
        Dict mapping tool_name -> dataset_name
    """
    client = get_langsmith_client()
    dataset_names = {}

    for tool_name, cases in cases_by_tool.items():
        if not cases:
            logger.warning(f"[uploader] No cases for tool '{tool_name}' — skipping")
            continue
        try:
            ds_name = upload_dataset(client, tool_name, cases, force=force)
            dataset_names[tool_name] = ds_name
        except Exception as e:
            logger.error(f"[uploader] Failed to upload dataset for '{tool_name}': {e}")

    return dataset_names


def ensure_datasets(force: bool = False) -> dict[str, str]:
    """
    Ensure all evaluation datasets exist in LangSmith.
    Loads from curated_cases.json and uploads if needed.

    Args:
        force: If True, recreate all datasets

    Returns:
        Dict mapping tool_name -> dataset_name
    """
    from evaluation.dataset.generator import load_curated_cases

    cases = load_curated_cases()
    return upload_all_datasets(cases, force=force)
