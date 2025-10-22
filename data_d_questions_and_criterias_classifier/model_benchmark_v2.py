"""
Model Benchmark Script v2
Benchmarks NLI, Sentence-BERT, Cross-Encoder, and traditional embedding models
"""

import logging
import os
import sys
import time
import json
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
import numpy as np
from tqdm import tqdm

MODULE_DIR = Path(__file__).resolve().parent
ENV_PATH = MODULE_DIR.parent / ".env"


def _load_env_from_file():
    if not ENV_PATH.exists():
        return

    try:
        content = ENV_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logging.getLogger(__name__).warning(f"Failed to read .env file at {ENV_PATH}: {exc}")
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


sys.path.append(str(MODULE_DIR))

from questions_classifier_v2 import QuestionsClassifierV2
from validation_metrics import ValidationMetrics
from qdrant_storage import QdrantStorage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelBenchmarkV2:
    """Enhanced benchmark supporting NLI, Sentence-BERT, and Cross-Encoder approaches"""

    def __init__(
        self,
        models_config_path: str = None,
        config_path: str = None
    ):
        """
        Initialize benchmark

        Args:
            models_config_path: Path to models_config_v2.yaml
            config_path: Path to config.yaml
        """
        _load_env_from_file()

        # Load models configuration
        if models_config_path is None:
            models_config_path = Path(__file__).parent / "models_config_v2.yaml"

        with open(models_config_path, 'r', encoding='utf-8') as f:
            self.models_config = yaml.safe_load(f)

        # Load main configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize ValidationMetrics
        self.metrics_calculator = ValidationMetrics()
        self.huggingface_config = self.config.get("huggingface", {})
        self._hf_env_var = self.huggingface_config.get("token_env_var")
        self._hf_token_cached: Optional[str] = None

        # Qdrant storage
        qdrant_config = self.config.get("qdrant", {})
        host = qdrant_config.get("host", "localhost")
        port = qdrant_config.get("port", 6333)
        qdrant_url = f"http://{host}:{port}"

        self.qdrant = QdrantStorage(qdrant_url=qdrant_url)
        self.qdrant.collection_name = qdrant_config.get("collection_name", "pubmed_papers")

        # Initialize Qdrant client (synchronously)
        from qdrant_client import QdrantClient
        self.qdrant.client = QdrantClient(url=qdrant_url)

        # Results
        self.results = {}

        # Results directory
        benchmark_config = self.models_config.get("benchmark", {})
        self.benchmark_config = benchmark_config
        self.show_progress = benchmark_config.get("show_progress", True)
        self.results_dir = Path(__file__).parent / benchmark_config.get("results_dir", "benchmark_results_v2")
        self.results_dir.mkdir(exist_ok=True)

        logger.info("ModelBenchmarkV2 initialized")

    def _get_hf_token(self, required: bool = False) -> Optional[str]:
        if self._hf_token_cached is None:
            token_value = None
            if self._hf_env_var:
                raw = os.environ.get(self._hf_env_var)
                if raw:
                    token_value = raw.strip() or None
            self._hf_token_cached = token_value

        if required and not self._hf_token_cached:
            env_hint = self._hf_env_var or "HF_TOKEN"
            raise RuntimeError(
                f"Hugging Face token is required but not set. "
                f"Set the environment variable {env_hint} before running the benchmark."
            )

        return self._hf_token_cached

    def _model_requires_auth(self, model_name: Optional[str], explicit_flag: Optional[bool]) -> bool:
        if explicit_flag is not None:
            return explicit_flag

        if not model_name:
            return False

        required_models = self.huggingface_config.get("require_token_for_models", []) or []
        model_lower = model_name.lower()
        return any(model_lower == item.lower() for item in required_models)

    def get_validation_papers(self) -> List[Dict[str, Any]]:
        """Get validation papers from Qdrant"""
        logger.info("Fetching validation papers from Qdrant...")

        papers = self.qdrant.get_validation_papers()

        if not papers:
            logger.error("No validation papers found!")
            return []

        logger.info(f"Found {len(papers)} validation papers")

        # Check data structure
        for paper in papers:
            if "validation_questions" not in paper:
                logger.warning(f"Paper {paper.get('paper_url')} missing validation_questions")
            if "questions_classification" not in paper:
                logger.warning(f"Paper {paper.get('paper_url')} missing questions_classification")

        return papers

    def benchmark_model(
        self,
        model_key: str,
        model_config: Dict[str, Any],
        validation_papers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Benchmark one model

        Args:
            model_key: Model key from config
            model_config: Model configuration
            validation_papers: Validation papers

        Returns:
            Benchmark results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARKING: {model_key}")
        logger.info(f"Model: {model_config.get('name')}")
        logger.info(f"Approach: {model_config.get('approach')}")
        logger.info(f"Description: {model_config.get('description')}")
        logger.info(f"{'='*60}\n")

        result = {
            "model_key": model_key,
            "model_name": model_config.get("name"),
            "approach": model_config.get("approach"),
            "description": model_config.get("description"),
            "params": model_config.get("params"),
            "training_data": model_config.get("training_data"),
            "timestamp": datetime.now().isoformat(),
            "error": None
        }

        try:
            requires_auth = self._model_requires_auth(model_config.get("name"), model_config.get("requires_auth"))
            quantize = bool(model_config.get("generation_config", {}).get("quantization", False))
            random_seed = model_config.get("random_config", {}).get("seed")
            provider_value = model_config.get("provider")
            provider_normalized = provider_value.lower() if isinstance(provider_value, str) else None
            if provider_normalized in (None, "", "huggingface"):
                self._get_hf_token(required=requires_auth)
            else:
                self._get_hf_token(required=False)
            result["requires_auth"] = requires_auth

            # Create temporary config (deep copy to avoid mutating the base config)
            temp_config = copy.deepcopy(self.config)
            temp_config.setdefault("classifier", {})
            temp_config["classifier"]["model_name"] = model_config.get("name")
            temp_config["classifier"]["quantize"] = quantize
            temp_config["classifier"]["show_progress"] = self.show_progress
            if random_seed is not None:
                temp_config["classifier"]["random_seed"] = random_seed
            provider = provider_value
            if provider:
                temp_config["classifier"]["provider"] = provider
            else:
                temp_config["classifier"].pop("provider", None)
            api_config = model_config.get("api_config")
            if api_config:
                temp_config["classifier"]["api"] = api_config
            else:
                temp_config["classifier"].pop("api", None)
            generation_overrides = model_config.get("generation_config")
            if generation_overrides:
                temp_config["classifier"]["generation_config"] = generation_overrides
            else:
                temp_config["classifier"].pop("generation_config", None)

            temp_config_path = self.results_dir / f"temp_config_{model_key}.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_config, f)

            # Initialize classifier
            logger.info("Initializing classifier...")
            start_init = time.time()

            classifier = QuestionsClassifierV2(
                config_path=str(temp_config_path),
                model_name=model_config.get("name"),
                approach=model_config.get("approach", "auto"),
                use_gpu=self.models_config["benchmark"].get("use_gpu", True),
                quantize=quantize,
                requires_auth=requires_auth,
                random_seed=random_seed
            )

            init_time = time.time() - start_init
            result["init_time_sec"] = round(init_time, 2)

            logger.info(f"Model initialized in {init_time:.2f} seconds")

            # Get model info
            model_info = classifier.get_model_info()
            result["model_info"] = model_info

            # Classify validation papers
            logger.info(f"Classifying {len(validation_papers)} validation papers...")

            start_classification = time.time()
            classified_papers = []

            total_papers = len(validation_papers)
            paper_iterator = tqdm(
                validation_papers,
                desc=f"{model_key} - papers",
                leave=False,
                disable=not self.show_progress
            )

            for index, paper in enumerate(paper_iterator, start=1):
                paper_url = paper.get("paper_url", f"paper_{index}")
                full_text = paper.get("full_text", "")

                if not full_text:
                    logger.warning(f"Paper {paper_url} has no full_text!")
                    continue

                logger.info(f"  [{index}/{total_papers}] Classifying {paper_url}...")

                paper_start = time.time()

                # Classify
                classification_result = classifier.classify_paper(full_text)

                paper_time = time.time() - paper_start

                # Add results to paper
                paper_copy = paper.copy()
                paper_copy["questions_classification"] = classification_result["questions"]
                paper_copy["classification_time_sec"] = round(paper_time, 2)

                classified_papers.append(paper_copy)

                logger.info(f"    Classified in {paper_time:.2f} sec")

            total_classification_time = time.time() - start_classification
            result["total_classification_time_sec"] = round(total_classification_time, 2)
            result["avg_time_per_paper_sec"] = round(total_classification_time / len(classified_papers), 2) if classified_papers else 0

            logger.info(f"Classification complete in {total_classification_time:.2f} seconds")

            # Calculate metrics
            logger.info("Calculating validation metrics...")

            metrics = self.metrics_calculator.calculate_validation_metrics(classified_papers)

            # Format metrics
            formatted_metrics = self.metrics_calculator.format_metrics_for_display(metrics)

            result["metrics"] = formatted_metrics
            result["num_papers_classified"] = len(classified_papers)

            # Main metrics
            overall = formatted_metrics.get("overall", {})
            result["f1_weighted"] = overall.get("f1_weighted", 0.0)
            result["f1_macro"] = overall.get("f1_macro", 0.0)
            result["f1_micro"] = overall.get("f1_micro", 0.0)
            result["accuracy_weighted"] = overall.get("accuracy_weighted", 0.0)

            # Compare with baseline
            baseline_f1 = self.models_config.get("targets", {}).get("current_f1_weighted", 0.5891)
            improvement = result["f1_weighted"] - baseline_f1
            improvement_percent = (improvement / baseline_f1 * 100) if baseline_f1 > 0 else 0

            result["improvement_absolute"] = round(improvement, 4)
            result["improvement_percent"] = round(improvement_percent, 2)

            logger.info(f"\n{'='*60}")
            logger.info(f"RESULTS FOR {model_key}")
            logger.info(f"{'='*60}")
            logger.info(f"Approach:            {model_config.get('approach')}")
            logger.info(f"F1-Score (Weighted): {result['f1_weighted']:.4f}")
            logger.info(f"F1-Score (Macro):    {result['f1_macro']:.4f}")
            logger.info(f"Accuracy (Weighted): {result['accuracy_weighted']:.4f}")
            logger.info(f"Improvement:         {improvement:+.4f} ({improvement_percent:+.2f}%)")
            logger.info(f"Total time:          {total_classification_time:.2f} sec")
            logger.info(f"Avg time/paper:      {result['avg_time_per_paper_sec']:.2f} sec")
            logger.info(f"{'='*60}\n")

            # Save detailed results
            if self.models_config["benchmark"].get("save_detailed_results", True):
                result_file = self.results_dir / f"{model_key}_detailed.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"Detailed results saved to {result_file}")

            # Delete temp config
            temp_config_path.unlink()

            # Clear memory
            del classifier
            import gc
            gc.collect()

            # Clear CUDA cache
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
            except:
                pass

        except Exception as e:
            logger.error(f"Error benchmarking {model_key}: {e}", exc_info=True)
            result["error"] = str(e)
            result["metrics"] = None

        return result

    def run_benchmark(self) -> Dict[str, Any]:
        """Run benchmark on all models"""
        logger.info("\n" + "="*80)
        logger.info("MODEL BENCHMARK V2 STARTED")
        logger.info("="*80 + "\n")

        validation_papers = self.get_validation_papers()

        if not validation_papers:
            logger.error("Cannot run benchmark without validation papers!")
            return {"error": "No validation papers found"}

        logger.info(f"Using {len(validation_papers)} validation papers\n")

        models = self.models_config.get("models", {})
        enabled_models = {
            key: config for key, config in models.items()
            if config.get("enabled", True)
        }

        if not enabled_models:
            logger.error("No models enabled!")
            return {"error": "No models enabled"}

        model_items = list(enabled_models.items())

        logger.info(f"Will benchmark {len(model_items)} models:")
        for key, config in model_items:
            logger.info(f"  - {key}: {config.get('name')} ({config.get('approach')})")
        logger.info("")

        benchmark_results: Dict[str, Any] = {}

        model_iterator = tqdm(
            model_items,
            desc="Models",
            leave=False,
            disable=not self.show_progress
        )

        for model_key, model_config in model_iterator:
            result = self.benchmark_model(model_key, model_config, validation_papers)
            benchmark_results[model_key] = result

            time.sleep(2)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_models_tested": len(benchmark_results),
            "num_validation_papers": len(validation_papers),
            "results": benchmark_results
        }

        summary_file = self.results_dir / "benchmark_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Summary saved to {summary_file}")

        self._create_comparison_table(benchmark_results)

        return summary

    def _create_comparison_table(self, results: Dict[str, Any]):
        """Create comparison table"""
        logger.info("\n" + "="*120)
        logger.info("BENCHMARK COMPARISON TABLE V2")
        logger.info("="*120 + "\n")

        # Header
        header = f"{'Model':<25} {'Approach':<15} {'F1 (W)':<10} {'F1 (Ma)':<10} {'Acc (W)':<10} {'Improve':<12} {'Time/paper':<12} {'Status':<15}"
        logger.info(header)
        logger.info("-"*120)

        # Sort by F1-Weighted
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get("f1_weighted", 0),
            reverse=True
        )

        best_f1 = 0
        best_model = None

        for model_key, result in sorted_results:
            if result.get("error"):
                row = f"{model_key:<25} {'-':<15} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<12} {'-':<12} {'Failed':<15}"
                logger.info(row)
                continue

            approach = result.get("approach", "unknown")
            f1_w = result.get("f1_weighted", 0)
            f1_ma = result.get("f1_macro", 0)
            acc_w = result.get("accuracy_weighted", 0)
            improvement = result.get("improvement_percent", 0)
            time_per_paper = result.get("avg_time_per_paper_sec", 0)

            # Status
            target_f1 = self.models_config.get("targets", {}).get("target_f1_weighted", 0.90)
            if f1_w >= target_f1:
                status = "⭐ TARGET MET!"
            elif f1_w > best_f1:
                status = "🏆 BEST"
                best_f1 = f1_w
                best_model = model_key
            else:
                status = "OK"

            improvement_str = f"{improvement:+.2f}%"

            row = f"{model_key:<25} {approach:<15} {f1_w:<10.4f} {f1_ma:<10.4f} {acc_w:<10.4f} {improvement_str:<12} {time_per_paper:<12.2f} {status:<15}"
            logger.info(row)

        logger.info("-"*120)

        # Best model
        if best_model:
            logger.info(f"\n🏆 WINNER: {best_model}")
            logger.info(f"   Approach: {results[best_model].get('approach')}")
            logger.info(f"   Model: {results[best_model].get('model_name')}")
            logger.info(f"   F1-Score (Weighted): {best_f1:.4f}")

            # Recommendation
            target_f1 = self.models_config.get("targets", {}).get("target_f1_weighted", 0.90)
            if best_f1 >= target_f1:
                logger.info(f"\n✅ TARGET ACHIEVED! F1 = {best_f1:.4f} >= {target_f1:.4f}")
                logger.info(f"   Recommendation: Update config.yaml with {results[best_model].get('model_name')}")
            else:
                gap = target_f1 - best_f1
                logger.info(f"\n⚠️  Target not met. Gap: {gap:.4f} ({gap/target_f1*100:.2f}%)")
                logger.info(f"   Recommendation: Try ensemble or hybrid approaches")

        logger.info("\n" + "="*120 + "\n")

        # Save table to file
        table_file = self.results_dir / "comparison_table.txt"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write("BENCHMARK COMPARISON TABLE V2\n")
            f.write("="*120 + "\n\n")
            f.write(header + "\n")
            f.write("-"*120 + "\n")

            for model_key, result in sorted_results:
                if result.get("error"):
                    row = f"{model_key:<25} {'-':<15} {'ERROR':<10} {'-':<10} {'-':<10} {'-':<12} {'-':<12} {'Failed':<15}"
                else:
                    approach = result.get("approach", "unknown")
                    f1_w = result.get("f1_weighted", 0)
                    f1_ma = result.get("f1_macro", 0)
                    acc_w = result.get("accuracy_weighted", 0)
                    improvement = result.get("improvement_percent", 0)
                    time_per_paper = result.get("avg_time_per_paper_sec", 0)
                    improvement_str = f"{improvement:+.2f}%"

                    target_f1 = self.models_config.get("targets", {}).get("target_f1_weighted", 0.90)
                    status = "TARGET MET!" if f1_w >= target_f1 else ("BEST" if model_key == best_model else "OK")

                    row = f"{model_key:<25} {approach:<15} {f1_w:<10.4f} {f1_ma:<10.4f} {acc_w:<10.4f} {improvement_str:<12} {time_per_paper:<12.2f} {status:<15}"

                f.write(row + "\n")

            f.write("-"*120 + "\n")

        logger.info(f"Comparison table saved to {table_file}")


def main():
    """Main function"""
    logger.info("Starting model benchmark v2...")

    benchmark = ModelBenchmarkV2()
    results = benchmark.run_benchmark()

    if "error" in results:
        logger.error(f"Benchmark failed: {results['error']}")
        return 1

    logger.info("\n✅ Benchmark v2 completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
