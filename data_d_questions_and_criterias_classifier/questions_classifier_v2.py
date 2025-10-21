"""
Questions and Criterias Classifier v2
Supports multiple approaches: NLI, Sentence-BERT, Cross-Encoder, and traditional embeddings
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class QuestionsClassifierV2:
    """
    Enhanced classifier supporting multiple approaches:
    - NLI (Natural Language Inference) for zero-shot classification
    - Sentence-BERT for proper semantic embeddings
    - Cross-Encoder for pairwise scoring
    - Traditional embeddings (baseline)
    """

    def __init__(
        self,
        config_path: str = None,
        model_name: str = None,
        approach: str = "auto",
        use_gpu: bool = True,
        quantize: bool = False
    ):
        """
        Initialize classifier

        Args:
            config_path: Path to config.yaml
            model_name: Model name (if None, uses config)
            approach: Classification approach (auto, nli, sentence_bert, cross_encoder, embedding)
            use_gpu: Use GPU if available
            quantize: Use quantization
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.questions = self.config.get("questions", {})
        self.criteria = self.config.get("criteria", {})
        self.classifier_config = self.config.get("classifier", {})

        # Model settings
        self.model_name = model_name or self.classifier_config.get("model_name", "bioformers/bioformer-8L")
        self.approach = approach
        self._use_gpu = use_gpu
        self._quantize = quantize

        # Models (lazy loading)
        self.model = None
        self.tokenizer = None
        self.device = None
        self._initialized = False

        # Approach-specific attributes
        self.nli_pipeline = None
        self.sbert_model = None
        self.cross_encoder = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.generation_config = None

        logger.info(f"QuestionsClassifierV2 created: model={self.model_name}, approach={self.approach}")

    def _detect_approach(self, model_name: str) -> str:
        """
        Auto-detect best approach based on model name

        Args:
            model_name: Model name

        Returns:
            Approach name
        """
        model_lower = model_name.lower()

        # LLM Generation models (3-4B autoregressive models)
        if any(keyword in model_lower for keyword in ['llama', 'mistral', 'phi', 'qwen', 'gemma', 'biomedlm', 'ministral']):
            return "llm_generation"

        # NLI models
        if any(keyword in model_lower for keyword in ['mnli', 'nli', 'bart-large-mnli', 'deberta']):
            return "nli"

        # Sentence-BERT models
        if any(keyword in model_lower for keyword in ['sentence-transformers', 'sbert', 'specter', 's-pubmedbert']):
            return "sentence_bert"

        # Cross-encoder models
        if 'cross-encoder' in model_lower:
            return "cross_encoder"

        # Default to embedding
        return "embedding"

    def _initialize(self):
        """Lazy initialization of model"""
        if self._initialized:
            return

        try:
            import torch

            # Detect approach if auto
            if self.approach == "auto":
                self.approach = self._detect_approach(self.model_name)
                logger.info(f"Auto-detected approach: {self.approach}")

            # Determine device
            if self._use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Initialize based on approach
            if self.approach == "nli":
                self._initialize_nli()
            elif self.approach == "sentence_bert":
                self._initialize_sentence_bert()
            elif self.approach == "cross_encoder":
                self._initialize_cross_encoder()
            elif self.approach == "embedding":
                self._initialize_embedding()
            elif self.approach == "llm_generation":
                self._initialize_llm_generation()
            else:
                raise ValueError(f"Unknown approach: {self.approach}")

            self._initialized = True
            logger.info(f"Model initialized successfully with {self.approach} approach")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    def _initialize_nli(self):
        """Initialize NLI (zero-shot classification) model"""
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading NLI model: {self.model_name}...")

        # Create zero-shot classification pipeline
        device_idx = 0 if self.device.type == "cuda" else -1

        # DeBERTa models require slow tokenizer
        if "deberta" in self.model_name.lower():
            logger.info("Detected DeBERTa model, using slow tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            self.nli_pipeline = pipeline(
                "zero-shot-classification",
                model=model,
                tokenizer=tokenizer,
                device=device_idx
            )
        else:
            # Other models can use fast tokenizer
            self.nli_pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device_idx
            )

        logger.info("NLI pipeline loaded successfully")

    def _initialize_sentence_bert(self):
        """Initialize Sentence-BERT model"""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading Sentence-BERT model: {self.model_name}...")

        self.sbert_model = SentenceTransformer(self.model_name)

        # Move to device
        if self.device.type == "cuda":
            self.sbert_model = self.sbert_model.to(self.device)

        logger.info("Sentence-BERT model loaded successfully")

    def _initialize_cross_encoder(self):
        """Initialize Cross-Encoder model"""
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading Cross-Encoder model: {self.model_name}...")

        self.cross_encoder = CrossEncoder(self.model_name)

        logger.info("Cross-Encoder loaded successfully")

    def _initialize_embedding(self):
        """Initialize traditional embedding model"""
        from transformers import AutoTokenizer, AutoModel
        import torch

        logger.info(f"Loading embedding model: {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        # Quantization
        if self._quantize and self.device.type == "cuda":
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                logger.info("Applied int8 quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        self.model.to(self.device)
        self.model.eval()

        logger.info("Embedding model loaded successfully")

    def _initialize_llm_generation(self):
        """Initialize LLM for generation-based classification"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
        import torch

        logger.info(f"Loading LLM generation model: {self.model_name}...")

        # Configure 8-bit quantization for GTX 1070 8GB
        quantization_config = None
        if self._quantize or self.device.type == "cuda":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                logger.info("Using 8-bit quantization for LLM")
            except Exception as e:
                logger.warning(f"8-bit quantization setup failed: {e}, loading without quantization")
                quantization_config = None

        # Load tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set padding token if not present
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        # Load model with quantization
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"

        try:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Move to device if not using device_map
            if quantization_config is None:
                self.llm_model.to(self.device)

            self.llm_model.eval()

            # Configure generation
            self.generation_config = GenerationConfig(
                max_new_tokens=5,  # Only need "Yes" or "No"
                temperature=0.0,   # Deterministic
                do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )

            logger.info("LLM generation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise

    def _classify_question_nli(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using NLI (zero-shot classification)

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text to avoid memory issues
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Binary classification with NLI
            # Hypothesis: "This paper {question}"
            hypothesis = f"suggests {question_text.lower()}"

            result = self.nli_pipeline(
                text_truncated,
                candidate_labels=["yes", "no"],
                hypothesis_template=f"This paper {{}}"
            )

            # Get scores
            labels = result['labels']
            scores = result['scores']

            # Find yes/no scores
            yes_idx = labels.index('yes') if 'yes' in labels else 0
            yes_score = scores[yes_idx]

            answer = yes_score > 0.5
            confidence = yes_score if answer else (1 - yes_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(1 - yes_score, 3)
                },
                "fragments": []  # TODO: implement fragment extraction
            }

        elif question_type == "multiclass":
            # Multiclass with NLI
            options = question_config.get("options", [])

            result = self.nli_pipeline(
                text_truncated,
                candidate_labels=options,
                hypothesis_template=f"This paper suggests {{}}"
            )

            best_label = result['labels'][0]
            best_score = result['scores'][0]

            return {
                "question_id": question_id,
                "answer": best_label,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": label, "score": round(score, 3)}
                    for label, score in zip(result['labels'], result['scores'])
                ],
                "fragments": []
            }

    def _classify_question_sentence_bert(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using Sentence-BERT (semantic similarity)

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        from sentence_transformers import util

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text
        max_text_length = 5000
        text_truncated = text[:max_text_length]

        # Get text embedding
        text_emb = self.sbert_model.encode(text_truncated, convert_to_tensor=True)

        if question_type == "binary":
            # Create contextualized prompts for yes/no
            yes_prompt = f"A scientific paper about aging that {question_text.lower().replace('does it', 'does').replace('?', '')}"
            no_prompt = f"A scientific paper about aging that does NOT {question_text.lower().replace('does it', '').replace('?', '')}"

            yes_emb = self.sbert_model.encode(yes_prompt, convert_to_tensor=True)
            no_emb = self.sbert_model.encode(no_prompt, convert_to_tensor=True)

            # Calculate similarities
            yes_sim = util.cos_sim(text_emb, yes_emb).item()
            no_sim = util.cos_sim(text_emb, no_emb).item()

            # Normalize scores to [0, 1]
            yes_score = (yes_sim + 1) / 2
            no_score = (no_sim + 1) / 2

            # Decision based on relative scores
            answer = yes_score > no_score
            confidence = max(yes_score, no_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(no_score, 3),
                    "yes_sim": round(yes_sim, 3),
                    "no_sim": round(no_sim, 3)
                },
                "fragments": []
            }

        elif question_type == "multiclass":
            # Multiclass with sentence embeddings
            options = question_config.get("options", [])

            option_prompts = [
                f"A scientific paper about aging where {question_text.lower().replace('?', '')}: {opt}"
                for opt in options
            ]

            option_embs = self.sbert_model.encode(option_prompts, convert_to_tensor=True)

            # Calculate similarities
            similarities = util.cos_sim(text_emb, option_embs)[0]

            # Normalize
            scores = [(sim.item() + 1) / 2 for sim in similarities]

            # Get best option
            best_idx = np.argmax(scores)
            best_option = options[best_idx]
            best_score = scores[best_idx]

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": opt, "score": round(score, 3)}
                    for opt, score in zip(options, scores)
                ],
                "fragments": []
            }

    def _classify_question_cross_encoder(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using Cross-Encoder

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Score with cross-encoder
            pairs = [
                [text_truncated, f"This paper suggests: {question_text}"],
                [text_truncated, f"This paper does NOT suggest: {question_text}"]
            ]

            scores = self.cross_encoder.predict(pairs)
            yes_score = (scores[0] + 1) / 2  # Normalize
            no_score = (scores[1] + 1) / 2

            answer = yes_score > no_score
            confidence = max(yes_score, no_score)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "scores": {
                    "yes": round(yes_score, 3),
                    "no": round(no_score, 3)
                },
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])

            pairs = [
                [text_truncated, f"This paper suggests: {question_text} Answer: {opt}"]
                for opt in options
            ]

            scores = self.cross_encoder.predict(pairs)
            normalized_scores = [(s + 1) / 2 for s in scores]

            best_idx = np.argmax(normalized_scores)
            best_option = options[best_idx]
            best_score = normalized_scores[best_idx]

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "all_options": [
                    {"option": opt, "score": round(score, 3)}
                    for opt, score in zip(options, normalized_scores)
                ],
                "fragments": []
            }

    def _classify_question_embedding(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using traditional embeddings (baseline approach)
        """
        self._initialize()

        import torch

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Get text embedding
        def get_embedding(txt: str) -> np.ndarray:
            inputs = self.tokenizer(
                txt,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

            return embedding.cpu().numpy()[0]

        text_emb = get_embedding(text[:2000])

        if question_type == "binary":
            yes_prompt = f"Question: {question_text}\n\nAnswer: Yes"
            no_prompt = f"Question: {question_text}\n\nAnswer: No"

            yes_emb = get_embedding(yes_prompt)
            no_emb = get_embedding(no_prompt)

            yes_sim = float(np.dot(text_emb, yes_emb))
            no_sim = float(np.dot(text_emb, no_emb))

            answer = yes_sim > no_sim
            confidence = max(yes_sim, no_sim)

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])
            option_embeddings = []

            for option in options:
                option_prompt = f"Question: {question_text}\n\nAnswer: {option}"
                option_emb = get_embedding(option_prompt)
                similarity = float(np.dot(text_emb, option_emb))
                option_embeddings.append((option, similarity))

            best_option, best_similarity = max(option_embeddings, key=lambda x: x[1])

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_similarity, 3),
                "all_options": [
                    {"option": opt, "similarity": round(sim, 3)}
                    for opt, sim in option_embeddings
                ],
                "fragments": []
            }

    def _parse_llm_response(self, response_text: str) -> Tuple[Optional[bool], float]:
        """
        Parse LLM response to extract Yes/No answer

        Args:
            response_text: Generated text from LLM

        Returns:
            Tuple of (answer: bool or None, confidence: float)
        """
        response_lower = response_text.lower().strip()

        # Try to extract Yes/No from response
        # Look for exact matches first
        if response_lower == "yes":
            return True, 1.0
        elif response_lower == "no":
            return False, 1.0

        # Look for Yes/No at the beginning
        if response_lower.startswith("yes"):
            return True, 0.9
        elif response_lower.startswith("no"):
            return False, 0.9

        # Look for Yes/No anywhere in response
        if "yes" in response_lower and "no" not in response_lower:
            return True, 0.7
        elif "no" in response_lower and "yes" not in response_lower:
            return False, 0.7

        # Could not determine answer
        logger.warning(f"Could not parse LLM response: {response_text}")
        return None, 0.0

    def _classify_question_llm(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify using LLM generation

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        import torch

        question_text = question_config.get("text", "")
        question_type = question_config.get("type", "binary")

        # Truncate text to fit context
        max_text_length = 2000
        text_truncated = text[:max_text_length]

        if question_type == "binary":
            # Create prompt for binary question
            prompt = f"""You are a scientific paper analyzer specialized in aging research.

Context: {text_truncated}

Question: {question_text}

Answer with ONLY "Yes" or "No": """

            # Tokenize
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode response
            generated_ids = outputs.sequences[0][len(inputs["input_ids"][0]):]
            response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse response
            answer, confidence = self._parse_llm_response(response_text)

            # Fallback to default if parsing failed
            if answer is None:
                logger.warning(f"LLM returned unparseable response for {question_id}: '{response_text}', defaulting to No")
                answer = False
                confidence = 0.1

            return {
                "question_id": question_id,
                "answer": answer,
                "confidence": round(confidence, 3),
                "llm_response": response_text,
                "fragments": []
            }

        elif question_type == "multiclass":
            options = question_config.get("options", [])

            # Create prompt for multiclass question
            options_str = ", ".join(options)
            prompt = f"""You are a scientific paper analyzer specialized in aging research.

Context: {text_truncated}

Question: {question_text}
Options: {options_str}

Answer with ONLY one of the options: """

            # Tokenize
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            # Decode response
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            response_text = self.llm_tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Find best matching option
            response_lower = response_text.lower().strip()
            best_option = None
            best_score = 0.0

            for option in options:
                if option.lower() in response_lower:
                    # Exact match
                    best_option = option
                    best_score = 1.0
                    break
                elif response_lower in option.lower():
                    # Partial match
                    if best_score < 0.7:
                        best_option = option
                        best_score = 0.7

            # Fallback to first option if no match
            if best_option is None:
                logger.warning(f"Could not match LLM response '{response_text}' to any option, using first option")
                best_option = options[0]
                best_score = 0.1

            return {
                "question_id": question_id,
                "answer": best_option,
                "confidence": round(best_score, 3),
                "llm_response": response_text,
                "all_options": [{"option": opt, "score": 1.0 if opt == best_option else 0.0} for opt in options],
                "fragments": []
            }

    def classify_question(
        self,
        text: str,
        question_id: str,
        question_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify text by question using the selected approach

        Args:
            text: Paper text
            question_id: Question ID
            question_config: Question configuration

        Returns:
            Classification result
        """
        self._initialize()

        if self.approach == "nli":
            return self._classify_question_nli(text, question_id, question_config)
        elif self.approach == "sentence_bert":
            return self._classify_question_sentence_bert(text, question_id, question_config)
        elif self.approach == "cross_encoder":
            return self._classify_question_cross_encoder(text, question_id, question_config)
        elif self.approach == "embedding":
            return self._classify_question_embedding(text, question_id, question_config)
        elif self.approach == "llm_generation":
            return self._classify_question_llm(text, question_id, question_config)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

    def classify_paper(self, full_text: str) -> Dict[str, Any]:
        """
        Classify paper by all questions and criteria

        Args:
            full_text: Full paper text

        Returns:
            Classification results
        """
        logger.info(f"Starting paper classification with {self.approach} approach...")

        results = {
            "questions": {},
            "criteria": {},
            "timestamp": None,
            "approach": self.approach,
            "model_name": self.model_name
        }

        # Classify questions
        for q_id, q_config in self.questions.items():
            logger.info(f"Classifying {q_id}...")
            result = self.classify_question(full_text, q_id, q_config)
            results["questions"][q_id] = result

        # Classify criteria
        for c_id, c_config in self.criteria.items():
            logger.info(f"Classifying {c_id}...")
            result = self.classify_question(full_text, c_id, c_config)
            results["criteria"][c_id] = result

        from datetime import datetime
        results["timestamp"] = datetime.now().isoformat()

        logger.info("Classification complete")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        self._initialize()

        import torch

        info = {
            "model_name": self.model_name,
            "approach": self.approach,
            "device": str(self.device),
            "initialized": self._initialized,
            "num_questions": len(self.questions),
            "num_criteria": len(self.criteria)
        }

        if self._initialized and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)

        return info


if __name__ == "__main__":
    # Test different approaches
    logging.basicConfig(level=logging.INFO)

    test_text = """
    Aging is characterized by progressive decline in cellular function and increased
    susceptibility to disease. Telomere shortening has been proposed as a biomarker
    of biological aging. Studies have shown that telomere length is inversely correlated
    with chronological age and mortality risk across different species.
    """

    # Test NLI approach
    print("\n=== Testing NLI Approach ===")
    try:
        classifier_nli = QuestionsClassifierV2(
            model_name="facebook/bart-large-mnli",
            approach="nli",
            use_gpu=False
        )
        results = classifier_nli.classify_paper(test_text)
        print(f"Q1 result: {results['questions']['Q1']}")
    except Exception as e:
        print(f"NLI test failed: {e}")

    # Test Sentence-BERT approach
    print("\n=== Testing Sentence-BERT Approach ===")
    try:
        classifier_sbert = QuestionsClassifierV2(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            approach="sentence_bert",
            use_gpu=False
        )
        results = classifier_sbert.classify_paper(test_text)
        print(f"Q1 result: {results['questions']['Q1']}")
    except Exception as e:
        print(f"Sentence-BERT test failed: {e}")
