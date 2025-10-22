"""
Backward-compatible shim that exposes PubmedBertClassifier under the legacy
BioformerClassifier name. The actual implementation lives in
``pubmedbert_classifier.py``.
"""

from pubmedbert_classifier import PubmedBertClassifier

BioformerClassifier = PubmedBertClassifier  # for legacy imports

__all__ = ["PubmedBertClassifier", "BioformerClassifier"]
