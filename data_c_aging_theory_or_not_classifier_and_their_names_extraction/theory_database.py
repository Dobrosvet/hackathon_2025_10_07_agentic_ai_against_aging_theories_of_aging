"""
Расширенная база данных теорий старения
Содержит 30+ теорий с множественными вариациями названий
"""

from typing import Dict, List, Set
import logging

logger = logging.getLogger(__name__)


class AgingTheoryDatabase:
    """
    База данных теорий старения с расширенными вариациями названий
    Включает классические теории и современные Hallmarks of Aging
    """

    def __init__(self):
        """Инициализация базы данных теорий"""
        self.theories = self._build_theory_database()
        self.all_patterns = self._flatten_patterns()

        logger.info(f"Loaded {len(self.theories)} aging theories")
        logger.info(f"Total patterns: {len(self.all_patterns)}")

    def _build_theory_database(self) -> Dict[str, List[str]]:
        """
        Построение полной базы данных теорий старения

        Returns:
            Dict[theory_name, List[pattern_variations]]
        """
        return {
            # ============ HALLMARKS OF AGING (López-Otín et al. 2013/2023) ============

            "Genomic Instability": [
                "genomic instability",
                "genome instability",
                "dna instability",
                "chromosomal instability",
                "genetic instability",
                "nuclear dna damage",
                "genomic maintenance",
                "dna damage accumulation",
                "somatic mutations accumulation"
            ],

            "Telomere Attrition": [
                "telomere attrition",
                "telomere shortening",
                "telomere erosion",
                "telomere dysfunction",
                "telomere length",
                "telomerase deficiency",
                "telomere crisis",
                "replicative senescence",
                "hayflick limit",
                "telomere theory",
                "telomere hypothesis"
            ],

            "Epigenetic Alterations": [
                "epigenetic alterations",
                "epigenetic changes",
                "epigenetic drift",
                "dna methylation changes",
                "histone modification",
                "chromatin remodeling",
                "epigenetic clock",
                "horvath clock",
                "hannum clock",
                "dna methylation age",
                "epigenetic aging",
                "epigenome aging"
            ],

            "Loss of Proteostasis": [
                "loss of proteostasis",
                "proteostasis collapse",
                "protein homeostasis",
                "protein misfolding",
                "protein aggregation",
                "unfolded protein response",
                "upr dysfunction",
                "chaperone dysfunction",
                "proteasome dysfunction",
                "protein quality control"
            ],

            "Disabled Macroautophagy": [
                "disabled macroautophagy",
                "autophagy decline",
                "autophagy dysfunction",
                "autophagic flux",
                "mitophagy dysfunction",
                "lysosomal dysfunction",
                "autophagosome formation",
                "autophagy theory",
                "chaperone-mediated autophagy"
            ],

            "Deregulated Nutrient Sensing": [
                "deregulated nutrient sensing",
                "nutrient sensing pathways",
                "mtor pathway",
                "insulin signaling",
                "igf-1 pathway",
                "ampk pathway",
                "sirtuin pathway",
                "nutrient response",
                "metabolic dysregulation",
                "growth hormone pathway"
            ],

            "Mitochondrial Dysfunction": [
                "mitochondrial dysfunction",
                "mitochondrial theory",
                "mitochondrial theory of aging",
                "mitochondrial damage",
                "mitochondrial dna mutations",
                "mtdna mutations",
                "oxidative phosphorylation",
                "oxphos dysfunction",
                "mitochondrial membrane potential",
                "mitochondrial biogenesis",
                "mitochondrial dynamics"
            ],

            "Cellular Senescence": [
                "cellular senescence",
                "senescent cells",
                "senescence-associated secretory phenotype",
                "sasp",
                "senolytics",
                "senomorphics",
                "p16 expression",
                "p21 expression",
                "replicative senescence",
                "stress-induced senescence",
                "oncogene-induced senescence"
            ],

            "Stem Cell Exhaustion": [
                "stem cell exhaustion",
                "stem cell depletion",
                "stem cell aging",
                "hematopoietic stem cell",
                "tissue regeneration decline",
                "progenitor cell dysfunction",
                "stem cell niche",
                "regenerative capacity",
                "stem cell senescence"
            ],

            "Altered Intercellular Communication": [
                "altered intercellular communication",
                "cell-cell communication",
                "paracrine signaling",
                "extracellular vesicles",
                "exosome signaling",
                "neurohormonal signaling",
                "inflammatory signaling",
                "intercellular crosstalk"
            ],

            "Chronic Inflammation": [
                "chronic inflammation",
                "inflammaging",
                "inflammation theory",
                "inflammatory aging",
                "chronic low-grade inflammation",
                "sterile inflammation",
                "il-6 elevation",
                "tnf-alpha elevation",
                "cytokine dysregulation",
                "nf-kb activation"
            ],

            "Dysbiosis": [
                "dysbiosis",
                "microbiome aging",
                "gut microbiota",
                "microbiome dysbiosis",
                "bacterial diversity",
                "intestinal permeability",
                "leaky gut",
                "microbiome-gut-brain axis"
            ],

            # ============ CLASSIC AGING THEORIES ============

            "Free Radical Theory of Aging": [
                "free radical theory",
                "free radical theory of aging",
                "oxidative stress theory",
                "reactive oxygen species",
                "ros theory",
                "oxidative damage",
                "lipid peroxidation",
                "protein oxidation",
                "dna oxidation",
                "antioxidant defense",
                "redox imbalance"
            ],

            "DNA Damage Theory": [
                "dna damage theory",
                "dna damage accumulation",
                "dna repair decline",
                "somatic mutation theory",
                "error catastrophe theory",
                "nucleotide excision repair",
                "base excision repair",
                "mismatch repair",
                "double-strand break repair"
            ],

            "Disposable Soma Theory": [
                "disposable soma theory",
                "disposable soma",
                "soma theory",
                "energy allocation theory",
                "reproductive fitness",
                "resource allocation",
                "life history theory",
                "trade-off theory"
            ],

            "Antagonistic Pleiotropy Theory": [
                "antagonistic pleiotropy",
                "antagonistic pleiotropy theory",
                "pleiotropy theory of aging",
                "pleiotropic genes",
                "evolutionary theory of aging",
                "williams theory"
            ],

            "Immunosenescence": [
                "immunosenescence",
                "immune aging",
                "immune system aging",
                "thymic involution",
                "t cell aging",
                "b cell aging",
                "adaptive immunity decline",
                "innate immunity aging",
                "inflamm-aging"
            ],

            "Caloric Restriction Theory": [
                "caloric restriction",
                "dietary restriction",
                "cr mimetics",
                "calorie restriction",
                "energy restriction",
                "intermittent fasting",
                "time-restricted feeding",
                "fasting-mimicking diet"
            ],

            "Glycation Theory": [
                "glycation theory",
                "advanced glycation end products",
                "age products",
                "protein glycation",
                "maillard reaction",
                "non-enzymatic glycation",
                "rage receptors",
                "glycemic stress"
            ],

            "Neuroendocrine Theory": [
                "neuroendocrine theory",
                "neuroendocrine aging",
                "hormonal theory",
                "hypothalamic-pituitary axis",
                "growth hormone decline",
                "dhea decline",
                "melatonin decline",
                "hormonal dysregulation"
            ],

            "Wear and Tear Theory": [
                "wear and tear theory",
                "wear and tear",
                "mechanical aging",
                "cumulative damage",
                "physical stress aging"
            ],

            "Cross-Linking Theory": [
                "cross-linking theory",
                "protein cross-linking",
                "collagen cross-linking",
                "extracellular matrix aging",
                "tissue stiffening",
                "advanced glycation cross-links"
            ],

            "Rate of Living Theory": [
                "rate of living theory",
                "rate of living",
                "metabolic rate theory",
                "metabolic theory of aging",
                "oxygen consumption",
                "energy expenditure theory"
            ],

            # ============ MODERN SYSTEMS THEORIES ============

            "Network Theory of Aging": [
                "network theory of aging",
                "systems biology of aging",
                "protein interaction networks",
                "gene regulatory networks",
                "network dysfunction",
                "network entropy",
                "network resilience"
            ],

            "Entropy Theory of Aging": [
                "entropy theory",
                "thermodynamic aging",
                "disorder accumulation",
                "organizational decline",
                "loss of complexity",
                "biological entropy"
            ],

            "Programmed Aging Theory": [
                "programmed aging",
                "programmed longevity",
                "aging program",
                "genetic program",
                "aging clock",
                "quasi-program theory",
                "programmed death"
            ],

            # ============ EMERGING THEORIES ============

            "Microbiome Theory of Aging": [
                "microbiome theory",
                "gut microbiome aging",
                "microbial aging",
                "bacterial metabolites",
                "short-chain fatty acids",
                "microbiota-host interaction"
            ],

            "Extracellular Matrix Theory": [
                "extracellular matrix aging",
                "ecm aging",
                "matrix remodeling",
                "basement membrane",
                "collagen aging",
                "elastin degradation"
            ],

            "Circadian Rhythm Theory": [
                "circadian rhythm disruption",
                "circadian clock aging",
                "clock gene dysfunction",
                "sleep-wake cycle",
                "circadian dysregulation"
            ],

            "Mitochondrial-Lysosomal Axis": [
                "mitochondrial-lysosomal axis",
                "mitochondria-lysosome crosstalk",
                "organelle communication",
                "mitochondrial quality control"
            ],

            "Protein Acetylation Theory": [
                "protein acetylation",
                "acetylation aging",
                "sirtuin deacetylation",
                "nad+ decline",
                "nad metabolism"
            ]
        }

    def _flatten_patterns(self) -> Set[str]:
        """
        Создать плоский набор всех паттернов для быстрого поиска

        Returns:
            Set всех паттернов (lowercase)
        """
        all_patterns = set()
        for patterns in self.theories.values():
            all_patterns.update(p.lower() for p in patterns)
        return all_patterns

    def get_theory_patterns(self, theory_name: str) -> List[str]:
        """
        Получить все паттерны для конкретной теории

        Args:
            theory_name: Название теории

        Returns:
            Список паттернов или пустой список
        """
        return self.theories.get(theory_name, [])

    def get_all_theories(self) -> Dict[str, List[str]]:
        """Получить всю базу данных теорий"""
        return self.theories

    def search_pattern(self, pattern: str) -> List[str]:
        """
        Найти теории, содержащие данный паттерн

        Args:
            pattern: Паттерн для поиска

        Returns:
            Список названий теорий
        """
        pattern_lower = pattern.lower()
        matching_theories = []

        for theory_name, patterns in self.theories.items():
            if any(pattern_lower in p.lower() for p in patterns):
                matching_theories.append(theory_name)

        return matching_theories

    def get_statistics(self) -> Dict[str, int]:
        """
        Получить статистику по базе данных

        Returns:
            Dict со статистикой
        """
        total_patterns = sum(len(patterns) for patterns in self.theories.values())
        avg_patterns = total_patterns // len(self.theories) if self.theories else 0

        return {
            "total_theories": len(self.theories),
            "total_patterns": total_patterns,
            "avg_patterns_per_theory": avg_patterns,
            "unique_patterns": len(self.all_patterns)
        }


# Создать глобальный экземпляр для импорта
theory_db = AgingTheoryDatabase()


if __name__ == "__main__":
    # Тестирование
    logging.basicConfig(level=logging.INFO)

    db = AgingTheoryDatabase()
    stats = db.get_statistics()

    print("\n=== Aging Theory Database Statistics ===")
    print(f"Total theories: {stats['total_theories']}")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Average patterns per theory: {stats['avg_patterns_per_theory']}")
    print(f"Unique patterns: {stats['unique_patterns']}")

    print("\n=== Sample Theories ===")
    for i, (theory, patterns) in enumerate(list(db.theories.items())[:5], 1):
        print(f"\n{i}. {theory}")
        print(f"   Patterns ({len(patterns)}): {', '.join(patterns[:3])}...")

    print("\n=== Pattern Search Test ===")
    test_patterns = ["mitochondrial", "telomere", "senescence"]
    for pattern in test_patterns:
        theories = db.search_pattern(pattern)
        print(f"Pattern '{pattern}' found in: {theories}")
