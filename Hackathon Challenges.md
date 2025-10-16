# Hackathon Challenges

Deep scientific challenges that push the boundaries of longevity research. These challenges focus on fundamental breakthroughs and novel approaches to understanding aging.

## Fundamental Track

### Phenotype Extractor

Build an AI system that assembles a cross-species database of phenotypic and morphological traits directly from scientific literature.

Tags: AI/ML, NLP, Data Engineering, Biology, Python, APIs

MAMMALIAN PHENOTYPES EXTRACTOR

Organismal trait data are scattered across papers and repositories, with no single standard for units, vocabularies, or provenance. We challenge participants to build an agentic AI system that assembles a cross-species database of phenotypic and morphological traits directly from the literature.

Your system should accept a species list (or taxon query) and a small trait vocabulary (e.g., adult body mass, head–body length, max lifespan, age at sexual maturity, gestation length, litter size) and autonomously produce a reproducible, versioned dataset with source provenance and a minimal web viewer suitable for immediate ML use.

#### System pipeline


1. Species Resolution  
    Resolving species names and IDs (e.g., NCBI TaxID/IUCN/GBIF)
2. Literature Search  
    Searching across PubMed, Crossref, and open repositories
3. Source Screening  
    Screening sources for relevance and open-access status
4. Data Extraction  
    Extracting numeric trait statements from PDFs/HTML
5. Provenance Capture  
    Capturing provenance with DOI/PMID (or publisher/repository URLs)
6. Normalization  
    Normalizing units and mapping to a controlled trait vocabulary
7. Context Handling  
    Handling context (wild vs. captive, sex, life stage) and deduplicating near-duplicates
8. Quality Assurance  
    Applying basic QA (outlier flags, license checks)
9. Version Control  
    Committing validated records with deterministic versioning (manifest hash)

#### Final Output

- Data Files  
  Normalized Parquet/CSV tables plus JSON provenance
- Web Viewer  
  A simple web page that displays traits with source links
- UI Controls  
  A UI control to trigger an agent run and view a live log

#### Evaluaton Framework

- Provenance & Traceability 25%  
  Each record includes source links. Agent run logs reviewed for source discovery process.
- Extraction & Normalization 25%  
  Correct unit conversions, plausible ranges, context handling, duplicate resolution.
- Reproducibility & Auditability 25%  
  Identical settings produce identical outputs. UI exposes re-run controls and logs.
- Coverage & Runtime 25%  
  Species × traits completion rate and end-to-end runtime performance.
- Bonus Considerations
  - Clade-aware applicability rules
  - Incremental updates as new papers appear
  - DwC-friendly exports/API
  - Clean viewer UX
  - Uncertainty-aware aggregation

#### Technical Specifications

- Source Handling
  Handle heterogeneous literature sources—peer-reviewed articles, open-access repositories, books/monographs, and institutional reports. When sources conflict, prioritize recency (most up-to-date, authoritative evidence).
- Format Processing
  Process heterogeneous formats—from structured database exports to HTML and PDFs. When tables or figures are embedded as images, use OCR/vision as needed to extract numeric trait statements.
- Uncertainty & Metadata
  All numeric outputs must include uncertainty metadata (e.g., point/range/percentile, source count and agreement, extractor confidence) and a clear record of normalization steps (unit conversions and assumptions).
- Transparency & Auditability
  The decision process must be transparent and auditable. For each skipped source, record a brief reason (e.g., not relevant, no extractable trait value, duplicate, parse failed). For included records, preserve a stable source link (DOI/PMID or publisher/repository URL) and, when detectable, a page/section hint.  
  Each run must emit a manifest and a structured event log so reviewers can trace retrieval → extraction → normalization → commit. Re-running with identical settings must produce identical outputs, enabling straightforward verification and trust.

#### Expected Impact

Linking our curated trait matrix to public reference genomes and regulatory annotations creates a tractable path from genotype to phenotype and, critically, to ranked intervention hypotheses for healthy lifespan.

- STEP 1 Join species via stable IDs
- STEP 2 Learn cross-species genome→trait mappings
- STEP 3 Run counterfactual intervention models

Long-term Vision

Over time, the same framework can absorb multi-omics (transcriptome, proteome, metabolome, epigenome, microbiome) and environmental/dietary covariates—including nutrition regimes, exposure profiles, and habitat/ecology descriptors.

With provenance on every datum and an active-learning loop to suggest the next most informative species/trait/perturbation, the resource evolves into a living, continuously improving map from genotype and context to organismal phenotype.

#### Resources & Guidelines

- A. TRAIT DATABASES
  - EOL TraitBank — cross-taxon trait repository
  - AnAge — longevity-related traits
  - PanTHERIA — mammal life-history/ecology
  - AVONET — avian morphology
  - FishBase — fishes and marine taxa
- B. TAXONOMY & JOIN KEYS
  - ASM Mammal Diversity Database
  - NCBI Taxonomy, GBIF Backbone
  - IUCN Red List — conservation status
- C. STANDARDS & ONTOLOGIES
  - Darwin Core (DwC) — trait record structure
  - OBO ontologies — VT, MP, PATO, UO, ENVO
  - EOL term URIs — consistent trait labels
- D. LITERATURE DISCOVERY
  - PubMed / PMC, Crossref, OpenAlex
  - Unpaywall — OA status
  - Repositories — Zenodo, Dryad, Figshare

Note: This list is neutral and non-exhaustive. We make no assumptions about how these databases are built internally; they are referenced as useful sources and compatibility targets.

#### Test Cases & Validation

- Required UI Functionality
  - The UI must be able to trigger an agent run for a selected species
  - The UI must be able to show a live run log during execution
  - On completion, the UI must be able to display the final trait table with provenance (DOI/PMID links)

### Singularis

Reforming Scientific Publishing

Create a cost-efficient knowledge extraction solution to represent research papers as interconnected knowledge graphs.

Tags: AI/ML, NLP, Graph Algorithms, Cost Optimization, Python, LLM


#### Mission Statement

The Singularis mission is to change how scientists interact with knowledge. Modern scientific literature is outdated; the so-called IMRAD format (which stands for Introduction, Method, Results, and Discussion) is highly obsolete.

It comes to us from the times when horses delivered the post. Singularis will change that and bring scientific data to a format that would correspond to the modern tempo of time, and immensely accelerate scientific progress.

The goal: boost the efficiency of both scientists and computers.

#### Why Couldn't it be Done Before?

All the scientific metrics that instruct funding, promotions, awards, and hiring are based on the reputation of the journal in which scientists publish. Everything depends on the journal's impact factors.

Representing data in a new format will impede funding, and therefore, the experiments in changing formats are as complex as changing the wheels of a moving vehicle.

The Singularis strategy: installing new wheels on the vehicle, transferring the traction to these new wheels, and then removing the old ones.

#### Central Idea

Our central idea is that the minimal publishable and citable unit should be smaller than a scientific paper. It could be one hypothesis, experiment, method, result, data set, or analysis.

- Hypothesis
- Experiment
- Method
- Result
- Dataset
- Analysis

To migrate the existing data to a new system, we will annotate the corpus of research papers with this new format. With the use of AI, papers will be represented as graphs consisting of elements connected with meaningful links.

#### Knowledge Graph Construction

1. STEP 1: Individual Graphs  
   Connect individual graphs to build a knowledge graph representing decades of scientific development with the flow of thought, breakthroughs, and failures.
2. STEP 2: Better Metrics  
   Provide much better metrics for scientific output that would characterize scientists based on this much more meaningful graph than the current papers connected with references.
3. STEP 3: True Impact  
   The position of scientists will not be influenced by hype, journal reputation, or political weight. Only the contribution to the ideas and results that people use will be accounted for.

#### Current Challenge

We have made significant progress in developing an algorithm based on LLM; however, these technologies are pretty expensive, and it might not be feasible or practical to apply them to the entire corpus of scientific data.
THE CHALLENGE FOR TEAMS

Make the algorithm more computationally efficient.

An example of paper annotation with a graph
![Singularis Example](images/singularis_example.png)
Example: Paper annotation represented as a knowledge graph with interconnected elements

#### Challenge Details

Currently, we are developing an algorithm to represent research papers as graphs. The task is to refine the LLM-based technology to improve its quality and, on this basis, build a more cost-efficient knowledge extraction solution, allowing for analysis of 50 million research papers.

Required Elements Structure
- Input Fact
- Hypothesis
- Experiment
- Technique
- Result
- Dataset
- Analysis
- Conclusion

With accurate representation of relationships between these elements.


Acceptable Approaches
- Pure Algoritmic  
  Regular expressions and rules without LLMs - welcome!
- Hybrid Approach  
  LLM + algorithms/regular expressions - acceptable
- LLM-Only  
  Not suitable due to cost and inference time

#### Evaluation Criteria

- Primary Metrics  
  - Code's Accuracy: building the graph correctly  
  - Computational Efficiency: cost and performance optimization

- Detailed Evaluation  
  - Completeness/accuracy: Precision/recall/F1 for each element type and relation
  - Correctness: Reference/citation extraction accuracy
  - Robustness: Handling different article formats
  - Cost analysis: CPU/GPU hours, $ per paper, tokens
  - Performance: Throughput (papers/hour)
  - Latency: Response time on fixed hardware

- Bonus Points  
  Fully algorithmic or hybrid solutions that demonstrate significant cost reduction at comparable quality relative to an LLM-only baseline.  
  Additional bonus points may be given to teams that improve the conceptual framework or provide additional ideas. These may play a role in tie situations.

#### Test Task

Algorithm Challenge

The algorithms built by the contestants will be challenged with research papers. The algorithms that will provide the best annotation (compared to the reference LLM-based algorithm) with the lowest computation requirements will be selected as the winner.

WINNING FORMULA: Best Annotation Quality + Lowest Computational Cost

#### Future Vision

In the future, we will add functionality for scientists to comment on the elements and publish their original results as elements outside the context of research papers.

Incentivized Publications
- Reproduced results
- Negative results
- Raw data
- Orphan results

Improved Peer Review
- Comment on granular elements
- Save time for reviewers
- Audience-evaluated quality
- Fair evaluation system

However, for now, the main aim is to build highly productive mechanisms for the annotation of existing papers.

### Aging Theories

Collect and classify aging theories from scientific literature to untangle the web of causality in aging research.

Tags: AI/ML, NLP, Data Mining, Biology, Literature Review, Python, Classification, Web Scraping, Data Analysis, Scientific Research

#### The problem

What do all five-year-olds understand that no senior academic does? Aging. As universal as it gets, yet its mechanisms remain unclear.

For decades, stubborn researchers from different fields have shared hundreds of opinions on why we age. These opinions are called the Theories of Aging. Some are groundless and unconvincing, while others may reveal underlying causes of age-related decline.

Together, they form pieces of a single puzzle – one whose solution could prevent 150,000 deaths every day. Let's solve it together!

🏆 WINNER REWARD  
The winner will be invited to write a paper together with ComputAge community.


#### Challenge Overview

In this challenge, you will collect and classify theories of aging. For example, you may find papers arguing that aging is driven by damage from free radicals, and others suggesting that the real cause lies in the diminished function of stem cells.

It may turn out that neither of these factors is central to aging. Or perhaps both represent independent mechanisms that contribute in parallel. Or could one even be the cause of the other?

The combinatorial complexity of such questions is huge, which is why we need to untangle the web of causality.

#### Challenge Structure

PART 1: COLLECT THEORIES  
Gather as many theories of aging and corresponding papers as possible. Create a comprehensive database linking theories to scientific literature.
- Literature Access Setup
- Classification Algorithm
- Crawler Implementation
- Theory Clustering

PART 2: EXTRACT DATA  
Use the collected papers to extract information about aging theories. For each paper, answer 9 research questions and write results in a table.
- Answer Q1-Q9 for each paper
- Build structured CSV table
- Include theory_id, paper_url, paper_name, paper_year
- Add Q1-Q9 answers for each paper

#### Research Questions

For each paper, answer these 9 critical questions:
- Q1: Does it suggest an aging biomarker (measurable entity reflecting aging pace or health state, associated with mortality or age-related conditions)? (Yes, quantitatively shown / Yes, but not shown / No)
- Q2: Does it suggest a molecular mechanism of aging? (Yes / No)
- Q3: Does it suggest a longevity intervention to test? (Yes / No)
- Q4: Does it claim that aging cannot be reversed? (Yes / No)
- Q5: Does it suggest a biomarker that predicts maximal lifespan differences between species? (Yes / No)
- Q6: Does it explain why the naked mole rat can live 40+ years despite its small size? (Yes / No)
- Q7: Does it explain why birds live much longer than mammals on average? (Yes / No)
- Q8: Does it explain why large animals live longer than small ones? (Yes / No)
- Q9: Does it explain why calorie restriction increases the lifespan of vertebrates? (Yes / No)

#### Evaluation System

Your final score will be the average of your ranks in two parts:

- PART 1: Theory Collection
  - Score = Σ log₁₀(papers per theory)
- PART 2: Paper Analysis
  - Accuracy vs ground truth

EXAMPLE CALCULATION:
Rank #1 in Part 1 + Rank #3 in Part 2 = (1+3)/2 = 2.0 final score

#### Submission Requirements

- PART 1: CSV TABLES
  - Table 1: Aging theories  
    Columns: theory_id, theory_name, number_of_collected_papers
  - Table 2: Collected papers  
    Columns: theory_id, paper_url, paper_name, paper_year
- PART 2: CSV TABLE  
  Single CSV table with extracted answers
  Columns: theory_id, paper_url, paper_name, paper_year, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9

GROUND TRUTH TEST:
We have 10 hidden papers with known correct answers. Missing papers from Part 1 = incorrect predictions.


#### Data Sources

PROVIDED DATA
We provide 15 example aging papers with Q1–Q9 already answered as a reference.

LITERATURE SOURCES
You will need to find sources yourself:
- PubMed
- bioRxiv
- Other scientific databases

#### Good Faith Disclaimer

We expect participants to apply new LLM techniques to untangle the complex network of aging theories. While we believe this is a good task for a hackathon, we also recognize that it may be difficult to design a perfectly fair scoring system on the first attempt.

It is possible to game the evaluation process to achieve higher scores at the expense of producing a meaningful solution. We ask participants not to exploit the scoring system but to act in the spirit of the task.

You will be asked to share your code, and your solution will be evaluated holistically. The jury may adjust the scoring system during the competition if necessary.


### Future of Evidence

Build an agentic AI system that automates the entire meta-analysis workflow from research question to publication-ready results.

Track: AI/ML, NLP, Statistics, Data Extraction, Medical Research, Python, R, PDF Processing, Meta-Analysis, Evidence Synthesis

#### The Challenge

Meta-analyses represent the gold standard for evidence synthesis in medicine, yet conducting them requires months of manual work by teams of experts. We challenge participants to build an agentic AI system that automates this entire workflow.

Your system should accept a research question as input, such as _"What is the effect of metformin on lifespan in animal models?"_, and autonomously produce a complete, publication-ready meta-analysis that matches the quality of expert human work.

The challenge will have a specific focus on biomedical interventions related to longevity and age-related diseases. This includes pharmaceuticals like metformin and rapamycin, as well as lifestyle interventions such as caloric restriction or the Mediterranean diet.

#### Meta Analysis Pipeline

AUTOMATED WORKFLOW

- SEARCH
  - PubMed, Embase, Cochrane
  - bioRxiv, medRxiv
  - Google Scholar
- SELECT
  - Inclusion/exclusion criteria
  - Relevance screening
  - Quality assessment
- EXTRACT
  - Statistical data from PDFs
  - Tables and figures
  - Effect sizes, confidence intervals
- ANALYZE
  - Statistical synthesis
  - Heterogeneity assessment
  - Forest plots, funnel plots

#### Evaluation Framework

- Study Selection Accuracy 25%
  Precision and recall against gold-standard dataset curated by expert reviewers.
- Data Extraction Accuracy 25%
  Agreement rates with manually extracted effect sizes, confidence intervals, and sample sizes.
- Statistical Validity 25%
  Reproduce published meta-analyses within acceptable margins, correct heterogeneity handling.
- Time Efficiency 25%
  Total computation time vs documented person-hours for manual meta-analysis.

ADVANCED CAPABILITIES (BONUS)
- ⚡Real-time updating as new studies are published
- ⚡Interactive visualization dashboards for subgroup analyses
- ⚡Integration with regulatory submission formats (FDA/EMA)


#### Technical Requirements


STUDY DESIGN HANDLING

Handle multiple study designs including randomized controlled trials, cohort studies, and case-control studies, as each requires different statistical approaches and quality assessment criteria.

Randomized Controlled TrialsCohort StudiesCase-Control Studies

AUTOMATED CLASSIFICATION

Core task: automated classification of all selected articles across multiple domains. For each article, the system should identify:

ARTICLE TYPE

Original research, systematic review, meta-analysis, case report, etc.

DATA TYPE

Blood biochemistry, RNA sequencing, DNA methylation, neurocognitive tests

BIOLOGICAL SPECIES

Homo sapiens, Mus musculus, etc.


STRUCTURED OUTPUT

Brief Intervention Description

3-4 sentences: active agent, molecular targets, delivery method

Agentic Study Selection

Automated study selection process

Intervention Effects

Mortality & disease risk (aggregated data using coding agents)

Evidence Basis

Methods used to generate data (clinical trials, animal studies)

Data Quality Assessment

Confidence score based on study type and journal quality

#### Test Cases & Validation


Participants will receive three well-characterized research questions from different medical domains, each with existing high-quality published meta-analyses that serve as ground truth:

TEST CASE 1
Pharmaceutical intervention with clear, standardized outcome measures (longevity intervention like metformin)

TEST CASE 2
Behavioral intervention with more heterogeneous outcome reporting

TEST CASE 3
Diagnostic test accuracy question requiring bivariate meta-analysis methods


PROVIDED FOR EACH TEST CASE

- Institutional database subscriptions
- Full text of all potentially relevant papers

- Published meta-analysis protocol (search strategy & inclusion criteria)
- Manually verified data extractions for validation

#### Expected Impact


TRANSFORMING EVIDENCE-BASED MEDICINE

Success in this challenge would fundamentally transform evidence-based medicine. Automated meta-analysis could reduce the typical timeline from months to hours, enabling real-time evidence synthesis as new studies emerge.

IMMEDIATE BENEFITS

- Critical acceleration during health emergencies
- Faster gerontology & preventive medicine research
- Democratized access for smaller research groups

BROADER IMPACT

- Revolutionize regulatory agency evaluations
- Transform clinical guidelines development
- Enable living systematic reviews

#### Resources & Guidelines


ESSENTIAL READING

- [Cochrane Handbook](https://training.cochrane.org/handbook)  
  Systematic Reviews of Interventions
- [PRISMA Statement](http://www.prisma-statement.org/)  
  Reporting standards for your system
- [Introduction to Meta-Analysis](https://www.meta-analysis.com/)  
  Borenstein et al. with code examples
    

TECHNICAL TOOLS

- [PubMed API](https://www.ncbi.nlm.nih.gov/home/develop/api/)
  Literature search and access  
- [Grobid](https://grobid.readthedocs.io/)
  PDF data extraction  
- [metafor R package](https://www.metafor-project.org/)
  Statistical analysis implementation  

ADDITIONAL RESOURCES

PDF PARSING

- [LlamaParse](https://github.com/run-llama/llama_parse) (LLM-native)
- [PyPDF2](https://pypi.org/project/PyPDF2/) (standard library)

VALIDATION & QUALITY

- [Validation Dataset](https://github.com/hyesunyun/llm-meta-analysis)
- Traffic light scoring (Green/Yellow/Red)

SUPPORT AVAILABLE

- Mentorship from meta-analysis experts
- Technical support for database API integration

- Compute credits for model training and inference
- Weekly office hours for team collaboration

## Rapid Adoption Track

Build market-ready AI tools that can immediately deliver value to the longevity industry. Focus on solutions with clear commercial pathways and real-world impact.

Choose from our curated challenges or propose your own innovative solution

### Challenges

#### Female/Reproductive Longevity

ALL

Leverage AI agents in female and reproductive longevity through data generation, intervention development, dissemination of knowledge, and attracting funding.

#### Global Human-Trials Intelligence Database

DATA

Parse papers/registries/regulatory agencies/datasets, dedupe across regions, and pool arm/endpoint/outcome data into a clean database of everything that was ever tried on humans. Enrich trials with publications linked to these trials. The purpose is to create a dataset of human clinical data that is as full as possible.

#### Vitalist Politician Ranker

ADVOCACY

Rank all politicians based on their vitalist views for each country. Assess outreach potential for the vitalist community and the outreach strategy.

#### Cross-Domain Methods Scout

INTERVENTIONS

Surface transferable methods (modalities/delivery/testing) from adjacent domains (oncology/transplant/space med) with aging use-cases, suggest cost-effective instruments to test these methods.

#### Longevity Intervention Evidence Tracker

ADVOCACY

Examine.com–style living profiles: efficacy, safety, cost, effect size of evidence for clinic-offered therapies with weekly (or monthly) updates. I.e. consumer advocate for applied longevity that provides truth over hype. Sample therapies - TPE, HBOT, Red light therapies (mine what is offered by longevity clinics and what is being discussed on biohacker forums).

#### LongBio Nonprofit Fundraising Agent

FUNDING

Automate fundraising for non-profits, internal A/B testing, trying of different methods and messaging with cost and efficacy tracker.

#### Combination Synergy Engine (LongBio + Modality-Agnostic)

INTERVENTIONS

Mine existing cases of successful synergies across literature. Create synergy scoring across drugs/devices/behaviors with MoA complementarity; propose trial-ready combinations.

#### An Extremophile Database to Combat Aging

INTERVENTIONS

Build an AI-first pipeline that mines extremophile biology (genes, proteins, pathways, metabolites) and prioritizes human-translatable interventions. Propose <$1–5k wet-lab assays to test the viability of the top hits.

### EVALUATION_MATRIX

Judging criteria breakdown:

01. Business Viability (or Social Impact) 30%  
    Demonstrate either business potential or social impact  
    Key factors:
    - Business Viability: Proof of Demand (qualified customer interviews, LOIs or paid-pilot intents, waitlist signups…)
    - Business Viability: Credible TAM/SOM with assumptions
    - Social Impact: Proof of Social Impact (waitlist signups, subscribers, social media effect, donations or donation intent)
    - Social Impact: Credible estimates of social effect with assumptions
02. Demo Quality 20%  
    Quality of implementation and presentation  
    Key factors:
    - Consistently Running Code
    - Clear Documentation
    - Engaging Demo Video with Clear Explanations 
03. Longevity Impact 20%  
    Impact on longevity research and industry  
    Key factors:
    - The effect on a High Level Purpose is justified (credible assumptions, effect estimates)
04. Agentic Fit 20%  
    Appropriate use of agentic AI
    Key factors:
    - Use of agentic AI is well justified (automation, can't be solved with just an LLM query, AI Workflow or basic script)

### Submission protocol

Technical Requirements:
- Submissions must be deployed and available at a public URL. The jury is not going to download and execute any code.
- Include a video presentation (up to 5 minutes) walking the jury through your project. Focus on the evaluation criteria (Business Viability, Demo Quality, Agentic Fit, Longevity Impact).
- We always appreciate open-source code available publicly. If you choose to make your repository public, we would love to take a look.

Demo Video Specs:
- The video portion of your entry must be no more than five (5) minutes in length
- A demonstration of your Submission
- An explanation of why you chose to create the Submission that you did and how this will impact the longevity field
- An answer to the question: "How will this entry earn money? (or impact society)"

Key Success Factors (What makes a winning submission):
- Business Viability Focus:
  - Qualified customer interviews or LOIs
  - Credible TAM/SOM with assumptions
  - Clear monetization strategy
- Agentic AI Requirements:
  - True automation, not just LLM queries
  - Complex AI workflows and decision-making
  - Justified use of agent frameworks


## 📅 Hackathon Timeline is Live!

October 7-24 | From kick-off to finals, here's your complete guide to the hackathon schedule.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 Kick-off Event — October 7
Time: 9 AM PT / 6 PM CET

Join us for the official hackathon launch where we'll:
• Present all available challenges
• Explain submission rules and evaluation criteria
• Answer your questions live

Can't make it? Recording will be shared afterwards.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💬 Challenge Deep-Dive Q&A — October 7-8-9
After the kick-off, join challenge-specific Q&A sessions to:
• Deep dive into each challenge track
• Ask technical questions to challenge owners
• Connect with potential teammates

📅 Detailed schedule coming next week

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ Hacking Period — October 8-22
14 days to build your solution

• Start: October 8 (after Q&A sessions)
• Code Freeze: October 22, 11:59 PM PT
• Use challenge channels for questions and updates
• Async support from organizers and mentors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎥 Project Submissions — Due October 22
What to submit:
• Video demo of your solution (3-5 minutes)
• Code repository with documentation
• Project brief and technical approach

All submissions will be reviewed by our jury panel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🏆 Finals & Winner Announcements
After submissions close:
• Internal jury evaluation
• Top projects selected for live finals
• All projects published for community viewing
• Join finals online or in San Francisco

💰 $20K prize pool distributed to winners

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Good luck! 🚀


## Dmitrii Kriukov 2025.10.05 18:33

Hey guys! This is Dima from ComputAge. I see the discussion on the challenge is already started. I foresee you to have a lot of questions about the challenge. [Here](https://docs.google.com/document/d/1XyAHnTXQh1pWKquYVsfegRrNPLdp2rwIaaiHAkff1WI/edit?tab=t.0) we tried briefly describe a challenge. We will also create a doc with frequent QAs later. 

Please, mind, we have a meeting immediately after the opening ceremony on 7-th Oct (19:00 CET time). There I'll briefly introduce you to the challenge with QA session afterwards.

Also, please, feel free to ask questions here - We (me and @Simon Steshin ) will do our best to answer them asap.