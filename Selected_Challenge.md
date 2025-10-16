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
