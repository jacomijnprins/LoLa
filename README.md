# Learning more with Less: Curriculum Learning for Natural Language Inference using Strategice Data Selection
This repository contains the scripts used for the paper "Learning more with Less: Curriculum Learning for Natural Language Inference using Strategice Data Selection". In this paper, curriculum learning was explored through mindfully selecting data from the SNLI train set by Bowman et al. (2015). Two different selection methods were explored: dataset carthography (Swayamdipta et al, 2020) and difficulty evaluation (Xu et al, 2020).

## Installation and dependencies
*All packages used:*
- pandas
- os
- random
- assigntools
- torch
- transformers
- sklearn
- tqdm
- logging
- re
- data_utils_glue (from [Cartography](https://github.com/allenai/cartography))


## Dataset Information
The Data used for this project is the SNLI dataset by Bowman et al (2015). The data can be acquired as a zip file named SNLI_raw.zip under ├── data/raw.

This data can also be downloaded through the following code:

```
# if assigntools not yet downloaded run line
# ! git clone https://github.com/kovvalsky/assigntools.git
from assigntools.LoLa.read_nli import snli_jsonl2dict, sen2anno_from_nli_problems
from assigntools.LoLa.sen_analysis import spacy_process_sen2tok, display_doc_dep
SNLI, S2A = snli_jsonl2dict('snli_1.0') 

```
From [Lasha Abzianidze](https://colab.research.google.com/drive/1cvOltz1eqA9QzzNCM5m7UsUhtw2_guxi?usp=sharing).

The meta subs sets created in the project are included under ├──baseline/data. The meta subsets are generated using ├──baseline/preprocess.py


## Results

The following accuracy scores on the SNLI development set were achieved using the developed curricula.

| Curriculum       | Accuracy                       |
|------------------|--------------------------------|
| Baseline         |  0.6654                        |
| Baseline-triplet |  0.6502                        |
|------------------|--------------------|
| Difficulty scoring | 0.6872   |
| Difficulty scoring reversed      | 0.6158|
| Carthography-Ambiguous | 0.3329                 |
| Carthography-Hard-to-learn        | 0.3307            |
| Carthography-Easy-Ambiguous       | 0.3329                    |
| Carthography-Mixed     | 0.3255      |

These were the scores for the 5k curricula:
| Curriculum       | Accuracy                       |
|------------------|--------------------------------|
| Ambiguous 5k     |  0.4388                        |
| Easy-ambiguous 5k |  0.6533                       |


## Directory structure
The structure of this project:
- README.md                 # Project documentation
- data/                     # Datasets and preprocessing scripts
- baseline/                 # Implementation, models and results of baselines
- cartography_selection/    # Implementation, models and results of cartography curricula
- difficulty_score/         # Implementation, models and results of difficulty scoring

Every directory dealing with implementations has a folder containing the preprocessed data or curricula; folder with the models and a folder containing the results of the models. All the other files are the code that are needed to recreate every step of the experiments. (Explanation on what to run can be found in section 'Usage')
- N.B: any files which were too large to upload after zipping/condensing have google drive download links. see "download.txt" for links (located in relevant subdirectories)

#### Difficulty_score directory 
Contains the entire process for the difficulty scoring curriculum learning method, adapted from Xu et al. (2020). 
- The teacher models are trained on 5 splits of SNLI_train.csv. Relevant code is in subdirectory "teacher_models".
- "cross_review" subdirectory contains the code for evaluating the trained teacher models and producing a dataset of instances with difficulty scores.
- "annealing" subdirectory contains the final model and evaluation, trained on both curriculum (easiest -> hardest) and reverse curriculum (hardest -> easiest) ways.

#### Cartography_selection directory
Contains the entire process for the cartography curriculum learning method, adapted from Swayamdipta et al. (2020).
- The cartography scores dataset was acquired from the [Cartography github](https://github.com/allenai/cartography/tree/main/data/data_map_coordinates) from the paper.
- Many curricula were produced for this selection method; all of them can be found in the curricula folder and every piece of code is made to run (produce/analyse) for a single curriculum, so the correct curriculum name must be given.


## Usage

**For recreating the best results of the experiments:**
- Baselines:
    - Download the model: baseline/models/download.txt
    - Run evaluation: baseline/baselineEval.py
- Difficulty scoring:
    - Download the model: difficulty_score/annealing/models/curriculum_learning/download.txt
    - Run evaluation: difficulty_score\annealing\curriculumLearning_Eval.py
- Cartography:
    - Download model: cartography_selection/models/download.txt
    - Run evaluation: cartography_selection/analysing_models.py

**To train the models yourself, run:**
- Baseline: baseline/baselineModels.py (trains the first random model not all five)
- Difficulty score: difficulty_score/annealing/curriculumTraining.py (trains the difficulty score model)
- Cartography: cartography_selection/training_models.py (trains the ambiguous easy triplet conserwing model on 5k)


# References
Samuel R. Bowman, Gabor Angeli, Christopher Potts,
and Christopher D. Manning. 2015a. A large anno-
tated corpus for learning natural language inference.
In Proceedings of the 2015 Conference on Empirical
Methods in Natural Language Processing, pages 632–
642, Lisbon, Portugal. Association for Computational
Linguistics.

Swabha Swayamdipta, Roy Schwartz, Nicholas
Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A.
Smith, and Yejin Choi. 2020. Dataset cartography:
Mapping and diagnosing datasets with training dy-
namics. In Proceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 9275–9293, Online. Association for
Computational Linguistics.

Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan
Wang, Hongtao Xie, and Yongdong Zhang. 2020.
Curriculum learning for natural language understand-
ing. In Proceedings of the 58th Annual Meeting of
the Association for Computational Linguistics, pages
6095–6104, Online. Association for Computational
Linguistics.
