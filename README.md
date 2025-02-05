# Learning more with Less: Curriculum Learning for Natural Language Inference using Strategice Data Selection
This repository contains the scripts used for the paper "Learning more with Less: Curriculum Learning for Natural Language Inference using Strategice Data Selection". In this paper, curriculum learning was explored through mindfully selecting data from the SNLI train set by Bowman et al. (2015). Two different selection methods were explored: dataset carthography (Swayamdipta et al, 2020) and difficulty evaluation (Xu et al, 2020).

## Installation and dependencies
*list all packages used*

## Dataset Information
The Data used for this project is the SNLI dataset by Bowman et al (2015). The data can be acquired as a zip file named SNLI_raw.zip under ├── data/raw.

(add code, link lasha's notebook) This data can also be downloaded through the following code by Lasha Abzianidze:
'''


'''

For the baseline models, the metasets created in the project are included under ├──baseline/data.

## Results

The following accuracy scores on the SNLI development set were achieved using the developed curricula.

| Curriculum       | Accuracy                       |
|------------------|--------------------------------|
| Baseline         |  0.6654                        |
| Baseline-triplet |  0.6502                        |
|------------------|--------------------------------|
|   | Difficulty-based, Self-paced    |
| Training Steps  | Number of epochs, batch size    |
| Optimizer      | Adam, SGD                        |
| Learning Rate  | 1e-5, 1e-4, etc.                 |
| Metrics        | Accuracy, F1-score               |
| Hardware       | GPU, TPU, CPU                    |
| Repository     | GitHub link or project name      |

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
