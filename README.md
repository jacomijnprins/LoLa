# LoLa

## Project: Curriculum for efficient NLI learning
Group members:
  - Pepijn: 
  - Jessica: 
  - Oliver:
  - Lui
  - Jaco: 1501178


This branch contains the entire process for the Curriculum Learning method, adapted from Xu et al. (2020). 

- The teacher models are trained on 5 splits of SNLI_train.csv. Relevant code is in subdirectory "teacher_models".
- "cross_review" subdirectory contains the code for evaluating the trained teacher models and producing a dataset of instances with difficulty scores.
- "annealing" subdirectory contains the final model and evaluation, trained on both curriculum (easiest -> hardest) and reverse curriculum (hardest -> easiest) ways.

- N.B: any files which were too large to upload after zipping/condensing have google drive download links. see "download.txt" for links (located in relevant subdirectories)
