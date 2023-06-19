# ncaa_mm_2023
Code to generate solution for Kaggle March Madness 2023 competition. Overall size of data is too large to push to Github, so it is not possible to run the full code after cloning.
`src\main.py` is the entry script for generating a new submission and provides the high level architecture of the solution.

For the Kaggle competition [March Madness 2023](https://www.kaggle.com/competitions/march-machine-learning-mania-2023), the objective is to predict the hypothetical results for every possible men and women's matchup as a probability. The competition is scored using Brier score. Correct, confident predictions will result in a smaller Brier Score compared to correct, less confident predictions. Incorrect, confident predictions will result in a larger Brier Score compared to incorrect, less confident predictions.

Solution was developed in a Dev Container using VSCode. This supported automated linting and formatting using flake8 and black, respectively. Specific flake8 adjustments can be found in 
`.flake8` and specific black adjustments can be found in `pyproject.toml`. Solution was setup to be object oriented so that code reuse between the men and women's competition was easy.
This was important for this year's Kaggle competition, where predictions for all men and women's games were required, instead of having 2 tournaments (one for men and one for women).
With additional time, I would have like to leverage an out of the box hyperparameter tuner (such as Keras Auto Tuner or Ray Tune) and setup MLflow for experiment tracking. However, this project
was done in roughly a weekend, so only 
