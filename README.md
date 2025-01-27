# protein-function-prediction
Predicting functional GO annotations given ProtT5 embeddings and BLAST results.

## Overview
This project was developed by [Flavio Agostini](https://github.com/biosaio), [Mikael Poli](https://github.com/mikaelpoli), and [Emanuele Quaglio](https://github.com/emanuele-quaglio) as their final exam for the Biological Data course at the University of Padova.   

We strongly suggest reading the [report](https://github.com/mikaelpoli/protein-function-prediction/blob/main/report.pdf) first in order to contextualize this repo, but essentially:
- Given a dataset containing 123,969 proteins, their ProtT5 embeddings, and their GO annotations, we built our own training and test sets, and trained three MLP models (one for each GO aspect) to learn association patterns between ProtT5 embeddings and GO terms;
- We also used BLAST matches to transfer GO annotations from the proteins in the training set to those in the test set independently of the MLP models;
- We merged the two types of predictions, weighing the MLP's contribution through an $\alpha$ coefficient;
- We tested the final results for different levels of $\alpha$ using [cafaeval](https://github.com/BioComputingUP/CAFA-evaluator).

We also tested an alternative approach that integrates BLAST results during MLP training using TF-IDF (term frequency–inverse document frequency). The repo for this approach, as well as the workflow for obtaining the BLAST results we integrated in both our methods, can be found [here](https://github.com/biosaio/Function-Prediction.git).

## Project repo structure
[root.]
  - models
  - results/
    - evaluations/
      - test-alpha-0.0/
      - test-alpha-0.1/
      - test-alpha-0.2/
      - test-alpha-0.3/
      - test-alpha-0.4/
      - test-alpha-0.5/
      - test-alpha-0.6/
      - test-alpha-0.7/
      - test-alpha-0.8/
      - test-alpha-0.9/
      - test-alpha-1.0/
    - plots
    - submission/
      - submission.tsv
  - src
  - src-notebooks  
config.yaml  
custom-dataset-builder.ipynb  
cafaeval.ipynb  
main.ipynb

## How to run
### Download data
Download the datasets from [MEGA](https://mega.nz/folder/gCkgDSDS#cetVnor0kUDwEAUtjnb1Kg). The directory has this structure:  

[root.]
  - data/
    - custom/
      - test
      - train
    - test
    - train
### Download repo
Download this repo and ***rename it to `pfp`***. This is very important, as `pfp` is the name of the root directory in the configuration file. If you don't want to rename the folder, or you choose to rename it to something else, you will need to manually edit the `config.yaml` file to account for the new name, or the notebooks won't work properly.
### Merge
Add the `data` folder to the `pfp` folder. The final structure of the `pfp` folder should be:  

[root.]  
  - **data/**  
    - **custom/**  
      - **test**  
      - **train**  
    - **test**  
    - **train**  
  - models
  - results/
    - evaluations/
      - test-alpha-0.0/
      - test-alpha-0.1/
      - test-alpha-0.2/
      - test-alpha-0.3/
      - test-alpha-0.4/
      - test-alpha-0.5/
      - test-alpha-0.6/
      - test-alpha-0.7/
      - test-alpha-0.8/
      - test-alpha-0.9/
      - test-alpha-1.0/
    - plots
    - submission/
      - submission.tsv
  - src
  - src-notebooks  
config.yaml  
custom-dataset-builder.ipynb  
cafaeval.ipynb  
main.ipynb

### Upload to Google Drive
Upload the whole directory to your Google Drive in the `My Drive` section.
### Understanding the `.ipynb` notebooks and choosing what to run
The root directory contains three `.ipynb` notebooks: one main one, and two supplementary ones.
- *Main*
   - `main.ipynb`: where it all comes together. As the name suggests, this is the main notebook containing the whole project pipeline, from loading the data to training the models to predicting protein function. This is what you want to run to reproduce model training for the post hoc approach. Since the integration of BLAST results doesn't require any training, the only training you may want to replicate is the one for the MLP models. But you can always try different integrations of BLAST results by initializing the scores to a different value in the `calculate_blast_based_scores()` function and changing the alpha value in the `update_predictions()` function!
- *Supplementary*
   - `custom-dataset-builder.ipynb`: contains the code that was used to obtain the custom training and test sets from the original ones we were supplied. Running this notebook is not necessary, as the output datasets generated with it are already saved in the `data/custom/` directory. 
   - `cafaeval.ipynb`: contains the code we used to evaluate the post hoc models. The results of the tests for each level of alpha are in the `results/evaluations/` directory.
### Reproducing MLP training
Run the `main.ipynb` notebook in its entirety. Training for each of the three MLP models takes around 15 minutes using Google Colab's T4 GPU.
### What's in `src`?
Aside from predefined Python libraries, we built our own custom ones: they're in the `src` directory. We've imported them in the "Setup" section of out notebooks. `src` contains the `.py` files. If you want to open them in Google Colab, you'll find the `.ipynb` notebooks we used to generate them in `src-notebooks`.
### What's in `results`?
We tested our approach at different levels of $\alpha$ to see how it would perform—this is where the results are. Each $\alpha$ we tested has a separate directory structued like so:  

[root. (e.g., test-alpha-0.0)]  
  - eval  
  - predictions

The predictions we tested with *cafaeval* are in `predictions`, while `eval` contains the *cafaeval* output .tsv files.
### What if I'm your professor?
You'll find our predictions for the original test set in a .tsv file named `submission.tsv` in the `results/submission directory`.



   
