# Steps to set up KSFinder 2.0
git clone git@github.com:manju-anandakrishnan/ksfinder-2.0.git <br>
cd ksfinder-2.0 <br>

# Create environment and load libraries
conda env create -f environment.yaml <br>
conda activate ksf2_env <br>

# Initialize the repository
sh init.sh <br>
Dataset and trained models are available in Zenodo repository, https://doi.org/10.5281/zenodo.14075070. Upon initialization, the models and dataset should be downloaded into appropriate folders in ksfinder-2.0 repository.

# To run assessments (1 thorugh 4)
sh assess.sh <br>

# KSFinder 2.0 predictions
We generated predictions by matching kinases and substrate, motifs in our dataset using KSFinder 2.0 model and the predictions are available in the output/batch folder. Or in the Zenodo repository, ksf2_predictions.zip

# To get KSFinder 2.0 predictions for your data, pass the arguments --k, --sp, --sm as required. 
--sp denotes the substrate protein, --sm denotes the phosphomotif (-/+ 4 mer),  --k denotes the kinase <br>
KSFinder 2.0 can make predictions only for those entities that are in its KG. <br>
For input substrate and motif <br>
    python classifier/predict/handler.py --sp Q14974 --sm RRSKTNKAK <br>
For an input substrate <br>
    python classifier/predict/handler.py --sp Q14974 <br>
For an input kinase <br>
    python classifier/predict/handler.py --k Q96GD4 <br>

# To train KGE models from scratch
    python kge/kge_model.py <br>

# To retrain classifier models from scratch
In assess.sh, include the argument --retrain=T with the python commands <br>
Example, python assessments/assessment2/nn_classifier_ksf2.py --retrain=T   <br>