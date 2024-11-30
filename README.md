# Steps to set up KSFinder 2.0 <br>
*Steps to set up KSFinder 2.0 --- begins ---*
1. Clone the repository <br>
git clone git@github.com:manju-anandakrishnan/ksfinder-2.0.git <br>
cd ksfinder-2.0 <br>

2. Create environment and load libraries
conda env create -f environment.yaml <br>
conda activate ksf2 <br>

python setup.py install <br>

If you have GPU compatability error message, "UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11060). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver." <br>

Install the below pytorch cudatoolkit package <br>
conda install pytorch=1.10.2 cudatoolkit=11.3 torchvision torchaudio -c pytorch <br>

3. Initialize the repository
sh init.sh <br>
Dataset and trained models are available in Zenodo repository, https://doi.org/10.5281/zenodo.14075070. Upon initialization, the models and dataset should be downloaded into appropriate folders in ksfinder-2.0 repository.

*Steps to set up KSFinder 2.0 --- ends ---*

# To run assessments (1 thorugh 4)
sh assess.sh <br>

# KSFinder 2.0 predictions
We generated predictions for all combinations of kianses and substrate_motifs using KSFinder 2.0 and the predictions are available in the Zenodo repository, ksf2_predictions.zip. If you have run init.sh and set up local workspace as described earlier, then predictions are also available under output/batch folder. 

# To get KSFinder 2.0 predictions for your data, pass the arguments --k, --sp, --sm as required. 
--sp denotes the substrate protein, --sm denotes the phosphomotif (-/+ 4 mer),  --k denotes the kinase <br>
KSFinder 2.0 can make predictions only for those entities that are in its KG. <br>
For input given substrate protein and motif at the phosphosite, <br>
python classifier/predict/handler.py --sp Q14974 --sm RRSKTNKAK <br>
For an input substrate protein and all motifs of the protein's phosphosites, <br>
python classifier/predict/handler.py --sp Q14974 <br>
For an input kinase, <br>
python classifier/predict/handler.py --k Q96GD4 <br>

# To train KGE models from scratch
python kge/kge_model.py <br>

# To retrain classifier models from scratch
In assess.sh, include the argument --retrain=T with the python commands <br>
Example, python assessments/assessment2/nn_classifier_ksf2.py --retrain=T   <br>

# To compare KSFinder 2.0 with other models using your own dataset
Follow the steps listed under 'Steps to set up KSFinder 2.0'. <br>
1. Either use the prediction data under output folder (or) <br>
2. Generate predictions using python classifier/predict/handler.py (by passing the arguments as described in the earlier steps) <br>
Note. To evaluate at the kinase-motif level without substrate protein information, use the highest probability prediction for a given kinase-motif combination. <br>
To evaluate at the kinase-substrate level without motif information, use the highest probability prediction for a given kinase-substrate combination. <br>

# To access datasets used for KSFinder 2.0's training and testing
1. Follow the steps listed under 'Steps to set up KSFinder 2.0'. <br>
    Training data is available under '/data/classifier' <br>
    Testing dataset1 is available under '/data/classifier/td1_ratio_1.1' <br>
    Testing dataset2 is available under '/data/classifier/td2_tr_dist' <br>
    (or) <br>
2. Download directly from Zenodo repository, https://zenodo.org/records/14075070/files/classifier_datasets.zip



