echo '(Assessment1) Assessing models developed using different KGE algorithms'

echo 'Evaluating model with transE embeddings'
python assessments/assessment1/nn_classifier_transE.py

echo 'Evaluating model with distmult embeddings'
python assessments/assessment1/nn_classifier_distmult.py

echo 'Evaluating model with complex embeddings'
python assessments/assessment1/nn_classifier_complex.py

echo 'Evaluating model with expressivE embeddings'
python assessments/assessment1/nn_classifier_expressivE.py

## To retrain these models, include argument retrain=T. E.g. python assessments/assessment1/nn_classifier_transE.py --retrain=T
echo '-----------------Assessment1 Ended---------------------'

echo '(Assessment2) KSFinder 2.0 KGE vs models developed using embeddings from other models'

echo 'Evaluating model with embeddings from ESM2'
python assessments/assessment2/nn_classifier_esm2.py

echo 'Evaluating model with embeddings from ESM3'
python assessments/assessment2/nn_classifier_esm3.py

echo 'Evaluating model with embeddings from ProtT5'
python assessments/assessment2/nn_classifier_protT5.py

echo 'Evaluating model with embeddings from KSFinder 2.0 KGE (transE)'
python assessments/assessment2/nn_classifier_ksf2.py

echo 'Evaluating model with embeddings from random generation'
python assessments/assessment2/nn_classifier_random.py

## To retrain these models, include argument retrain=T. E.g. python assessments/assessment2/nn_classifier_esm2.py --retrain=T
echo '-----------------Assessment2 Ended---------------------'

echo '(Assessment3) Influence of embeddings from models - Phosformer and ProstT5'

echo 'Evaluating model with KSFinder 2.0 embeddings, ProsT5 and Phosformer (structure & sequence only, no KSF2)'
python assessments/assessment3/prostT5_phosformer_only/nn_classifier.py

echo 'Evaluating model with KSFinder 2.0 embeddings, ProsT5 and KSF2 (structure & function)'
python assessments/assessment3/w_prostT5/nn_classifier.py

echo 'Evaluating model with KSFinder 2.0 embeddings, Phosformer and KSF2 (sequence & function)'
python assessments/assessment3/w_phosformer/nn_classifier.py

echo 'Evaluating model with KSFinder 2.0 embeddings, ProstT5, Phosformer and KSF2 (structure, sequence & function)'
python assessments/assessment3/w_prostT5_phosformer/nn_classifier.py

echo 'Evaluating model with KSFinder 2.0 embeddings, KSF2 only (function only)'
python assessments/assessment3/ksf2_only/nn_classifier.py

## To retrain these models, include argument retrain=T. E.g. python assessments/assessment3/prostT5_phosformer_only/nn_classifier.py --retrain=T
echo '-----------------Assessment3 Ended---------------------'

echo '(Assessment4) Comparative evaluation of KSFinder 2.0 with other kinase-substrate tools'

echo 'Comparing KSFinder 2.0 and Phosformer-ST (subassessment 1)'
python assessments/assessment4/phosformer-ST/compare_subassess1.py

echo 'Comparing KSFinder 2.0 and Phosformer-ST (subassessment 2)'
python assessments/assessment4/phosformer-ST/compare_subassess2.py

echo 'Comparing KSFinder 2.0 and LinkPhinder'
python assessments/assessment4/link_phinder/compare.py

echo 'Comparing KSFinder 2.0 and ksfinder'
python assessments/assessment4/ksfinder/compare.py

echo 'Comparing KSFinder 2.0 and predkinkg'
python assessments/assessment4/predkinkg/compare.py

echo '-----------------Assessment4 Ended---------------------'