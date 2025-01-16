import torch

from sklearn.metrics import roc_auc_score, average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Evaluates the model with the input testing_data

Parameters
------------
testing_data: pd.DataFrame

Returns
------------
ROC_SCORE and PR_SCORE based on input testing_data predictions

"""
def evaluate_model(model,testing_data):   
    kd_emb_test = torch.tensor(testing_data[0],dtype=torch.float32).to(device)
    s11_emb_test = torch.tensor(testing_data[1],dtype=torch.float32).to(device)
    k_st_emb_test = torch.tensor(testing_data[2],dtype=torch.float32).to(device)
    s_st_emb_test = torch.tensor(testing_data[3],dtype=torch.float32).to(device)
    y_test = torch.tensor(testing_data[4],dtype=torch.float32).view(-1,1)

    model.eval()
    with torch.no_grad():        
        outputs = model(kd_emb_test,s11_emb_test,k_st_emb_test,s_st_emb_test)
        outputs = outputs.to('cpu')
        y_pred = outputs.numpy().flatten() if isinstance(outputs, torch.Tensor) else outputs.flatten()
        roc_score = round(roc_auc_score(y_test, y_pred),3)        
        pr_score = round(average_precision_score(y_test, y_pred),3)
        return roc_score, pr_score