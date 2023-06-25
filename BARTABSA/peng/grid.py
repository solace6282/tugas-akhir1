from model.modeling_bart import BartModel
from transformers import AutoTokenizer
from sklearn.model_selection import GridSearchCV

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')
tokens = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')

def bart_model_fn(params):
    model = BartModel.from_pretrained('facebook/bart-base', **params)
    # Add any other necessary model configuration
    return model

param_grid = {
    'num_layers': [1, 2, 3],
    'dropout': [0.1, 0.2, 0.3],
    # Add any other parameters you want to search over
}

grid_search = GridSearchCV(estimator=bart_model_fn, param_grid=param_grid)

grid_search.fit(tokens['input_ids'], tokens['attention_mask'])

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(best_model, best_params)