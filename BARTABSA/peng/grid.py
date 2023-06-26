
from data.pipe import BartBPEABSAPipe
from model.modeling_bart import BartModel
from transformers import AutoTokenizer, BartTokenizer
from sklearn.model_selection import GridSearchCV

def get_data():
    pipe = BartBPEABSAPipe(tokenizer='facebook/bart-base', opinion_first=True, dataset = f'../final_data/opener_en/')
    data_bundle = pipe.process_from_file(f'../final_data/opener_en/', demo=False)
    return data_bundle, pipe.tokenizer, pipe.mapping2id

data_bundle, tokenizer, mapping2id = get_data()
model = BartModel.from_pretrained('facebook/bart-base')
num_tokens, _ = model.encoder.embed_tokens.weight.shape
model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
encoder = model.encoder
decoder = model.decoder

_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
for token in tokenizer.unique_no_split_tokens:
    if token[:2] == '<<':
        index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        if len(index)>1:
            raise RuntimeError(f"{token} wrong split")
        else:
            index = index[0]
        assert index>=num_tokens, (index, num_tokens, token)
        indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
        embed = model.encoder.embed_tokens.weight.data[indexes[0]]
        for i in indexes[1:]:
            embed += model.decoder.embed_tokens.weight.data[i]
        embed /= len(indexes)
        model.decoder.embed_tokens.weight.data[index] = embed

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

print(data_bundle.get_dataset('train'))
grid_search.fit(data_bundle.get_dataset('train')['src_tokens'], data_bundle.get_dataset('train')['aspects'])

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(best_model, best_params)