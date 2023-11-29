import json
import os

import esprima
import pandas as pd
import torch
import torch_geometric

from classifier import classifier

def get_important_nodes(explanation_subgraph):
    proportion_threshold = 0.4
    
    node_importances = dict(enumerate(explanation_subgraph.node_mask, start=0))
    sorted_node_importances = dict(sorted(node_importances.items(), key=lambda item: item[1], reverse=True))
    
    important_nodes = []
    for node_rank, (node_idx, node_importance) in enumerate(sorted_node_importances.items(), start=1):
        proportion = node_rank / len(sorted_node_importances)
        if proportion < proportion_threshold:
            important_nodes.append(node_idx)
    
    return important_nodes

def highlight_important_code(explanation_path, tokens, node_data_path, important_nodes):
    important_locations = set()
    for idx, row in pd.read_csv(node_data_path, header=0).iterrows():
        node_idx = int(row['node_idx'])
        locations = str(row['locations'])
        
        if node_idx in important_nodes:
            important_locations.update(locations.split('&'))
    
    important_code_list = []
    for token in tokens:
        if token['location'] in important_locations:
            important_code_list.append({
                'location': token['location'],
                'code': token['code'],
            })
    df_important_code_list = pd.DataFrame(important_code_list)
    df_important_code_list.to_csv(explanation_path)

def generate_explanation(explanation_path, tokens, graph, node_data_path, explainer):
    loader = torch_geometric.loader.DataLoader([graph], batch_size=1)
    data = next(iter(loader))
    
    explanation = explainer(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch, explain=True)
    explanation_subgraph = explanation.get_explanation_subgraph()
    
    important_nodes = get_important_nodes(explanation_subgraph)
    
    highlight_important_code(explanation_path, tokens, node_data_path, important_nodes)

def tokenize_code(code_path):
    with open(code_path, 'r') as f:
        code = f.read()
    
    tokens = []
    for token in esprima.tokenize(code, options={'loc': True}):
        if token.type in ['Identifier', 'Boolean', 'Numeric', 'String', 'RegularExpression']:
            tokens.append({
                'code': token.value,
                'location': f'{token.loc.start.line}:{token.loc.start.column}',
            })
    
    return tokens

def main():
    data_path = os.path.abspath('./data')
    scripts_path = os.path.join(data_path, 'scripts')
    results_path = os.path.join(data_path, 'results')
    
    explanations_path = os.path.join(results_path, 'explanations')
    os.makedirs(explanations_path)
    
    folds_path = os.path.join(results_path, 'folds.json')
    with open(folds_path, 'r') as f:
        folds = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for test_idx in folds:
        model_path = os.path.join(results_path, f'model-{test_idx}.pt')
        model = torch.load(model_path)
        model = model.to(device)
        
        explainer = torch_geometric.explain.Explainer(
            model=model,
            algorithm=torch_geometric.explain.GNNExplainer(epochs=500, lr=0.001),
            explanation_type='model',
            model_config=torch_geometric.explain.ModelConfig(
                mode='multiclass_classification',
                task_level='graph',
                return_type='probs',
            ),
            node_mask_type='object',
            edge_mask_type='object',
        )
        
        for ast_hash in folds[test_idx]:
            script_path = os.path.join(scripts_path, ast_hash)
            code_path = os.path.join(script_path, 'code.js')
            graph_path = os.path.join(script_path, 'graph.pt')
            node_data_path = os.path.join(script_path, 'node_data.csv')
            
            explanation_path = os.path.join(explanations_path, f'explanation-{ast_hash}.csv')
            
            tokens = tokenize_code(code_path)
            
            graph = torch.load(graph_path)
            graph = graph.to(device)
            
            generate_explanation(explanation_path, tokens, graph, node_data_path, explainer)

if __name__ == '__main__':
    main()
