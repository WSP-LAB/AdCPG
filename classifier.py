import json
import os
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import torch
import torch_geometric

class classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()
        self.conv_1 = torch_geometric.nn.GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim)
        self.norm_1 = torch.nn.BatchNorm1d(hidden_channels)
        self.act_1 = torch.nn.PReLU()
        self.conv_2 = torch_geometric.nn.GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        self.norm_2 = torch.nn.BatchNorm1d(hidden_channels)
        self.act_2 = torch.nn.PReLU()
        self.conv_3 = torch_geometric.nn.GATv2Conv(hidden_channels * 2, hidden_channels, edge_dim=edge_dim)
        self.norm_3 = torch.nn.BatchNorm1d(hidden_channels)
        self.act_3 = torch.nn.PReLU()
        self.conv_4 = torch_geometric.nn.GATv2Conv(hidden_channels * 3, hidden_channels, edge_dim=edge_dim)
        
        self.readout = torch_geometric.nn.global_add_pool
        
        self.classification = torch_geometric.nn.MLP([hidden_channels * 4, hidden_channels, out_channels], act='PReLU')
        self.softmax = torch.nn.Softmax(dim=1)
    
    def apply_layer(self, x, edge_index, edge_attr, conv, norm, act):
        x = conv(x, edge_index, edge_attr)
        if norm is not None:
            x = norm(x)
        if act is not None:
            x = act(x)
        
        return x
    
    def forward(self, x, edge_index, edge_attr, batch, explain=False):
        h_1 = self.apply_layer(x, edge_index, edge_attr, self.conv_1, self.norm_1, self.act_1)
        h_2 = self.apply_layer(h_1, edge_index, edge_attr, self.conv_2, self.norm_2, self.act_2)
        h_3 = self.apply_layer(torch.cat([h_1, h_2], dim=1), edge_index, edge_attr, self.conv_3, self.norm_3, self.act_3)
        h_4 = self.apply_layer(torch.cat([h_1, h_2, h_3], dim=1), edge_index, edge_attr, self.conv_4, None, None)
        
        x = self.readout(torch.cat([h_1, h_2, h_3, h_4], dim=1), batch)
        
        x = self.classification(x)
        prob = self.softmax(x)
        pred = prob.argmax(dim=1)
        
        return x if explain else (x, prob, pred)

class early_stopping:
    def __init__(self, patience):
        self.early_stop = False
        self.patience = patience
        self.counter = 0
        self.best_score = 0
    
    def __call__(self, score, model, model_path):
        if score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_score = score
            
            torch.save(model, model_path)

def train(model_path, device, train_loader, test_loader):
    model = classifier(333, 16, 2, 3)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    stopping = early_stopping(50)
    
    epoch = 0
    while True:
        epoch += 1
        
        model.train()
        for data in train_loader:
            data = data.to(device)
            
            out, prob, pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            
            loss = criterion(out, data.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        ys = []
        preds = []
        for data in test_loader:
            data = data.to(device)
            
            with torch.no_grad():
                out, prob, pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            ys += data.y.tolist()
            preds += pred.tolist()
        accuracy = round(accuracy_score(ys, preds), 4)
        
        stopping(accuracy, model, model_path)
        if stopping.early_stop:
            break

def test(model_path, performance_path, device, test_loader):
    model = torch.load(model_path)
    model = model.to(device)
    
    model.eval()
    ys = []
    probs = []
    preds = []
    for data in test_loader:
        data = data.to(device)
        
        with torch.no_grad():
            out, prob, pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        ys += data.y.tolist()
        probs += prob[:, 1].tolist()
        preds += pred.tolist()
    accuracy = round(accuracy_score(ys, preds), 4)
    precision = round(precision_score(ys, preds), 4)
    recall = round(recall_score(ys, preds), 4)
    try:
        auc = round(roc_auc_score(ys, probs), 4)
    except:
        auc = 0
    
    performance = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }
    with open(performance_path, 'w') as f:
        json.dump(performance, f, indent=4)

def evaluate(folds, scripts_path, results_path):
    os.makedirs(results_path)
    
    folds_path = os.path.join(results_path, 'folds.json')
    with open(folds_path, 'w') as f:
        json.dump(folds, f, indent=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    graphs = {}
    for idx in folds:
        graphs[idx] = []
        for ast_hash in folds[idx]:
            script_path = os.path.join(scripts_path, ast_hash)
            graph_path = os.path.join(script_path, 'graph.pt')
            graph = torch.load(graph_path)
            graphs[idx].append(graph)
    
    for test_idx in folds:
        model_path = os.path.join(results_path, f'model-{test_idx}.pt')
        performance_path = os.path.join(results_path, f'performance-{test_idx}.json')
        
        train_set = []
        test_set = []
        for idx in folds:
            if idx == test_idx:
                test_set += graphs[idx]
            else:
                train_set += graphs[idx]
        train_loader = torch_geometric.loader.DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
        test_loader = torch_geometric.loader.DataLoader(test_set, batch_size=16, shuffle=False, drop_last=False)
        
        train(model_path, device, train_loader, test_loader)
        
        test(model_path, performance_path, device, test_loader)

def get_folds(scripts_path):
    ats_ratio = 0.5
    
    dataset = {
        0: [],
        1: [],
    }
    for ast_hash in os.listdir(scripts_path):
        script_path = os.path.join(scripts_path, ast_hash)
        graph_path = os.path.join(script_path, 'graph.pt')
        if os.path.exists(graph_path):
            label_path = os.path.join(script_path, 'label')
            with open(label_path, 'r') as f:
                label = int(f.read())
            
            dataset[label].append(ast_hash)
    dataset[0] = random.sample(dataset[0], len(dataset[1]))
    dataset[1] = random.sample(dataset[1], int(len(dataset[1]) * ats_ratio / (1 - ats_ratio)))
    
    folds = {}
    for idx in range(10):
        folds[idx] = []
        folds[idx] += dataset[0][idx * int(len(dataset[0]) / 10):(idx + 1) * int(len(dataset[0]) / 10)]
        folds[idx] += dataset[1][idx * int(len(dataset[1]) / 10):(idx + 1) * int(len(dataset[1]) / 10)]
    
    return folds

def main():
    data_path = os.path.abspath('./data')
    scripts_path = os.path.join(data_path, 'scripts')
    results_path = os.path.join(data_path, 'results')
    
    folds = get_folds(scripts_path)
    
    evaluate(folds, scripts_path, results_path)

if __name__ == '__main__':
    main()
