import os
import subprocess
import sys

import networkx as nx
import pandas as pd
import torch
import torch_geometric

from features import node_features, edge_features

class graph_builder:
    def __init__(self, cpg_csv_path, graph_path, node_data_path, label, node_features, edge_features):
        try:
            self.graph_data = {
                'node': {},
                'edge': [],
            }
            
            self.load_data(cpg_csv_path, node_features, edge_features)
            
            self.prune_function_subgraphs(node_features)
            
            self.prune_intermediate_nodes()
            
            if len(self.graph_data['node']) > 0 and len(self.graph_data['edge']) > 0:
                self.save_data(graph_path, node_data_path, label, node_features, edge_features)
        except:
            pass
    
    def get_node_idx(self, node_ids, node_id):
        return node_ids.index(node_id) if node_id in node_ids else None
    
    def refine_graph(self, valid_node_ids):
        self.graph_data['node'] = {node_id: node_info for node_id, node_info in self.graph_data['node'].items() if node_id in valid_node_ids}
        self.graph_data['edge'] = [edge_info for edge_info in self.graph_data['edge'] if edge_info['src_node_id'] in valid_node_ids and edge_info['dst_node_id'] in valid_node_ids and edge_info['src_node_id'] != edge_info['dst_node_id']]
    
    def load_data(self, cpg_csv_path, node_features, edge_features):
        node_ids_from_nodes = set()
        for node_type in ['BLOCK', 'CALL', 'CONTROL_STRUCTURE', 'FIELD_IDENTIFIER', 'IDENTIFIER', 'LITERAL', 'LOCAL', 'METHOD', 'METHOD_PARAMETER_IN', 'METHOD_REF', 'RETURN']:
            header_csv_path = os.path.join(cpg_csv_path, f'nodes_{node_type}_header.csv')
            if os.path.exists(header_csv_path):
                headers = pd.read_csv(header_csv_path, header=None).loc[0, :].tolist()
                
                data_csv_path = os.path.join(cpg_csv_path, f'nodes_{node_type}_data.csv')
                if os.path.exists(data_csv_path):
                    for idx, row in pd.read_csv(data_csv_path, header=None, names=headers).iterrows():
                        node_id = int(row[':ID'])
                        
                        code = ''
                        if node_type == 'CALL':
                            code = str(row['CODE:string'].split('.')[-1] if row['NAME:string'] == '<operator>.fieldAccess' else row['NAME:string'])
                        elif node_type == 'METHOD':
                            code = str(row['NAME:string'])
                            if code == ':program':
                                self.ast_root = node_id
                        elif node_type in ['LITERAL', 'LOCAL', 'METHOD_PARAMETER_IN']:
                            code = str(row['CODE:string'])
                        
                        locations = []
                        if node_type != 'LOCAL' and pd.notna(row['LINE_NUMBER:int']) and pd.notna(row['COLUMN_NUMBER:int']):
                            line = int(row['LINE_NUMBER:int'])
                            column = int(row['COLUMN_NUMBER:int'])
                            locations.append(f'{line}:{column}')
                        
                        node_feature = {}
                        for component, component_info in node_features['TYPE']['All'].items():
                            types = component_info['types']
                            node_feature[component] = [0] * len(types)
                            for name in types:
                                if node_type == name:
                                    node_feature[component][types.index(name)] = 1
                                    break
                        for component, component_info in node_features['API']['LOCAL'].items():
                            apis = component_info['apis']
                            node_feature[component] = [0] * len(apis)
                            if node_type == 'LOCAL':
                                for name in apis:
                                    if code.lower() == name.lower():
                                        node_feature[component][apis.index(name)] = 1
                                        break
                        for component, component_info in node_features['API']['CALL'].items():
                            apis = component_info['apis']
                            node_feature[component] = [0] * (1 if component_info['is_grouped'] else len(apis))
                            if node_type == 'CALL':
                                for name in apis:
                                    if code == name:
                                        node_feature[component][0 if component_info['is_grouped'] else apis.index(name)] = 1
                                        break
                        for component, component_info in node_features['STR']['LITERAL'].items():
                            strings = component_info['strings']
                            node_feature[component] = [0] * 1
                            if node_type == 'LITERAL':
                                for name in strings:
                                    if name.lower() in code.lower():
                                        node_feature[component][0] = 1
                                        break
                        
                        self.graph_data['node'][node_id] = {
                            'node_type': node_type,
                            'code': code,
                            'locations': locations,
                            'node_feature': node_feature,
                        }
                        
                        node_ids_from_nodes.add(node_id)
        
        node_ids_from_edges = set()
        for edge_type in ['AST', 'CDG', 'CFG', 'REACHING_DEF', 'REF']:
            header_csv_path = os.path.join(cpg_csv_path, f'edges_{edge_type}_header.csv')
            if os.path.exists(header_csv_path):
                headers = pd.read_csv(header_csv_path, header=None).loc[0, :].tolist()
                
                data_csv_path = os.path.join(cpg_csv_path, f'edges_{edge_type}_data.csv')
                if os.path.exists(data_csv_path):
                    for idx, row in pd.read_csv(data_csv_path, header=None, names=headers).iterrows():
                        src_node_id = int(row[':START_ID'])
                        dst_node_id = int(row[':END_ID'])
                        
                        if self.graph_data['node'].get(src_node_id) is not None and self.graph_data['node'].get(dst_node_id) is not None:
                            src_node_type = self.graph_data['node'][src_node_id]['node_type']
                            dst_node_type = self.graph_data['node'][dst_node_id]['node_type']
                            
                            pruned_edge_type = None
                            if edge_type == 'AST':
                                pruned_edge_type = 'AST'
                            elif edge_type == 'CFG':
                                pruned_edge_type = 'CFG'
                            elif edge_type in ['CDG', 'REACHING_DEF']:
                                pruned_edge_type = 'PDG'
                            
                            edge_feature = {}
                            for component, component_info in edge_features['TYPE']['All'].items():
                                types = component_info['types']
                                edge_feature[component] = [0] * len(types)
                                for name in types:
                                    if pruned_edge_type == name:
                                        edge_feature[component][types.index(name)] = 1
                                        break
                            
                            self.graph_data['edge'].append({
                                'edge_type': edge_type,
                                'src_node_id': src_node_id,
                                'src_node_type': src_node_type,
                                'dst_node_id': dst_node_id,
                                'dst_node_type': dst_node_type,
                                'edge_feature': edge_feature,
                            })
                            
                            node_ids_from_edges.add(src_node_id)
                            node_ids_from_edges.add(dst_node_id)
        
        valid_node_ids = node_ids_from_nodes.intersection(node_ids_from_edges)
        self.refine_graph(valid_node_ids)
        
        self.ast = nx.DiGraph()
        for node_id, node_info in self.graph_data['node'].items():
            self.ast.add_node(node_id)
        for edge_info in self.graph_data['edge']:
            if edge_info['edge_type'] == 'AST':
                self.ast.add_edge(edge_info['src_node_id'], edge_info['dst_node_id'])
        
        valid_node_ids = set(list(nx.dfs_preorder_nodes(self.ast, source=self.ast_root)))
        self.refine_graph(valid_node_ids)
    
    def prune_function_subgraphs(self, node_features):
        unnecessary_node_ids = set()
        
        subgraph_roots = [node_id for node_id, node_info in self.graph_data['node'].items() if node_info['node_type'] == 'METHOD']
        for subgraph_root in subgraph_roots:
            subgraph_node_ids = list(nx.dfs_preorder_nodes(self.ast, source=subgraph_root))
            
            is_necessary_subgraph = False
            for subgraph_node_id in subgraph_node_ids:
                subgraph_node_feature = self.graph_data['node'][subgraph_node_id]['node_feature']
                
                is_necessary_node = any(max(subgraph_node_feature[component]) != 0 for component in list(node_features['API']['LOCAL']) + list(node_features['API']['CALL']))
                if is_necessary_node:
                    is_necessary_subgraph = True
                    break
            if not is_necessary_subgraph:
                unnecessary_node_ids.update(subgraph_node_id for subgraph_node_id in subgraph_node_ids if subgraph_node_id != subgraph_root)
        
        valid_node_ids = set(self.graph_data['node']).difference(unnecessary_node_ids)
        self.refine_graph(valid_node_ids)
    
    def prune_intermediate_nodes(self):
        unnecessary_node_ids = set()
        
        transfer_lookup_table = {}
        for edge_info in self.graph_data['edge']:
            if edge_info['edge_type'] == 'REF' and edge_info['src_node_type'] in ['IDENTIFIER', 'METHOD_REF']:
                unnecessary_node_ids.add(edge_info['src_node_id'])
                transfer_lookup_table[edge_info['src_node_id']] = (edge_info['dst_node_id'], edge_info['dst_node_type'])
            elif edge_info['edge_type'] == 'AST' and edge_info['dst_node_type'] in ['BLOCK', 'CONTROL_STRUCTURE', 'FIELD_IDENTIFIER']:
                unnecessary_node_ids.add(edge_info['dst_node_id'])
                transfer_lookup_table[edge_info['dst_node_id']] = (edge_info['src_node_id'], edge_info['src_node_type'])
        
        edge_data = []
        for edge_info in self.graph_data['edge']:
            transferred_src_node_id = edge_info['src_node_id']
            transferred_src_node_type = edge_info['src_node_type']
            while transferred_src_node_id in transfer_lookup_table:
                transferred_src_node_id, transferred_src_node_type = transfer_lookup_table[transferred_src_node_id]
            transferred_dst_node_id = edge_info['dst_node_id']
            transferred_dst_node_type = edge_info['dst_node_type']
            while transferred_dst_node_id in transfer_lookup_table:
                transferred_dst_node_id, transferred_dst_node_type = transfer_lookup_table[transferred_dst_node_id]
            
            if edge_info['edge_type'] == 'REF':
                for location in self.graph_data['node'][edge_info['src_node_id']]['locations']:
                    if location not in self.graph_data['node'][transferred_dst_node_id]['locations']:
                        self.graph_data['node'][transferred_dst_node_id]['locations'].append(location)
            
            edge_data.append({
                'edge_type': edge_info['edge_type'],
                'src_node_id': transferred_src_node_id,
                'src_node_type': transferred_src_node_type,
                'dst_node_id': transferred_dst_node_id,
                'dst_node_type': transferred_dst_node_type,
                'edge_feature': edge_info['edge_feature'],
            })
        self.graph_data['edge'] = edge_data
        
        valid_node_ids = set(self.graph_data['node']).difference(unnecessary_node_ids)
        self.refine_graph(valid_node_ids)
    
    def save_data(self, graph_path, node_data_path, label, node_features, edge_features):
        node_ids = list(self.graph_data['node'])
        
        node_feature_matrix = [[]] * len(node_ids)
        for node_id, node_info in self.graph_data['node'].items():
            node_idx = self.get_node_idx(node_ids, node_id)
            if node_idx is not None:
                if node_info['node_type'] not in node_features['TYPE']['All']['Node_type']['types']:
                    return
                
                node_feature = []
                for component in list(node_features['TYPE']['All']) + list(node_features['API']['LOCAL']) + list(node_features['API']['CALL']) + list(node_features['STR']['LITERAL']):
                    node_feature += node_info['node_feature'][component]
                node_feature_matrix[node_idx] = node_feature
        
        edge_index = []
        edge_feature_matrix = []
        for edge_info in self.graph_data['edge']:
            src_node_idx = self.get_node_idx(node_ids, edge_info['src_node_id'])
            dst_node_idx = self.get_node_idx(node_ids, edge_info['dst_node_id'])
            if src_node_idx is not None and dst_node_idx is not None:
                edge_index.append([src_node_idx, dst_node_idx])
                
                edge_feature = []
                for component in edge_features['TYPE']['All']:
                    edge_feature += edge_info['edge_feature'][component]
                edge_feature_matrix.append(edge_feature)
        
        graph = torch_geometric.data.Data(
            x=torch.tensor(node_feature_matrix, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_feature_matrix, dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
        )
        torch.save(graph, graph_path)
        
        node_data = []
        for node_id, node_info in self.graph_data['node'].items():
            node_data.append({
                'node_idx': self.get_node_idx(node_ids, node_id),
                'node_type': node_info['node_type'],
                'code': node_info['code'],
                'locations': '&'.join(str(location) for location in node_info['locations']),
            })
        df_node_data = pd.DataFrame(node_data)
        df_node_data.to_csv(node_data_path)

def extract_cpg(script_path, cpg_path, cpg_csv_path, joern_path):
    try:
        command = [f'{joern_path}/joern-cli/joern-parse', script_path, f'--output={cpg_path}']
        subprocess.run(command, timeout=600)
        
        if os.path.exists(cpg_path):
            command = [f'{joern_path}/joern-cli/joern-export', cpg_path, '--repr=all', '--format=neo4jcsv', f'--out={cpg_csv_path}']
            subprocess.run(command)
    except:
        pass

def build_cpg(script_path, joern_path, node_features, edge_features):
    label_path = os.path.join(script_path, 'label')
    cpg_path = os.path.join(script_path, 'cpg.bin')
    cpg_csv_path = os.path.join(script_path, 'cpg_csv')
    graph_path = os.path.join(script_path, 'graph.pt')
    node_data_path = os.path.join(script_path, 'node_data.csv')
    
    with open(label_path, 'r') as f:
        label = int(f.read())
    
    extract_cpg(script_path, cpg_path, cpg_csv_path, joern_path)
    
    if os.path.exists(cpg_csv_path):
        graph_builder(cpg_csv_path, graph_path, node_data_path, label, node_features, edge_features)

def main(joern_path):
    data_path = os.path.abspath('./data')
    scripts_path = os.path.join(data_path, 'scripts')
    for ast_hash in os.listdir(scripts_path):
        script_path = os.path.join(scripts_path, ast_hash)
        build_cpg(script_path, joern_path, node_features, edge_features)

if __name__ == '__main__':
    main(sys.argv[1])
