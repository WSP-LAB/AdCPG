import hashlib
import os

import adblockparser
import escodegen
import esprima
import pandas as pd
import requests
from seleniumwire import webdriver
from seleniumwire.utils import decode

class script_extractor:
    def __init__(self, request, scripts_path, filter_lists):
        try:
            script_url = request.url
            if script_url:
                response = request.response
                if 'javascript' in str(response.headers.get('Content-Type', '')):
                    body = decode(response.body, response.headers.get('Content-Encoding', 'identity')).decode('utf-8')
                    
                    self.ast_structure = ''
                    ast = esprima.parseScript(body, delegate=self.get_ast_structure)
                    if self.ast_structure:
                        ast_hash = hashlib.sha256(self.ast_structure.encode()).hexdigest()
                        if ast_hash not in os.listdir(scripts_path):
                            code = escodegen.generate(ast, options={'parse': esprima.parseScript})
                            if code:
                                count = sum(filter_lists[name].should_block(script_url) for name in filter_lists)
                                label = 1 if count >= 2 else 0 if count == 0 else None
                                if label is not None:
                                    self.save_data(scripts_path, ast_hash, code, label)
        except:
            pass
    
    def get_ast_structure(self, node, metadata):
        self.ast_structure += node.type
    
    def save_data(self, scripts_path, ast_hash, code, label):
        script_path = os.path.join(scripts_path, ast_hash)
        os.makedirs(script_path)
        
        code_path = os.path.join(script_path, 'code.js')
        with open(code_path, 'w') as f:
            f.write(code)
        
        label_path = os.path.join(script_path, 'label')
        with open(label_path, 'w') as f:
            f.write(str(label))

def get_requests(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(60)
    
    requests = []
    try:
        driver.get(url)
        requests = driver.requests
    except:
        pass
    finally:
        driver.quit()
    
    return requests

def crawl(domain, scripts_path, filter_lists):
    url = f'http://{domain}'
    requests = get_requests(url)
    for request in requests:
        script_extractor(request, scripts_path, filter_lists)

def get_filter_lists(filter_lists_path):
    os.makedirs(filter_lists_path)
    
    filter_lists = {
        'EasyList': 'https://easylist.to/easylist/easylist.txt',
        'EasyPrivacy': 'https://easylist.to/easylist/easyprivacy.txt',
        'Fanboy\'s Annoyance List': 'https://easylist.to/easylist/fanboy-annoyance.txt',
        'Peter Lowe\'s List': 'https://pgl.yoyo.org/adservers/serverlist.php?hostformat=adblockplus&mimetype=plaintext',
    }
    for name, url in filter_lists.items():
        filter_list_path = os.path.join(filter_lists_path, f'{name}.txt')
        
        content = requests.get(url).content
        with open(filter_list_path, 'wb') as f:
            f.write(content)
        
        with open(filter_list_path, 'r') as f:
            raw_rules = f.read().splitlines()
        
        filter_lists[name] = adblockparser.AdblockRules(raw_rules)
    
    return filter_lists

def get_domains(tranco_list_path):
    content = requests.get('https://tranco-list.eu/top-1m.csv.zip').content
    with open(tranco_list_path, 'wb') as f:
        f.write(content)
    
    domains = []
    for idx, row in pd.read_csv(tranco_list_path, header=None, names=['rank', 'domain'], nrows=10000).iterrows():
        domain = row['domain']
        domains.append(domain)
    
    return domains

def main():
    data_path = os.path.abspath('./data')
    os.makedirs(data_path)
    
    scripts_path = os.path.join(data_path, 'scripts')
    os.makedirs(scripts_path)
    
    filter_lists_path = os.path.join(data_path, 'filter_lists')
    filter_lists = get_filter_lists(filter_lists_path)
    
    tranco_list_path = os.path.join(data_path, 'tranco_list.zip')
    domains = get_domains(tranco_list_path)
    for domain in domains:
        crawl(domain, scripts_path, filter_lists)

if __name__ == '__main__':
    main()
