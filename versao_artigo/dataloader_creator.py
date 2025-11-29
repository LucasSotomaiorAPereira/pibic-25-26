import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class CreatorDL:
    def __init__(self, seed, bs):
        self.seed = seed
        self.bs = bs

    def reader(self, filename):
        df = pd.read_csv(f'../db/{filename}.csv')
        
        df.drop(['IPV4_SRC_ADDR',
                 'IPV4_DST_ADDR',
                 'L4_SRC_PORT',
                 'L4_DST_PORT',
                 'L7_PROTO',
                 'TCP_FLAGS',
                 'CLIENT_TCP_FLAGS',
                 'SERVER_TCP_FLAGS',
                 'MIN_TTL', 
                 'MAX_TTL',
                 'SHORTEST_FLOW_PKT',
                 'MIN_IP_PKT_LEN', 
                 'TCP_WIN_MAX_IN', 
                 'TCP_WIN_MAX_OUT', 
                 'DNS_QUERY_ID', 
                 'DNS_TTL_ANSWER',
                 'FTP_COMMAND_RET_CODE',
                 'SRC_TO_DST_SECOND_BYTES',
                 'DST_TO_SRC_SECOND_BYTES',
                 'FLOW_START_MILLISECONDS',
                 'FLOW_END_MILLISECONDS',], inplace=True, axis=1)
        return df
        
    def splitter(self, df):
        dictionary_sets_by_attack_type = {}
        attack_types = df['Attack'].unique()

        for attack_type in attack_types:
            print(f"Processando a categoria: '{attack_type}'")
            df_current_attack = df[df['Attack'] == attack_type]
        
            df_train_current_attack, df_aux_current_attack = train_test_split(df_current_attack, train_size=0.5, random_state=self.seed)
            df_test_current_attack, df_val_current_attack = train_test_split(df_aux_current_attack, train_size=0.5, random_state=self.seed)
        
            dictionary_sets_by_attack_type[attack_type] = {
                'treino': df_train_current_attack,
                'teste': df_test_current_attack,
                'validacao': df_val_current_attack
            }
            print(f"  -> Treino: {len(df_train_current_attack)} | Teste: {len(df_test_current_attack)} | Validação: {len(df_val_current_attack)}")

        list_train = [dictionary_sets_by_attack_type[attack_type]['treino'] for attack_type in attack_types]
        df_train = pd.concat(list_train)
        df_train = shuffle(df_train, random_state=self.seed).reset_index(drop=True)
        
        list_test = [dictionary_sets_by_attack_type[attack_type]['teste'] for attack_type in attack_types]
        df_test = pd.concat(list_test)
        df_test = shuffle(df_test, random_state=self.seed).reset_index(drop=True)
        
        list_val = [dictionary_sets_by_attack_type[attack_type]['validacao'] for attack_type in attack_types]
        df_val = pd.concat(list_val)
        df_val = shuffle(df_val, random_state=self.seed).reset_index(drop=True)
        
        print(f"\n--- Base de Treino ---")
        print(f"Tamanho: {len(df_train)} linhas")
        print(f"Categorias presentes: {df_train['Attack'].unique()}")
        print(df_train['Attack'].value_counts())
        print("-" * 25)
        
        print(f"\n--- Base de Teste ---")
        print(f"Tamanho: {len(df_test)} linhas")
        print(f"Categorias presentes: {df_test['Attack'].unique()}")
        print(df_test['Attack'].value_counts())
        print("-" * 25)
        
        print(f"\n--- Base de Validação ---")
        print(f"Tamanho: {len(df_val)} linhas")
        print(f"Categorias presentes: {df_val['Attack'].unique()}")
        print(df_val['Attack'].value_counts())
        print("-" * 25)

        return df_train, df_test, df_val

    def balancer(self, df_train, df_test, df_val):
        scaler = MinMaxScaler()

        df_train_benign = df_train[df_train['Attack'] == 'Benign']
        df_train_attacks = df_train[df_train['Attack'] != 'Benign']
        
        rus = df_train_attacks['Attack'].value_counts().min()
        if rus < 1000:
            rus = 1000
        
        df_train_attacks_balanced = df_train_attacks.groupby('Attack').sample(n=rus, replace=True, random_state=self.seed)
        
        num_attack_classes = len(df_train_attacks['Attack'].unique())
        num_benign_samples = num_attack_classes * rus
        df_train_benign_sampled = df_train_benign.sample(n=num_benign_samples, random_state=self.seed)
        
        df_train = pd.concat([df_train_attacks_balanced, df_train_benign_sampled])
        df_train = shuffle(df_train, random_state=self.seed).reset_index(drop=True)
        
        
        X_train = df_train.drop(['Label', 'Attack'], axis=1)
        y_train = df_train['Label'].to_numpy()
        
        X_train = scaler.fit_transform(X_train)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        print(f"\n--- train ---")
        print(df_train['Label'].value_counts())
        print()
        print(df_train['Attack'].value_counts())
        print()
        print(X_train.shape)
        print()
        print(y_train.unique(return_counts=True))
        print(X_train.min(), X_train.max(), X_train.mean())
        print("-" * 25)

        df_test_benign = df_test[df_test['Attack'] == 'Benign']
        df_test_attacks = df_test[df_test['Attack'] != 'Benign']
        
        rus = df_test_attacks['Attack'].value_counts().min()
        if rus < 1000:
            rus = 1000
        
        df_test_attacks_balanced = df_test_attacks.groupby('Attack').sample(n=rus, replace=True, random_state=self.seed)
        
        num_attack_classes = len(df_test_attacks['Attack'].unique())
        num_benign_samples = num_attack_classes * rus
        df_test_benign_sampled = df_test_benign.sample(n=num_benign_samples, random_state=self.seed)
        
        df_test = pd.concat([df_test_attacks_balanced, df_test_benign_sampled])
        df_test = shuffle(df_test, random_state=self.seed).reset_index(drop=True)
        
        
        X_test = df_test.drop(['Label', 'Attack'], axis=1)
        y_test = df_test['Label'].to_numpy()
        
        X_test = scaler.fit_transform(X_test)
        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        print(f"\n--- test ---")
        print(df_test['Label'].value_counts())
        print()
        print(df_test['Attack'].value_counts())
        print()
        print(X_test.shape)
        print()
        print(y_test.unique(return_counts=True))
        print(X_test.min(), X_test.max(), X_test.mean())
        print("-" * 25)

        df_val_benign = df_val[df_val['Attack'] == 'Benign']
        df_val_attacks = df_val[df_val['Attack'] != 'Benign']
        
        rus = df_val_attacks['Attack'].value_counts().min()
        if rus < 1000:
            rus = 1000
        
        df_val_attacks_balanced = df_val_attacks.groupby('Attack').sample(n=rus, replace=True, random_state=self.seed)
        
        num_attack_classes = len(df_val_attacks['Attack'].unique())
        num_benign_samples = num_attack_classes * rus
        df_val_benign_sampled = df_val_benign.sample(n=num_benign_samples, random_state=self.seed)
        
        df_val = pd.concat([df_val_attacks_balanced, df_val_benign_sampled])
        df_val = shuffle(df_val, random_state=self.seed).reset_index(drop=True)
        
        
        X_val = df_val.drop(['Label', 'Attack'], axis=1)
        y_val = df_val['Label'].to_numpy()
        
        X_val = scaler.fit_transform(X_val)
        
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)

        print(f"\n--- val ---")
        print(df_val['Label'].value_counts())
        print()
        print(df_val['Attack'].value_counts())
        print()
        print(X_val.shape)
        print()
        print(y_val.unique(return_counts=True))
        print(X_val.min(), X_val.max(), X_val.mean())

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        val_dataset = TensorDataset(X_val, y_val)
                
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True, num_workers=80)
        test_loader = DataLoader(test_dataset, batch_size=self.bs, shuffle=False, num_workers=80)
        val_loader = DataLoader(val_dataset, batch_size=self.bs, shuffle=False, num_workers=80)

        return train_loader, test_loader, val_loader