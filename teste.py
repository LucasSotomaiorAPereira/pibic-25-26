import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('NF-UNSW-NB15-v3.csv')


# --- Dicionário para armazenar as bases de cada ataque ---
bases_por_ataque = {}
categorias_unicas = df['Attack'].unique()

# --- ETAPA 1: Dividir os dados DENTRO de cada categoria de ataque ---
print("--- ETAPA 1: Divisão por categoria (50% treino, 25% teste, 25% validação) ---\n")
for ataque in categorias_unicas:
    print(f"Processando a categoria: '{ataque}'")

    # Filtra o DataFrame para pegar apenas os dados do ataque atual
    df_ataque = df[df['Attack'] == ataque]

    # Primeira divisão: 50% treino, 50% temp (para teste/validação)
    df_treino_ataque, df_temp = train_test_split(
        df_ataque,
        train_size=0.5,
        random_state=42
        # Não precisa de 'stratify' aqui pois já estamos dentro de uma única categoria
    )

    # Segunda divisão: divide o temp em 50% teste e 50% validação
    df_teste_ataque, df_validacao_ataque = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=42
    )

    # Armazena os 3 DataFrames resultantes no dicionário
    bases_por_ataque[ataque] = {
        'treino': df_treino_ataque,
        'teste': df_teste_ataque,
        'validacao': df_validacao_ataque
    }

    print(f"  -> Treino: {len(df_treino_ataque)} | Teste: {len(df_teste_ataque)} | Validação: {len(df_validacao_ataque)}")

print("\n" + "="*40 + "\n")


# --- ETAPA 2: Escolher as categorias para cada base final ---
print("--- ETAPA 2: Sorteando as categorias para as bases finais ---\n")

# Embaralha a lista de categorias para garantir a aleatoriedade
np.random.seed(42) # Usando seed para o resultado ser sempre o mesmo
np.random.shuffle(categorias_unicas)

# Divide a LISTA de categorias nas proporções 50%, 25%, 25%
# np.split(array, [índice_corte_1, índice_corte_2])
cat_para_treino, cat_para_teste, cat_para_validacao = np.split(
    categorias_unicas,
    [
        int(len(categorias_unicas) * 0.5),      # Corte para os 50%
        int(len(categorias_unicas) * 0.75)     # Corte para os 75% (50% + 25%)
    ]
)

print(f"Categorias para a base de TREINO FINAL: {list(cat_para_treino)}")
print(f"Categorias para a base de TESTE FINAL: {list(cat_para_teste)}")
print(f"Categorias para a base de VALIDAÇÃO FINAL: {list(cat_para_validacao)}")

print("\n" + "="*40 + "\n")


# --- ETAPA 3: Montar as bases finais ---
print("--- ETAPA 3: Montando as bases finais ---\n")

# Pega as bases de TREINO das categorias sorteadas para TREINO
lista_df_treino = [bases_por_ataque[cat]['treino'] for cat in cat_para_treino]
base_treino_final = pd.concat(lista_df_treino)

# Pega as bases de TESTE das categorias sorteadas para TESTE
lista_df_teste = [bases_por_ataque[cat]['teste'] for cat in cat_para_teste]
base_teste_final = pd.concat(lista_df_teste)

# Pega as bases de VALIDAÇÃO das categorias sorteadas para VALIDAÇÃO
lista_df_validacao = [bases_por_ataque[cat]['validacao'] for cat in cat_para_validacao]
base_validacao_final = pd.concat(lista_df_validacao)


# --- Verificação dos resultados ---
print(f"--- Base de Treino Final ---")
print(f"Tamanho: {len(base_treino_final)} linhas")
print(f"Categorias presentes: {base_treino_final['Attack'].unique()}")
print(base_treino_final['Attack'].value_counts())
print("-" * 25)

print(f"\n--- Base de Teste Final ---")
print(f"Tamanho: {len(base_teste_final)} linhas")
print(f"Categorias presentes: {base_teste_final['Attack'].unique()}")
print(base_teste_final['Attack'].value_counts())
print("-" * 25)

print(f"\n--- Base de Validação Final ---")
print(f"Tamanho: {len(base_validacao_final)} linhas")
print(f"Categorias presentes: {base_validacao_final['Attack'].unique()}")
print(base_validacao_final['Attack'].value_counts())
print("-" * 25)