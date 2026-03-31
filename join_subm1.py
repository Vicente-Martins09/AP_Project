import pandas as pd

try:
    # 1. Nomes dos ficheiros (ajusta o nome do ficheiro do professor se for diferente)
    ficheiro_treino_antigo = 'dataset_final_v2.csv'
    ficheiro_novas_labels = 'subm2_labels_revealed.csv' 
    ficheiro_treino_atualizado = 'dataset_final_v3.csv' # Criamos um v3 para não esmagar o original sem querer
    
    # 2. Carregar os dados
    df_antigo = pd.read_csv(ficheiro_treino_antigo, sep=';')
    df_novo = pd.read_csv(ficheiro_novas_labels, sep=';')
    
    print(f"📊 Linhas no dataset antigo: {len(df_antigo)}")
    print(f"📈 Linhas nos novos dados do professor: {len(df_novo)}")
    
    # Garantir que o ficheiro do professor tem apenas as 3 colunas necessárias
    # (caso ele tenha enviado com alguma coluna extra que não precisamos)
    df_novo = df_novo[['ID', 'Text', 'Label']] 
    
    # 3. Juntar tudo num único DataFrame
    df_combinado = pd.concat([df_antigo, df_novo], ignore_index=True)
    
    # 4. Remover duplicados (olhando estritamente para o texto)
    # Assim evitamos que textos iguais tenham peso a dobrar no treino
    tamanho_antes = len(df_combinado)
    df_combinado = df_combinado.drop_duplicates(subset=['Text']).reset_index(drop=True)
    duplicados_removidos = tamanho_antes - len(df_combinado)
    if duplicados_removidos > 0:
        print(f"🧹 Foram removidos {duplicados_removidos} textos duplicados (coincidências de geração).")
    
    # 5. Baralhar (Shuffle)
    # É crucial misturar para que os 150 dados do professor não fiquem todos no fim do dataset
    df_misturado = df_combinado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. Reescrever os IDs para manter o formato impecável (D1-0001...)
    df_misturado['ID'] = [f"D1-{i+1:04d}" for i in range(len(df_misturado))]
    
    # 7. Guardar o novo mega-dataset
    df_misturado.to_csv(ficheiro_treino_atualizado, sep=';', index=False)
    
    print(f"\n✅ SUCESSO ABSOLUTO! O teu novo dataset de treino tem agora {len(df_misturado)} linhas.")
    print(f"💾 Ficheiro guardado como '{ficheiro_treino_atualizado}'.")
    
    print("\n📈 Nova distribuição de classes para os próximos treinos:")
    print(df_misturado['Label'].value_counts())
    
except FileNotFoundError as e:
    print(f"❌ Erro: Não encontrei o ficheiro. Detalhe: {e}")
except KeyError as e:
    print(f"❌ Erro de colunas: O ficheiro do professor tem os cabeçalhos exatos (ID, Text, Label)? Detalhe: {e}")