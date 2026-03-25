import pandas as pd

# 1. Carregar o teu dataset atual
ficheiro_atual = 'dataset_final.csv' 
ficheiro_novo = 'dataset_distribuido.csv'

print("⚖️ A ajustar a distribuição do dataset para a regra (1/3 Humanos)...\n")

try:
    df = pd.read_csv(ficheiro_atual, sep=';')
    
    # 2. Separar os Humanos
    df_human = df[df['Label'] == 'Human'].copy()
    n_human = len(df_human)
    print(f"👤 Textos Humanos (1/3 do total): {n_human}")
    
    # 3. Calcular a quota para cada IA
    # Se Humanos = 1/3, então as 4 IAs juntas = 2/3.
    # Cada IA será metade dos humanos (1/6 do total).
    n_por_ia = n_human // 2
    print(f"🤖 Quota calculada para CADA classe de IA: {n_por_ia}")
    
    # 4. Fazer o Undersampling das IAs (cortar o excesso aleatoriamente)
    df_google = df[df['Label'] == 'Google'].sample(n=n_por_ia, random_state=42)
    df_openai = df[df['Label'] == 'OpenAI'].sample(n=n_por_ia, random_state=42)
    df_meta = df[df['Label'] == 'Meta'].sample(n=n_por_ia, random_state=42)
    df_anthropic = df[df['Label'] == 'Anthropic'].sample(n=n_por_ia, random_state=42)
    
    # 5. Juntar tudo
    df_balanceado = pd.concat([df_human, df_google, df_openai, df_meta, df_anthropic], ignore_index=True)
    
    # 6. Baralhar bem para não ficarem agrupados
    df_final = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 7. Reescrever os IDs
    df_final['ID'] = [f"D1-{i+1:04d}" for i in range(len(df_final))]
    
    # 8. Guardar o novo ficheiro
    df_final.to_csv(ficheiro_novo, sep=';', index=False)
    
    print(f"\n✅ SUCESSO! O dataset '{ficheiro_novo}' foi gerado.")
    print(f"   Tamanho Total: {len(df_final)} linhas.")
    
    print("\n📊 Nova distribuição exata:")
    print(df_final['Label'].value_counts())

except FileNotFoundError:
    print(f"❌ Erro: O ficheiro '{ficheiro_atual}' não foi encontrado.")