import pandas as pd
import io

nome_ficheiro = 'dataset_final.csv' 

print("FASE 1: Pré-verificação estrutural do ficheiro...\n")

erros_formato = 0
linhas_validas = []

try:
    with open(nome_ficheiro, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    if not linhas:
        raise ValueError("O ficheiro está completamente vazio!")

    # A primeira linha define a regra: quantos ';' existem no cabeçalho
    cabecalho = linhas[0].strip()
    ponto_virgula_esperado = cabecalho.count(';')
    linhas_validas.append(cabecalho)

    for i in range(1, len(linhas)):
        linha_original = linhas[i]
        linha_strip = linha_original.strip()

        # 1. VERIFICAR LINHAS VAZIAS
        if not linha_strip:
            print(f"AVISO (Linha {i+1}): Linha completamente vazia detetada e ignorada.")
            erros_formato += 1
            continue
            
        # 2. VERIFICAR PONTO E VÍRGULA A MAIS/A MENOS
        contagem_pv = linha_strip.count(';')
        if contagem_pv > ponto_virgula_esperado:
            print(f"ERRO (Linha {i+1}): Tem {contagem_pv} ';' (esperados {ponto_virgula_esperado}). Há um ';' no meio do texto!")
            print(f"   Conteúdo: {linha_strip[:60]}...")
            erros_formato += 1
            continue
        elif contagem_pv < ponto_virgula_esperado:
            print(f"ERRO (Linha {i+1}): Faltam separadores. Tem {contagem_pv} ';' (esperados {ponto_virgula_esperado}).")
            erros_formato += 1
            continue
            
        # Se passar nestes testes, a linha é estruturalmente válida
        linhas_validas.append(linha_strip)

    if erros_formato > 0:
        print(f"\nEncontrados {erros_formato} erros estruturais. Remove os ';' a mais ou linhas vazias no teu CSV original antes de prosseguir.")
    else:
        print("Fase 1 concluída: Nenhuma linha vazia nem separadores ';' a mais!\n")
        
        print("FASE 2: Verificação de Regras...\n")
        # Simular um ficheiro limpo para o pandas ler, usando as linhas válidas
        ficheiro_limpo = io.StringIO("\n".join(linhas_validas))
        df = pd.read_csv(ficheiro_limpo, sep=';')
        
        # 3. VERIFICAR LABELS
        labels_permitidas = ['Human', 'Anthropic', 'Google', 'OpenAI', 'Meta']
        if 'Label' in df.columns:
            labels_encontradas = df['Label'].dropna().unique()
            labels_invalidas = [lbl for lbl in labels_encontradas if lbl not in labels_permitidas]
            if not labels_invalidas:
                print("Labels corretas.\n")
            else:
                print(f"ERRO nas Labels. Encontradas labels inválidas: {labels_invalidas}\n")

        # 4. VERIFICAR TAMANHO DO TEXTO (80 a 120 Palavras)
        df['Contagem_Palavras'] = df['Text'].astype(str).apply(lambda x: len(x.split()))
        
        textos_validos = df['Contagem_Palavras'].between(80, 120).sum()
        textos_invalidos = len(df) - textos_validos
        
        print(f"Textos com tamanho correto (80-120 palavras): {textos_validos}")
        print(f"Textos com tamanho INCORRETO: {textos_invalidos}")
        
        if textos_invalidos > 0:
            print("\nTens linhas fora do limite! (menos de 80 ou mais de 120 palavras)")
        else:
            print("Todos os textos cumprem a regra de tamanho.")

except FileNotFoundError:
    print(f"O ficheiro '{nome_ficheiro}' não foi encontrado.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")