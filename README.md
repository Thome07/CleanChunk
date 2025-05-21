## 📘 CleanChunk Transforma arquivos .txt, .pdf e .docx em chunks semânticos limpos e organizados. Ideal para pipelines de NLP, embeddings e análise textual.

🚀 Funcionalidades
Interface simples com Tkinter.

Aceita arquivos .txt, .pdf e .docx.

Realiza:

Remoção de ruído (números soltos, palavras quebradas, etc).

Correção de caracteres com ftfy e unidecode.

Divisão em chunks semânticos com SentenceTransformer.

Exporta os chunks como JSON com chunk_id e conteudo.

📂 Estrutura do JSON gerado
[ { "pagina": 5, "chunk_id": 102, "conteudo": "Texto processado aqui." } ] pagina será null em arquivos .txt ou .docx.

🛠️ Instalação
pip install -r requirements.txt

▶️ Uso
Execute o script Python principal.

Clique em Selecionar arquivo e escolha seu .txt, .pdf ou .docx.

Aguarde o processamento.

O JSON final será salvo no mesmo diretório do arquivo original.

⚠️ Observações
A chunkificação usa IA apenas em trechos curtos com baixa legibilidade.

SentenceTransformer é usado com verificação de confiança.

O código é tolerante a falhas (try/except nos pontos críticos).
