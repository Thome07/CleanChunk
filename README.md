## 🧠 CleanChunk – Chunk Semântico Inteligente em JSON
Transforme documentos extensos (PDF, DOCX, TXT) em chunks semânticos limpos e estruturados em JSON, prontos para NLP, RAG, chatbots ou análise de conteúdo.

### ⚙️ Funcionalidades
Suporte a arquivos: PDF, DOCX, TXT

3 modos de chunking:

📖 Texto Livre: Chunks coesos validados semanticamente

❓ FAQ: Perguntas e respostas detectadas automaticamente

🎤 Entrevista: Pares pergunta–resposta extraídos por turnos de fala

Saída JSON estruturada: Inclui metadados como página, coesão e similaridade

NLP Inteligente:

Sentence Transformers

Threshold ajustável de similaridade (slider)

Validação gramatical, semântica e remoção de ruído

### 🚀 Como Usar
Instale as dependências:
```
pip install -r requirements.txt
``` 
Rode a interface:

```
python Main.py
```
No app:

Selecione o arquivo (PDF, DOCX ou TXT)

Escolha o tipo de conteúdo (Texto, FAQ, Entrevista)

Ajuste o nível de similaridade (0.5–0.9)

Clique em Processar Arquivo

Salve o resultado em JSON

### 📤 Exemplo de Saída (JSON)
```json
{
  "chunk_id": 1,
  "conteudo": "Texto processado...",
  "pagina": 3,
  "similaridade_media": 0.82,
  "coesao_interna": 0.79
}
```
### 🧪 Tecnologias
NLP: sentence-transformers, transformers (opcional)

PDFs: pdfplumber

Interface: tkinter (com tema escuro)

Limpeza de texto: unidecode, ftfy

### 💡 Dicas
Use o modo "Contém Cabeçalhos" para melhores resultados em documentos estruturados

Recomendado manter os modelos NLP atualizados

Arquivos grandes (>100 páginas) podem exigir mais RAM

