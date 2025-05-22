from tkinter import *
from tkinter import ttk, filedialog, messagebox
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import json
import pdfplumber
import docx
from unidecode import unidecode
import ftfy
import logging
from typing import List, Dict, Any, Optional

try:
    from transformers import pipeline
    gramatica_checker = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")
except ImportError:
    gramatica_checker = None
    print("AVISO: Biblioteca 'transformers' não instalada. A validação de sentido estará desativada.")

class DocumentProcessor:
    def __init__(self):
        # Modelo maior para melhor qualidade (tradeoff performance/qualidade)
        self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.threshold = 0.82  # Aumentei a sensibilidade
        self.min_chunk_size = 80  # Tamanho mínimo aumentado
        self.max_chunk_size = 600  # Tamanho máximo ajustado
        
        # Novos padrões regex para melhor detecção
        self.header_re = re.compile(
            r'^(?:Capítulo|Seção|Parte)\s+\d+[\.:]?\s*.*|^#{2,}\s+.*|^[A-Z][\w\s]{15,}$',
            re.IGNORECASE | re.MULTILINE
        )
        self.page_re = re.compile(r'\[Página (\d+)\]')

       # Regex para FAQ
        self.faq_q_re = re.compile(r'^\s*\d+\.\s*(.*)')
        self.faq_a_re = re.compile(r'^\s*[Rr]:?\s*(.*)')
        
        # Regex para Entrevista
        self.header_interview_re = re.compile(r'^\d+\.\s+\w+')
        self.speaker_re = re.compile(r'^([^:]{1,50}):\s*(.*)')

    def tem_sentido(self, texto: str) -> bool:
        """Verifica se o texto tem sentido usando modelo de gramática"""
        if not gramatica_checker or len(texto) < 3:
            return len(texto) >= 10  # Fallback mais permissivo
        try:
            resultado = gramatica_checker(texto[:512])
            return resultado[0]['label'] == 'LABEL_1' and resultado[0]['score'] > 0.6  # Threshold mais baixo
        except Exception:
            return len(texto) >= 10

    def limpar_texto(self, texto: str) -> str:
        """Limpeza robusta de texto"""
        if not texto:
            return ""
        
        # Fix encoding issues
        texto = ftfy.fix_text(texto)
        texto = unidecode(texto)
        
        # Remove escapes e caracteres problemáticos
        texto = re.sub(r'\\+["\']', '"', texto)  # Substitui \" por "
        texto = re.sub(r'-\s*\n', '', texto)
        texto = re.sub(r'\r\n|\r|\n', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        texto = re.sub(r'\s+([,\.!?;:])', r'\1', texto)
        texto = re.sub(r'\\(["\'\\])', r'\1', texto)  # Remove escapes desnecessários
        texto = texto.replace('\\n', ' ')  # Substitui quebras de linha escapadas
        
        # Tratamento completo de escapes
        replacements = {
        '\\"': '"',
        "\\'": "'",
        '\\\\': '\\',
        '\u201c': '"',  # Aspas curvas esquerdas
        '\u201d': '"',  # Aspas curvas direitas
        '\u2018': "'",  # Aspas simples esquerda
        '\u2019': "'"   # Aspas simples direita
        }
    
        for esc, normal in replacements.items():
            texto = texto.replace(esc, normal)
        # Escape de aspas duplas e simples
        texto = texto.replace('\\', '\\\\')  # Primeiro escapa as barras
        texto = texto.replace('"', '\\"')
        texto = texto.replace("'", "\\'")
        return texto.strip()

    def preprocess_text(self, text: str) -> str:
        """Preprocessamento específico para limpeza de ruído"""
        if not text:
            return ""
            
        # Remove linhas numéricas isoladas
        text = re.sub(r'^\s*\d+\.?\s*$', '', text, flags=re.MULTILINE)
        # Remove palavras em CAPS muito curtas
        text = re.sub(r'\b[A-Z]{1,4}\b\.?', '', text)
        # Remove pontuação repetida
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()

    def load_document(self, path: str) -> str:
        """Carrega documento de diferentes formatos"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            
        ext = os.path.splitext(path)[1].lower()
        content = ""
        
        try:
            if ext == '.pdf':
                with pdfplumber.open(path) as pdf:
                    parts = []
                    for i, page in enumerate(pdf.pages, 1):
                        raw = page.extract_text()
                        if raw:
                            raw = raw.replace('\\"', '"').replace("\\'", "'")
                        if raw.strip():  # Só adiciona se tem conteúdo
                            parts.append(f"[Página {i}]\n{raw}")
                    content = "\n\n".join(parts)
            elif ext == '.docx':
                doc = docx.Document(path)
                parts = [para.text for para in doc.paragraphs if para.text.strip()]
                content = "\n\n".join(parts)
            else:  # .txt
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            raise Exception(f"Erro ao ler arquivo: {str(e)}")
        
        return content

    def load_lines(self, path: str) -> List[str]:
        """Carrega documento preservando linhas"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
            
        ext = os.path.splitext(path)[1].lower()
        lines = []
        
        try:
            if ext == '.pdf':
                with pdfplumber.open(path) as pdf:
                    for i, page in enumerate(pdf.pages, 1):
                        raw = page.extract_text() or ''
                        if raw.strip():
                            lines.append(f"[Página {i}]")
                            lines.extend([l for l in raw.splitlines() if l.strip()])
            elif ext == '.docx':
                doc = docx.Document(path)
                lines = [para.text for para in doc.paragraphs if para.text.strip()]
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [l.rstrip('\n') for l in f if l.strip()]
        except Exception as e:
            raise Exception(f"Erro ao ler arquivo: {str(e)}")
        
        return lines

    def split_sentences(self, text: str) -> List[str]:
        """Divide texto em sentenças"""
        if not text:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_chunk_valid(self, chunk_sentences: List[str]) -> bool:
        """Valida se um chunk tem conteúdo semântico válido"""
        if not chunk_sentences:
            return False
            
        chunk_text = ' '.join(chunk_sentences).strip()
        
        # Critérios básicos de tamanho
        if len(chunk_text) < self.min_chunk_size:
            return False
        
        # Verifica se tem palavras suficientes (não só pontuação/números)
        words = re.findall(r'\b\w+\b', chunk_text)
        if len(words) < 10:  # Pelo menos 10 palavras
            return False
        
        # Verifica se não termina no meio de uma frase
        if not chunk_text.rstrip().endswith(('.', '!', '?', ':', ';')):
            # Se não termina com pontuação, verifica se a última sentença está completa
            last_sentence = chunk_sentences[-1].strip()
            if len(last_sentence) < 20 or not self._sentence_seems_complete(last_sentence):
                return False
        
        # Verifica se não tem muitos caracteres estranhos ou fragmentação
        clean_chars = re.sub(r'[^\w\s]', '', chunk_text)
        if len(clean_chars) < len(chunk_text) * 0.7:  # Muito símbolo/pontuação
            return False
        
        # Verifica diversidade vocabular básica
        unique_words = set(word.lower() for word in words if len(word) > 2)
        if len(unique_words) < len(words) * 0.4:  # Muito repetitivo
            return False
        
        # Verifica se não é só números ou códigos
        if re.match(r'^[\d\s\-\.]+$', chunk_text):
            return False
            
        return True

    def _sentence_seems_complete(self, sentence: str) -> bool:
        """Verifica se uma sentença parece estar completa"""
        sentence = sentence.strip()
        
        # Muito curta
        if len(sentence) < 15:
            return False
        
        # Termina com pontuação
        if sentence.endswith(('.', '!', '?', ':', ';')):
            return True
        
        # Verifica padrões de sentença incompleta
        incomplete_patterns = [
            r'\b(e|mas|que|quando|se|porque|para|com|em|de|da|do|na|no)\s*$',  # Termina com conectivo
            r'^[a-z]',  # Começa com minúscula (meio de frase)
            r',$',  # Termina com vírgula
        ]
        
        for pattern in incomplete_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return False
        
        return True

    def _clean_chunk_text(self, text: str) -> str:
        """Limpeza final e validação do texto do chunk"""
        if not text:
            return ""
        
        # Limpeza básica
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Remove caracteres de escape problemáticos
        text = text.replace('\\n', ' ').replace('\\t', ' ')
        text = re.sub(r'\\+', '', text)
        
        # Corrige pontuação duplicada
        text = re.sub(r'([.!?])\s*\1+', r'\1', text)
        
        # Remove espaços antes de pontuação
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Remove fragmentos órfãos no início/fim
        text = re.sub(r'^[,;:\-]\s*', '', text)  # Remove pontuação inicial órfã
        text = re.sub(r'\s*[,;]\s*$', '.', text)  # Substitui vírgula final por ponto
        
        text = text.strip()
        
        # Validação final - se ficou muito pequeno ou inválido, retorna vazio
        if len(text) < 30 or not re.search(r'\b\w{3,}\b.*\b\w{3,}\b', text):
            return ""
        
        return text

    def chunk_semantic_pairwise(self, sentences: List[str], pagina: Optional[int] = None) -> List[Dict]:
        """Chunking semântico melhorado com validação rigorosa"""
        if not sentences:
            return []

        # Pré-filtragem: remove sentenças claramente inválidas
        valid_sentences = []
        for sentence in sentences:
            cleaned = self.limpar_texto(sentence)
            if (len(cleaned) >= 20 and 
                self._sentence_seems_complete(cleaned) and
                not re.match(r'^[\d\s\-\.]+$', cleaned)):
                valid_sentences.append(cleaned)
        
        if not valid_sentences:
            return []

        try:
            embeddings = self.model.encode(
                valid_sentences,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            chunks = []
            current_chunk = [valid_sentences[0]]
            current_embeddings = [embeddings[0]]
            
            for i in range(1, len(valid_sentences)):
                current_sentence = valid_sentences[i]
                current_embedding = embeddings[i]
                
                # Cálculos de similaridade (mantidos iguais)
                similarities = []
                last_sim = np.dot(current_embedding, embeddings[i-1])
                similarities.append(last_sim)
                
                if len(current_embeddings) > 0:
                    chunk_centroid = np.mean(current_embeddings, axis=0)
                    centroid_sim = np.dot(current_embedding, chunk_centroid)
                    similarities.append(centroid_sim)
                
                if len(current_embeddings) >= 3:
                    recent_centroid = np.mean(current_embeddings[-3:], axis=0)
                    recent_sim = np.dot(current_embedding, recent_centroid)
                    similarities.append(recent_sim)
                
                # Média ponderada
                if len(similarities) == 1:
                    avg_similarity = similarities[0]
                elif len(similarities) == 2:
                    avg_similarity = 0.6 * similarities[0] + 0.4 * similarities[1]
                else:
                    avg_similarity = 0.4 * similarities[0] + 0.4 * similarities[1] + 0.2 * similarities[2]
                
                # Threshold dinâmico
                base_threshold = self.threshold
                chunk_size_factor = min(len(current_chunk) / 8, 0.1)
                dynamic_threshold = base_threshold - chunk_size_factor
                
                # Detecções de quebra
                is_header = self.header_re.match(current_sentence.strip())
                is_too_long = len(' '.join(current_chunk)) + len(current_sentence) > self.max_chunk_size
                
                theme_change = False
                if len(current_embeddings) >= 2:
                    prev_similarities = [np.dot(emb, embeddings[i-1]) for emb in current_embeddings[-2:]]
                    avg_prev_sim = np.mean(prev_similarities)
                    similarity_drop = avg_prev_sim - avg_similarity
                    theme_change = similarity_drop > 0.2
                
                should_break = (
                    avg_similarity < dynamic_threshold or
                    is_too_long or
                    is_header or
                    theme_change
                )
                
                if should_break:
                    # Valida chunk antes de adicionar
                    if self._is_chunk_valid(current_chunk):
                        chunk_text = self._clean_chunk_text(' '.join(current_chunk))
                        if chunk_text:  # Só adiciona se limpeza foi bem-sucedida
                            chunks.append({
                                'pagina': pagina,
                                'conteudo': chunk_text,
                                'similaridade_media': float(np.mean([np.dot(emb, current_embeddings[0]) 
                                                                   for emb in current_embeddings]))
                            })
                    
                    current_chunk = [current_sentence]
                    current_embeddings = [current_embedding]
                else:
                    current_chunk.append(current_sentence)
                    current_embeddings.append(current_embedding)
            
            # Último chunk
            if current_chunk and self._is_chunk_valid(current_chunk):
                chunk_text = self._clean_chunk_text(' '.join(current_chunk))
                if chunk_text:
                    chunks.append({
                        'pagina': pagina,
                        'conteudo': chunk_text,
                        'similaridade_media': float(np.mean([np.dot(emb, current_embeddings[0]) 
                                                           for emb in current_embeddings]))
                    })
            
            return self._post_process_semantic_chunks(chunks)
            
        except Exception as e:
            print(f"Erro no chunking semântico: {e}")
            return self._fallback_chunking(valid_sentences, pagina)

    def _post_process_semantic_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Pós-processamento para otimizar chunks semânticos"""
        if not chunks:
            return []
        
        processed = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Se o chunk é muito pequeno, tenta merge com o próximo
            if (len(current['conteudo']) < self.min_chunk_size * 1.5 and 
                i + 1 < len(chunks) and
                chunks[i + 1]['pagina'] == current['pagina']):
                
                next_chunk = chunks[i + 1]
                merged_content = current['conteudo'] + ' ' + next_chunk['conteudo']
                
                # Só faz merge se o resultado não ficar muito grande
                if len(merged_content) <= self.max_chunk_size:
                    processed.append({
                        'pagina': current['pagina'],
                        'conteudo': merged_content,
                        'similaridade_media': (current.get('similaridade_media', 0.8) + 
                                             next_chunk.get('similaridade_media', 0.8)) / 2
                    })
                    i += 2  # Pula os dois chunks que foram unidos
                    continue
            
            processed.append(current)
            i += 1
        
        return processed

    def _fallback_chunking(self, sentences: List[str], pagina: Optional[int] = None) -> List[Dict]:
        """Chunking de fallback quando há erro no semântico"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'pagina': pagina,
                        'conteudo': self._clean_chunk_text(chunk_text)
                    })
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Último chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'pagina': pagina,
                    'conteudo': self._clean_chunk_text(chunk_text)
                })
        
        return chunks

    def chunk_structured(self, path: str) -> List[Dict]:
        """Versão melhorada com extração de metadados e processamento em batches de sentenças"""
        lines = self.load_lines(path)
        all_chunks = []
        current_page = 1
        header = None
        sentences_batch = []
        
        for line in lines:
            # Atualiza página atual
            page_match = self.page_re.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                continue
                
            # Detecta cabeçalhos
            if self.header_re.match(line):
                header = line.strip()
                continue
                
            # Processa conteúdo
            if self.is_valid_content(line):
                cleaned = self.limpar_texto(line)
                sentences = self.split_sentences(cleaned)
                sentences_batch.extend(sentences)  # Acumula sentenças
                
                # Processa em batches de 100 sentenças
                if len(sentences_batch) >= 100:
                    chunks = self.chunk_semantic_pairwise(sentences_batch, current_page)
                    for chunk in chunks:
                        if header:
                            chunk['cabecalho'] = header
                        all_chunks.append(chunk)
                    sentences_batch = []
                    header = None  # Reset header após uso
        
        # Processa restante
        if sentences_batch:
            chunks = self.chunk_semantic_pairwise(sentences_batch, current_page)
            for chunk in chunks:
                if header:
                    chunk['cabecalho'] = header
                all_chunks.append(chunk)
        
        # Pós-processamento final
        return self._post_process_chunks(all_chunks)

    def _post_process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Etapas finais de validação e formatação"""
        final_chunks = []
        for i, chunk in enumerate(chunks, 1):
            # Remove cabeçalhos duplicados
            if 'cabecalho' in chunk:
                chunk['conteudo'] = chunk['conteudo'].replace(chunk['cabecalho'], '').strip()
            
            # Substitui aspas curvas e entidades HTML por aspas normais
            chunk['conteudo'] = chunk['conteudo'].replace('\u201c', '"').replace('\u201d', '"')
            chunk['conteudo'] = chunk['conteudo'].replace('&quot;', '"').replace('&#39;', "'")
            
            # Validação final rigorosa
            if self.is_valid_content(chunk['conteudo']):
                chunk['chunk_id'] = i
                chunk['comprimento'] = len(chunk['conteudo'])
                final_chunks.append(chunk)
        return final_chunks

    def is_valid_content(self, content: str) -> bool:
        """Validação mais rigorosa com múltiplos critérios"""
        if not content or len(content) < self.min_chunk_size:
            return False
            
        # Verifica padrões inválidos
        invalid_patterns = [
            r'^\d+$',  # Apenas números
            r'^[^a-zA-Z0-9]{5,}$',  # Símbolos
            r'^\s*[A-Z]{2,}\s*$'  # Siglas
        ]
        
        if any(re.match(p, content) for p in invalid_patterns):
            return False
            
        # Verificação gramatical com fallback
        return self.tem_sentido(content) if len(content) < 120 else True

    def chunk_faq(self, lines: List[str]) -> List[Dict]:
        """Processa FAQ format"""
        chunks = []
        chunk_id = 1
        
        for i in range(len(lines) - 1):
            question_match = self.faq_q_re.match(lines[i])  # Nome correto
            answer_match = self.faq_a_re.match(lines[i + 1])  # Nome correto
            
            if question_match and answer_match:
                question = question_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                if question and answer:  # Valida se tem conteúdo
                    chunks.append({
                        'chunk_id': chunk_id,
                        'pergunta': question,
                        'resposta': answer
                    })
                    chunk_id += 1
        
        return chunks

    def parse_interview_blocks(self, lines: List[str]) -> List[tuple]:
        """Parse blocos de entrevista/diálogo"""
        blocks = []
        current_block = None
        current_header = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detecta cabeçalho
            if self.header_interview_re.match(line):
                current_header = line
                continue
            
            # Detecta speaker
            speaker_match = self.speaker_re.match(line)
            if speaker_match:
                name, text = speaker_match.groups()
                name, text = name.strip(), text.strip()
                
                if current_block and current_block[0] == name:
                    current_block[1].append(text)
                else:
                    if current_block:
                        blocks.append((current_header, current_block))
                    current_block = [name, [text]]
        
        if current_block:
            blocks.append((current_header, current_block))
        
        return blocks

    def chunk_interview(self, blocks: List[tuple], has_header: bool = False) -> List[Dict]:
        """Processa formato de entrevista"""
        chunks = []
        chunk_id = 1
        
        # Processa pares de blocos (pergunta-resposta)
        for i in range(0, len(blocks) - 1, 2):
            header_a, block_a = blocks[i]
            header_b, block_b = blocks[i + 1]
            
            pergunta = f"{block_a[0]}: {' '.join(block_a[1])}"
            resposta = f"{block_b[0]}: {' '.join(block_b[1])}"
            
            chunk = {
                'chunk_id': chunk_id,
                'pergunta': pergunta,
                'resposta': resposta
            }
            
            if has_header and header_a:
                chunk['cabecalho'] = header_a
            
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks


class Application:
    def __init__(self, master):
        self.master = master
        self.processor = DocumentProcessor()
        self.json_data = None
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores do tema dark (mantidas as mesmas)
        dark_bg = '#23272e'
        dark_panel = '#2c313a'
        accent = '#5e81ac'
        accent_hover = '#81a1c1'
        text_color = '#e5e9f0'
        label_color = '#bfc9db'
        entry_bg = '#181a20'
        border_color = '#444857'
        card_bg = '#252a33'
        card_border = '#3b4252'
        shadow = '#181a20'

        # Configuração de estilos (mantida igual)
        style.configure('TFrame', background=dark_bg)
        style.configure('TLabel', background=dark_bg, foreground=label_color, font=('Segoe UI', 11))
        style.configure('Title.TLabel', background=dark_bg, foreground=accent, font=('Segoe UI', 20, 'bold'))
        style.configure('Subtitle.TLabel', background=dark_bg, foreground=label_color, font=('Segoe UI', 12, 'italic'))
        style.configure('TButton', font=('Segoe UI', 12, 'bold'), padding=10, background=accent, foreground=text_color, borderwidth=0)
        style.map('TButton', background=[('active', accent_hover)], foreground=[('active', text_color)])
        style.configure('TCheckbutton', background=dark_bg, foreground=label_color, font=('Segoe UI', 10))
        style.configure('TEntry', fieldbackground=entry_bg, foreground=text_color, bordercolor=border_color, font=('Segoe UI', 12))
        style.configure('TCombobox', fieldbackground=entry_bg, foreground=text_color, background=entry_bg, bordercolor=border_color, font=('Segoe UI', 12))
        style.map('TCombobox', fieldbackground=[('readonly', entry_bg)], foreground=[('readonly', text_color)])

        self.master.title("Processador de Documentos - Versão Melhorada")
        self.master.geometry("950x750")
        self.master.configure(bg=dark_bg)

        # Interface (mantida similar, mas com melhorias)
        title = ttk.Label(self.master, text="Processador de Documentos", style='Title.TLabel')
        title.pack(pady=(18, 2))
        subtitle = ttk.Label(self.master, text="Chunking semântico inteligente com validação de conteúdo", style='Subtitle.TLabel')
        subtitle.pack(pady=(0, 12))

        # Frame principal
        main_frame = ttk.Frame(self.master, padding=24, style='TFrame')
        main_frame.pack(fill=X, padx=40, pady=(0, 10))

        # Linha do arquivo
        file_frame = ttk.Frame(main_frame, style='TFrame')
        file_frame.pack(fill=X, pady=10)
        ttk.Label(file_frame, text="Arquivo:").pack(side=LEFT, padx=(0, 8))
        self.file_entry = ttk.Entry(file_frame, width=48)
        self.file_entry.pack(side=LEFT, padx=(0, 8))
        btn_browse = ttk.Button(file_frame, text="🔍 Procurar", command=self.browse_file)
        btn_browse.pack(side=LEFT)
        download_btn = ttk.Button(file_frame, text="⬇️ Baixar JSON", command=self.save_json)
        download_btn.pack(side=LEFT, padx=(12, 0))

        # Configurações avançadas
        config_frame = ttk.Frame(main_frame, style='TFrame')
        config_frame.pack(fill=X, pady=10)
        
        ttk.Label(config_frame, text="Estilo:").pack(side=LEFT, padx=(0, 8))
        self.doc_type = ttk.Combobox(config_frame, values=["FAQ", "Pergunta-Resposta/Entrevista", "Texto Puro"], 
                                   state="readonly", width=24)
        self.doc_type.pack(side=LEFT, padx=(0, 8))
        self.doc_type.bind("<<ComboboxSelected>>", self.toggle_header_check)
        
        self.header_var = IntVar()
        self.header_check = ttk.Checkbutton(config_frame, text="Contém cabeçalhos", variable=self.header_var)
        
        # Controles de threshold
        ttk.Label(config_frame, text="Similaridade:").pack(side=LEFT, padx=(15, 5))
        self.threshold_var = DoubleVar(value=0.75)
        threshold_scale = ttk.Scale(config_frame, from_=0.5, to=0.9, variable=self.threshold_var, 
                                  orient=HORIZONTAL, length=100)
        threshold_scale.pack(side=LEFT, padx=(0, 5))
        self.threshold_label = ttk.Label(config_frame, text="0.75")
        self.threshold_label.pack(side=LEFT)
        threshold_scale.configure(command=self.update_threshold_label)

        # Status e progresso
        self.status_var = StringVar(value="Pronto para processar")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                               font=('Segoe UI', 10, 'italic'))
        status_label.pack(pady=(10, 0))

        # Card de exemplo (mantido)
        card_frame = Frame(main_frame, bg=card_bg, highlightbackground=card_border, highlightthickness=1, bd=0)
        card_frame.pack(fill=X, pady=(10, 16), padx=2)
        self.example_label = Label(card_frame, text="", font=("Segoe UI", 11, "italic"), 
                                 fg=accent, bg=card_bg, justify=LEFT, anchor=W, padx=12, pady=8)
        self.example_label.pack(fill=X, anchor=W)
        self.update_example_label()

        # Botão de processar
        process_btn = ttk.Button(main_frame, text="⚡ Processar Arquivo", command=self.process_file)
        process_btn.pack(pady=(0, 10), ipadx=10)

        # Frame de visualização
        view_frame = Frame(self.master, bg=shadow)
        view_frame.pack(fill=BOTH, expand=1, padx=40, pady=(0, 10))
        result_card = Frame(view_frame, bg=card_bg, highlightbackground=card_border, highlightthickness=2, bd=0)
        result_card.pack(fill=BOTH, expand=1, padx=0, pady=0)
        
        # Header do resultado
        result_header = ttk.Frame(result_card, style='TFrame')
        result_header.configure(style='Card.TFrame')
        result_header.pack(fill=X, pady=(8, 0), padx=12)
        
        ttk.Label(result_header, text="Resultado:", font=("Segoe UI", 12, "bold"), 
                foreground=accent, background=card_bg).pack(side=LEFT)
        
        self.chunk_count_label = ttk.Label(result_header, text="", font=("Segoe UI", 10), 
                                         foreground=label_color, background=card_bg)
        self.chunk_count_label.pack(side=RIGHT)
        
        # Área de texto
        text_frame = Frame(result_card, bg=card_bg)
        text_frame.pack(fill=BOTH, expand=1, padx=12, pady=(5, 12))
        
        self.json_text = Text(text_frame, wrap=NONE, font=("Consolas", 11), 
                            bg=dark_panel, fg=text_color, insertbackground=accent, 
                            relief=GROOVE, borderwidth=2, height=18)
        self.json_text.pack(side=LEFT, fill=BOTH, expand=1)
        
        scroll_y = ttk.Scrollbar(text_frame, command=self.json_text.yview)
        scroll_y.pack(side=RIGHT, fill=Y)
        self.json_text.configure(yscrollcommand=scroll_y.set)

    def update_threshold_label(self, value):
        """Atualiza label do threshold"""
        self.threshold_label.config(text=f"{float(value):.2f}")
        self.processor.threshold = float(value)

    def toggle_header_check(self, event=None):
        """Toggle checkbox de cabeçalho"""
        if self.doc_type.get() == "Pergunta-Resposta/Entrevista":
            self.header_check.pack(side=LEFT, padx=(8, 0))
        else:
            self.header_check.pack_forget()
        self.update_example_label()

    def update_example_label(self):
        """Atualiza exemplo baseado no tipo selecionado"""
        estilo = self.doc_type.get()
        if estilo == "FAQ":
            exemplo = "Exemplo:\n1. Qual é a capital do Brasil?\nR: Brasília é a capital."
        elif estilo == "Pergunta-Resposta/Entrevista":
            exemplo = "Exemplo:\nEntrevistador: Como você começou?\nEntrevistado: Comecei há 5 anos..."
        else:
            exemplo = "Exemplo:\nTexto livre será dividido em chunks semânticos baseado na similaridade."
        self.example_label.config(text=exemplo)

    def browse_file(self):
        """Seleciona arquivo"""
        filepath = filedialog.askopenfilename(
            title="Selecionar documento",
            filetypes=[
                ("Todos os suportados", "*.txt *.pdf *.docx"),
                ("Arquivos de texto", "*.txt"),
                ("PDFs", "*.pdf"),
                ("Word", "*.docx")
            ]
        )
        if filepath:
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, filepath)
            self.status_var.set(f"Arquivo selecionado: {os.path.basename(filepath)}")

    def process_file(self):
        """Processa o arquivo selecionado"""
        path = self.file_entry.get().strip()
        
        if not path:
            messagebox.showerror("Erro", "Selecione um arquivo primeiro!")
            return
            
        if not os.path.isfile(path):
            messagebox.showerror("Erro", "Arquivo não encontrado!")
            return

        doc_type = self.doc_type.get()
        if not doc_type:
            messagebox.showerror("Erro", "Selecione o tipo de documento!")
            return

        self.status_var.set("Processando arquivo...")
        self.master.update()

        try:
            if doc_type == "FAQ":
                lines = self.processor.load_lines(path)
                chunks = self.processor.chunk_faq(lines)
            elif doc_type == "Pergunta-Resposta/Entrevista":
                lines = self.processor.load_lines(path)
                blocks = self.processor.parse_interview_blocks(lines)
                has_header = self.header_var.get() == 1
                chunks = self.processor.chunk_interview(blocks, has_header)
            else:  # Texto Puro
                chunks = self.processor.chunk_structured(path)

            if not chunks:
                messagebox.showwarning("Aviso", "Nenhum chunk válido foi gerado. Verifique o formato do arquivo.")
                self.status_var.set("Nenhum resultado gerado")
                return

            # Serialização JSON com parâmetros especificados
            self.json_data = json.dumps(
                chunks,
                ensure_ascii=False,
                indent=2,
                separators=(',', ': '),
                sort_keys=False
            )
            
            # Exibe resultado cru
            self.json_text.delete(1.0, END)
            self.json_text.insert(END, self.json_data)
            
            # Atualiza status
            self.chunk_count_label.config(text=f"{len(chunks)} chunks gerados")
            self.status_var.set(f"Processamento concluído - {len(chunks)} chunks")
            
            messagebox.showinfo("Sucesso", f"Arquivo processado com sucesso!\n{len(chunks)} chunks gerados.")

        except Exception as e:
            error_msg = f"Erro durante o processamento:\n{str(e)}"
            messagebox.showerror("Erro", error_msg)
            self.status_var.set("Erro no processamento")
            logging.error(f"Erro no processamento: {e}")

    def save_json(self):
        """Salva o JSON processado"""
        if not self.json_data:
            messagebox.showwarning("Aviso", "Nenhum dado processado para salvar!")
            return

        entrada = self.file_entry.get()
        if entrada:
            base = os.path.splitext(os.path.basename(entrada))[0]
            sugestao = f"{base}_chunks.json"
        else:
            sugestao = "chunks.json"

        filepath = filedialog.asksaveasfilename(
            title="Salvar chunks como JSON",
            defaultextension=".json",
            initialfile=sugestao,
            filetypes=[("JSON files", "*.json"), ("Todos os arquivos", "*.")]
        )
        
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    try:
                        json.loads(self.json_data)  # Valida o JSON
                        f.write(self.json_data)
                    except Exception as e:
                        messagebox.showerror("Erro", f"JSON inválido: {str(e)}")
                        return
                messagebox.showinfo("Sucesso", f"Arquivo salvo em:\n{filepath}")
                self.status_var.set(f"Salvo: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar arquivo:\n{str(e)}")


def main():
    """Função principal"""
    # Configura logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Suprime logs verbosos do pdfminer
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    
    # Cria e executa aplicação
    root = Tk()
    app = Application(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Aplicação interrompida pelo usuário")
    except Exception as e:
        logging.error(f"Erro fatal na aplicação: {e}")
        messagebox.showerror("Erro Fatal", f"Erro inesperado:\n{str(e)}")


if __name__ == '__main__':
    main()
