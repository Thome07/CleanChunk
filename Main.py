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

try:
    from transformers import pipeline
    gramatica_checker = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")
except ImportError:
    gramatica_checker = None
    print("AVISO: Biblioteca 'transformers' não instalada. A validação de sentido estará desativada.")

def tem_sentido(texto):
    if not gramatica_checker or len(texto) < 3:
        return False
    try:
        resultado = gramatica_checker(texto[:512])  # Limita ao tamanho máximo do modelo
        return resultado[0]['label'] == 'LABEL_1' and resultado[0]['score'] > 0.7
    except:
        return False
    
def preprocess_text(text):
    # Remove linhas com padrão numérico isolado (ex: "1.", "2.") 
    text = re.sub(r'^\s*\d+\.?\s*$', '', text, flags=re.MULTILINE)
    # Remove palavras isoladas em CAIXA ALTA com menos de 5 letras
    text = re.sub(r'\b[A-Z]{1,4}\b\.?', '', text)
    # Remove pontuação repetida
    text = re.sub(r'([.!?])\1+', r'\1', text)
    return text.strip()

# Modifique a função chunk_structured
def chunk_structured(path):
    raw = load_raw(path)
    raw = limpar_texto(raw)
    raw = preprocess_text(raw)
    parts = re.split(r'\n{2,}', raw)
    all_chunks = []
    
    for part in parts:
        lines = part.splitlines()
        pagina, header, idx = None, None, 0
        
        if lines and lines[0].startswith('[Página'):
            pagina_match = re.search(r'\d+', lines[0])
            if pagina_match:
                pagina = int(pagina_match.group())
                idx = 1
        
        if idx < len(lines) and header_re.match(lines[idx]):
            header = lines[idx]
            idx += 1
        
        body = ' '.join(lines[idx:]).strip()
        
        # Filtros rigorosos
        if not body or len(body) < 25 or re.match(r'^[\d\W]+$', body):
            continue
            
        if len(body) > 300:
            subs = chunk_semantic_pairwise(split_sentences(body), pagina)
        else:
            subs = [{'pagina': pagina, 'conteudo': body}]
        
        for c in subs:
            conteudo = c['conteudo']
            
            # Verificação em camadas
            if len(conteudo) < 5:  # Bloqueia totalmente textos mínimos
                continue
                
            if 5 <= len(conteudo) < 15:
                # Filtros adicionais antes de usar IA
                if re.match(r'^\W*\d+\.?\W*$', conteudo):  # Números isolados
                    continue
                if conteudo.isupper() and len(conteudo.split()) < 2:  # Palavras soltas em CAIXA ALTA
                    continue
                if not tem_sentido(conteudo):
                    continue
                    
            if header:
                c['conteudo'] = f"[{header}] {conteudo}"
                
            all_chunks.append(c)
    
    for i, c in enumerate(all_chunks, 1):
        c['chunk_id'] = i
    
    return all_chunks

# Configurações
t_model = SentenceTransformer('all-MiniLM-L6-v2')
THRESHOLD = 0.75
speaker_re = re.compile(r'^([^:]{1,50}):\s*(.*)')
faq_q_re = re.compile(r'^\s*\d+\.\s*(.*)')  # Pergunta
faq_a_re = re.compile(r'^\s*[Rr]:?\s*(.*)')  # Resposta (aceita "R:", "r:" ou "R ")
header_re = re.compile(r'^(Capítulo\s*\d+[:\-]?.*|Exercício\b.*|Tópico\b.*)', re.IGNORECASE)
header_interview_re = re.compile(r'^\d+\.\s+\w+')  # Detecta padrões como "1. Introdução"

# Limpeza robusta de texto using ftfy + unidecode
def limpar_texto(texto):
    texto = ftfy.fix_text(texto)
    texto = unidecode(texto)
    texto = re.sub(r'\\["\']', '', texto)  # Remove escapes de aspas
    texto = re.sub(r'-\s*\n', '', texto)
    texto = re.sub(r'\r\n|\r|\n', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'\s+([,\.!?;:])', r'\1', texto)
    return texto.strip()

# Lê linhas preserving original spacing for .txt, .pdf, .docx
def load_lines(path):
     ext = os.path.splitext(path)[1].lower()
     lines = []
     if ext == '.pdf':
         with pdfplumber.open(path) as pdf:
             for i, page in enumerate(pdf.pages, 1):
                 raw = page.extract_text() or ''
                 lines.append(f"[Página {i}]")
                 for l in raw.splitlines():
                     lines.append(l)
     elif ext == '.docx':
         doc = docx.Document(path)
         for para in doc.paragraphs:
             lines.append(para.text)
     else:
         with open(path, encoding='utf-8') as f:
             for l in f:
                 lines.append(l.rstrip('\n'))
     return [l for l in lines if l]

# Reads raw text for structured chunks
def load_raw(path):
     ext = os.path.splitext(path)[1].lower()
     parts = []
     if ext == '.pdf':
         with pdfplumber.open(path) as pdf:
             for i, page in enumerate(pdf.pages, 1):
                 raw = page.extract_text() or ''
                 parts.append(f"[Página {i}]\n" + raw)
     elif ext == '.docx':
         doc = docx.Document(path)
         for para in doc.paragraphs:
             parts.append(para.text)
     else:
         parts.append(open(path, encoding='utf-8').read())
     return "\n\n".join(parts)

# Chunkers
def chunk_faq(lines):
    chunks = []
    chunk_id = 1
    for i in range(len(lines)-1):
        qm = faq_q_re.match(lines[i])
        am = faq_a_re.match(lines[i+1])
        if qm and am:
            chunks.append({
                'chunk_id': chunk_id,
                'pergunta': qm.group(1).strip(),
                'resposta': am.group(1).strip()
            })
            chunk_id += 1
    return chunks

def parse_blocks(lines):
    blocks, curr, current_header = [], None, None
    for l in lines:
        # Verifica se é um cabeçalho
        if header_interview_re.match(l):
            current_header = l.strip()
            continue
            
        m = speaker_re.match(l)
        if m:
            name, txt = m.groups()
            if curr and curr[0] == name:
                curr[1].append(txt)
            else:
                if curr: blocks.append((current_header, curr))
                curr = [name, [txt]]
    if curr: blocks.append((current_header, curr))
    return blocks

def chunk_interview(blocks, has_header=False):
    out, chunk_id = [], 1
    for i in range(0, len(blocks)-1, 2):
        header, a = blocks[i]
        _, b = blocks[i+1]
        
        chunk = {
            'chunk_id': chunk_id,
            'pergunta': f"{a[0]}: {' '.join(a[1])}",
            'resposta': f"{b[0]}: {' '.join(b[1])}"
        }
        
        if has_header and header:
            chunk['cabecalho'] = header
            
        out.append(chunk)
        chunk_id += 1
    return out

def split_sentences(text):
     return [s for s in re.split(r'(?<=[.!?])\s+', text) if s]

def chunk_semantic_pairwise(sents, pagina=None):
     embs = t_model.encode(sents)
     res, curr = [], [sents[0]]
     for idx in range(1, len(sents)):
         sim = (embs[idx] @ embs[idx-1])/(np.linalg.norm(embs[idx])*np.linalg.norm(embs[idx-1]))
         if sim < THRESHOLD:
             res.append({'pagina': pagina, 'conteudo': ' '.join(curr)})
             curr = [sents[idx]]
         else:
             curr.append(sents[idx])
     res.append({'pagina': pagina, 'conteudo': ' '.join(curr)})
     return res

def chunk_structured(path):
     raw = load_raw(path)
     parts = re.split(r'\n{2,}', raw)
     all_chunks = []
     for part in parts:
         lines = part.splitlines()
         pagina, header, idx = None, None, 0
         if lines[0].startswith('[Página'):
             pagina = int(re.search(r'\d+', lines[0]).group())
             idx = 1
         if idx < len(lines) and header_re.match(lines[idx]):
             header = lines[idx]; idx += 1
         body = ' '.join(lines[idx:]).strip()
         if not body: continue
         if len(body) > 300:
             subs = chunk_semantic_pairwise(split_sentences(body), pagina)
         else:
             subs = [{'pagina': pagina, 'conteudo': body}]
         for c in subs:
             if header:
                 c['conteudo'] = f"[{header}] {c['conteudo']}"
             all_chunks.append(c)
     for i, c in enumerate(all_chunks, 1): c['chunk_id'] = i
     return all_chunks

class Application:
    def __init__(self, master):
        self.master = master
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        # Cores do tema dark
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

        self.master.title("Processador de Documentos")
        self.master.geometry("900x700")
        self.master.configure(bg=dark_bg)

        # Título
        title = ttk.Label(self.master, text="Processador de Documentos", style='Title.TLabel')
        title.pack(pady=(18, 2))
        subtitle = ttk.Label(self.master, text="Quebre arquivos em chunks semânticos de forma fácil e visual", style='Subtitle.TLabel')
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
        self.add_tooltip(btn_browse, "Selecione o arquivo para processar")
        # Botão de download na mesma linha
        download_btn = ttk.Button(file_frame, text="⬇️ Baixar JSON", command=self.save_json)
        download_btn.pack(side=LEFT, padx=(12, 0))

        # Linha das opções
        options_frame = ttk.Frame(main_frame, style='TFrame')
        options_frame.pack(fill=X, pady=10)
        ttk.Label(options_frame, text="Estilo:").pack(side=LEFT, padx=(0, 11))
        self.doc_type = ttk.Combobox(options_frame, values=["FAQ", "Pergunta-Resposta/Entrevista", "Texto Puro"], state="readonly", width=24)
        self.doc_type.pack(side=LEFT, padx=(0, 11))
        self.doc_type.bind("<<ComboboxSelected>>", self.toggle_header_check)
        self.header_var = IntVar()
        self.header_check = ttk.Checkbutton(options_frame, text="Contém cabeçalhos", variable=self.header_var)
        self.header_check.pack(side=LEFT, padx=(0, 8))
        self.header_check.pack_forget()

        # Exemplo em card
        card_frame = Frame(main_frame, bg=card_bg, highlightbackground=card_border, highlightthickness=1, bd=0)
        card_frame.pack(fill=X, pady=(0, 16), padx=2)
        self.example_label = Label(card_frame, text="", font=("Segoe UI", 11, "italic"), fg=accent, bg=card_bg, justify=LEFT, anchor=W, padx=12, pady=8)
        self.example_label.pack(fill=X, anchor=W)
        self.update_example_label()

        # Divisor visual
        divider = Frame(main_frame, height=2, bg=border_color, bd=0)
        divider.pack(fill=X, pady=(0, 18))

        # Botão de processar centralizado
        process_btn = ttk.Button(main_frame, text="⚡ Processar Arquivo", command=self.process_file)
        process_btn.pack(pady=(0, 10), ipadx=10)
        self.add_tooltip(process_btn, "Clique para processar o arquivo selecionado")

        # Frame de visualização com sombra
        view_frame = Frame(self.master, bg=shadow)
        view_frame.pack(fill=BOTH, expand=1, padx=40, pady=(0, 10))
        result_card = Frame(view_frame, bg=card_bg, highlightbackground=card_border, highlightthickness=2, bd=0)
        result_card.pack(fill=BOTH, expand=1, padx=0, pady=0)
        ttk.Label(result_card, text="Resultado em JSON:", font=("Segoe UI", 12, "bold"), foreground=accent, background=card_bg).pack(anchor=W, pady=(8, 2), padx=12)
        text_frame = Frame(result_card, bg=card_bg)
        text_frame.pack(fill=BOTH, expand=1, padx=12, pady=(0, 12))
        self.json_text = Text(text_frame, wrap=NONE, font=("Consolas", 11), bg=dark_panel, fg=text_color, insertbackground=accent, relief=GROOVE, borderwidth=2, height=18, highlightbackground=border_color, highlightcolor=border_color)
        self.json_text.pack(side=LEFT, fill=BOTH, expand=1)
        scroll_y = ttk.Scrollbar(text_frame, command=self.json_text.yview)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x = ttk.Scrollbar(result_card, orient=HORIZONTAL, command=self.json_text.xview)
        scroll_x.pack(fill=X, padx=12, pady=(0, 8))
        self.json_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

    def add_tooltip(self, widget, text):
        # Tooltip simples para widgets
        def on_enter(event):
            self.tooltip = Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{widget.winfo_rootx()+20}+{widget.winfo_rooty()+20}")
            label = Label(self.tooltip, text=text, background="#222", foreground="#eee", borderwidth=1, relief="solid", font=("Segoe UI", 9))
            label.pack(ipadx=6, ipady=2)
        def on_leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def browse_file(self):
        filepath = filedialog.askopenfilename(filetypes=[
            ("Documentos", "*.txt *.pdf *.docx")
        ])
        if filepath:
            self.file_entry.delete(0, END)
            self.file_entry.insert(0, filepath)
            
    def update_example_label(self):
        estilo = self.doc_type.get()
        if estilo == "Pergunta e Resposta":
            exemplo = "Cabeçalho\nPergunta:\nResposta:"
        elif estilo == "FAQ":
            exemplo = "1. Pergunta\nR: Resposta"
        else:
            exemplo = "Texto livre, Exemplo um livro"
        self.example_label.config(text=exemplo)

    def toggle_header_check(self, event=None):
        if self.doc_type.get() == "Pergunta e Resposta":
            self.header_check.pack()
        else:
            self.header_check.pack_forget()
        self.update_example_label()
    
    def process_file(self):
        path = self.file_entry.get()
        if not os.path.isfile(path):
            messagebox.showerror("Erro", "Arquivo não encontrado!")
            return
            
        choice = {"FAQ": "1", "Pergunta e Resposta": "2", "Texto Puro": "3"}[self.doc_type.get()]
        has_header = self.header_var.get() == 1
        
        try:
            lines = load_lines(path)
            if choice == '1':
                chunks = chunk_faq(lines)
            elif choice == '2':
                blocks = parse_blocks(lines)
                chunks = chunk_interview(blocks, has_header)
            else:
                chunks = chunk_structured(path)
                
            self.json_data = json.dumps(
                chunks,
                ensure_ascii=False,
                indent=2
            )
            self.json_text.delete(1.0, END)
            self.json_text.insert(END, self.json_data)
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro:\n{str(e)}")
    
    def save_json(self):
        if not hasattr(self, 'json_data'):
            messagebox.showwarning("Aviso", "Nenhum dado processado para salvar!")
            return

        # Sugerir nome baseado no arquivo de entrada
        entrada = self.file_entry.get()
        if entrada:
            base = os.path.splitext(os.path.basename(entrada))[0]
            sugestao = base + ".json"
        else:
            sugestao = "resultado.json"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile=sugestao,
            filetypes=[("JSON files", "*.json")]
        )
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.json_data)
            messagebox.showinfo("Sucesso", "Arquivo salvo com sucesso!")

# Main
if __name__ == '__main__':
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    root = Tk()
    app = Application(root)
    root.mainloop()
    