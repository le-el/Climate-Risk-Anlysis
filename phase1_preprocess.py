"""
Phase 1: Preprocess all documents - Convert PDFs/HTML to searchable text chunks
"""

import os
import json
from pathlib import Path
import pdfplumber
from bs4 import BeautifulSoup
from tqdm import tqdm
import re

def extract_text_from_pdf(pdf_path, chunk_size=1000):
    """Extract text from PDF and split into chunks"""
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue
                
                # Split into sentences/paragraphs
                paragraphs = re.split(r'\n\s*\n|\.\s+', text)
                current_chunk = ""
                chunk_num = 0
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) < chunk_size:
                        current_chunk += para + " "
                    else:
                        if current_chunk:
                            chunks.append({
                                "page": page_num,
                                "chunk_id": f"{Path(pdf_path).stem}_{page_num}_{chunk_num}",
                                "text": current_chunk.strip(),
                                "language": "en"
                            })
                            chunk_num += 1
                        current_chunk = para + " "
                
                # Add remaining chunk
                if current_chunk.strip():
                    chunks.append({
                        "page": page_num,
                        "chunk_id": f"{Path(pdf_path).stem}_{page_num}_{chunk_num}",
                        "text": current_chunk.strip(),
                        "language": "en"
                    })
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    
    return chunks

def extract_text_from_html(html_path, chunk_size=1000):
    """Extract text from HTML and split into chunks"""
    chunks = []
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text(separator='\n')
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            chunk_num = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < 50:  # Skip very short paragraphs
                    continue
                
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + " "
                else:
                    if current_chunk:
                        chunks.append({
                            "page": 0,  # HTML doesn't have pages
                            "chunk_id": f"{Path(html_path).stem}_0_{chunk_num}",
                            "text": current_chunk.strip(),
                            "language": "en"
                        })
                        chunk_num += 1
                    current_chunk = para + " "
            
            if current_chunk.strip():
                chunks.append({
                    "page": 0,
                    "chunk_id": f"{Path(html_path).stem}_0_{chunk_num}",
                    "text": current_chunk.strip(),
                    "language": "en"
                })
    except Exception as e:
        print(f"Error processing {html_path}: {e}")
    
    return chunks

def preprocess_company_data(data_folder, output_folder="preprocessed_chunks"):
    """Preprocess all documents for all companies"""
    os.makedirs(output_folder, exist_ok=True)
    
    for company_folder in os.listdir(data_folder):
        company_path = os.path.join(data_folder, company_folder)
        if not os.path.isdir(company_path) or company_folder.startswith('.'):
            continue
        
        print(f"\nProcessing company: {company_folder}")
        all_chunks = []
        
        # Process PDFs
        documents_path = os.path.join(company_path, 'documents')
        if os.path.exists(documents_path):
            pdf_files = [f for f in os.listdir(documents_path) if f.endswith('.pdf')]
            print(f"  Found {len(pdf_files)} PDF documents")
            
            for pdf_file in tqdm(pdf_files, desc="  Processing PDFs"):
                pdf_path = os.path.join(documents_path, pdf_file)
                chunks = extract_text_from_pdf(pdf_path)
                for chunk in chunks:
                    chunk["company_id"] = company_folder
                    chunk["doc_id"] = pdf_file
                all_chunks.extend(chunks)
        
        # Process HTML files
        webpages_path = os.path.join(company_path, 'webpages')
        if os.path.exists(webpages_path):
            html_files = [f for f in os.listdir(webpages_path) if f.endswith('.html')]
            print(f"  Found {len(html_files)} HTML files")
            
            for html_file in tqdm(html_files, desc="  Processing HTML"):
                html_path = os.path.join(webpages_path, html_file)
                chunks = extract_text_from_html(html_path)
                for chunk in chunks:
                    chunk["company_id"] = company_folder
                    chunk["doc_id"] = html_file
                all_chunks.extend(chunks)
        
        # Save chunks for this company
        output_file = os.path.join(output_folder, f"{company_folder}_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(all_chunks)} chunks to {output_file}")
    
    print("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    preprocess_company_data("Collected_Data")
