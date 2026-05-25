import os
import zipfile
import xml.etree.ElementTree as ET

def convert_docx_to_md(docx_path, md_path):
    NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    PARA = NAMESPACE + 'p'
    TEXT = NAMESPACE + 't'
    TABLE = NAMESPACE + 'tbl'
    ROW = NAMESPACE + 'tr'
    CELL = NAMESPACE + 'tc'
    
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return
        
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            body = root.find(f'.//{NAMESPACE}body')
            if body is None:
                body = root
                
            md_lines = []
            
            def get_text(el):
                return "".join(t.text for t in el.iter(TEXT) if t.text)
                
            for child in body:
                tag = child.tag
                if tag == PARA:
                    txt = get_text(child).strip()
                    if txt:
                        md_lines.append(txt + "\n")
                elif tag == TABLE:
                    table_rows = []
                    for row in child.iter(ROW):
                        row_cells = []
                        for cell in row.iter(CELL):
                            cell_text = " ".join(get_text(p).strip() for p in cell.iter(PARA))
                            row_cells.append(cell_text.replace("\n", " ").replace("|", "\\|"))
                        table_rows.append(row_cells)
                    
                    if table_rows:
                        col_count = max(len(r) for r in table_rows)
                        for r_idx, r in enumerate(table_rows):
                            r += [""] * (col_count - len(r))
                            md_lines.append("| " + " | ".join(r) + " |")
                            if r_idx == 0:
                                md_lines.append("| " + " | ".join(["---"] * col_count) + " |")
                        md_lines.append("")
                else:
                    if TABLE not in child.tag:
                        for p in child.iter(PARA):
                            txt = get_text(p).strip()
                            if txt:
                                md_lines.append(txt + "\n")
                            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
            print(f"Successfully converted {docx_path} to {md_path}")
            
    except Exception as e:
        print(f"Error parsing {docx_path}: {e}")

if __name__ == "__main__":
    artifacts_dir = r"f:\APPS_PJ\mamba-forecast-ad\Hydrid-Mamba-for-Predictive-Bearing-Fault\artifacts"
    files = [
        "references.docx",
        "report_nckh - vn.docx",
        "report_nckh.docx"
    ]
    for file in files:
        docx_path = os.path.join(artifacts_dir, file)
        md_name = file.replace(" - vn", "_vn").replace(" ", "_").replace(".docx", ".md")
        md_path = os.path.join(artifacts_dir, md_name)
        convert_docx_to_md(docx_path, md_path)
