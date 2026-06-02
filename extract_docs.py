import os
import zipfile
import xml.etree.ElementTree as ET

NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
p_tag = NAMESPACE + 'p'
t_tag = NAMESPACE + 't'
tbl_tag = NAMESPACE + 'tbl'
tr_tag = NAMESPACE + 'tr'
tc_tag = NAMESPACE + 'tc'
pPr_tag = NAMESPACE + 'pPr'
pStyle_tag = NAMESPACE + 'pStyle'
numPr_tag = NAMESPACE + 'numPr'
ilvl_tag = NAMESPACE + 'ilvl'
numId_tag = NAMESPACE + 'numId'

def to_roman(n, uppercase=True):
    if n <= 0:
        return str(n)
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ''
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            roman_num += syb[i]
            n -= val[i]
        i += 1
    return roman_num if uppercase else roman_num.lower()

def to_alpha(n, uppercase=True):
    if n <= 0:
        return str(n)
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr((65 if uppercase else 97) + remainder) + result
    return result

def format_number(val, fmt):
    if fmt == 'decimal':
        return str(val)
    elif fmt == 'upperRoman':
        return to_roman(val, True)
    elif fmt == 'lowerRoman':
        return to_roman(val, False)
    elif fmt == 'upperLetter':
        return to_alpha(val, True)
    elif fmt == 'lowerLetter':
        return to_alpha(val, False)
    elif fmt == 'bullet':
        return '-'
    else:
        return str(val)

def parse_numbering(docx_file):
    if 'word/numbering.xml' not in docx_file.namelist():
        return {}, {}, {}, {}
        
    xml_content = docx_file.read('word/numbering.xml')
    root = ET.fromstring(xml_content)
    
    # 1. Parse Abstract Nums
    abstract_nums = {}
    style_to_level = {}
    for abstractNum in root.findall(NAMESPACE + 'abstractNum'):
        absId = abstractNum.get(NAMESPACE + 'abstractNumId')
        levels = {}
        for lvl in abstractNum.findall(NAMESPACE + 'lvl'):
            ilvl = int(lvl.get(NAMESPACE + 'ilvl'))
            start = lvl.find(NAMESPACE + 'start')
            numFmt = lvl.find(NAMESPACE + 'numFmt')
            lvlText = lvl.find(NAMESPACE + 'lvlText')
            pStyle = lvl.find(NAMESPACE + 'pStyle')
            
            start_val = int(start.get(NAMESPACE + 'val')) if start is not None else 1
            fmt = numFmt.get(NAMESPACE + 'val') if numFmt is not None else 'decimal'
            text_tmpl = lvlText.get(NAMESPACE + 'val') if lvlText is not None else '%1.'
            style = pStyle.get(NAMESPACE + 'val') if pStyle is not None else None
            
            levels[ilvl] = {
                'start': start_val,
                'fmt': fmt,
                'text': text_tmpl,
                'style': style
            }
            if style:
                style_to_level[style] = (absId, ilvl)
        abstract_nums[absId] = levels
        
    # 2. Parse Nums
    num_to_abs = {}
    abs_to_num = {}
    for num in root.findall(NAMESPACE + 'num'):
        numId = num.get(NAMESPACE + 'numId')
        absVal = num.find(NAMESPACE + 'abstractNumId').get(NAMESPACE + 'val')
        num_to_abs[numId] = absVal
        if absVal not in abs_to_num:
            abs_to_num[absVal] = numId
            
    return abstract_nums, num_to_abs, style_to_level, abs_to_num

def resolve_paragraph(p, abstract_nums, num_to_abs, style_to_level, abs_to_num, list_states):
    def get_text(el):
        return "".join(t.text for t in el.iter(t_tag) if t.text)
        
    txt = get_text(p).strip()
    if not txt:
        return ""
        
    pPr = p.find(pPr_tag)
    pStyle = pPr.find(pStyle_tag) if pPr is not None else None
    style_val = pStyle.get(NAMESPACE + 'val') if pStyle is not None else None
    
    numPr = pPr.find(numPr_tag) if pPr is not None else None
    
    ilvl = None
    numId = None
    
    # Check explicit numPr
    if numPr is not None:
        ilvl_el = numPr.find(ilvl_tag)
        numId_el = numPr.find(numId_tag)
        if ilvl_el is not None and numId_el is not None:
            ilvl = int(ilvl_el.get(NAMESPACE + 'val'))
            numId = numId_el.get(NAMESPACE + 'val')
    # Check style-based numbering mapping
    elif style_val in style_to_level:
        absId, ilvl = style_to_level[style_val]
        numId = abs_to_num.get(absId)
        
    prefix = ""
    if ilvl is not None and numId is not None:
        absId = num_to_abs.get(numId)
        if absId in abstract_nums:
            levels_def = abstract_nums[absId]
            
            # Init state for this list instance (numId)
            if numId not in list_states:
                list_states[numId] = {}
            state = list_states[numId]
            
            for lvl_idx, lvl_def in levels_def.items():
                if lvl_idx not in state:
                    state[lvl_idx] = lvl_def['start'] - 1
                    
            state[ilvl] += 1
            
            # Reset sub-levels
            for lvl_idx in list(state.keys()):
                if lvl_idx > ilvl:
                    state[lvl_idx] = levels_def[lvl_idx]['start'] - 1
                    
            # Resolve formatting
            template = levels_def[ilvl]['text']
            fmt = levels_def[ilvl]['fmt']
            
            if fmt == 'bullet':
                prefix = "  " * ilvl + "-"
            else:
                resolved_prefix = template
                for k in range(1, 10):
                    ph = f"%{k}"
                    if ph in resolved_prefix:
                        lvl_k = k - 1
                        k_val = state.get(lvl_k, 1)
                        k_fmt = levels_def[lvl_k]['fmt']
                        formatted_k = format_number(k_val, k_fmt)
                        resolved_prefix = resolved_prefix.replace(ph, formatted_k)
                # Keep indent for list items but not for headings
                is_heading = style_val and style_val.startswith('Heading')
                if is_heading:
                    prefix = resolved_prefix
                else:
                    prefix = "  " * ilvl + resolved_prefix
                    
    if prefix:
        return f"{prefix} {txt}"
    return txt

def convert_docx_to_md(docx_path, md_path):
    if not os.path.exists(docx_path):
        print(f"File not found: {docx_path}")
        return
        
    try:
        with zipfile.ZipFile(docx_path) as docx:
            abstract_nums, num_to_abs, style_to_level, abs_to_num = parse_numbering(docx)
            
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            
            body = root.find(f'.//{NAMESPACE}body')
            if body is None:
                body = root
                
            md_lines = []
            list_states = {}
            
            for child in body:
                tag = child.tag
                if tag == p_tag:
                    txt = resolve_paragraph(child, abstract_nums, num_to_abs, style_to_level, abs_to_num, list_states).strip()
                    if txt:
                        md_lines.append(txt + "\n")
                elif tag == tbl_tag:
                    table_rows = []
                    for row in child.iter(tr_tag):
                        row_cells = []
                        for cell in row.iter(tc_tag):
                            cell_text = " ".join(
                                resolve_paragraph(p, abstract_nums, num_to_abs, style_to_level, abs_to_num, list_states).strip()
                                for p in cell.iter(p_tag)
                            )
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
                    if tbl_tag not in child.tag:
                        for p in child.iter(p_tag):
                            txt = resolve_paragraph(p, abstract_nums, num_to_abs, style_to_level, abs_to_num, list_states).strip()
                            if txt:
                                md_lines.append(txt + "\n")
                                
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
            print(f"Successfully converted {docx_path} to {md_path}")
            
    except Exception as e:
        print(f"Error parsing {docx_path}: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(current_dir, "artifacts")
    
    if not os.path.exists(artifacts_dir):
        artifacts_dir = r"f:\APPS_PJ\mamba-forecast-ad\Hydrid-Mamba-for-Predictive-Bearing-Fault\artifacts"
        
    if not os.path.exists(artifacts_dir):
        artifacts_dir = "/mnt/f/APPS_PJ/mamba-forecast-ad/Hydrid-Mamba-for-Predictive-Bearing-Fault/artifacts"

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
