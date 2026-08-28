import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# 1. 제목 및 저자 (직접 입력하도록 자리 확보)
doc.add_heading('Modeling Personalized Digital Bedtime Routines for Sleep and Stress Prediction Using Multimodal Lifelogs', 0)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Heeyoung Jung\nDept. of AI Big Data, Soonchunhyang University\nAsan, South Korea\njung1222@sch.ac.kr").bold = True

# 2. Abstract
doc.add_heading('Abstract', level=1)
doc.add_paragraph("In this paper, we evaluate the conventional 'Digital Bedtime Routine Window' (DBRW) by modeling personalized digital routines using multimodal lifelogs. Our analysis reveals that macroscopic physical environment variables are not the primary predictors of sleep and stress. Instead, contextual network footprints and reward-seeking application usage exhibit superior predictive power.")

# 3. I. INTRODUCTION ~ III. METHOD (이하 상세 내용 삽입)
doc.add_heading('I. INTRODUCTION', level=1)
doc.add_paragraph("The proliferation of multimodal lifelogs offers unprecedented opportunities to predict sleep and stress. Our contributions: 1) Quantifying DBRW efficacy, 2) Robust ExtraTrees framework, 3) Evidence that dopaminergic digital behaviors are critical indicators.")

doc.add_heading('III. METHOD', level=1)
doc.add_paragraph("A. Data Collection\nWe utilize hardware states as baselines. Missing values are imputed as zero.\nB. Predictive Modeling\nWe deployed ExtraTrees (n=100, depth=10). We applied calibration: P' = αP + (1 - α)μ.")

# 4. IV. EXPERIMENTS (표 자동 삽입)
doc.add_heading('IV. EXPERIMENTS', level=1)
t = doc.add_table(rows=2, cols=6)
t.style = 'Light Shading Accent 1'
data = [["Dataset", "Instances", "Features", "Labels", "Validation", "Metric"], ["Multimodal", "450", "12", "7", "Subject-wise", "Log-Loss"]]
for r in range(2):
    for c in range(6): t.rows[r].cells[c].text = data[r][c]

# 5. V. RESULTS (표 자동 삽입 - 결과 파일이 있는 경우)
doc.add_heading('V. RESULTS', level=1)
if os.path.exists('table_result.csv'):
    df = pd.read_csv('table_result.csv')
    table = doc.add_table(rows=1, cols=len(df.columns))
    for i, col in enumerate(df.columns): table.cell(0, i).text = col
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row): cells[i].text = str(val)

# 6. 결론, 사사, 레퍼런스
doc.add_heading('VII. CONCLUSION', level=1)
doc.add_paragraph("1) Physical variables are not primary. 2) Contextual features outperform hardware. 3) Digital behaviors are critical.")

doc.add_heading('ACKNOWLEDGMENT', level=5)
doc.add_paragraph("This research was conducted as part of the 5th ETRI Human Understanding AI Paper Challenge (IWETRIAI).")

doc.add_heading('REFERENCES', level=5)
refs = [
    "[1] A. Sano and R. W. Picard, \"Stress recognition using wearable sensors...\", 2013.",
    "[2] R. Wang et al., \"StudentLife: Assessing mental health...\", 2014.",
    "[3] L. Lund et al., \"Machine learning for the prediction of sleep quality...\", 2020.",
    "[4] Y. Chen et al., \"Understanding the relationship...\", 2020.",
    "[5] J. Park and H. Kim, \"Feature extraction and tree-based ensemble...\", 2021."
]
for r in refs: doc.add_paragraph(r)

doc.save('final_submit.docx')