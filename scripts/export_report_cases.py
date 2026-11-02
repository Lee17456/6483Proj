"""Export the six original report cases on the server, preserving their source IDs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image

CASES=[('val','dog','dog.8507','val_dog_8507'),('val','dog','dog.2199','val_dog_2199'),
       ('test',None,'2','test_2'),('test',None,'5','test_5'),
       ('test',None,'428','test_428'),('test',None,'244','test_244')]
CAPTIONS=['Validation dog.8507: true dog; predicted cat.',
          'Validation dog.2199: true dog; predicted dog.',
          'Test 2: predicted cat; confidence 1.000.',
          'Test 5: predicted dog; confidence 1.000.',
          'Test 428: predicted dog; confidence 0.556.',
          'Test 244: predicted dog; confidence 0.686.']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('report_case_export'))
    a=p.parse_args()
    root=a.data_dir.resolve(); dest=a.output.resolve()/'cases'
    sources=[]
    for split,cls,ident,stem in CASES:
        folder=root/split if cls is None else root/split/cls
        matches=[x for x in folder.iterdir() if x.is_file() and x.stem==ident and x.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')]
        if len(matches)!=1: raise ValueError(f'{folder}: expected exactly one image with original ID {ident}, found {len(matches)}')
        sources.append(matches[0])
    dest.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for (_,_,ident,stem),src in zip(CASES,sources):
        with Image.open(src) as im:
            rgb=im.convert('RGB'); rgb.save(dest/(stem+'.png'))
            size=list(rgb.size)
        manifest.append({'original_id':ident,'source_path':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'export':stem+'.png','size':size})
    tex=[r'\begin{figure}[!tbp]',r'\centering']
    for i,(case,caption) in enumerate(zip(CASES,CAPTIONS)):
        tex.extend([r'\begin{minipage}[t]{0.31\linewidth}',r'\centering',
                    r'\includegraphics[width=\linewidth,height=30mm,keepaspectratio]{figures/cases/'+case[3]+'.png}',
                    r'\par\scriptsize '+f'({chr(97+i)}) '+caption,r'\end{minipage}'])
        tex.append(r'\par\medskip' if i==2 else (r'\hfill' if i not in (5,) else ''))
    tex.extend([r'\caption{Original validation and test images corresponding to Tables~\ref{tab:cases} and~\ref{tab:test-cases}. Predictions and confidence refer to pretrained ResNet18, seed 44. Only the validation cases have supplied ground-truth labels. Images are displayed at their original aspect ratios.}',r'\label{fig:case-images}',r'\end{figure}'])
    (dest/'panel.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    note='Inspect the original test images before assigning correctness. Record the visible animal, ambiguity, cropping/occlusion/background observations and the reviewer. Do not revise the trained model or threshold from these cases.\n'
    (dest/'VISUAL_REVIEW.txt').write_text(note,encoding='utf-8')
    archive=a.output.resolve()/'report_case_images.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for f in sorted(dest.iterdir()): z.write(f,'cases/'+f.name)
    print(archive)
    print('Download this small ZIP, then put cases/ inside the LaTeX figures/ directory.')
if __name__=='__main__': main()
