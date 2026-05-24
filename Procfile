"""
MedAgente — versão Railway (frontend embutido, sem pasta separada)
"""
import os, json, sqlite3, datetime, smtplib, re, math, base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import anthropic

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_OK = True
except ImportError:
    TWILIO_OK = False

# ── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
TWILIO_SID     = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN   = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM    = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
GMAIL_USER     = os.getenv("GMAIL_USER", "")
GMAIL_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
PIXEON_URL     = os.getenv("PIXEON_FHIR_URL", "")
PIXEON_KEY     = os.getenv("PIXEON_SUBSCRIPTION_KEY", "")
PORT           = int(os.getenv("PORT", 8000))

DB_PATH = Path("/tmp/medagente.db")  # Railway usa /tmp para escrita
claude  = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
MODEL   = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """Você é um assistente médico clínico especializado, auxiliando um médico brasileiro com expertise em Cardiologia, Reumatologia e Clínica Médica.

Seu papel durante as consultas:
1. HIPÓTESES DIAGNÓSTICAS: Gere DDx estruturada com probabilidades e raciocínio clínico baseado em evidências.
2. GUIDELINES: Busque e cite diretrizes atuais (ACC/AHA, ACR, EULAR, SBC, SBR). Indique ano e grau de evidência.
3. RECOMENDAÇÕES: Sugira exames, medicações com doses, interações e alertas de segurança.
4. MATERIAIS PARA PACIENTES: Quando solicitado, gere textos em linguagem acessível.
5. EVOLUÇÃO SOAP: Gere rascunhos estruturados prontos para prontuário.
6. CALCULADORAS: Quando mencionar escores, explique o significado clínico do resultado.

Sempre cite a fonte. Seja objetivo e clínico. Nunca tome decisões pelo médico.
Responda SEMPRE em português brasileiro."""

# ── Banco ──────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente TEXT, data TEXT, resumo TEXT, historico TEXT, soap TEXT
        );
        CREATE TABLE IF NOT EXISTS mensagens_enviadas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consulta_id INTEGER, canal TEXT, destinatario TEXT, conteudo TEXT, enviada_em TEXT
        );
        CREATE TABLE IF NOT EXISTS calculos_salvos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consulta_id INTEGER, calculadora TEXT, inputs TEXT, resultado TEXT, salvo_em TEXT
        );
    """)
    con.commit(); con.close()

def db(): return sqlite3.connect(DB_PATH)

def salvar_consulta(paciente, resumo, historico):
    con = db()
    cur = con.cursor()
    cur.execute("INSERT INTO consultas (paciente,data,resumo,historico) VALUES (?,?,?,?)",
        (paciente, datetime.date.today().isoformat(), resumo, json.dumps(historico, ensure_ascii=False)))
    cid = cur.lastrowid; con.commit(); con.close(); return cid

def listar_consultas():
    con = db()
    rows = con.execute("SELECT id,paciente,data,resumo,soap FROM consultas ORDER BY id DESC LIMIT 100").fetchall()
    con.close()
    return [{"id":r[0],"paciente":r[1],"data":r[2],"resumo":r[3],"tem_soap":bool(r[4])} for r in rows]

def carregar_historico(cid):
    con = db()
    row = con.execute("SELECT historico FROM consultas WHERE id=?", (cid,)).fetchone()
    con.close(); return json.loads(row[0]) if row else []

def atualizar_consulta(cid, historico, resumo, soap=None):
    con = db()
    if soap:
        con.execute("UPDATE consultas SET historico=?,resumo=?,soap=? WHERE id=?",
            (json.dumps(historico, ensure_ascii=False), resumo, soap, cid))
    else:
        con.execute("UPDATE consultas SET historico=?,resumo=? WHERE id=?",
            (json.dumps(historico, ensure_ascii=False), resumo, cid))
    con.commit(); con.close()

def salvar_calculo(cid, calc, inputs, resultado):
    con = db()
    con.execute("INSERT INTO calculos_salvos (consulta_id,calculadora,inputs,resultado,salvo_em) VALUES (?,?,?,?,?)",
        (cid, calc, json.dumps(inputs,ensure_ascii=False), json.dumps(resultado,ensure_ascii=False), datetime.datetime.now().isoformat()))
    con.commit(); con.close()

# ── Calculadoras ───────────────────────────────────────────────────────────────
def calc_das28(tj28, ta28, vhs_ou_pcr, eva_paciente, usar_vhs=True):
    if usar_vhs:
        s = 0.56*math.sqrt(tj28)+0.28*math.sqrt(ta28)+0.70*math.log(vhs_ou_pcr)+0.014*eva_paciente
    else:
        s = 0.56*math.sqrt(tj28)+0.28*math.sqrt(ta28)+0.36*math.log(vhs_ou_pcr+1)+0.014*eva_paciente+0.96
    s = round(s,2)
    a = "Remissão" if s<2.6 else "Baixa atividade" if s<3.2 else "Atividade moderada" if s<5.1 else "Alta atividade"
    return {"score":s,"atividade":a,"interpretacao":f"DAS28: {s} → {a}"}

def calc_sledai(campos: dict):
    pesos = {"convulsao":8,"psicose":8,"sindrome_organica_cerebral":8,"perturbacao_visual":8,
             "nervo_craniano":8,"cefaleia_lupica":8,"acidente_vascular":8,"vasculite":8,
             "artrite":4,"miosite":4,"cilindros_urinarios":4,"hematuria":4,"proteinuria":4,
             "piuria":4,"rash_novo":2,"alopecia":2,"ulceras_mucosas":2,"pleurite":2,
             "pericardite":2,"complemento_baixo":2,"anticorpos_dna_aumentado":2,
             "febre":1,"trombocitopenia":1,"leucopenia":1}
    total = sum(pesos.get(k,0) for k,v in campos.items() if v)
    a = "Inativo" if total==0 else "Atividade leve" if total<=4 else "Atividade moderada" if total<=8 else "Atividade grave"
    return {"score":total,"atividade":a,"interpretacao":f"SLEDAI-2K: {total} → {a}"}

def calc_cha2ds2(idade, sexo_feminino, hf_ou_avc, dm, has, ic_disfuncao, doenca_vascular):
    s = 0
    if idade>=75: s+=2
    elif idade>=65: s+=1
    if sexo_feminino: s+=1
    if hf_ou_avc: s+=2
    if dm: s+=1
    if has: s+=1
    if ic_disfuncao: s+=1
    if doenca_vascular: s+=1
    tab={0:0.0,1:1.3,2:2.2,3:3.2,4:4.0,5:6.7,6:9.8,7:9.6,8:12.5,9:15.2}
    r=tab.get(min(s,9),15.2)
    if s==0 or (s==1 and sexo_feminino): rec="Anticoagulação não recomendada"
    elif s==1 and not sexo_feminino: rec="Considerar anticoagulação"
    else: rec="Anticoagulação oral recomendada (NOAC preferível)"
    return {"score":s,"risco_anual_pct":r,"recomendacao":rec,"interpretacao":f"CHA₂DS₂-VASc: {s} → risco {r}%/ano → {rec}"}

def calc_has_bled(has,funcao_renal_hepatica,avc,sangramento_historico,inr_instavel,idoso,drogas_alcool):
    s=int(has)+int(funcao_renal_hepatica)+int(avc)+int(sangramento_historico)+int(inr_instavel)+int(idoso)+int(drogas_alcool)
    r="Baixo" if s<=1 else "Moderado" if s==2 else "Alto (≥3%/ano)"
    return {"score":s,"risco":r,"nota":"Score ≥3: corrigir fatores modificáveis, não contraindica anticoag.","interpretacao":f"HAS-BLED: {s} → {r}"}

def calc_wells_tep(sintomas_tvp,dx_alternativo_menos_provavel,fc_maior_100,imobilizacao_cirurgia,tvp_tep_previo,hemoptise,cancer):
    s=3.0*int(sintomas_tvp)+3.0*int(dx_alternativo_menos_provavel)+1.5*int(fc_maior_100)+1.5*int(imobilizacao_cirurgia)+1.5*int(tvp_tep_previo)+int(hemoptise)+int(cancer)
    p="Baixa (~1.3%)" if s<=1 else "Intermediária (~16%)" if s<=6 else "Alta (~41%)"
    return {"score":s,"probabilidade":p,"conduta":"Score ≤4: D-dímero. Score >4: AngioTC direta.","interpretacao":f"Wells TEP: {s} → {p}"}

def calc_wells_tvp(cancer,paralisia_paresia,acamado_cirurgia,dor_trajeto_venoso,edema_toda_perna,edema_assimetrico,edema_cacifo,circulacao_colateral,tvp_previa,dx_alternativo_igualmente_provavel):
    s=int(cancer)+int(paralisia_paresia)+int(acamado_cirurgia)+int(dor_trajeto_venoso)+int(edema_toda_perna)+int(edema_assimetrico)+int(edema_cacifo)+int(circulacao_colateral)+int(tvp_previa)+(-2)*int(dx_alternativo_igualmente_provavel)
    p="Baixa (~3%)" if s<=0 else "Intermediária (~17%)" if s<=2 else "Alta (~75%)"
    return {"score":s,"probabilidade":p,"interpretacao":f"Wells TVP: {s} → {p}"}

def calc_score2(idade,sexo_masculino,pas,colesterol_nao_hdl,fumante,dm2,regiao="europa_baixo"):
    b=15 if idade>=70 else 8 if idade>=60 else 4 if idade>=50 else 2
    if sexo_masculino: b*=1.3
    if pas>=160: b*=2.0
    elif pas>=140: b*=1.5
    if fumante: b*=1.6
    if dm2: b*=1.4
    if colesterol_nao_hdl>=5.7: b*=1.3
    b=round(min(b,50),1)
    if idade<50: r="Baixo" if b<2.5 else "Moderado" if b<7.5 else "Alto" if b<10 else "Muito alto"
    else: r="Baixo" if b<5 else "Moderado" if b<10 else "Alto" if b<15 else "Muito alto"
    return {"risco_estimado_pct":b,"categoria":r,"nota":"Estimativa — use tabelas ESC 2021 para decisão final.","interpretacao":f"SCORE2: ~{b}% → {r}"}

def calc_framingham(idade,sexo_masculino,colesterol_total,hdl,pas,tratamento_has,fumante,dm):
    pts=0
    if sexo_masculino:
        pts+=[−1,0,1,2,3,4,5,6,7][min(max(0,int((idade-30)/5)),8)] if idade>=30 else -1
    else:
        base=[-9,-4,0,3,6,7,8,8,8]
        pts+=base[min(max(0,int((idade-30)/5)),8)] if idade>=30 else -9
    if colesterol_total<4.7: pts+=0
    elif colesterol_total<5.7: pts+=1
    elif colesterol_total<6.7: pts+=2
    else: pts+=3
    if hdl>=1.6: pts-=2
    elif hdl>=1.3: pts-=1
    elif hdl<0.9: pts+=2
    elif hdl<1.2: pts+=1
    if pas>=160: pts+=3
    elif pas>=140: pts+=(2 if not tratamento_has else 2)
    elif pas>=130: pts+=(1 if not tratamento_has else 2)
    if dm: pts+=(3 if sexo_masculino else 4)
    if fumante: pts+=2
    tab_m={-3:1,-2:2,-1:2,0:3,1:4,2:4,3:6,4:7,5:9,6:11,7:14,8:18,9:22,10:27,11:33,12:40,13:47}
    tab_f={-2:1,-1:2,0:2,1:2,2:3,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:11,11:13,12:15,13:17,14:20,15:24,16:27}
    tab=tab_m if sexo_masculino else tab_f
    r=tab.get(pts,47 if pts>13 else 1)
    c="Baixo" if r<10 else "Intermediário" if r<20 else "Alto"
    return {"pontos":pts,"risco_10anos_pct":r,"categoria":c,"interpretacao":f"Framingham: {pts}pts → {r}% em 10 anos → {c}"}

CALCULADORAS={"das28":calc_das28,"sledai":calc_sledai,"cha2ds2":calc_cha2ds2,
              "has_bled":calc_has_bled,"wells_tep":calc_wells_tep,"wells_tvp":calc_wells_tvp,
              "score2":calc_score2,"framingham":calc_framingham}

# ── Ferramentas ────────────────────────────────────────────────────────────────
TOOLS=[
    {"type":"web_search_20250305","name":"web_search"},
    {"name":"calcular_escore","description":"Executa calculadora clínica. Disponíveis: das28, sledai, cha2ds2, has_bled, wells_tep, wells_tvp, score2, framingham",
     "input_schema":{"type":"object","properties":{"calculadora":{"type":"string"},"parametros":{"type":"object"}},"required":["calculadora","parametros"]}},
    {"name":"gerar_material_paciente","description":"Gera material educativo para o paciente em linguagem acessível",
     "input_schema":{"type":"object","properties":{"titulo":{"type":"string"},"conteudo":{"type":"string"},"tipo":{"type":"string"}},"required":["titulo","conteudo","tipo"]}},
    {"name":"gerar_soap","description":"Gera evolução SOAP estruturada para prontuário",
     "input_schema":{"type":"object","properties":{"subjetivo":{"type":"string"},"objetivo":{"type":"string"},"avaliacao":{"type":"string"},"plano":{"type":"string"}},"required":["subjetivo","objetivo","avaliacao","plano"]}}
]

_s={}

def handle_tool(name,inputs,cid):
    if name=="calcular_escore":
        fn=CALCULADORAS.get(inputs["calculadora"])
        if not fn: return f"Calculadora '{inputs['calculadora']}' não encontrada."
        try:
            res=fn(**inputs["parametros"]); salvar_calculo(cid,inputs["calculadora"],inputs["parametros"],res)
            if "calculos" not in _s: _s["calculos"]=[]
            _s["calculos"].append({"calc":inputs["calculadora"],"resultado":res})
            return json.dumps(res,ensure_ascii=False)
        except TypeError as e: return f"Parâmetros incorretos: {e}"
    if name=="gerar_material_paciente":
        _s["material"]=inputs; return f"Material gerado: '{inputs['titulo']}'"
    if name=="gerar_soap":
        soap=f"EVOLUÇÃO MÉDICA — {datetime.date.today().strftime('%d/%m/%Y')}\n\nS — SUBJETIVO\n{inputs['subjetivo']}\n\nO — OBJETIVO\n{inputs['objetivo']}\n\nA — AVALIAÇÃO\n{inputs['avaliacao']}\n\nP — PLANO\n{inputs['plano']}"
        _s["soap"]=soap
        con=db(); con.execute("UPDATE consultas SET soap=? WHERE id=?",(soap,cid)); con.commit(); con.close()
        return soap
    return "Ferramenta desconhecida."

def run_agent(historico,msg,cid):
    _s.clear(); historico.append({"role":"user","content":msg})
    while True:
        resp=claude.messages.create(model=MODEL,max_tokens=4096,system=SYSTEM_PROMPT,tools=TOOLS,messages=historico)
        historico.append({"role":"assistant","content":resp.content})
        if resp.stop_reason=="end_turn":
            return "\n".join(b.text for b in resp.content if hasattr(b,"text")),historico
        results=[]
        for b in resp.content:
            if b.type=="tool_use" and b.name!="web_search":
                results.append({"type":"tool_result","tool_use_id":b.id,"content":handle_tool(b.name,b.input,cid)})
        if results: historico.append({"role":"user","content":results})
        if resp.stop_reason!="tool_use":
            return "\n".join(b.text for b in resp.content if hasattr(b,"text")),historico

def enviar_whatsapp(numero,msg):
    if not TWILIO_OK: return "Erro: twilio não instalado"
    if not TWILIO_SID: return "Erro: TWILIO_ACCOUNT_SID não configurado"
    try:
        tc=TwilioClient(TWILIO_SID,TWILIO_TOKEN); n=re.sub(r'\D','',numero)
        if not n.startswith('55'): n='55'+n
        m=tc.messages.create(body=msg,from_=TWILIO_FROM,to=f"whatsapp:+{n}")
        return f"✅ WhatsApp enviado (SID: {m.sid})"
    except Exception as e: return f"Erro: {e}"

def enviar_email(dest,assunto,corpo):
    if not GMAIL_USER: return "Erro: GMAIL_USER não configurado"
    try:
        msg=MIMEMultipart("alternative"); msg["Subject"]=assunto; msg["From"]=GMAIL_USER; msg["To"]=dest
        msg.attach(MIMEText(corpo,"plain","utf-8"))
        msg.attach(MIMEText(f"<html><body style='font-family:Arial;font-size:15px'>{corpo.replace(chr(10),'<br>')}</body></html>","html","utf-8"))
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s:
            s.login(GMAIL_USER,GMAIL_PASSWORD); s.sendmail(GMAIL_USER,dest,msg.as_string())
        return f"✅ E-mail enviado para {dest}"
    except Exception as e: return f"Erro: {e}"

def gerar_pdf(soap,paciente):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer
        import io
        buf=io.BytesIO()
        doc=SimpleDocTemplate(buf,pagesize=A4,topMargin=2*cm,bottomMargin=2*cm,leftMargin=2.5*cm,rightMargin=2.5*cm)
        styles=getSampleStyleSheet()
        story=[
            Paragraph(f"Evolução Médica — {paciente}",ParagraphStyle('t',parent=styles['Heading1'],fontSize=14)),
            Paragraph(datetime.date.today().strftime('%d/%m/%Y'),styles['Normal']),
            Spacer(1,.4*cm),
            Paragraph(soap.replace('\n','<br/>'),ParagraphStyle('b',parent=styles['Normal'],fontSize=10,leading=16,fontName='Courier')),
        ]
        doc.build(story); return buf.getvalue()
    except: return soap.encode('utf-8')

# ── FastAPI ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    init_db(); yield

app=FastAPI(title="MedAgente",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

fp=Path(__file__).parent/"frontend"
if fp.exists(): app.mount("/app",StaticFiles(directory=str(fp),html=True),name="frontend")

class NovaConsulta(BaseModel): paciente:str
class Chat(BaseModel): consulta_id:int; mensagem:str
class Wpp(BaseModel): consulta_id:int; numero:str; mensagem:str
class Email(BaseModel): consulta_id:int; destinatario:str; assunto:str; mensagem:str
class Calc(BaseModel): consulta_id:int; calculadora:str; parametros:dict
class PixeonEnviar(BaseModel): consulta_id:int; paciente_id:str; crm:Optional[str]=""

@app.get("/")
def root(): return HTMLResponse('<meta http-equiv="refresh" content="0;url=/app/"/>')

@app.post("/consultas/nova")
def nova(b:NovaConsulta):
    cid=salvar_consulta(b.paciente,f"Consulta de {b.paciente}",[])
    return {"consulta_id":cid,"paciente":b.paciente}

@app.get("/consultas")
def consultas(): return listar_consultas()

@app.get("/consultas/{cid}")
def get_consulta(cid:int):
    con=db()
    row=con.execute("SELECT id,paciente,data,resumo,soap FROM consultas WHERE id=?",(cid,)).fetchone()
    calcs=[{"id":r[0],"calculadora":r[1],"inputs":json.loads(r[2]),"resultado":json.loads(r[3]),"salvo_em":r[4]}
           for r in con.execute("SELECT id,calculadora,inputs,resultado,salvo_em FROM calculos_salvos WHERE consulta_id=? ORDER BY id DESC",(cid,)).fetchall()]
    con.close()
    if not row: raise HTTPException(404,"Consulta não encontrada")
    return {"id":row[0],"paciente":row[1],"data":row[2],"resumo":row[3],"soap":row[4],"calculos":calcs,"historico":carregar_historico(cid)}

@app.post("/chat")
def chat(b:Chat):
    hist=carregar_historico(b.consulta_id)
    resp,hist2=run_agent(hist,b.mensagem,b.consulta_id)
    atualizar_consulta(b.consulta_id,hist2,b.mensagem[:80],_s.get("soap"))
    return {"resposta":resp,"material_gerado":_s.get("material"),"soap":_s.get("soap"),"calculos":_s.get("calculos",[])}

@app.post("/calcular")
def calcular(b:Calc):
    fn=CALCULADORAS.get(b.calculadora)
    if not fn: raise HTTPException(400,f"Calculadora '{b.calculadora}' não existe")
    try:
        res=fn(**b.parametros); salvar_calculo(b.consulta_id,b.calculadora,b.parametros,res); return res
    except TypeError as e: raise HTTPException(422,str(e))

@app.get("/evolucao/pdf/{cid}")
def pdf(cid:int):
    con=db(); row=con.execute("SELECT paciente,soap FROM consultas WHERE id=?",(cid,)).fetchone(); con.close()
    if not row or not row[1]: raise HTTPException(404,"Evolução não encontrada")
    data=gerar_pdf(row[1],row[0])
    ct="application/pdf" if data[:4]==b"%PDF" else "text/plain"
    return Response(content=data,media_type=ct,headers={"Content-Disposition":f"attachment; filename=evolucao_{cid}.pdf"})

@app.post("/enviar/whatsapp")
def wpp(b:Wpp):
    res=enviar_whatsapp(b.numero,b.mensagem)
    con=db(); con.execute("INSERT INTO mensagens_enviadas VALUES (NULL,?,?,?,?,?)",(b.consulta_id,"whatsapp",b.numero,b.mensagem,datetime.datetime.now().isoformat())); con.commit(); con.close()
    return {"resultado":res}

@app.post("/enviar/email")
def email(b:Email):
    res=enviar_email(b.destinatario,b.assunto,b.mensagem)
    con=db(); con.execute("INSERT INTO mensagens_enviadas VALUES (NULL,?,?,?,?,?)",(b.consulta_id,"email",b.destinatario,b.mensagem,datetime.datetime.now().isoformat())); con.commit(); con.close()
    return {"resultado":res}

@app.get("/health")
def health(): return {"status":"ok","versao":"2.0"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="0.0.0.0",port=PORT,reload=False)
