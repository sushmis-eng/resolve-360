import os, json, sqlite3, uuid, re, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parent.parent
DB=Path(os.getenv('DATABASE_PATH',str(ROOT/'data'/'escalations.db')))
RULES=json.loads((ROOT/'data'/'routing_rules.json').read_text())
FAQ=json.loads((ROOT/'data'/'banking_faq.json').read_text())
app=FastAPI(title='Customer Escalation AI',version='1.0')
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('CORS_ORIGINS','http://localhost:5173').split(','),allow_methods=['*'],allow_headers=['*'])

@contextmanager
def conn():
 DB.parent.mkdir(parents=True,exist_ok=True)
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row
 try: yield c; c.commit()
 finally:c.close()

def now():return datetime.now(timezone.utc).isoformat()
def init():
 with conn() as c:
  c.executescript('''CREATE TABLE IF NOT EXISTS interactions(id TEXT PRIMARY KEY,customer_id TEXT NOT NULL,channel TEXT NOT NULL,complaint TEXT NOT NULL,transaction_ref TEXT,created_at TEXT NOT NULL,analysis TEXT NOT NULL,duplicate_of TEXT,decision TEXT NOT NULL,case_id TEXT);CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY,interaction_id TEXT NOT NULL,team TEXT NOT NULL,status TEXT NOT NULL,priority TEXT NOT NULL,sla_deadline TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,interaction_id TEXT NOT NULL,event_type TEXT NOT NULL,details TEXT NOT NULL,created_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS notifications(id TEXT PRIMARY KEY,interaction_id TEXT NOT NULL,channel TEXT NOT NULL,message TEXT NOT NULL,created_at TEXT NOT NULL);''')
init()

class Submission(BaseModel):
 customer_id:str=Field(min_length=2,max_length=80)
 channel:str=Field(pattern='^(WhatsApp|Email|Mobile App)$')
 complaint:str=Field(min_length=10,max_length=4000)
 transaction_ref:str|None=None
class Update(BaseModel):status:str=Field(pattern='^(Open|In Progress|Resolved)$')

def tokenize(s):return set(re.findall(r'[a-z0-9]+',s.lower()))-{'i','my','the','a','for','was','is','but','and','to','of','on','in'}
def cosine(a,b):
 x,y=tokenize(a),tokenize(b)
 return len(x&y)/math.sqrt(len(x)*len(y)) if x and y else 0

def semantic_scores(text, candidates):
 # Real sentence-transformer embeddings when installed; deterministic lexical fallback is clearly labeled.
 try:
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer(os.getenv('EMBEDDING_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
  vectors=model.encode([text]+[r['complaint'] for r in candidates],normalize_embeddings=True)
  return [(float(vectors[0]@vectors[i+1]),'embedding') for i in range(len(candidates))]
 except (ImportError,OSError):return [(cosine(text,r['complaint']),'lexical_fallback') for r in candidates]

def triage(text):
 key=os.getenv('OPENAI_API_KEY')
 if key:
  try:
   from openai import OpenAI
   client=OpenAI(api_key=key,timeout=20)
   response=client.chat.completions.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'),messages=[{'role':'system','content':'Classify a banking complaint. Return only JSON with keys category (Transaction Dispute, Account Services, Card Services, Other), product (Credit Card, Debit Card, Savings Account, Other), sentiment (Negative, Neutral, Positive), severity (Low, Medium, High, Critical), confidence (number 0 to 1), explanation (short string). Financial loss or failed debit is High. Do not promise refunds.'},{'role':'user','content':text}],response_format={'type':'json_object'},temperature=0)
   obj=json.loads(response.choices[0].message.content)
   if obj['category'] not in {'Transaction Dispute','Account Services','Card Services','Other'} or obj['severity'] not in {'Low','Medium','High','Critical'}:raise ValueError('Invalid classification')
   obj['confidence']=max(0,min(1,float(obj['confidence'])));obj['source']='llm';return obj
  except Exception as e:
   raise HTTPException(503,detail=f'LLM triage unavailable: {type(e).__name__}. Check credentials and connectivity.')
 # Explicit demo fallback, never misrepresented as LLM.
 t=text.lower();dispute=any(w in t for w in ['failed','deduct','debited','charged','unsuccessful','refund','fraud','unauthoriz']);statement='statement' in t or 'download' in t
 cat='Transaction Dispute' if dispute else 'Account Services' if statement else 'Card Services' if 'card' in t else 'Other'
 return {'category':cat,'product':'Credit Card' if 'credit card' in t else 'Debit Card' if 'debit card' in t else 'Savings Account' if 'account' in t else 'Other','sentiment':'Negative' if dispute or 'problem' in t else 'Neutral','severity':'High' if dispute else 'Low' if statement else 'Medium','confidence':0.65,'explanation':'Rule-based demo fallback; configure OPENAI_API_KEY for live LLM classification.','source':'rules_fallback'}

def event(c,i,kind,details):c.execute('INSERT INTO events VALUES (?,?,?,?,?)',(str(uuid.uuid4()),i,kind,json.dumps(details),now()))

@app.post('/interactions',status_code=201)
def submit(payload:Submission):
 analysis=triage(payload.complaint);iid='INT-'+uuid.uuid4().hex[:10].upper()
 with conn() as c:
  rows=c.execute('SELECT * FROM interactions WHERE customer_id=? ORDER BY created_at DESC LIMIT 100',(payload.customer_id,)).fetchall()
  matches=[]
  for row,(score,method) in zip(rows,semantic_scores(payload.complaint,rows)):
   ref_match=bool(payload.transaction_ref and row['transaction_ref'] and payload.transaction_ref.strip().lower()==row['transaction_ref'].strip().lower())
   if ref_match or score>=float(os.getenv('DUPLICATE_THRESHOLD','0.70')):
    matches.append({'interaction_id':row['id'],'similarity':round(score,3),'method':method,'transaction_reference_match':ref_match,'case_id':row['case_id']})
  matches.sort(key=lambda m:(m['transaction_reference_match'],m['similarity']),reverse=True)
  duplicate=matches[0] if matches else None
  severity=analysis['severity'];category=analysis['category'];rule=RULES['categories'].get(category,RULES['default'])
  faq_hits=[f for f in FAQ if all(word in payload.complaint.lower() for word in f['keywords'])]
  eligible=severity in RULES['auto_resolution_severities'] and bool(faq_hits) and not duplicate and analysis['confidence']>=RULES['minimum_auto_confidence']
  decision='Auto-Resolved' if eligible else 'Escalated'
  case_id=None
  if not eligible:
   # Link to an existing case without creating a second case.
   case_id=duplicate['case_id'] if duplicate and duplicate['case_id'] else 'CASE-'+uuid.uuid4().hex[:10].upper()
   if not duplicate or not duplicate['case_id']:
    deadline=(datetime.now(timezone.utc)+timedelta(hours=RULES['sla_hours'][severity])).isoformat()
    c.execute('INSERT INTO cases VALUES (?,?,?,?,?,?,?,?)',(case_id,iid,rule['team'],'Open',severity,deadline,now(),now()))
  analysis['duplicate_matches']=matches
  analysis['decision_reason']='Approved FAQ matched and automation eligibility passed' if eligible else 'Probable duplicate linked to existing case' if duplicate and duplicate['case_id'] else 'Human review required by severity, confidence or knowledge-base policy'
  if eligible:analysis['approved_answer']=faq_hits[0]['answer'];analysis['faq_id']=faq_hits[0]['id']
  c.execute('INSERT INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?)',(iid,payload.customer_id,payload.channel,payload.complaint,payload.transaction_ref,now(),json.dumps(analysis),duplicate['interaction_id'] if duplicate else None,decision,case_id))
  event(c,iid,'Submitted',{'channel':payload.channel})
  event(c,iid,'AI Triage',{'category':category,'severity':severity,'source':analysis['source']})
  event(c,iid,'Duplicate Check',{'matches':matches})
  event(c,iid,decision,{'case_id':case_id,'reason':analysis['decision_reason']})
  message=analysis.get('approved_answer') if eligible else f'Your complaint {iid} is registered. Case reference: {case_id}. Our {rule["team"]} team will review it.'
  c.execute('INSERT INTO notifications VALUES (?,?,?,?,?)',(str(uuid.uuid4()),iid,payload.channel,message,now()))
 return detail(iid)

def detail(iid):
 with conn() as c:
  row=c.execute('SELECT * FROM interactions WHERE id=?',(iid,)).fetchone()
  if not row:raise HTTPException(404,'Interaction not found')
  d=dict(row);d['analysis']=json.loads(d['analysis'])
  d['case']=dict(c.execute('SELECT * FROM cases WHERE id=?',(d['case_id'],)).fetchone()) if d['case_id'] else None
  if d['case']:d['case']['sla_breached']=d['case']['status']!='Resolved' and datetime.fromisoformat(d['case']['sla_deadline'])<datetime.now(timezone.utc)
  d['events']=[dict(e) for e in c.execute('SELECT * FROM events WHERE interaction_id=? ORDER BY created_at',(iid,))]
  d['notifications']=[dict(n) for n in c.execute('SELECT * FROM notifications WHERE interaction_id=? ORDER BY created_at',(iid,))]
  return d

@app.get('/interactions')
def interactions():
 with conn() as c:ids=[r['id'] for r in c.execute('SELECT id FROM interactions ORDER BY created_at DESC LIMIT 100')]
 return [detail(i) for i in ids]
@app.get('/interactions/{iid}')
def interaction(iid:str):return detail(iid)
@app.get('/dashboard')
def dashboard():
 with conn() as c:
  total=c.execute('SELECT COUNT(*) FROM interactions').fetchone()[0];cases=c.execute('SELECT COUNT(*) FROM cases').fetchone()[0];resolved=c.execute("SELECT COUNT(*) FROM interactions WHERE decision='Auto-Resolved'").fetchone()[0];duplicates=c.execute('SELECT COUNT(*) FROM interactions WHERE duplicate_of IS NOT NULL').fetchone()[0]
  breached=c.execute("SELECT COUNT(*) FROM cases WHERE status!='Resolved' AND sla_deadline<?",(now(),)).fetchone()[0]
 return {'total_interactions':total,'cases':cases,'auto_resolved':resolved,'duplicates':duplicates,'sla_breached':breached}
@app.patch('/cases/{case_id}')
def update_case(case_id:str,payload:Update):
 with conn() as c:
  case=c.execute('SELECT * FROM cases WHERE id=?',(case_id,)).fetchone()
  if not case:raise HTTPException(404,'Case not found')
  c.execute('UPDATE cases SET status=?,updated_at=? WHERE id=?',(payload.status,now(),case_id))
  event(c,case['interaction_id'],'Case Status Updated',{'status':payload.status})
 return {'case_id':case_id,'status':payload.status}
@app.get('/health')
def health():return {'status':'ok','llm_configured':bool(os.getenv('OPENAI_API_KEY'))}

# Serve the built React SPA and API from the same origin on Hugging Face Spaces.
from fastapi.staticfiles import StaticFiles
if (ROOT / "frontend" / "dist").is_dir():
 app.mount("/", StaticFiles(directory=str(ROOT / "frontend" / "dist"), html=True), name="frontend")
