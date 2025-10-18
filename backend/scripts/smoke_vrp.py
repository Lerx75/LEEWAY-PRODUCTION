import io, time, json, requests, pandas as pd
from pathlib import Path

# Prepare tiny datasets
calls = pd.DataFrame({
    'Postcode': ['SW1A 1AA','EC1A 1BB','W1A 0AX'],
    'Duration (mins)': [30, 45, 20],
    'Days': ['YYYYYNN','YYYYYNN','YYYYYNN'],
})
resources = pd.DataFrame({
    'Name': ['Tech 1'],
    'Postcode': ['SW1A 2AA'],
    'Days': ['YYYYYNN'],
    'Start': ['09:00'],
    'End': ['17:00'],
})

outdir = Path('testdata'); outdir.mkdir(parents=True, exist_ok=True)
(calls).to_excel(outdir/'calls.xlsx', index=False)
(resources).to_excel(outdir/'resources.xlsx', index=False)

files = {
    'callsFile': ('calls.xlsx', open(outdir/'calls.xlsx','rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    'resourcesFile': ('resources.xlsx', open(outdir/'resources.xlsx','rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
}
form = {'projectName': 'dev-smoke'}
headers = {'X-Service-Token': 'dev-project-service-token','X-User-Id': 'test-user'}

resp = requests.post('http://localhost:8000/api/vehicle-route', files=files, data=form, headers=headers, timeout=180)
print('SUBMIT', resp.status_code)
print(resp.text)
if resp.status_code not in (200,202):
    raise SystemExit(1)
job = resp.json(); status_url = job.get('status_url'); result_url = job.get('result_url')
for _ in range(90):
    s = requests.get(status_url, timeout=30).json()
    if s.get('ready') and s.get('state') == 'SUCCESS':
        r = requests.get(result_url, timeout=60).json()
        print('RESULT', json.dumps({'state': r.get('state'), 'msg': r.get('result',{}).get('message'), 'unscheduled': r.get('result',{}).get('unscheduled_count')}, indent=2))
        break
    time.sleep(2)
else:
    print('Timed out waiting for job')
    raise SystemExit(2)
